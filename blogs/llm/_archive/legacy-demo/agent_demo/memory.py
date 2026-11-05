"""
§02 - Memory 记忆系统

Five-level memory architecture:
  L0 — Working (sliding window of recent messages)
  L1 — Session (summary of current conversation)
  L2 — Long-Term (persisted facts, preferences, knowledge)
  L3 — Episodic (event-based memories with importance scores)
  L4 — Vector DB (semantic retrieval via embeddings)

Features:
  - Sliding window with configurable size
  - Importance-based filtering & decay
  - Semantic retrieval (keyword fallback, vector-ready interface)
  - Consolidation pipeline (short → long-term)
  - Persistence (JSON-based, pluggable backend)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ──────────────────────────────────────────────
# 1. Memory Entry Model
# ──────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """单条记忆。"""

    content: str
    role: str = "user"
    importance: float = 0.5  # 0.0 – 1.0
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def touch(self):
        """Mark as recently accessed."""
        self.accessed_at = time.time()
        self.access_count += 1

    def decay(self, half_life_hours: float = 24.0) -> float:
        """Time-based importance decay. Returns current effective importance."""
        age_hours = (time.time() - self.created_at) / 3600
        decay_factor = 0.5 ** (age_hours / half_life_hours)
        self.importance = max(0.01, self.importance * decay_factor)
        return self.importance

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "role": self.role,
            "importance": self.importance,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "tags": self.tags,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            content=d["content"],
            role=d.get("role", "user"),
            importance=d.get("importance", 0.5),
            created_at=d.get("created_at", time.time()),
            accessed_at=d.get("accessed_at", time.time()),
            access_count=d.get("access_count", 0),
            tags=d.get("tags", []),
            meta=d.get("meta", {}),
        )


# ──────────────────────────────────────────────
# 2. Importance Estimator
# ──────────────────────────────────────────────

class ImportanceEstimator:
    """估算记忆的重要性分数。"""

    # Heuristic importance signals
    IMPORTANCE_KEYWORDS = {
        "记住": 0.9, "remember": 0.9, "重要": 0.85, "important": 0.85,
        "关键": 0.8, "critical": 0.8, "不要": 0.75, "never": 0.75,
        "总是": 0.7, "always": 0.7, "偏好": 0.7, "preference": 0.7,
        "喜欢": 0.65, "讨厌": 0.65, "name": 0.6, "名字": 0.6,
    }

    def estimate(self, content: str, role: str = "user") -> float:
        """估算内容的重要性。"""
        score = 0.5  # base score

        # Keyword matching
        lower = content.lower()
        for kw, weight in self.IMPORTANCE_KEYWORDS.items():
            if kw.lower() in lower:
                score = max(score, weight)
                break

        # Role bonus: user memories > assistant memories
        if role == "user":
            score = min(1.0, score + 0.05)

        # Length bonus: very short messages are less important
        if len(content) < 10:
            score *= 0.7
        elif len(content) > 200:
            score = min(1.0, score + 0.1)

        return round(score, 2)


# ──────────────────────────────────────────────
# 3. Vector Index (semantic retrieval)
# ──────────────────────────────────────────────

class KeywordVectorIndex:
    """
    关键词向量索引 — 轻量语义检索。

    In production, replace this with FAISS / Chroma / pgvector.
    This implementation uses keyword hashing as a simple
    semantic similarity approximation.
    """

    def __init__(self):
        self._entries: dict[str, MemoryEntry] = {}

    def add(self, entry_id: str, entry: MemoryEntry):
        self._entries[entry_id] = entry

    def remove(self, entry_id: str):
        self._entries.pop(entry_id, None)

    def search(self, query: str, top_k: int = 5, threshold: float = 0.1) -> list[tuple[str, float]]:
        """
        检索与查询最相关的记忆。

        Returns list of (entry_id, similarity_score).
        """
        if not self._entries:
            return []

        query_words = set(self._tokenize(query))
        if not query_words:
            return []

        scores = []
        for eid, entry in self._entries.items():
            entry_words = set(self._tokenize(entry.content))
            if not entry_words:
                continue

            # Jaccard similarity
            intersection = query_words & entry_words
            union = query_words | entry_words
            jaccard = len(intersection) / len(union) if union else 0

            # Bonus for importance
            combined = jaccard * 0.7 + entry.importance * 0.3

            if combined >= threshold:
                scores.append((eid, round(combined, 3)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenizer: split on whitespace + punctuation, keep Chinese chars."""
        import re
        # Keep Chinese characters, English words, and numbers
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', text.lower())
        return tokens

    def __len__(self):
        return len(self._entries)


# ──────────────────────────────────────────────
# 4. Full Memory System (backward compatible)
# ──────────────────────────────────────────────

class MemoryStore:
    """
    完整记忆系统 — 向后兼容旧 API。

    Layers:
      short_term  — sliding window (L0 working memory)
      session_summary — conversation summary (L1)
      long_term — persisted facts with importance (L2)
      episodic  — event-based memories (L3)
      vector_index — semantic retrieval (L4)
    """

    def __init__(
        self,
        max_short: int = 10,
        importance_threshold: float = 0.3,
        decay_half_life: float = 24.0,
        persist_path: Optional[str] = None,
    ):
        # L0 — Working memory (sliding window)
        self.short_term: list[dict] = []
        self.max_short = max_short

        # L1 — Session summary
        self.session_summary: str = ""

        # L2 — Long-term memory (key → MemoryEntry)
        self.long_term: dict[str, MemoryEntry] = {}

        # L3 — Episodic memory
        self.episodic: list[MemoryEntry] = []

        # L4 — Vector index for semantic retrieval
        self.vector_index = KeywordVectorIndex()

        # Config
        self.importance_threshold = importance_threshold
        self.decay_half_life = decay_half_life
        self.estimator = ImportanceEstimator()
        self.persist_path = persist_path

        # Stats
        self._total_adds = 0
        self._total_retrievals = 0
        self._consolidations = 0

        # Load persisted data if available
        if persist_path and os.path.exists(persist_path):
            self._load(persist_path)

    # ── Backward-compatible API ──

    def add(self, role: str, content: str, importance: float | None = None):
        """写入短期记忆，超过窗口则淘汰到会话摘要。"""
        estimated = importance or self.estimator.estimate(content, role)

        self.short_term.append({"role": role, "content": content})
        self._total_adds += 1

        # Add to vector index
        entry = MemoryEntry(content=content, role=role, importance=estimated)
        entry_id = f"st_{uuid.uuid4().hex[:6]}"
        self.vector_index.add(entry_id, entry)

        # Sliding window: evict oldest
        if len(self.short_term) > self.max_short:
            old = self.short_term.pop(0)
            self.session_summary += f"[{old['role'][:4]}] {old['content'][:50]}... | "

        # Auto-consolidate if importance is high
        if estimated >= self.importance_threshold:
            self._promote_to_longterm(entry_id, entry)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """从记忆中检索（语义搜索）。"""
        self._total_retrievals += 1
        results = self.vector_index.search(query, top_k=top_k)
        return [
            f"[score={s:.2f}] {self.vector_index._entries[eid].content}"
            for eid, s in results
        ]

    def consolidate(self):
        """记忆巩固——短期→长期。"""
        self._consolidations += 1
        for msg in self.short_term[-3:]:
            importance = self.estimator.estimate(msg["content"], msg["role"])
            if importance >= self.importance_threshold:
                entry = MemoryEntry(
                    content=msg["content"],
                    role=msg["role"],
                    importance=importance,
                )
                self._promote_to_longterm(f"mem_{uuid.uuid4().hex[:6]}", entry)

    def get_window(self) -> list[dict]:
        return self.short_term.copy()

    def stats(self) -> dict:
        return {
            "short_term": len(self.short_term),
            "long_term": len(self.long_term),
            "episodic": len(self.episodic),
            "vector_index": len(self.vector_index),
            "total_adds": self._total_adds,
            "total_retrievals": self._total_retrievals,
            "consolidations": self._consolidations,
        }

    # ── Extended API ──

    def add_episodic(self, event: str, tags: list[str] | None = None, importance: float = 0.6):
        """添加事件记忆。"""
        entry = MemoryEntry(
            content=event,
            role="system",
            importance=importance,
            tags=tags or [],
        )
        self.episodic.append(entry)
        self.vector_index.add(f"ep_{uuid.uuid4().hex[:6]}", entry)

    def add_fact(self, key: str, value: str, importance: float = 0.7):
        """添加长期事实。"""
        entry = MemoryEntry(
            content=value,
            role="fact",
            importance=importance,
        )
        self.long_term[key] = entry
        self.vector_index.add(f"lt_{key}", entry)

    def forget(self, min_importance: float = 0.1):
        """遗忘低重要性的长期记忆。"""
        to_remove = []
        for key, entry in self.long_term.items():
            effective = entry.decay(self.decay_half_life)
            if effective < min_importance:
                to_remove.append(key)
        for key in to_remove:
            self.vector_index.remove(f"lt_{key}")
            del self.long_term[key]
        return len(to_remove)

    def apply_decay(self):
        """对所有长期记忆应用衰减。"""
        for entry in self.long_term.values():
            entry.decay(self.decay_half_life)

    def persist(self, path: str | None = None):
        """持久化记忆到 JSON 文件。"""
        path = path or self.persist_path
        if not path:
            return
        data = {
            "short_term": self.short_term,
            "session_summary": self.session_summary,
            "long_term": {k: v.to_dict() for k, v in self.long_term.items()},
            "episodic": [e.to_dict() for e in self.episodic],
            "stats": {
                "total_adds": self._total_adds,
                "total_retrievals": self._total_retrievals,
                "consolidations": self._consolidations,
            },
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self, path: str):
        """从 JSON 文件加载记忆。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.short_term = data.get("short_term", [])
            self.session_summary = data.get("session_summary", "")
            for k, v in data.get("long_term", {}).items():
                entry = MemoryEntry.from_dict(v)
                self.long_term[k] = entry
                self.vector_index.add(f"lt_{k}", entry)
            for e in data.get("episodic", []):
                entry = MemoryEntry.from_dict(e)
                self.episodic.append(entry)
                self.vector_index.add(f"ep_{uuid.uuid4().hex[:6]}", entry)
        except Exception:
            pass  # Silently ignore load failures

    def _promote_to_longterm(self, entry_id: str, entry: MemoryEntry):
        """将高重要性记忆提升到长期存储。"""
        key = f"mem_{entry_id}"
        if key not in self.long_term:
            self.long_term[key] = entry
            self.vector_index.add(key, entry)
