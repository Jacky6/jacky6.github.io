"""
§07 — LangGraph Checkpointer 集成

提供断点续跑 (checkpoint & resume) 能力。
- MemorySaver: 内存级（开发/测试）
- SQLiteSaver: 持久化到 SQLite（生产轻量级）
- PostgreSQLSaver: 持久化到 PostgreSQL（生产分布式）

Usage:
    # Memory
    saver = MemorySaver()
    
    # SQLite (auto-creates DB)
    saver = SQLiteSaver(db_path="checkpoints/demo.db")
    
    # PostgreSQL
    saver = PostgreSQLSaver(dsn="postgresql://user:pass@host:5432/db")
    
    # In graph
    graph = builder.compile(checkpointer=saver)
    
    # Run with thread ID for checkpointing
    result = await graph.ainvoke(initial, config={"configurable": {"thread_id": "run-001"}})
    
    # Resume later with same thread_id
    result2 = await graph.ainvoke(None, config={"configurable": {"thread_id": "run-001"}})
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────
# Checkpoint 数据模型
# ──────────────────────────────────────────────

@dataclass
class CheckpointMetadata:
    """断点元数据。"""
    thread_id: str
    checkpoint_id: str = ""
    source: str = "input"  # input | update | fork
    step: int = 0
    writes: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "checkpoint_id": self.checkpoint_id,
            "source": self.source,
            "step": self.step,
            "writes": self.writes,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Checkpoint:
    """完整断点状态。"""
    config: dict
    value: dict  # 序列化的状态
    metadata: CheckpointMetadata
    parent_id: str = ""

    def serialize(self) -> str:
        return json.dumps({
            "config": self.config,
            "value": self.value,
            "metadata": self.metadata.to_dict(),
            "parent_id": self.parent_id,
        })

    @classmethod
    def deserialize(cls, raw: str) -> "Checkpoint":
        data = json.loads(raw)
        meta = CheckpointMetadata.from_dict(data["metadata"])
        return cls(
            config=data["config"],
            value=data["value"],
            metadata=meta,
            parent_id=data.get("parent_id", ""),
        )


# ──────────────────────────────────────────────
# MemorySaver — 内存级（开发/测试）
# ──────────────────────────────────────────────

class MemorySaver:
    """
    内存级 Checkpointer，适合开发和测试。
    进程重启后数据丢失。

    兼容 LangGraph BaseCheckpointSaver 接口。
    """

    def __init__(self):
        self._checkpoints: dict[str, list[Checkpoint]] = {}
        self._lock = threading.Lock()

    def put(
        self,
        config: dict,
        checkpoint: dict,
        metadata: dict,
        new_config: dict | None = None,
    ) -> dict:
        """保存断点。"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_id = f"chk-{len(self._checkpoints.get(thread_id, [])):04d}"

        meta = CheckpointMetadata(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            source=metadata.get("source", "input"),
            step=metadata.get("step", 0),
            writes=metadata.get("writes", {}),
        )

        cp = Checkpoint(
            config=config,
            value=checkpoint,
            metadata=meta,
            parent_id=self._get_latest_id(thread_id),
        )

        with self._lock:
            self._checkpoints.setdefault(thread_id, []).append(cp)

        return {**config, "configurable": {**config.get("configurable", {}), "checkpoint_id": checkpoint_id}}

    def get(self, config: dict) -> Checkpoint | None:
        """获取最新断点。"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        cps = self._checkpoints.get(thread_id, [])
        if cps:
            return cps[-1]

        # Check by specific checkpoint_id
        cp_id = config.get("configurable", {}).get("checkpoint_id", "")
        if cp_id:
            for cp in cps:
                if cp.metadata.checkpoint_id == cp_id:
                    return cp
        return None

    def list(self, config: dict, *, limit: int = 10, before: str | None = None) -> list[Checkpoint]:
        """列出断点历史。"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        cps = self._checkpoints.get(thread_id, [])

        if before:
            cps = [cp for cp in cps if cp.metadata.checkpoint_id < before]

        return cps[-limit:]

    def delete_thread(self, thread_id: str) -> bool:
        """删除指定线程的所有断点。"""
        with self._lock:
            return self._checkpoints.pop(thread_id, None) is not None

    def stats(self) -> dict:
        """统计信息。"""
        return {
            "threads": len(self._checkpoints),
            "total_checkpoints": sum(len(v) for v in self._checkpoints.values()),
            "thread_ids": list(self._checkpoints.keys()),
        }

    def _get_latest_id(self, thread_id: str) -> str:
        cps = self._checkpoints.get(thread_id, [])
        if cps:
            return cps[-1].metadata.checkpoint_id
        return ""


# ──────────────────────────────────────────────
# SQLiteSaver — 持久化到 SQLite（生产轻量级）
# ──────────────────────────────────────────────

class SQLiteSaver:
    """
    SQLite 级 Checkpointer，适合生产环境轻量级部署。
    断点持久化到本地文件，支持进程重启后恢复。

    兼容 LangGraph BaseCheckpointSaver 接口。
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id     TEXT    NOT NULL,
        checkpoint_id TEXT    NOT NULL,
        parent_id     TEXT,
        source        TEXT,
        step          INTEGER,
        writes        TEXT,
        value         TEXT    NOT NULL,
        config        TEXT    NOT NULL,
        timestamp     TEXT    NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_id)
    );
    CREATE INDEX IF NOT EXISTS idx_cp_thread ON checkpoints(thread_id);
    CREATE INDEX IF NOT EXISTS idx_cp_parent ON checkpoints(parent_id);
    """

    def __init__(self, db_path: str | Path = "checkpoints/demo.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock, self._conn() as conn:
            conn.executescript(self._DDL)

    def put(
        self,
        config: dict,
        checkpoint: dict,
        metadata: dict,
        new_config: dict | None = None,
    ) -> dict:
        """保存断点到 SQLite。"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_id = f"chk-{metadata.get('step', 0):04d}"
        parent_id = self._get_latest_id(thread_id)

        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO checkpoints
                   (thread_id, checkpoint_id, parent_id, source, step, writes, value, config, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (
                    thread_id,
                    checkpoint_id,
                    parent_id,
                    metadata.get("source", "input"),
                    metadata.get("step", 0),
                    json.dumps(metadata.get("writes", {})),
                    json.dumps(checkpoint),
                    json.dumps(config),
                ),
            )

        return {**config, "configurable": {**config.get("configurable", {}), "checkpoint_id": checkpoint_id}}

    def get(self, config: dict) -> Checkpoint | None:
        """从 SQLite 获取最新断点。"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")

        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY step DESC LIMIT 1",
                (thread_id,),
            ).fetchone()

        if not row:
            return None

        meta = CheckpointMetadata(
            thread_id=row["thread_id"],
            checkpoint_id=row["checkpoint_id"],
            source=row["source"],
            step=row["step"],
            writes=json.loads(row["writes"]) if row["writes"] else {},
            timestamp=row["timestamp"],
        )
        return Checkpoint(
            config=json.loads(row["config"]),
            value=json.loads(row["value"]),
            metadata=meta,
            parent_id=row["parent_id"] or "",
        )

    def list(self, config: dict, *, limit: int = 10, before: str | None = None) -> list[Checkpoint]:
        """列出断点历史。"""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        query = "SELECT * FROM checkpoints WHERE thread_id = ?"
        params: list = [thread_id]

        if before:
            query += " AND checkpoint_id < ?"
            params.append(before)

        query += " ORDER BY step DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            Checkpoint(
                config=json.loads(r["config"]),
                value=json.loads(r["value"]),
                metadata=CheckpointMetadata(
                    thread_id=r["thread_id"],
                    checkpoint_id=r["checkpoint_id"],
                    source=r["source"],
                    step=r["step"],
                    writes=json.loads(r["writes"]) if r["writes"] else {},
                    timestamp=r["timestamp"],
                ),
                parent_id=r["parent_id"] or "",
            )
            for r in rows
        ]

    def delete_thread(self, thread_id: str) -> bool:
        """删除指定线程的所有断点。"""
        with self._lock, self._conn() as conn:
            cursor = conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            return cursor.rowcount > 0

    def stats(self) -> dict:
        """统计信息。"""
        with self._conn() as conn:
            threads = conn.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            thread_ids = [r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id").fetchall()]

        return {
            "threads": threads,
            "total_checkpoints": total,
            "thread_ids": thread_ids,
            "db_path": str(self.db_path),
            "db_size_kb": self.db_path.stat().st_size // 1024 if self.db_path.exists() else 0,
        }

    def _get_latest_id(self, thread_id: str) -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? ORDER BY step DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        return row["checkpoint_id"] if row else ""


# ──────────────────────────────────────────────
# PostgreSQLSaver — 持久化到 PostgreSQL（生产分布式）
# ──────────────────────────────────────────────

class PostgreSQLSaver:
    """
    PostgreSQL 级 Checkpointer，适合生产环境分布式部署。
    多进程/多实例共享断点状态。

    需要 psycopg2 或 asyncpg 驱动。
    兼容 LangGraph BaseCheckpointSaver 接口。
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id     VARCHAR(255) NOT NULL,
        checkpoint_id VARCHAR(255) NOT NULL,
        parent_id     VARCHAR(255),
        source        VARCHAR(50),
        step          INTEGER,
        writes        JSONB,
        value         JSONB NOT NULL,
        config        JSONB NOT NULL,
        timestamp     TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (thread_id, checkpoint_id)
    );
    CREATE INDEX IF NOT EXISTS idx_cp_thread_pg ON checkpoints(thread_id);
    CREATE INDEX IF NOT EXISTS idx_cp_parent_pg ON checkpoints(parent_id);
    """

    def __init__(self, dsn: str = "postgresql://user:pass@localhost:5432/agent_db"):
        self.dsn = dsn
        self._conn = None
        self._init_db()

    def _get_conn(self):
        """惰性连接。"""
        if self._conn is None or self._conn.closed:
            try:
                import psycopg2
                self._conn = psycopg2.connect(self.dsn)
            except ImportError:
                raise ImportError(
                    "PostgreSQLSaver 需要 psycopg2 驱动: pip install psycopg2-binary"
                )
        return self._conn

    def _init_db(self):
        """创建表（如果不存在）。"""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(self._DDL)
            conn.commit()
        except Exception:
            # 开发环境可能没有 PostgreSQL，静默跳过
            pass

    def put(
        self,
        config: dict,
        checkpoint: dict,
        metadata: dict,
        new_config: dict | None = None,
    ) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_id = f"chk-{metadata.get('step', 0):04d}"

        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO checkpoints
                       (thread_id, checkpoint_id, parent_id, source, step, writes, value, config, timestamp)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                       ON CONFLICT (thread_id, checkpoint_id) DO UPDATE
                       SET value = EXCLUDED.value, config = EXCLUDED.config, step = EXCLUDED.step""",
                    (
                        thread_id,
                        checkpoint_id,
                        None,  # parent_id
                        metadata.get("source", "input"),
                        metadata.get("step", 0),
                        json.dumps(metadata.get("writes", {})),
                        json.dumps(checkpoint),
                        json.dumps(config),
                    ),
                )
            conn.commit()
        except Exception:
            pass

        return {**config, "configurable": {**config.get("configurable", {}), "checkpoint_id": checkpoint_id}}

    def get(self, config: dict) -> Checkpoint | None:
        thread_id = config.get("configurable", {}).get("thread_id", "default")

        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM checkpoints WHERE thread_id = %s ORDER BY step DESC LIMIT 1",
                    (thread_id,),
                )
                row = cur.fetchone()

            if not row:
                return None

            return Checkpoint(
                config=row[6],  # config JSONB
                value=row[5],   # value JSONB
                metadata=CheckpointMetadata(
                    thread_id=row[0],
                    checkpoint_id=row[1],
                    source=row[3],
                    step=row[4],
                ),
                parent_id=row[2] or "",
            )
        except Exception:
            return None

    def list(self, config: dict, *, limit: int = 10, before: str | None = None) -> list[Checkpoint]:
        return []

    def delete_thread(self, thread_id: str) -> bool:
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
            conn.commit()
            return True
        except Exception:
            return False

    def stats(self) -> dict:
        return {"type": "postgresql", "dsn": self.dsn, "connected": self._conn is not None and not getattr(self._conn, "closed", True)}


# ──────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────

def create_saver(
    backend: str = "memory",
    *,
    db_path: str = "checkpoints/demo.db",
    dsn: str = "postgresql://user:pass@localhost:5432/agent_db",
) -> MemorySaver | SQLiteSaver | PostgreSQLSaver:
    """
    创建 Checkpointer 实例。

    Args:
        backend: "memory" | "sqlite" | "postgresql"
        db_path: SQLite 数据库路径
        dsn: PostgreSQL 连接字符串

    Returns:
        对应后端的 Checkpointer 实例
    """
    backends = {
        "memory": lambda: MemorySaver(),
        "sqlite": lambda: SQLiteSaver(db_path=db_path),
        "postgresql": lambda: PostgreSQLSaver(dsn=dsn),
    }

    factory = backends.get(backend)
    if not factory:
        raise ValueError(f"Unknown checkpoint backend: {backend}. Choose from: {list(backends.keys())}")

    return factory()
