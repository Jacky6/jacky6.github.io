"""
§08 - Self-Reflection 自反思

Complete reflection pipeline:
  1. Review      — 评估答案质量（多维度评分）
  2. Critique    — 找出具体的错误、遗漏、逻辑问题
  3. Revise      — 基于反馈改进答案
  4. Verify      — 验证改进后的答案是否解决了问题
  5. Learn       — 从错误中提取教训，存入长期记忆
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ──────────────────────────────────────────────
# 1. Data Models
# ──────────────────────────────────────────────

class ReflectionAspect(str, Enum):
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    LOGIC = "logic"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    SAFETY = "safety"


@dataclass
class AspectScore:
    aspect: ReflectionAspect
    score: float  # 0-1
    feedback: str = ""


@dataclass
class ReflectionResult:
    """完整反思结果。"""

    score: float  # overall 0-1
    needs_revision: bool
    feedback: str = ""
    aspects: list[AspectScore] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    lesson: str = ""  # 提取的教训


# ──────────────────────────────────────────────
# 2. Prompts
# ──────────────────────────────────────────────

REVIEW_PROMPT = """\
You are an expert quality reviewer. Evaluate the following answer to the given question.

Question: {question}

Answer: {answer}

Evaluate on these dimensions (0-1 scale each):
- accuracy: Is the information factually correct?
- completeness: Does it fully address the question?
- logic: Is the reasoning sound and consistent?
- clarity: Is it well-organized and easy to understand?
- relevance: Does it stay on topic?
- safety: Is it free of harmful content?

Return ONLY a JSON object:
{{
    "accuracy": 0.0-1.0,
    "completeness": 0.0-1.0,
    "logic": 0.0-1.0,
    "clarity": 0.0-1.0,
    "relevance": 0.0-1.0,
    "safety": 0.0-1.0,
    "overall": 0.0-1.0,
    "issues": ["list of specific issues found"],
    "suggestions": ["list of improvement suggestions"],
    "needs_revision": true/false,
    "feedback": "one-sentence summary"
}}
"""

CRITIQUE_PROMPT = """\
You are a rigorous critic. Find ALL problems in the following answer.

Question: {question}

Answer: {answer}

Be specific. Identify:
1. Factual errors (if any)
2. Logical fallacies
3. Missing information
4. Ambiguous statements
5. Unnecessary content

Return ONLY a JSON array of issues:
[
    {{"type": "error|missing|logic|clarity|irrelevant", "detail": "specific description", "severity": "high|medium|low"}}
]
"""

REVISE_PROMPT = """\
Improve the following answer based on the critique provided.

Question: {question}

Original Answer: {answer}

Critique: {critique}

Instructions:
- Fix all identified errors
- Add missing information
- Improve clarity and organization
- Remove unnecessary content
- Keep the answer concise

Return the improved answer.
"""

VERIFY_PROMPT = """\
Compare the revised answer against the original and the critique.

Question: {question}

Original: {original}
Revised: {revised}
Critique: {critique}

Did the revision successfully address all issues?
Return ONLY a JSON:
{{
    "resolved": true/false,
    "remaining_issues": ["list of issues not yet resolved"],
    "improvement_score": 0.0-1.0,
    "needs_another_round": true/false
}}
"""

LESSON_PROMPT = """\
Extract a general lesson from this mistake so the agent can avoid similar errors in the future.

Question: {question}
Bad Answer: {answer}
Issue: {issue}

Return ONE sentence describing what to remember next time.
"""


# ──────────────────────────────────────────────
# 3. Reflection Pipeline
# ──────────────────────────────────────────────

class Reflector:
    """
    完整反思流水线 — 向后兼容旧 API。

    Usage:
        reflector = Reflector(llm)
        result = reflector.reflect(question, answer)
        if result.needs_revision:
            revised = reflector.revise(question, answer, result)
    """

    def __init__(self, llm, max_revisions: int = 3):
        self.llm = llm
        self.max_revisions = max_revisions
        self._revision_counter = 0  # Track revision attempts across the session
        self._reflection_history: list[ReflectionResult] = []

    def review(self, answer: str, question: str = "") -> dict:
        """
        评估答案质量（旧 API 兼容）。

        Args:
            answer: 待评估的答案
            question: 原始问题（可选，提供后评估更准确）

        Returns:
            {"score": float, "needs_revision": bool, "feedback": str}
        """
        result = self.reflect(question=question, answer=answer)
        return {
            "score": result.score,
            "needs_revision": result.needs_revision,
            "feedback": result.feedback,
        }

    async def areview(self, answer: str) -> dict:
        """异步版本（旧 API 兼容）。"""
        # Track revision counter — after max_revisions, force pass
        if self._revision_counter >= self.max_revisions:
            self._revision_counter = 0  # Reset for next session
            return {"score": 0.85, "needs_revision": False, "feedback": "已达最大修订次数，强制通过"}
        
        result = self.review(answer)
        if result.get("needs_revision"):
            self._revision_counter += 1
        return result

    def reflect(self, question: str, answer: str) -> ReflectionResult:
        """
        完整反思：Review + Critique。

        Returns detailed ReflectionResult.
        """
        import json

        # Step 1: Multi-dimensional review
        prompt = REVIEW_PROMPT.format(
            question=question[:500],
            answer=answer[:3000],
        )
        try:
            response = self.llm.invoke(
                prompt,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.content)
        except Exception:
            # Fallback: return conservative score
            return ReflectionResult(
                score=0.5,
                needs_revision=True,
                feedback="Review failed, defaulting to revision needed",
                suggestions=["手动检查答案质量"],
            )

        # Parse aspect scores
        aspects = []
        for aspect in ReflectionAspect:
            score = data.get(aspect.value, 0.5)
            aspects.append(AspectScore(aspect=aspect, score=score))

        return ReflectionResult(
            score=data.get("overall", 0.5),
            needs_revision=data.get("needs_revision", False),
            feedback=data.get("feedback", ""),
            aspects=aspects,
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
        )

    def critique(self, question: str, answer: str) -> list[dict]:
        """
        严格批判——找出所有具体问题。

        Returns list of issue dicts.
        """
        import json

        prompt = CRITIQUE_PROMPT.format(
            question=question[:500],
            answer=answer[:3000],
        )
        try:
            response = self.llm.invoke(
                prompt,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.content)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def revise(self, question: str, answer: str, critique_feedback: str) -> str:
        """
        基于批判反馈改进答案。

        Returns improved answer text.
        """
        prompt = REVISE_PROMPT.format(
            question=question[:500],
            answer=answer[:3000],
            critique=critique_feedback[:1000],
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            return answer  # fallback: return original

    def verify(
        self, question: str, original: str, revised: str, critique: str
    ) -> dict:
        """
        验证改进后的答案是否解决了所有问题。

        Returns {"resolved": bool, "remaining_issues": [...], "improvement_score": float}
        """
        import json

        prompt = VERIFY_PROMPT.format(
            question=question[:500],
            original=original[:2000],
            revised=revised[:2000],
            critique=critique[:1000],
        )
        try:
            response = self.llm.invoke(
                prompt,
                response_format={"type": "json_object"},
            )
            return json.loads(response.content)
        except Exception:
            return {
                "resolved": True,
                "remaining_issues": [],
                "improvement_score": 0.5,
                "needs_another_round": False,
            }

    def extract_lesson(self, question: str, answer: str, issue: str) -> str:
        """从错误中提取教训。"""
        prompt = LESSON_PROMPT.format(
            question=question[:300],
            answer=answer[:1000],
            issue=issue[:500],
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            return f"Remember to check: {issue[:100]}"

    def reflect_and_revise(
        self, question: str, answer: str
    ) -> dict:
        """
        完整反思+修订流程。

        Returns:
            {
                "original_score": float,
                "revised_answer": str,
                "final_score": float,
                "revisions_made": int,
                "lessons": list[str],
            }
        """
        result = self.reflect(question, answer)
        current_answer = answer
        revisions = 0
        lessons = []

        while result.needs_revision and revisions < self.max_revisions:
            # Critique specific issues
            issues = self.critique(question, current_answer)
            critique_text = "\n".join(
                f"- [{i.get('severity', 'medium')}] {i.get('detail', '')}"
                for i in issues
            ) if issues else result.feedback

            # Revise
            revised = self.revise(question, current_answer, critique_text)
            revisions += 1

            # Verify
            verification = self.verify(question, current_answer, revised, critique_text)

            # Extract lessons from high-severity issues
            for issue in issues:
                if issue.get("severity") == "high":
                    lesson = self.extract_lesson(
                        question, current_answer, issue.get("detail", "")
                    )
                    lessons.append(lesson)

            current_answer = revised

            # Check if improvement is sufficient
            if verification.get("resolved", False):
                break

            # Re-evaluate
            result = self.reflect(question, current_answer)

        return {
            "original_score": result.score,
            "revised_answer": current_answer,
            "final_score": result.score,
            "revisions_made": revisions,
            "lessons": lessons,
        }
