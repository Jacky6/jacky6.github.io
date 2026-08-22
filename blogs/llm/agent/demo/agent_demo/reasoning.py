"""
§03 - Reasoning 推理引擎

推理策略层次:
  Level 0 — Direct           直接回答
  Level 1 — CoT              逐步推理 (Chain of Thought)
  Level 2 — ReAct            边想边做 (Reason + Act)
  Level 3 — ToT              多路径探索 (Tree of Thoughts)
  Level 4 — Self-Consistency 多次采样取共识

自动路由: 根据输入复杂度自动选择推理策略。
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Callable, Optional


# ──────────────────────────────────────────────
# 1. Reasoning Strategy
# ──────────────────────────────────────────────

class ReasoningStrategy:
    """推理策略枚举。"""

    DIRECT = "direct"
    COT = "cot"              # Chain of Thought
    REACT = "react"          # Reason + Act
    TOT = "tot"              # Tree of Thoughts
    SELF_CONSISTENCY = "self_consistency"


# ──────────────────────────────────────────────
# 2. Strategy Router
# ──────────────────────────────────────────────

class StrategyRouter:
    """根据输入自动选择推理策略。"""

    def __init__(self, simple_threshold: int = 50, medium_threshold: int = 200):
        self.simple_threshold = simple_threshold
        self.medium_threshold = medium_threshold

    def route(self, question: str, has_tools: bool = False) -> ReasoningStrategy:
        """
        路由逻辑:
          - 短文本 + 无工具 → direct
          - 短文本 + 有工具 → react
          - 中等长度 → cot
          - 长文本/复杂 → self_consistency
          - 多步骤规划 → tot
        """
        length = len(question)

        if has_tools and length < self.simple_threshold:
            return ReasoningStrategy.REACT
        if length < self.simple_threshold:
            return ReasoningStrategy.DIRECT
        if length < self.medium_threshold:
            return ReasoningStrategy.COT

        # 检测是否需要多路径探索
        complex_keywords = [
            "方案", "选择", "对比", "评估", "决策",
            "plan", "choose", "compare", "evaluate", "decide",
        ]
        if any(kw in question for kw in complex_keywords):
            return ReasoningStrategy.TOT

        return ReasoningStrategy.SELF_CONSISTENCY


# ──────────────────────────────────────────────
# 3. CoT (Chain of Thought)
# ──────────────────────────────────────────────

class ChainOfThought:
    """
    Zero-shot Chain of Thought.

    Prompt the LLM to think step by step before answering.
    """

    COT_PROMPT = """\
{question}

Let's think step by step.
1. First, identify the key elements of the problem.
2. Then, reason through each element carefully.
3. Finally, provide your answer.

Be thorough and show your reasoning.
"""

    def __init__(self, llm):
        self.llm = llm

    def reason(self, question: str) -> str:
        """执行 CoT 推理。"""
        prompt = self.COT_PROMPT.format(question=question)
        response = self.llm([{"role": "user", "content": prompt}])
        return response.get("content", "")

    async def areason(self, question: str) -> str:
        """异步 CoT 推理。"""
        prompt = self.COT_PROMPT.format(question=question)
        if hasattr(self.llm, "ainvoke"):
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            return response.content if hasattr(response, "content") else str(response)
        return self.reason(question)


# ──────────────────────────────────────────────
# 4. ToT (Tree of Thoughts)
# ──────────────────────────────────────────────

class TreeOfThoughts:
    """
    Tree of Thoughts — 多路径推理探索。

    生成多个推理路径，分别评估，选择最优路径的答案。
    """

    BRANCH_PROMPT = """\
Given the following question, generate {n_branches} different approaches to solving it.

Question: {question}

Return each approach as a separate paragraph, labeled Approach 1, Approach 2, etc.
"""

    EVALUATE_PROMPT = """\
Evaluate the quality of this reasoning approach:

Question: {question}
Approach: {approach}

Score from 0-10 based on:
- Logical soundness
- Completeness
- Feasibility

Return ONLY a JSON: {{"score": 0-10, "reason": "brief explanation"}}
"""

    def __init__(self, llm, n_branches: int = 3):
        self.llm = llm
        self.n_branches = n_branches

    def reason(self, question: str) -> dict:
        """
        执行 ToT 推理。

        Returns:
            {"answer": str, "branches": [{"approach": str, "score": float}], "best_approach": str}
        """
        import json

        # Step 1: Generate branches
        branch_prompt = self.BRANCH_PROMPT.format(
            question=question,
            n_branches=self.n_branches,
        )
        branch_response = self.llm([{"role": "user", "content": branch_prompt}])
        branch_text = branch_response.get("content", "")

        # Parse approaches (simple split by "Approach")
        approaches = []
        for line in branch_text.split("\n"):
            if line.strip().startswith(("Approach", "approach")):
                approaches.append(line.strip())
        if not approaches:
            # Fallback: use whole response as single approach
            approaches = [branch_text]

        # Step 2: Evaluate each approach
        scored = []
        for approach in approaches[:self.n_branches]:
            eval_prompt = self.EVALUATE_PROMPT.format(
                question=question,
                approach=approach,
            )
            try:
                eval_response = self.llm.invoke(
                    eval_prompt,
                    response_format={"type": "json_object"},
                )
                eval_data = json.loads(eval_response.content)
                scored.append({
                    "approach": approach,
                    "score": eval_data.get("score", 5),
                    "reason": eval_data.get("reason", ""),
                })
            except Exception:
                scored.append({
                    "approach": approach,
                    "score": 5.0,
                    "reason": "evaluation failed",
                })

        # Step 3: Select best approach and generate final answer
        best = max(scored, key=lambda x: x["score"]) if scored else {"approach": branch_text, "score": 0}

        final_prompt = f"""\
Based on the best approach, provide a complete answer.

Question: {question}

Best approach (score: {best['score']}/10):
{best['approach']}

Reasoning: {best.get('reason', '')}

Provide the final answer.
"""
        final_response = self.llm([{"role": "user", "content": final_prompt}])

        return {
            "answer": final_response.get("content", ""),
            "branches": scored,
            "best_approach": best["approach"],
            "best_score": best["score"],
        }


# ──────────────────────────────────────────────
# 5. Self-Consistency
# ──────────────────────────────────────────────

class SelfConsistency:
    """
    Self-Consistency — 多次采样取共识。

    对同一问题多次采样，选择出现频率最高的答案。
    """

    def __init__(self, llm, n_samples: int = 5, temperature: float = 0.7):
        self.llm = llm
        self.n_samples = n_samples
        self.temperature = temperature

    def reason(self, question: str) -> dict:
        """
        执行 Self-Consistency 推理。

        Returns:
            {"answer": str, "samples": list[str], "consensus": str, "confidence": float}
        """
        # Generate multiple samples
        samples = []
        for _ in range(self.n_samples):
            prompt = f"{question}\n\nThink step by step and provide your answer."
            try:
                response = self.llm.invoke(
                    [{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                samples.append(response.get("content", "") if isinstance(response, dict) else response.content)
            except Exception:
                samples.append(self._fallback_sample(question))

        # Find consensus (simple: most frequent first sentence)
        first_sentences = []
        for s in samples:
            first_sent = s.split("\n")[0].strip() if s else ""
            if first_sent:
                first_sentences.append(first_sent)

        if not first_sentences:
            consensus = samples[0] if samples else "No response"
            confidence = 0.0
        else:
            # Group by similarity (simple: exact match or contains)
            groups: list[list[str]] = []
            for sent in first_sentences:
                found_group = False
                for group in groups:
                    if sent in group[0] or group[0] in sent:
                        group.append(sent)
                        found_group = True
                        break
                if not found_group:
                    groups.append([sent])

            best_group = max(groups, key=len)
            consensus = best_group[0]
            confidence = len(best_group) / len(first_sentences)

        return {
            "answer": consensus,
            "samples": samples,
            "consensus": consensus,
            "confidence": round(confidence, 2),
        }

    def _fallback_sample(self, question: str) -> str:
        """Fallback when LLM sampling fails."""
        return f"[Sample failed] I cannot answer: {question[:100]}"


# ──────────────────────────────────────────────
# 6. ReasoningEngine (unified entry point)
# ──────────────────────────────────────────────

class ReasoningEngine:
    """
    统一推理引擎 — 自动路由到最佳策略。

    Usage:
        engine = ReasoningEngine(llm)
        result = engine.reason("复杂问题...")
        print(result["strategy"], result["answer"])
    """

    def __init__(self, llm, max_iter: int = 5):
        self.llm = llm
        self.max_iter = max_iter
        self.router = StrategyRouter()
        self.cot = ChainOfThought(llm)
        self.tot = TreeOfThoughts(llm)
        self.sc = SelfConsistency(llm)

    def reason(
        self,
        question: str,
        strategy: ReasoningStrategy | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        """
        执行推理，自动或手动选择策略。

        Args:
            question: 问题
            strategy: 手动指定策略（None 则自动路由）
            tools: 可用工具列表

        Returns:
            {"strategy": str, "answer": str, **strategy_specific_data}
        """
        if strategy is None:
            strategy = self.router.route(question, has_tools=bool(tools))

        if strategy == ReasoningStrategy.DIRECT:
            response = self.llm([{"role": "user", "content": question}])
            return {"strategy": "direct", "answer": response.get("content", "")}

        if strategy == ReasoningStrategy.COT:
            answer = self.cot.reason(question)
            return {"strategy": "cot", "answer": answer}

        if strategy == ReasoningStrategy.REACT:
            return self._react(question, tools)

        if strategy == ReasoningStrategy.TOT:
            result = self.tot.reason(question)
            return {"strategy": "tot", **result}

        if strategy == ReasoningStrategy.SELF_CONSISTENCY:
            result = self.sc.reason(question)
            return {"strategy": "self_consistency", **result}

        # Fallback
        response = self.llm([{"role": "user", "content": question}])
        return {"strategy": "direct", "answer": response.get("content", "")}

    def _react(self, question: str, tools: list[dict] | None = None) -> dict:
        """ReAct 单步推理（兼容旧 API）。"""
        messages = [{"role": "user", "content": question}]
        response = self.llm(messages, tools=tools)
        tc = response.get("tool_calls")
        if tc:
            return {"strategy": "react", "tool_calls": tc, "answer": ""}
        return {"strategy": "react", "answer": response.get("content", "")}

    # ── Backward compatible methods ──

    def react_step(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """ReAct 单步（旧 API 兼容）。"""
        response = self.llm(messages, tools=tools)
        tc = response.get("tool_calls")
        if tc:
            return {"tool_calls": tc}
        return {"answer": response.get("content", "")}

    def react_standalone(self, question: str, tools: list[dict] | None = None) -> dict:
        """独立 ReAct 循环（旧 API 兼容）。"""
        messages = [{"role": "user", "content": question}]
        tool_calls = []
        results = []

        for i in range(self.max_iter):
            response = self.llm(messages, tools=tools)
            tc = response.get("tool_calls")
            if not tc:
                return {
                    "answer": response.get("content", ""),
                    "tool_calls": tool_calls,
                    "tool_results": results,
                    "iterations": i + 1,
                }
            for t in tc:
                tool_calls.append(t)
                result = f"[{t['name']}] {t.get('arguments', {})}"
                results.append({"tool": t["name"], "result": result})
                messages.append({"role": "user", "content": f"Observation: {result}"})

        return {
            "answer": "达到最大迭代次数",
            "tool_calls": tool_calls,
            "tool_results": results,
            "iterations": self.max_iter,
        }

    def cot(self, question: str) -> str:
        """Zero-shot CoT（旧 API 兼容）。"""
        return self.cot.reason(question)
