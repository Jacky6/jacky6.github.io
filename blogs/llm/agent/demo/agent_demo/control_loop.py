"""
§07 - Control Loop 控制循环

Five-layer loop defense:
  1. 最大迭代次数 (MAX_ITER)
  2. Token 预算 (TokenBudget)
  3. 超时控制 (TimeoutGuard)
  4. 状态收敛检测 (StateConvergenceDetector) — 防死循环
  5. 终止条件检测 (TerminationChecker)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopConfig:
    """循环配置。"""

    max_iterations: int = 8
    max_tokens: int = 10000
    max_time_seconds: float = 60.0
    convergence_window: int = 3  # 最近 N 步用于收敛检测
    convergence_threshold: float = 0.1  # 状态变化低于此值视为收敛


class LoopGuard:
    """循环守卫——五层防御。"""

    def __init__(self, config: LoopConfig | None = None):
        self.config = config or LoopConfig()
        self.used_tokens: int = 0
        self.iteration: int = 0
        self.start_time: float = time.time()
        self._stopped: bool = False
        self._stop_reason: str = ""

        # Convergence detection
        self._state_hashes: deque[str] = deque(maxlen=self.config.convergence_window)

    # ── Layer 1: 最大迭代 ──

    def check_max_iterations(self) -> tuple[bool, str]:
        if self.iteration > self.config.max_iterations:
            return False, f"达到最大迭代次数 ({self.iteration}/{self.config.max_iterations})"
        return True, ""

    # ── Layer 2: Token 预算 ──

    def check_token_budget(self) -> tuple[bool, str]:
        if self.used_tokens > self.config.max_tokens:
            return False, f"超出 Token 预算 ({self.used_tokens}/{self.config.max_tokens})"
        return True, ""

    # ── Layer 3: 超时控制 ──

    def check_timeout(self) -> tuple[bool, str]:
        elapsed = time.time() - self.start_time
        if elapsed > self.config.max_time_seconds:
            return False, f"运行超时 ({elapsed:.1f}s / {self.config.max_time_seconds}s)"
        return True, ""

    # ── Layer 4: 状态收敛检测 ──

    def check_convergence(self, state: dict[str, Any] | None = None) -> tuple[bool, str]:
        """
        检测状态是否陷入死循环。

        如果最近 N 步的状态摘要变化很小，认为 Agent 在打转。
        """
        if state is None:
            return True, ""

        # 生成当前步的状态摘要
        summary = self._state_summary(state)
        state_hash = self._hash(summary)

        # 记录状态
        self._state_hashes.append(state_hash)

        # 窗口满了才检测
        if len(self._state_hashes) < self.config.convergence_window:
            return True, ""

        # 检查是否有变化
        unique_hashes = set(self._state_hashes)
        if len(unique_hashes) == 1:
            return False, "状态收敛：连续 3 步状态相同，疑似死循环"

        # 检查是否高度重复（超过 80% 相同）
        most_common = max(set(self._state_hashes), key=list(self._state_hashes).count)
        repeat_ratio = list(self._state_hashes).count(most_common) / len(self._state_hashes)
        if repeat_ratio > 0.8:
            return False, f"状态高度重复（{repeat_ratio:.0%}），疑似无效循环"

        return True, ""

    # ── Layer 5: 终止条件检测 ──

    def check_termination(self, state: dict[str, Any] | None = None) -> tuple[bool, str]:
        """检查是否有明确的终止信号。"""
        if state is None:
            return True, ""

        if state.get("is_done"):
            return False, "任务完成标记"
        if state.get("stop_requested"):
            return False, "用户请求停止"
        if state.get("error") and state.get("error") != "":
            return False, f"发生错误: {state.get('error')}"

        return True, ""

    # ── Main Check ──

    def check(self, state: dict[str, Any] | None = None, token_delta: int = 0) -> tuple[bool, str]:
        """
        执行所有检查。任一失败即终止。

        Args:
            state: 当前状态（用于收敛/终止检测）
            token_delta: 本轮新增 token 数
        """
        if self._stopped:
            return False, f"循环已停止: {self._stop_reason}"

        self.iteration += 1
        self.used_tokens += token_delta

        checks = [
            ("迭代次数", self.check_max_iterations),
            ("Token 预算", self.check_token_budget),
            ("超时控制", self.check_timeout),
            ("状态收敛", lambda: self.check_convergence(state)),
            ("终止条件", lambda: self.check_termination(state)),
        ]

        for name, check_fn in checks:
            ok, reason = check_fn()
            if not ok:
                self._stopped = True
                self._stop_reason = reason
                return False, reason

        return True, f"OK iter={self.iteration} tokens={self.used_tokens}"

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def reset(self):
        self.used_tokens = 0
        self.iteration = 0
        self.start_time = time.time()
        self._stopped = False
        self._stop_reason = ""
        self._state_hashes.clear()

    # ── Helpers ──

    @staticmethod
    def _state_summary(state: dict[str, Any]) -> str:
        """生成状态摘要（用于收敛检测）。"""
        parts = []
        # 关键信号字段
        for key in ("answer", "route", "current_step", "reflection"):
            val = state.get(key)
            if val is not None:
                parts.append(f"{key}={str(val)[:100]}")
        # 消息数量
        msgs = state.get("messages", [])
        parts.append(f"msg_count={len(msgs)}")
        return "|".join(parts)

    @staticmethod
    def _hash(text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:8]


# ── Backward compatible alias ──
ControlLoop = LoopGuard
