"""
§06 - Supervisor 架构

Supervisor 模式: LLM 作为项目经理，动态分配任务给 Worker。
架构选择:
  - Single Agent: 简单任务
  - Workflow: 固定流程
  - Team (Supervisor): 复杂多角色协作
"""

from __future__ import annotations
from typing import Callable

MAX_REVISIONS = 2


class Supervisor:
    """Supervisor 节点——动态路由到不同 Agent。"""

    def __init__(self, llm: Callable):
        self.llm = llm

    def route(self, state: dict) -> str:
        """
        根据当前状态决定下一步。
        返回值 = 下一个要执行的节点名。
        """
        reflection = state.get("reflection", {})
        revision_count = state.get("revision_count", 0)
        
        # 修订次数超限 → 强制结束
        if revision_count >= MAX_REVISIONS:
            return "end"

        # 自检不通过 → 重新研究
        if reflection.get("needs_revision"):
            return "researcher"

        # 没有计划 → 先规划
        if not state.get("plan"):
            return "planner"

        # 有答案且通过自检 → 完成
        if state.get("answer") and not reflection.get("needs_revision"):
            return "end"

        # 默认继续研究
        return "researcher"
