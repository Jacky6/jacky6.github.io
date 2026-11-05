"""
§00 - State 定义 (教程 00 - 架构总览)

LangGraph 的核心是 StateGraph + TypedDict。
所有节点通过返回 dict 来增量更新共享状态（Delta 更新）。
"""

from __future__ import annotations
from typing import TypedDict, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """
    Agent 共享状态。
    total=False = 所有字段可选，节点只返回需要更新的 Delta。
    """
    messages: list[dict]                    # 对话历史
    plan: list[dict]                        # 分解后的步骤
    current_step: Optional[dict]            # 当前执行步骤
    tool_calls: list[dict]                  # LLM 发出的工具调用
    tool_results: list[dict]               # 工具执行结果
    answer: str                             # 最终答案
    reflection: dict                        # {"score", "needs_revision", "feedback"}
    memory: dict                            # 记忆元数据
    token_usage: int                        # 累计 Token 消耗
    iteration: int                          # 当前迭代轮数
    revision_count: int                     # 自检修订次数（防无限循环）
    route: str                              # 下一个要执行的节点名
    approval_required: bool                 # 是否需要人工审批
    approved: bool                          # 是否已审批
    quality_score: float                    # 最终质量评分
    # 流式输出
    stream_content: str                     # 流式生成的累积内容
