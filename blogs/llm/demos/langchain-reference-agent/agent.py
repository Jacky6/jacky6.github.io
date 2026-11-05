"""LangChain 参考实现 — 当前 API 基线（create_agent）。

对应教程：
- agent/concepts/01-Agent-Loop.md（循环）
- agent/concepts/02-Input-Tools-Structured-Output.md（工具）
- agent/concepts/03-State-Context-Memory.md（Checkpointer）
- agent/concepts/07-Safety-and-HITL.md（HITL）

与 minimal-agent-loop 的对照：
- minimal-agent-loop 手写循环，展示原理（零依赖）；
- 本 Demo 用 langchain.agents.create_agent（v1 API），展示生产基线。

版本基线（verified_at 2026-10-29）：
- langchain 1.3.4 / langchain-core 1.4.0 / langgraph 1.2.4

测试策略（concepts/08）：
- 用 GenericFakeChatModel 脚本化模型，无需 API Key；
- 断言轨迹（调用了哪些工具、顺序）而不是只看最终输出。
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# ── 工具层（concepts/02：契约质量决定调用质量）────────────

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。

    当问题涉及城市天气时使用。不要用于时间、数学或其他问题。

    Args:
        city: 城市名，如"北京"。
    """
    fake_db = {"北京": "晴，25°C", "上海": "多云，22°C"}
    return fake_db.get(city, f"{city}：暂无数据")


@tool
def add(a: float, b: float) -> str:
    """计算两个数之和。仅用于加法。"""
    return str(a + b)


TOOLS = [get_weather, add]


# ── 模型层：脚本化替身（concepts/08 的测试姿势）───────────

class ScriptedToolModel(GenericFakeChatModel):
    """支持 bind_tools 的脚本化模型替身。

    GenericFakeChatModel 不实现 bind_tools（create_agent 需要它），
    子类化后直接返回 self——工具 Schema 由 Agent 注入，替身忽略即可。
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedToolModel":
        return self


def make_scripted_model(script: list[AIMessage]) -> ScriptedToolModel:
    """按顺序回放预设响应的模型替身。

    messages 参数是 GenericFakeChatModel 的标准接口：
    传入一个迭代器，每次调用弹出一条。
    """
    return ScriptedToolModel(messages=iter(script))


def weather_script() -> list[AIMessage]:
    """场景 1：模型请求工具 → 收到结果 → 给出最终答案。"""
    return [
        AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "北京"}}],
        ),
        AIMessage(content="北京今天晴，25°C，适合出行。"),
    ]


def error_recovery_script() -> list[AIMessage]:
    """场景 2：错误也是观察——工具失败后模型换策略。"""
    return [
        AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "深圳"}}],
        ),
        # 收到"暂无数据"（错误观察），换城市
        AIMessage(
            content="",
            tool_calls=[{"id": "call_2", "name": "get_weather", "args": {"city": "上海"}}],
        ),
        AIMessage(content="深圳暂无数据；上海多云 22°C。"),
    ]


# ── 场景 1：基础循环（自然终止）────────────────────────────

def demo_basic_loop():
    model = make_scripted_model(weather_script())
    agent = create_agent(model=model, tools=TOOLS)
    result = agent.invoke({"messages": [HumanMessage(content="北京今天天气怎么样？")]})
    return result


# ── 场景 2：错误恢复 ──────────────────────────────────────

def demo_error_recovery():
    model = make_scripted_model(error_recovery_script())
    agent = create_agent(model=model, tools=TOOLS)
    result = agent.invoke({"messages": [HumanMessage(content="深圳和上海天气怎么样？")]})
    return result


# ── 场景 3：Checkpointer 会话记忆（concepts/03）───────────

def demo_session_memory():
    """同一 thread_id 跨调用保留状态（InMemorySaver 仅用于测试）。"""
    model = make_scripted_model([
        AIMessage(content="你好！我是天气助手。"),
        AIMessage(content="你刚才问过北京，它今天晴 25°C。"),
    ])
    agent = create_agent(
        model=model,
        tools=TOOLS,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "session-1"}}

    # 第一轮
    r1 = agent.invoke({"messages": [HumanMessage(content="你好")]}, config=config)
    # 第二轮：同一 thread，模型能看到历史
    r2 = agent.invoke(
        {"messages": [HumanMessage(content="我刚才问了哪个城市？")]}, config=config
    )
    return r1, r2


# ── 轨迹提取（concepts/08：评估看轨迹不只看结果）──────────

def extract_trajectory(result: dict) -> list[dict]:
    """从结果消息中提取轨迹：工具调用、结果、最终答案。"""
    trajectory = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                trajectory.append({"type": "tool_call", "name": tc["name"], "args": tc["args"]})
        elif isinstance(msg, ToolMessage):
            trajectory.append({"type": "tool_result", "name": msg.name, "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            trajectory.append({"type": "final", "content": msg.content})
    return trajectory


if __name__ == "__main__":
    print("=" * 60)
    print("场景 1：基础循环（自然终止）")
    result = demo_basic_loop()
    for step in extract_trajectory(result):
        print(f"  {step}")

    print("=" * 60)
    print("场景 2：错误恢复（错误也是观察）")
    result = demo_error_recovery()
    for step in extract_trajectory(result):
        print(f"  {step}")

    print("=" * 60)
    print("场景 3：Checkpointer 会话记忆")
    r1, r2 = demo_session_memory()
    print(f"  第一轮消息数: {len(r1['messages'])}")
    print(f"  第二轮消息数: {len(r2['messages'])}（含历史，证明状态保留）")
    print(f"  第二轮最终回答: {r2['messages'][-1].content}")
