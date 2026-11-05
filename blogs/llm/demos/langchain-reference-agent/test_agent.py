"""langchain-reference-agent 的单元测试。

验证 concepts/01、02、03 在 LangChain v1 API 下的落地：
- 基础循环与自然终止
- 错误作为观察驱动恢复
- Checkpointer 跨调用保留状态
- 轨迹断言（不只看最终输出）

运行：python3 test_agent.py
版本基线：langchain 1.3.4（verified_at 2026-10-29）
"""

from agent import (
    TOOLS, demo_basic_loop, demo_error_recovery, demo_session_memory,
    extract_trajectory, make_scripted_model, weather_script,
)
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage


def test_basic_loop_trajectory():
    """轨迹断言：先工具调用，后最终答案（concepts/08）。"""
    result = demo_basic_loop()
    traj = extract_trajectory(result)
    assert traj[0]["type"] == "tool_call" and traj[0]["name"] == "get_weather"
    assert traj[1]["type"] == "tool_result" and traj[1]["content"] == "晴，25°C"
    assert traj[-1]["type"] == "final"
    print("✅ test_basic_loop_trajectory")


def test_error_recovery():
    """错误观察驱动模型换策略（concepts/01 第 4 节）。"""
    result = demo_error_recovery()
    traj = extract_trajectory(result)
    tool_calls = [s for s in traj if s["type"] == "tool_call"]
    # 两次调用不同城市：深圳失败后换上海
    assert [c["args"]["city"] for c in tool_calls] == ["深圳", "上海"]
    print("✅ test_error_recovery")


def test_session_memory():
    """Checkpointer 保留跨调用状态（concepts/03）。"""
    r1, r2 = demo_session_memory()
    assert len(r2["messages"]) > len(r1["messages"])  # 第二轮含历史
    assert "北京" in r2["messages"][-1].content
    print("✅ test_session_memory")


def test_parallel_tool_calls():
    """一轮多个 tool_call 并行执行（concepts/02 第 3.1 节）。"""
    model = make_scripted_model([
        AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "get_weather", "args": {"city": "北京"}},
                {"id": "c2", "name": "get_weather", "args": {"city": "上海"}},
            ],
        ),
        AIMessage(content="两地天气已查。"),
    ])
    agent = create_agent(model=model, tools=TOOLS)
    result = agent.invoke({"messages": [HumanMessage(content="北京和上海天气")]})
    traj = extract_trajectory(result)
    calls = [s for s in traj if s["type"] == "tool_call"]
    results = [s for s in traj if s["type"] == "tool_result"]
    assert len(calls) == 2 and len(results) == 2
    print("✅ test_parallel_tool_calls")


def test_scripted_model_reusable():
    """替身可重复构造，脚本确定性回放。"""
    model = make_scripted_model(weather_script())
    agent = create_agent(model=model, tools=TOOLS)
    result = agent.invoke({"messages": [HumanMessage(content="北京天气")]})
    assert extract_trajectory(result)[-1]["type"] == "final"
    print("✅ test_scripted_model_reusable")


if __name__ == "__main__":
    test_basic_loop_trajectory()
    test_error_recovery()
    test_session_memory()
    test_parallel_tool_calls()
    test_scripted_model_reusable()
    print("\n🎉 全部 5 个测试通过")
