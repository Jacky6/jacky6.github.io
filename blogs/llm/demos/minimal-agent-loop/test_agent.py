"""minimal-agent-loop 的单元测试。

验证 concepts/01、02、06 的关键设计：
- 循环自然终止与兜底终止
- 工具结果配对、错误作为观察
- 参数校验、幻觉工具名处理
- 重复动作检测

运行：python3 test_agent.py
"""

from agent import (
    Agent, ScriptedModel, ModelResponse, ToolCall, make_tools,
)


def test_natural_stop():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "北京"})]),
        ModelResponse(text="北京晴 25°C"),
    ])
    result = Agent(model, make_tools()).run("北京天气")
    assert result["stop_reason"] == "natural"
    assert result["answer"] == "北京晴 25°C"
    assert len(result["trace"]) == 2  # 1 次工具 + 1 次终止
    print("✅ test_natural_stop")


def test_max_iterations():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall(id=f"c{i}", name="add", arguments={"a": i, "b": 1})])
        for i in range(20)
    ])
    result = Agent(model, make_tools(), max_iterations=5).run("死循环")
    assert result["stop_reason"].startswith("max_iterations")
    assert result["answer"] is None
    print("✅ test_max_iterations")


def test_repeated_action():
    same = ToolCall(id="c", name="get_weather", arguments={"city": "北京"})
    model = ScriptedModel([ModelResponse(tool_calls=[same]) for _ in range(10)])
    result = Agent(model, make_tools()).run("重复")
    assert result["stop_reason"] == "repeated_action"
    print("✅ test_repeated_action")


def test_error_as_observation():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "深圳"})]),
        ModelResponse(text="深圳暂无数据"),
    ])
    result = Agent(model, make_tools()).run("深圳天气")
    # 错误观察进入轨迹，模型基于它回答
    assert result["trace"][0]["result"] == "深圳：暂无数据"
    assert result["answer"] == "深圳暂无数据"
    print("✅ test_error_as_observation")


def test_param_validation():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1})]),
        ModelResponse(text="缺参数"),
    ])
    result = Agent(model, make_tools()).run("加法")
    assert result["trace"][0]["is_error"] is True
    assert "参数错误" in result["trace"][0]["result"]
    print("✅ test_param_validation")


def test_unknown_tool():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="search_web", arguments={})]),
        ModelResponse(text="已纠正"),
    ])
    result = Agent(model, make_tools()).run("幻觉")
    assert result["trace"][0]["is_error"] is True
    assert "未知工具" in result["trace"][0]["result"]
    assert "get_weather" in result["trace"][0]["result"]  # 返回可用列表
    print("✅ test_unknown_tool")


def test_tool_call_id_pairing():
    """工具结果必须与请求配对（concepts/02 第 4 节）。"""
    model = ScriptedModel([
        ModelResponse(tool_calls=[
            ToolCall(id="c1", name="get_weather", arguments={"city": "北京"}),
            ToolCall(id="c2", name="get_weather", arguments={"city": "上海"}),
        ]),
        ModelResponse(text="done"),
    ])
    result = Agent(model, make_tools()).run("两地天气")
    # 并行两个调用，各自有结果
    tool_steps = [s for s in result["trace"] if "tool" in s]
    assert len(tool_steps) == 2
    print("✅ test_tool_call_id_pairing")


if __name__ == "__main__":
    test_natural_stop()
    test_max_iterations()
    test_repeated_action()
    test_error_as_observation()
    test_param_validation()
    test_unknown_tool()
    test_tool_call_id_pairing()
    print("\n🎉 全部 7 个测试通过")
