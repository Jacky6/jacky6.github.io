"""最小 Agent Loop — 概念对应实现。

对应教程：agent/concepts/01-Agent-Loop.md

实现 Agent Loop 的核心 10 行：
    Think（模型决策）→ Act（工具执行）→ Observe（结果入上下文）→ Stop（终止）

设计要点：
- 零外部依赖：不依赖任何 LLM SDK，模型用 ScriptedModel（脚本化回放）
- 终止兜底在运行时（LoopGuard），不依赖模型自觉
- 工具结果带 tool_call_id 配对
- 错误也作为观察返回给模型
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class ToolCall:
    """模型请求的一次工具调用。"""
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """工具执行结果，带 id 与请求配对。"""
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ModelResponse:
    """模型一轮的输出：要么请求工具，要么给出最终答案。"""
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str | None = None


# ── 工具层 ────────────────────────────────────────────────

@dataclass
class Tool:
    """工具 = 名称 + 描述 + 参数 Schema + 执行函数（见 concepts/02）。"""
    name: str
    description: str
    parameters: dict
    func: callable


def make_tools() -> dict[str, Tool]:
    """演示用工具集：只读、无副作用、确定性。"""

    def get_weather(city: str) -> str:
        fake_db = {"北京": "晴，25°C", "上海": "多云，22°C"}
        return fake_db.get(city, f"{city}：暂无数据")

    def add(a: float, b: float) -> str:
        return str(a + b)

    def read_file(path: str) -> str:
        # 演示"错误也是观察"：失败返回结构化错误而不是抛异常
        if path.startswith("/etc") or path.startswith("~/.ssh"):
            return f"权限拒绝：{path} 不在允许范围内"
        return f"（模拟读取 {path} 成功）"

    return {
        "get_weather": Tool(
            name="get_weather",
            description="查询城市天气。当问题涉及实时天气时使用。",
            parameters={"city": {"type": "string", "description": "城市名"}},
            func=get_weather,
        ),
        "add": Tool(
            name="add",
            description="计算两个数的和。仅用于加法，其他运算不要调用。",
            parameters={"a": {"type": "number"}, "b": {"type": "number"}},
            func=add,
        ),
        "read_file": Tool(
            name="read_file",
            description="读取 workspace 内的文件。路径以 /etc 或 ~/.ssh 开头会被拒绝。",
            parameters={"path": {"type": "string", "description": "文件路径"}},
            func=read_file,
        ),
    }


def execute_tool(tool: Tool, call: ToolCall) -> ToolResult:
    """执行工具：校验参数 → 执行 → 包装结果（含错误）。"""
    # 参数校验（Host 责任，不是模型责任）
    missing = [k for k in tool.parameters if k not in call.arguments]
    if missing:
        return ToolResult(
            tool_call_id=call.id,
            content=f"参数错误：缺少 {missing}，请检查后重试",
            is_error=True,
        )
    try:
        content = str(tool.func(**call.arguments))
        return ToolResult(tool_call_id=call.id, content=content)
    except Exception as e:
        return ToolResult(
            tool_call_id=call.id,
            content=f"工具执行失败：{type(e).__name__}: {e}",
            is_error=True,
        )


# ── 模型层（脚本化替身）────────────────────────────────────

class ScriptedModel:
    """脚本化模型：按调用顺序返回预设响应。

    这是测试 Agent 的正确姿势（见 concepts/08）：
    不用关键词模拟"智能"，而是确定性回放，可以断言每轮行为。
    """

    def __init__(self, script: list[ModelResponse]):
        self.script = list(script)
        self.calls: list[list[ToolCall]] = []  # 记录每轮收到的上下文

    def __call__(self, messages: list[dict]) -> ModelResponse:
        self.calls.append([m for m in messages if m.get("role") == "tool"])
        if not self.script:
            return ModelResponse(text="（脚本耗尽，强制结束）")
        return self.script.pop(0)


# ── 运行时层：循环 + 终止兜底 ──────────────────────────────

@dataclass
class Stop:
    reason: str


class LoopGuard:
    """终止兜底（见 concepts/06）：防御结果必须接入路由。"""

    def __init__(self, max_iterations: int = 10, repeat_window: int = 3):
        self.max_iterations = max_iterations
        self.repeat_window = repeat_window
        self.history: list[str] = []

    def check(self, iteration: int, response: ModelResponse) -> Stop | None:
        if iteration >= self.max_iterations:
            return Stop(f"max_iterations({self.max_iterations})")
        # 重复检测：连续 N 次完全相同的工具调用签名
        for tc in response.tool_calls:
            self.history.append(f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True)}")
        window = self.history[-self.repeat_window:]
        if len(window) == self.repeat_window and len(set(window)) == 1:
            return Stop("repeated_action")
        return None


class Agent:
    """最小 Agent：循环 + 工具 + 终止兜底。"""

    def __init__(self, model, tools: dict[str, Tool], max_iterations: int = 10):
        self.model = model
        self.tools = tools
        self.guard = LoopGuard(max_iterations=max_iterations)

    def run(self, task: str) -> dict:
        messages = [{"role": "user", "content": task}]
        trace = []  # 简化版 Trace（见 concepts/08）

        for iteration in range(self.guard.max_iterations + 1):
            response = self.model(messages)

            # 终止兜底：防御结果接入路由
            stop = self.guard.check(iteration, response)
            if stop:
                trace.append({"iteration": iteration, "stop": stop.reason})
                return {"answer": None, "stop_reason": stop.reason, "trace": trace}

            if not response.tool_calls:
                # 自然终止：模型给出最终答案
                trace.append({"iteration": iteration, "final": True})
                return {"answer": response.text, "stop_reason": "natural", "trace": trace}

            # Act + Observe：执行工具，结果入上下文
            for tc in response.tool_calls:
                tool = self.tools.get(tc.name)
                if tool is None:
                    result = ToolResult(
                        tool_call_id=tc.id,
                        content=f"未知工具：{tc.name}，可用工具：{list(self.tools)}",
                        is_error=True,
                    )
                else:
                    result = execute_tool(tool, tc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                })
                trace.append({
                    "iteration": iteration,
                    "tool": tc.name,
                    "arguments": tc.arguments,
                    "result": result.content,
                    "is_error": result.is_error,
                })

        return {"answer": None, "stop_reason": "max_iterations", "trace": trace}


# ── 演示场景 ──────────────────────────────────────────────

def demo_success():
    """场景 1：正常两轮循环（查天气 → 汇总回答）。"""
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "北京"})]),
        ModelResponse(text="北京今天晴，25°C，适合出行。"),
    ])
    agent = Agent(model, make_tools())
    return agent.run("北京今天天气怎么样？")


def demo_error_recovery():
    """场景 2：错误也是观察——模型收到错误后换策略。"""
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "深圳"})]),
        # 第一轮"暂无数据"（错误观察），模型换城市重试
        ModelResponse(tool_calls=[ToolCall(id="c2", name="get_weather", arguments={"city": "上海"})]),
        ModelResponse(text="深圳暂无数据；上海多云 22°C。"),
    ])
    agent = Agent(model, make_tools())
    return agent.run("深圳和上海天气怎么样？")


def demo_loop_detection():
    """场景 3：循环防御——重复调用触发兜底终止。"""
    same_call = ToolCall(id="c", name="get_weather", arguments={"city": "北京"})
    model = ScriptedModel([
        ModelResponse(tool_calls=[same_call]) for _ in range(10)
    ])
    agent = Agent(model, make_tools())
    return agent.run("查天气（模型陷入重复）")


def demo_param_validation():
    """场景 4：参数校验——Host 拦截缺参调用并返回结构化错误。"""
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1})]),  # 缺 b
        ModelResponse(tool_calls=[ToolCall(id="c2", name="add", arguments={"a": 1, "b": 2})]),
        ModelResponse(text="1 + 2 = 3"),
    ])
    agent = Agent(model, make_tools())
    return agent.run("算一下 1+2")


def demo_unknown_tool():
    """场景 5：幻觉工具名——返回可用列表让模型纠正。"""
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall(id="c1", name="search_web", arguments={})]),
        ModelResponse(tool_calls=[ToolCall(id="c2", name="get_weather", arguments={"city": "北京"})]),
        ModelResponse(text="北京晴 25°C。"),
    ])
    agent = Agent(model, make_tools())
    return agent.run("北京天气（模型先幻觉了工具名）")


if __name__ == "__main__":
    demos = [
        ("正常循环", demo_success),
        ("错误恢复", demo_error_recovery),
        ("循环防御", demo_loop_detection),
        ("参数校验", demo_param_validation),
        ("幻觉工具名", demo_unknown_tool),
    ]
    for name, fn in demos:
        result = fn()
        print(f"\n{'=' * 50}")
        print(f"场景：{name}")
        print(f"终止原因：{result['stop_reason']}")
        print(f"最终答案：{result['answer']}")
        print(f"轨迹（{len(result['trace'])} 步）：")
        for step in result["trace"]:
            print(f"  {step}")
