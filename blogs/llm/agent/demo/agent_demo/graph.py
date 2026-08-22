"""
§00 + 07 - LangGraph StateGraph 编排

默认使用真实 LLM (qwen3-max), 传 MockLLM() 可回退到模拟模式。
支持异步流式输出：模型生成内容实时通过 callback 吐出。
"""

from __future__ import annotations

import asyncio
from loguru import logger
from langgraph.graph import StateGraph, END

from agent_demo.state import AgentState
from agent_demo.memory import MemoryStore
from agent_demo.perception import PerceptionRouter, PerceptionInput, PerceptionResult
from agent_demo.planning import Planner, DependencyGraph
from agent_demo.tools import ToolRegistry, search_web, calculate, check_weather, Sandbox
from agent_demo.mcp_client import MCPClient, MCPToolConfig, AsyncMCPWrapper
from agent_demo.architecture import Supervisor
from agent_demo.control_loop import ControlLoop
from agent_demo.reflection import Reflector
from agent_demo.safety import SafetyGuard
from agent_demo.evaluation import generate_report, LLMJudge, SystemMetrics
from agent_demo.reasoning import ReasoningEngine, StrategyRouter
from agent_demo.llm_real import RealLLM, StreamCallback
from agent_demo.llm_mock import MockLLM


def _log(node: str, detail: str = ""):
    msg = f"📦 {node}"
    if detail:
        msg += f" — {detail}"
    logger.info(msg)


MAX_REVISIONS = 2


# ──────────────────────────────────────────────────────────────
# 异步流式回调 — 带节点名称前缀
# ──────────────────────────────────────────────────────────────

class AgentStreamCallback(StreamCallback):
    """带节点标注的流式回调。"""

    def __init__(self, node_name: str = "Agent"):
        self.node_name = node_name
        self._started = False
        self._pending_tool = None  # (tool_name,) 等待参数

    async def on_token(self, text: str) -> None:
        # 清除思考提示（用退格覆盖）
        if self._started == "thinking":
            # 退格清除 "🤔 [Researcher] 思考中..."
            thinking_text = f"\n🤔 [{self.node_name}] 思考中..."
            print("\r" + " " * len(thinking_text) + "\r", end="", flush=True)
        if not self._started or self._started == "thinking":
            print(f"\n💬 [{self.node_name}] → ", end="", flush=True)
            self._started = True
        print(text, end="", flush=True)

    async def on_tool_call(self, tool_name: str, args: dict) -> None:
        # 清除思考提示
        if self._started == "thinking":
            thinking_text = f"\n🤔 [{self.node_name}] 思考中..."
            print("\r" + " " * len(thinking_text) + "\r", end="", flush=True)
        # 工具调用意图检测到了，但 args 可能为空 — 先打印占位
        self._pending_tool = tool_name
        print(f"\n🔧 [{self.node_name}] 调用工具: {tool_name}(...)", flush=True)
        self._started = False

    async def update_tool_call(self, tool_name: str, args: dict) -> None:
        """更新工具调用参数（覆盖之前的占位行）。"""
        # 用退格覆盖上一行的占位，打印完整参数
        # 简单做法：不换行，直接在后面打印完整信息
        print(f"   → {tool_name}({args})", flush=True)
        self._pending_tool = None

    async def on_tool_result(self, tool_name: str, result: str) -> None:
        preview = result[:120] + ("..." if len(result) > 120 else "")
        print(f"\n✅ [{self.node_name}] 工具结果: {tool_name} → {preview}", flush=True)
        self._started = False

    async def on_done(self) -> None:
        if self._started:
            print()  # 换行
        self._started = False


# ──────────────────────────────────────────────────────────────
# 节点工厂 — 通过闭包引用 self，延迟绑定 tools
# ──────────────────────────────────────────────────────────────

def _make_planner_node(agent):
    async def node(state: AgentState) -> dict:
        _log("Planner", "分解任务")
        question = state["messages"][-1]["content"]
        callback = AgentStreamCallback("Planner")
        plan_result = await agent.planner.adecompose_task(question, callback=callback)
        # 兼容 list 或 dict{__steps, request_id} 返回
        if isinstance(plan_result, dict):
            plan = plan_result.get("__steps", [])
            req_id = plan_result.get("request_id", "")
        else:
            plan = plan_result
            req_id = ""
        if req_id:
            _log("🆔 plan_request_id", req_id)
        return {"plan": plan, "current_step": plan[0] if plan else None}
    return node


def _make_researcher_node(agent):
    """
    异步 ReAct 单步:
      - 无 Observation → 带 tools, LLM 决定调用（流式输出）
      - 有 Observation → 不带 tools, LLM 生成最终答案（流式输出）
      - 有 revision → 清除旧 Observation, 追加 feedback, 重新带 tools
    """
    async def node(state: AgentState) -> dict:
        tools = agent.tools  # 延迟读取（在 arun 中已更新）
        step = state.get("current_step", {})
        _log("Researcher", f"执行: {step.get('title', 'N/A')}")

        messages = list(state.get("messages", []))
        revision = state.get("reflection", {})
        needs_revision = revision.get("needs_revision", False)
        
        if needs_revision:
            # 修订模式：保留消息但追加 feedback
            messages.append({
                "role": "user",
                "content": f"之前的回答不够好。{revision.get('feedback', '请改进。')}" 
                f"\n\n已有工具结果，请直接基于工具结果生成更好的中文总结。"
            })
            has_observation = True  # 有工具结果 → 不传 tools
        else:
            # 检测是否已经有 Observation（说明工具调用过，需要生成答案）
            has_observation = any("Observation:" in m.get("content", "") for m in messages)
        
        # 有 Observation 时不传 tools → LLM 生成最终答案
        # 有 revision 时也不传 tools → 重新生成
        tool_schemas = None if (has_observation or needs_revision) else tools.list_tools()
        callback = AgentStreamCallback("Researcher")

        # ── 复杂问题自动走 CoT/ToT 推理 ──
        if not needs_revision and not has_observation:
            question = messages[-1].get("content", "") if messages else ""
            strategy = agent.strategy_router.route(question, has_tools=bool(tool_schemas))
            if strategy in ("cot", "tot", "self_consistency"):
                _log("🧠 Strategy", f"{strategy} — 复杂推理")
                cot_result = agent.reasoning_engine.reason(question, strategy=strategy)
                answer = cot_result.get("answer", "")
                return {
                    "answer": answer,
                    "stream_content": answer,
                    "messages": messages + [{"role": "assistant", "content": answer}],
                    "reasoning_strategy": strategy,
                }

        # ── 异步流式调用 LLM ──
        if needs_revision:
            # 修订模式也使用流式
            response = await agent.llm.astream(messages, tools=tool_schemas, callback=callback)
        else:
            # 有 tools 时，先打印一个思考提示，避免 LLM 决策期间的空白
            if tool_schemas:
                print(f"\n🤔 [{callback.node_name}] 思考中...", end="", flush=True)
            response = await agent.llm.astream(messages, tools=tool_schemas, callback=callback)

        tool_calls = response.get("tool_calls", [])
        answer = response.get("content", "")
        request_id = response.get("request_id", "")
        if request_id:
            _log("🆔 request_id", request_id)

        if needs_revision:
            # 修订模式：保留已有工具结果
            tool_results = list(state.get("tool_results", []))
        else:
            tool_results = list(state.get("tool_results", []))
        new_messages = messages

        if tool_calls:
            _log(f"🔧 tool_call", f"{len(tool_calls)} 个")
            for tc in tool_calls:
                tname = tc["name"]
                targs = tc.get("arguments", {})
                # 跳过：astream 里已经通过 on_tool_call 通知过了
                # 用 update_tool_call 更新完整参数
                await callback.update_tool_call(tname, targs)

                extra = {
                    "query": targs.get("query", ""),
                    "expression": targs.get("expression", ""),
                    "city": targs.get("city", ""),
                }
                output = await tools.acall(tname, **{**targs, **extra})
                await callback.on_tool_result(tname, output)

                tool_results.append({"tool": tname, "result": output})
                new_messages = new_messages + [
                    {"role": "assistant", "content": f"Action: {tname}({targs})"},
                    {"role": "user", "content": f"Observation: {output}"},
                ]
                agent.memory.add("assistant", f"[{tname}] {output[:80]}")

        updates: dict = {
            "messages": new_messages,
            "tool_calls": list(state.get("tool_calls", [])) if not needs_revision else tool_calls,
            "tool_results": tool_results,
            "memory": agent.memory.stats(),
        }
        if answer:
            updates["answer"] = answer
            updates["stream_content"] = answer
        if needs_revision:
            updates["revision_count"] = state.get("revision_count", 0) + 1

        return updates
    return node


def _make_reflection_node(agent):
    async def node(state: AgentState) -> dict:
        _log("Reflection", "评估质量")
        review = await agent.reflector.areview(state.get("answer", ""))
        if review.get("score", 0) >= 0.8:
            review["needs_revision"] = False
        req_id = review.get("request_id", "")
        if req_id:
            _log("🆔 review_request_id", req_id)
        _log("📊 Score", f"{review['score']:.0%} | 修订: {review['needs_revision']}")
        return {"reflection": review}
    return node


def _make_safety_node(agent):
    async def node(state: AgentState) -> dict:
        _log("Safety", "循环防御检查")
        token_delta = len(str(state.get("messages", ""))) // 4  # 粗略估算
        ok, msg = agent.loop.check(state=state, token_delta=token_delta)
        _log("✅" if ok else "⚠️", msg)
        return {"iteration": agent.loop.iteration, "token_usage": agent.loop.used_tokens}
    return node


def _make_perception_node(agent):
    """感知节点：意图分类 + 复杂度评估 + 路由建议。"""
    async def node(state: AgentState) -> dict:
        question = state["messages"][-1].get("content", "")
        _log("Perception", f"意图分析: {question[:60]}")

        result = agent.perception_router.process(
            PerceptionInput(content=question, input_type="text")
        )

        _log(
            f"🎯 Intent={result.intent.value} | "
            f"复杂度={result.complexity.value} | "
            f"紧急度={result.urgency.value} | "
            f"需工具={result.requires_tool}"
        )

        # ── LoopGuard check ──
        loop_ok, loop_msg = agent.loop.check(
            state=state,
            token_delta=len(question) // 4
        )
        if not loop_ok:
            _log("⚠️ LoopGuard", loop_msg)

        return {
            "perception": result.model_dump(),
            "route": result.route_to(),
            "iteration": agent.loop.iteration,
            "token_usage": agent.loop.used_tokens,
        }
    return node


def _make_supervisor_node(agent):
    async def node(state: AgentState) -> dict:
        route = agent.supervisor.route(state)
        _log("Supervisor", f"→ {route}")
        return {"route": route}
    return node


# ──────────────────────────────────────────────────────────────
# 路由函数
# ──────────────────────────────────────────────────────────────

def route_after_perception(state: AgentState) -> str:
    """根据感知结果路由到不同处理路径。"""
    perception = state.get("perception", {})
    intent = perception.get("intent", "question")
    route = perception.get("route", "researcher")

    # 安全检查
    if state.get("token_usage", 0) > 9000:
        return "end"

    # 闲聊直接结束（不需要工具/规划）
    if intent == "chitchat":
        return "end"

    # 根据感知路由建议分发
    if route == "tool_use":
        return "planner"  # 需要规划 + 工具
    return "planner"  # 默认都走规划


def route_after_research(state: AgentState) -> str:
    prev_tool_count = len(state.get("tool_results", []))
    all_tools = state.get("tool_calls", [])
    has_answer = bool(state.get("answer"))
    has_new_tools = len(all_tools) > prev_tool_count

    if has_answer and not has_new_tools:
        return "reflection"
    return "researcher"


def route_after_reflection(state: AgentState) -> str:
    revision = state.get("reflection", {})
    revision_count = state.get("revision_count", 0)

    if revision.get("needs_revision") and revision_count < MAX_REVISIONS:
        return "researcher"
    return "supervisor"


def route_supervisor(state: AgentState) -> str:
    revision_count = state.get("revision_count", 0)
    revision = state.get("reflection", {})

    if revision_count >= MAX_REVISIONS:
        return "end"
    if revision.get("needs_revision"):
        return "researcher"
    if state.get("answer"):
        return "end"
    return "researcher"


# ──────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────

def build_graph(agent: ResearchAgent, saver=None):
    """
    构建并编译 LangGraph StateGraph。

    Args:
        agent: ResearchAgent 实例
        saver: 可选的 Checkpointer (MemorySaver / SQLiteSaver)

    Returns:
        编译后的 CompiledStateGraph
    """
    return agent._build_graph(checkpointer=saver)


# ──────────────────────────────────────────────────────────────
# Agent 类
# ──────────────────────────────────────────────────────────────

class ResearchAgent:
    """完整的研 Agent（异步流式）。"""

    def __init__(self, llm=None, mcp_configs: list[MCPToolConfig] | None = None):
        self.llm = llm or RealLLM()
        self.mcp_configs = mcp_configs or []
        self.mcp_wrappers: list[AsyncMCPWrapper] = []

        # 初始化各子模块
        self.perception_router = PerceptionRouter(self.llm)
        self.planner = Planner(self.llm)
        self.reflector = Reflector(self.llm)
        self.supervisor = Supervisor(self.llm)
        self.memory = MemoryStore(max_short=10)
        self.safety = SafetyGuard(max_tokens=10000)
        self.loop = ControlLoop()

        # 推理引擎 + 策略路由器
        self.reasoning_engine = ReasoningEngine(self.llm)
        self.strategy_router = StrategyRouter()

        # 评估 + 系统指标
        self.judge = LLMJudge(self.llm)
        self.metrics = SystemMetrics()

        # 初始只注册本地工具
        self.tools = self._build_local_tools()

        # 图编译延迟到 arun() 中（MCP 连接之后）
        self.graph = None

    def _build_local_tools(self) -> ToolRegistry:
        """仅注册本地工具。"""
        reg = ToolRegistry()
        reg.register(search_web, "搜索网络", {"query": {"type": "string"}})
        reg.register(calculate, "数学计算", {"expression": {"type": "string"}})
        reg.register(check_weather, "查询天气", {"city": {"type": "string"}})
        return reg

    async def _setup_mcp(self):
        """异步连接 MCP 并注册远端工具。"""
        for cfg in self.mcp_configs:
            mcp = MCPClient(cfg)
            await mcp.connect()
            if mcp.is_connected:
                wrapper = AsyncMCPWrapper(mcp)
                self.mcp_wrappers.append(wrapper)
                self.tools.register_mcp(wrapper)
                remote_tools = await wrapper.list_tools()
                self.tools.register_mcp_tools(remote_tools)
                logger.info(f"🔧 MCP 工具已注册: {cfg.name} ({len(remote_tools)} 个远端工具)")
            else:
                logger.warning(f"⚠️ MCP 连接失败，跳过: {cfg.name}")

        all_tools = self.tools.list_tools()
        local_count = sum(1 for t in all_tools if t.get("_source") != "mcp")
        mcp_count = sum(1 for t in all_tools if t.get("_source") == "mcp")
        logger.info(f"📦 工具就绪: {local_count} 本地 + {mcp_count} MCP 远端")

    def _build_graph(self, checkpointer=None):
        """构建并编译 StateGraph。"""
        builder = StateGraph(AgentState)

        builder.add_node("perception", _make_perception_node(self))
        builder.add_node("safety", _make_safety_node(self))
        builder.add_node("planner", _make_planner_node(self))
        builder.add_node("researcher", _make_researcher_node(self))
        builder.add_node("reflection", _make_reflection_node(self))
        builder.add_node("supervisor", _make_supervisor_node(self))

        builder.set_entry_point("perception")
        builder.add_edge("perception", "safety")
        builder.add_conditional_edges("safety", route_after_perception, {
            "planner": "planner", "end": END,
        })
        builder.add_edge("planner", "researcher")
        builder.add_conditional_edges("researcher", route_after_research, {
            "researcher": "researcher",
            "reflection": "reflection",
        })
        builder.add_conditional_edges("reflection", route_after_reflection, {
            "researcher": "researcher",
            "supervisor": "supervisor",
        })
        builder.add_conditional_edges("supervisor", route_supervisor, {
            "researcher": "researcher",
            "planner": "planner",
            "end": END,
        })

        return builder.compile(checkpointer=checkpointer)

    async def arun(self, question: str) -> dict:
        """异步运行 Agent，支持流式输出。"""
        # 1. 连接 MCP + 注册远端工具
        await self._setup_mcp()

        # 2. 编译图（此时 tools 已包含 MCP 工具）
        self.graph = self._build_graph()

        # 3. 重置状态
        self.loop.reset()
        self.memory = MemoryStore(max_short=10)

        logger.info("=" * 50)
        logger.info("🤖 Research Agent — qwen3-max (异步流式)")
        logger.info("=" * 50)

        self.memory.add("user", question)

        initial: AgentState = {
            "messages": [{"role": "user", "content": question}],
            "plan": [], "current_step": None,
            "tool_calls": [], "tool_results": [],
            "answer": "", "reflection": {},
            "memory": {}, "token_usage": 0, "iteration": 0,
            "revision_count": 0,
            "route": "", "approval_required": False,
            "approved": True, "quality_score": 0.0,
            "stream_content": "",
        }

        result = await self.graph.ainvoke(initial, config={"recursion_limit": 25})

        # 优雅断开 MCP 连接（避免 asyncio.run() 退出时的 cleanup 报错）
        for wrapper in self.mcp_wrappers:
            try:
                await wrapper._client.close()
            except Exception:
                pass

        report = generate_report(result, self.memory.stats(), judge=self.judge)
        logger.info(report)

        # System metrics summary
        metrics = self.metrics.snapshot()
        _log("📊 Metrics", f"avg_latency={metrics['avg_latency']}s | error_rate={metrics['error_rate']}")
        
        return result

    def run(self, question: str) -> dict:
        """同步入口。"""
        return asyncio.run(self.arun(question))


if __name__ == "__main__":
    agent = ResearchAgent()
    agent.run("LangGraph 2026 有什么新特性？对比 AutoGen 有什么优势？")
