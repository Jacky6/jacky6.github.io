#!/usr/bin/env python3
"""
Agent Demo — CLI Entry Point

Usage:
    python3 -m agent_demo "北京今天天气怎么样"
    python3 -m agent_demo --help
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure demo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="AI Agent Demo — 智能体架构演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 -m agent_demo "北京今天天气怎么样"
  python3 -m agent_demo "如何设计一个分布式缓存系统？" --mock
  python3 -m agent_demo "搜索 Python 异步编程最佳实践" --verbose
        """,
    )

    parser.add_argument("question", nargs="?", default="如何使用 LangGraph 构建多 Agent 系统？", help="要问的问题")
    parser.add_argument("--mock", action="store_true", help="使用 MockLLM（默认）")
    parser.add_argument("--real", action="store_true", help="使用真实 LLM (需要 OPENAI_API_KEY)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志输出")
    parser.add_argument("--checkpoints", action="store_true", help="启用 Checkpointer")
    parser.add_argument("--traces", action="store_true", help="启用可观测性追踪")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    parser.add_argument("--max-steps", type=int, default=20, help="最大步骤数 (默认 20)")
    parser.add_argument("--max-tokens", type=int, default=8000, help="最大 Token 预算 (默认 8000)")

    args = parser.parse_args()

    # LLM 选择
    if args.real:
        from agent_demo.llm_real import RealLLM
        llm = RealLLM()
        print("🔵 使用真实 LLM")
    else:
        from agent_demo.llm_mock import MockLLM
        llm = MockLLM()
        print("🟡 使用 MockLLM")

    # 构建 Agent
    from agent_demo.graph import ResearchAgent, build_graph
    from agent_demo.checkpointer import create_saver
    from agent_demo.observability import create_tracer

    agent = ResearchAgent(llm=llm)

    # Checkpointer
    saver = None
    if args.checkpoints:
        saver = create_saver("sqlite", db_path="checkpoints/demo.db")
        print(f"🟢 Checkpointer: SQLiteSaver")

    # Tracer
    tracer = None
    if args.traces:
        tracer = create_tracer("simple", output_dir="traces/")
        print(f"🟢 Tracer: SimpleTracer")

    # 构建图
    graph = build_graph(agent, saver=saver)
    print(f"\n🤖 Agent Demo 启动")
    print(f"   问题: {args.question}")
    print(f"   最大步骤: {args.max_steps}")
    print(f"   最大 Token: {args.max_tokens}")
    print(f"   流式输出: {'❌ 关闭' if args.no_stream else '✅ 开启'}")
    print(f"   Checkpointer: {'✅ 启用' if saver else '❌ 未启用'}")
    print(f"   Tracer: {'✅ 启用' if tracer else '❌ 未启用'}")
    print(f"{'─' * 50}")

    # 运行
    initial_state = {
        "question": args.question,
        "messages": [{"role": "user", "content": args.question}],
        "answer": "",
        "stream_content": "",
        "current_step": {"title": "分析中...", "status": "pending"},
        "iteration": 0,
        "revision_count": 0,
        "reflection": {"score": 0, "needs_revision": False},
        "route": "researcher",
        "tool_calls": [],
        "observations": [],
        "perception": {},
        "reasoning_strategy": "default",
        "safety_status": "clean",
        "loop_status": "ok",
    }

    try:
        result = asyncio.run(graph.ainvoke(
            initial_state,
            config={
                "recursion_limit": args.max_steps,
                "callbacks": [tracer] if tracer else None,
                "configurable": {"thread_id": "demo-run-001"} if saver else {},
            },
        ))

        print(f"\n{'─' * 50}")
        print(f"📝 回答:")
        print(result.get("answer", "(无回答)"))
        print(f"\n{'─' * 50}")
        print(f"✅ 完成 | 步骤: {result.get('iteration', '?')} | 修订: {result.get('revision_count', 0)}")

        if tracer:
            summary = tracer.summary()
            print(f"   追踪: {summary.get('traces_loaded', 0)} traces, {summary.get('total_spans', 0)} spans")

        if saver:
            stats = saver.stats()
            print(f"   Checkpoints: {stats.get('total_checkpoints', 0)} saved")

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
