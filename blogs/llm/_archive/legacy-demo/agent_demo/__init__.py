"""
Research Agent Demo — 基于 LangGraph 的研究助手

模块化结构:
  agent_demo/
  ├── __init__.py              # 包入口
  ├── state.py                 # 00-State 定义
  ├── memory.py                # 02-Memory 三层存储
  ├── reasoning.py             # 03-推理模式 (ReAct/CoT)
  ├── planning.py              # 04-任务规划
  ├── tools.py                 # 05-工具注册与调用
  ├── architecture.py          # 06-Supervisor 架构
  ├── control_loop.py          # 07-控制循环 & 防循环
  ├── reflection.py            # 08-自检重试
  ├── safety.py                # 09-安全护栏
  ├── evaluation.py            # 10-评估报告
  ├── llm_mock.py              # Mock LLM (开发用)
  └── graph.py                 # LangGraph StateGraph 编排

Usage: python3 -m agent_demo
"""

# Lazy import: ResearchAgent requires langgraph + mcp packages.
# Use `from agent_demo.graph import ResearchAgent` directly for explicit access.
def __getattr__(name):
    if name == "ResearchAgent":
        from agent_demo.graph import ResearchAgent
        return ResearchAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ResearchAgent"]
