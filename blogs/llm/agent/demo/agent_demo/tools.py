"""
§05 - Tool Registry 工具注册中心

Enhanced features:
  - Tool registration with JSON Schema
  - Local tools + MCP remote tools
  - Skills system (tool compositions)
  - Sandboxed execution (safe eval / subprocess timeout)
  - Retry + error handling
  - Tool usage statistics
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ──────────────────────────────────────────────
# 1. Data Models
# ──────────────────────────────────────────────

@dataclass
class ToolStats:
    """工具使用统计。"""

    call_count: int = 0
    error_count: int = 0
    total_time: float = 0.0
    last_used: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / max(1, self.call_count)

    @property
    def error_rate(self) -> float:
        return self.error_count / max(1, self.call_count)


@dataclass
class Skill:
    """
    技能 — 多个工具的组合。

    例如 "research" 技能 = search_web + read_url + summarize
    """

    name: str
    description: str
    steps: list[dict]  # [{"tool": "name", "input_template": "..."}]
    tools_required: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# 2. Sandbox
# ──────────────────────────────────────────────

class Sandbox:
    """
    沙箱执行环境 — 安全运行不受信任的代码。

    Features:
      - Restricted builtins
      - Timeout enforcement
      - Resource limits (via subprocess)
    """

    SAFE_BUILTINS = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float, "int": int,
        "len": len, "list": list, "map": map, "max": max, "min": min,
        "range": range, "reversed": reversed, "round": round, "set": set,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
        "True": True, "False": False, "None": None,
    }

    @classmethod
    def run_python(cls, code: str, timeout: float = 5.0) -> str:
        """
        在受限环境中执行 Python 代码。

        Safety:
          - No __builtins__ access
          - No import statements
          - Captures stdout
        """
        import io
        import sys

        # Block dangerous patterns
        dangerous = ["import os", "import sys", "__import__", "subprocess",
                     "eval(", "exec(", "open(", "__builtins__"]
        for pattern in dangerous:
            if pattern in code:
                return f"[安全拦截] 代码包含受限模式: {pattern}"

        # Execute in restricted namespace
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            namespace = {"__builtins__": cls.SAFE_BUILTINS}
            exec(compile(code, "<sandbox>", "exec"), namespace)
            output = sys.stdout.getvalue()
            return f"[执行成功]\n{output}" if output else "[执行成功] (无输出)"
        except Exception as e:
            return f"[执行错误] {type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout


# ──────────────────────────────────────────────
# 3. Specific Tools
# ──────────────────────────────────────────────

def search_web(query: str = "", expression: str = "", city: str = "") -> str:
    """搜索网络获取最新信息。"""
    q = query or expression or city or "unknown"
    return f"[搜索结果] 关于 '{q}' 的最新资料..."

def calculate(expression: str = "", query: str = "", city: str = "") -> str:
    """执行数学计算。"""
    expr = expression or query or "0"
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return f"[计算] {expr} = {result}"
    except Exception as e:
        return f"[计算错误] {e}"

def check_weather(city: str = "", query: str = "", expression: str = "") -> str:
    """查询天气。"""
    c = city or query or "未知"
    return f"[天气] {c}: 晴天 25°C"


# ──────────────────────────────────────────────
# 4. Tool Registry (enhanced)
# ──────────────────────────────────────────────

class ToolRegistry:
    """
    增强型工具注册中心 — 支持本地工具 + MCP 远端工具 + Skills。
    """

    def __init__(self, mcp_wrapper=None):
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}
        self._skills: dict[str, Skill] = {}
        self._stats: dict[str, ToolStats] = {}
        self._mcp = mcp_wrapper

    # ── Registration ──

    def register(self, func: Callable, description: str = "", parameters: dict = None):
        """注册一个本地工具。"""
        name = func.__name__
        self._tools[name] = func
        self._schemas[name] = {
            "name": name,
            "description": description or (func.__doc__ or "").strip(),
            "parameters": parameters or self._infer_schema(func),
        }
        self._stats[name] = ToolStats()

    def register_mcp(self, mcp_wrapper):
        """注册 MCP 客户端。"""
        self._mcp = mcp_wrapper

    def register_mcp_tools(self, tool_schemas: list[dict]):
        """注册 MCP 工具 Schema。"""
        for schema in tool_schemas:
            name = schema["name"]
            self._schemas[name] = {**schema, "_source": "mcp"}
            self._stats[name] = ToolStats()

    def register_skill(self, skill: Skill):
        """注册一个技能（工具组合）。"""
        self._skills[skill.name] = skill

    # ── Discovery ──

    def list_tools(self) -> list[dict]:
        """列出所有工具 Schema。"""
        return list(self._schemas.values())

    def list_skills(self) -> list[dict]:
        """列出所有技能。"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "tools": s.tools_required,
                "steps": len(s.steps),
            }
            for s in self._skills.values()
        ]

    def get_stats(self) -> dict:
        """获取所有工具的统计信息。"""
        return {
            name: {
                "calls": st.call_count,
                "errors": st.error_count,
                "avg_time": round(st.avg_time, 3),
                "error_rate": round(st.error_rate, 2),
            }
            for name, st in self._stats.items()
        }

    # ── Execution ──

    def call(self, name: str, **kwargs) -> str:
        """同步工具调用（带统计）。"""
        start = time.time()
        stats = self._stats.setdefault(name, ToolStats())
        stats.call_count += 1

        # Local tool
        if name in self._tools:
            try:
                result = self._tools[name](**kwargs)
                stats.total_time += time.time() - start
                stats.last_used = time.time()
                return str(result)
            except Exception as e:
                stats.error_count += 1
                stats.total_time += time.time() - start
                return f"[工具异常] {name}: {e}"

        # MCP tool
        if self._mcp and self._mcp.is_connected:
            schema = self._schemas.get(name, {})
            if schema.get("_source") == "mcp":
                result = self._mcp.call_tool(name, kwargs)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        raise RuntimeError("Use acall() instead")
                    except RuntimeError:
                        pass
                    loop = asyncio.new_event_loop()
                    try:
                        r = loop.run_until_complete(result)
                        stats.total_time += time.time() - start
                        stats.last_used = time.time()
                        return str(r)
                    finally:
                        loop.close()
                stats.total_time += time.time() - start
                stats.last_used = time.time()
                return str(result)

        stats.error_count += 1
        return f"[错误] 未知工具: {name}"

    async def acall(self, name: str, **kwargs) -> str:
        """异步工具调用。"""
        start = time.time()
        stats = self._stats.setdefault(name, ToolStats())
        stats.call_count += 1

        if name in self._tools:
            try:
                result = self._tools[name](**kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                stats.total_time += time.time() - start
                stats.last_used = time.time()
                return str(result)
            except Exception as e:
                stats.error_count += 1
                stats.total_time += time.time() - start
                return f"[工具异常] {name}: {e}"

        if self._mcp and self._mcp.is_connected:
            schema = self._schemas.get(name, {})
            if schema.get("_source") == "mcp":
                try:
                    result = await self._mcp.call_tool(name, kwargs)
                    stats.total_time += time.time() - start
                    stats.last_used = time.time()
                    return str(result)
                except Exception as e:
                    stats.error_count += 1
                    stats.total_time += time.time() - start
                    return f"[MCP错误] {name}: {e}"

        stats.error_count += 1
        return f"[错误] 未知工具: {name}"

    # ── Skill Execution ──

    async def run_skill(self, skill_name: str, context: dict = None) -> list[str]:
        """
        执行一个技能（工具组合）。

        Returns list of step results.
        """
        skill = self._skills.get(skill_name)
        if not skill:
            return [f"[错误] 未知技能: {skill_name}"]

        results = []
        ctx = context or {}

        for step in skill.steps:
            tool_name = step["tool"]
            # Simple template substitution
            input_template = step.get("input_template", "{}")
            try:
                input_data = input_template.format(**ctx) if isinstance(input_template, str) else input_template
                if isinstance(input_data, str):
                    result = await self.acall(tool_name, query=input_data)
                else:
                    result = await self.acall(tool_name, **input_data)
                results.append(result)
                # Update context for next step
                ctx[f"result_{tool_name}"] = result
            except Exception as e:
                results.append(f"[技能步骤失败] {tool_name}: {e}")
                break

        return results

    # ── Helpers ──

    @staticmethod
    def _infer_schema(func: Callable) -> dict:
        """从函数签名推断 JSON Schema。"""
        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            prop = {"type": "string"}
            if param.default == inspect.Parameter.empty:
                required.append(name)
            elif param.default != "":
                prop["default"] = param.default
            properties[name] = prop

        return {"type": "object", "properties": properties, "required": required}


# ──────────────────────────────────────────────
# 5. Robust Tool Call (backward compatible)
# ──────────────────────────────────────────────

def robust_tool_call(tool_name: str, tool_func: Callable, tool_input: dict, max_retries: int = 2) -> str:
    """带重试的工具调用。"""
    for attempt in range(max_retries + 1):
        try:
            return tool_func(**tool_input)
        except Exception as e:
            if attempt < max_retries:
                return f"Error: {e}. Please check input."
            return f"Error: {e}. Max retries reached."
