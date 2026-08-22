"""
§07 + §10 — LangSmith / OpenTelemetry 可观测性集成

提供 Agent 执行追踪、评估和可视化能力。
- LangSmithTracer: LangSmith 原生追踪（推荐）
- SimpleTracer: 轻量级本地追踪（无外部依赖）
- OTelExporter: OpenTelemetry 标准导出

Usage:
    # LangSmith (requires LANGCHAIN_API_KEY env var)
    tracer = LangSmithTracer(project_name="agent-demo")
    
    # Simple (local only)
    tracer = SimpleTracer(output_dir="traces/")
    
    # In graph
    from agent_demo.observability import SimpleTracer
    tracer = SimpleTracer()
    result = await graph.ainvoke(initial, config={"callbacks": [tracer]})
    
    # Manual trace
    with tracer.trace("research_step", metadata={"step": step_data}):
        result = await do_research()
        tracer.record_tool_call("search_web", {"query": "xxx"}, 0.5, "success")
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────
# Trace 数据模型
# ──────────────────────────────────────────────

@dataclass
class SpanEvent:
    """Span 内的事件。"""
    name: str
    timestamp: float
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """单个追踪 Span（对应一个操作/工具调用/LLM 请求）。"""
    trace_id: str
    span_id: str
    name: str
    start_time: float
    end_time: float = 0.0
    status: str = "pending"  # pending | ok | error
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    error: str = ""
    parent_span_id: str = ""

    @property
    def duration_ms(self) -> float:
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "error": self.error,
        }

    def complete(self, status: str = "ok", error: str = ""):
        self.end_time = time.time()
        self.status = status
        self.error = error


@dataclass
class Trace:
    """完整追踪记录。"""
    trace_id: str
    name: str
    start_time: float
    end_time: float = 0.0
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────
# SimpleTracer — 轻量级本地追踪（无外部依赖）
# ──────────────────────────────────────────────

class SimpleTracer:
    """
    轻量级本地追踪器，不需要外部服务。
    追踪数据保存到本地 JSON 文件，支持离线分析。

    兼容 LangChain Callback 接口子集。
    """

    def __init__(self, output_dir: str | Path = "traces", max_traces: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_traces = max_traces

        self._active_traces: dict[str, Trace] = {}
        self._active_spans: dict[str, Span] = {}
        self._current_trace: Trace | None = None
        self._current_span: Span | None = None

    # ── 上下文管理 ──

    def start_trace(self, name: str, metadata: dict | None = None) -> Trace:
        """开始新的追踪。"""
        trace_id = str(uuid.uuid4())[:8]
        trace = Trace(
            trace_id=trace_id,
            name=name,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self._active_traces[trace_id] = trace
        self._current_trace = trace
        return trace

    def end_trace(self, trace_id: str | None = None) -> Trace | None:
        """结束追踪并保存。"""
        tid = trace_id or (self._current_trace.trace_id if self._current_trace else None)
        if not tid or tid not in self._active_traces:
            return None

        trace = self._active_traces[tid]
        trace.end_time = time.time()
        del self._active_traces[tid]

        # 保存为 JSON
        self._save_trace(trace)

        # 清理旧 traces
        self._prune_traces()

        return trace

    def start_span(self, name: str, attributes: dict | None = None, parent_id: str | None = None) -> Span:
        """开始新的 Span。"""
        if not self._current_trace:
            self.start_trace("auto")

        trace_id = self._current_trace.trace_id
        span_id = str(uuid.uuid4())[:8]
        parent_span_id = parent_id or (self._current_span.span_id if self._current_span else "")

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            name=name,
            start_time=time.time(),
            attributes=attributes or {},
            parent_span_id=parent_span_id,
        )

        self._active_spans[span_id] = span
        self._current_trace.spans.append(span)
        self._current_span = span

        return span

    def end_span(self, span_id: str | None = None, status: str = "ok", error: str = "") -> Span | None:
        """结束 Span。"""
        sid = span_id or (self._current_span.span_id if self._current_span else None)
        if not sid or sid not in self._active_spans:
            return None

        span = self._active_spans[sid]
        span.complete(status=status, error=error)
        del self._active_spans[sid]

        return span

    def add_event(self, name: str, attributes: dict | None = None):
        """在当前 Span 添加事件。"""
        if self._current_span:
            event = SpanEvent(name=name, timestamp=time.time(), attributes=attributes or {})
            self._current_span.events.append(event)

    # ── LangChain Callback 兼容 ──

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        self.start_span("llm_call", attributes={
            "model": serialized.get("kwargs", {}).get("model_name", "unknown"),
            "prompt_length": sum(len(p) for p in prompts),
        })

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        if self._current_span:
            token_usage = getattr(response, "usage_metadata", {})
            self._current_span.attributes["output_tokens"] = token_usage.get("total_tokens", 0)
            self.end_span(status="ok")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        self.end_span(status="error", error=str(error))

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        self.start_span("tool_call", attributes={
            "tool_name": serialized.get("name", "unknown"),
            "input": input_str[:200],
        })

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        if self._current_span:
            self._current_span.attributes["output_preview"] = output[:200]
            self.end_span(status="ok")

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        self.end_span(status="error", error=str(error))

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:
        self.start_span("chain", attributes={
            "chain_name": serialized.get("name", "unknown"),
            "input_keys": list(inputs.keys()),
        })

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        self.end_span(status="ok")

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        self.end_span(status="error", error=str(error))

    # ── 便捷方法 ──

    def record_tool_call(self, tool_name: str, args: Any, duration_s: float, status: str) -> Span:
        """记录工具调用（手动模式）。"""
        span = self.start_span(tool_name, attributes={"args": str(args)[:500]})
        span.end_time = span.start_time + duration_s
        span.status = status
        return span

    def record_llm_call(self, model: str, prompt_len: int, output_len: int, duration_s: float) -> Span:
        """记录 LLM 调用（手动模式）。"""
        span = self.start_span("llm_call", attributes={
            "model": model,
            "prompt_len": prompt_len,
            "output_len": output_len,
        })
        span.end_time = span.start_time + duration_s
        span.status = "ok"
        return span

    # ── 内部 ──

    def _save_trace(self, trace: Trace):
        path = self.output_dir / f"trace-{trace.trace_id}-{trace.name.replace(' ', '_')}.json"
        with open(path, "w") as f:
            json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)

    def _prune_traces(self):
        """保留最近 max_traces 个文件。"""
        files = sorted(self.output_dir.glob("trace-*.json"), key=lambda p: p.stat().st_mtime)
        for f in files[:-self.max_traces]:
            f.unlink()

    def summary(self) -> dict:
        """汇总最近 traces 的统计。"""
        files = sorted(self.output_dir.glob("trace-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
        total_spans = 0
        total_duration = 0
        errors = 0

        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            total_spans += len(data.get("spans", []))
            total_duration += data.get("duration_ms", 0)
            errors += sum(1 for s in data.get("spans", []) if s.get("status") == "error")

        return {
            "traces_loaded": len(files),
            "total_spans": total_spans,
            "total_duration_ms": round(total_duration, 2),
            "avg_duration_ms": round(total_duration / len(files), 2) if files else 0,
            "errors": errors,
            "output_dir": str(self.output_dir),
        }


# ──────────────────────────────────────────────
# LangSmithTracer — LangSmith 原生追踪
# ──────────────────────────────────────────────

class LangSmithTracer:
    """
    LangSmith 原生追踪器。
    需要设置环境变量:
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=ls_xxx
      - LANGCHAIN_PROJECT=agent-demo (可选)

    提供自动追踪 LLM 调用、工具调用、链执行。
    支持 LangSmith 可视化面板。
    """

    def __init__(self, project_name: str = "agent-demo"):
        self.project_name = project_name
        self._enabled = self._check_env()
        self._tracer = None

        if self._enabled:
            try:
                from langsmith import Client
                from langsmith.wrappers import wrap_openai
                self._client = Client()
                self._enabled = True
            except ImportError:
                self._enabled = False

    def _check_env(self) -> bool:
        """检查环境变量是否配置。"""
        return bool(
            os.environ.get("LANGCHAIN_TRACING_V2") == "true"
            and os.environ.get("LANGCHAIN_API_KEY", "").startswith("ls_")
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_callback(self):
        """
        获取 LangChain Callback Handler。
        用法: graph.ainvoke(initial, config={"callbacks": [tracer.get_callback()]})
        """
        if not self._enabled:
            return None

        try:
            from langchain.callbacks.tracers import LangChainTracer
            return LangChainTracer(project_name=self.project_name)
        except ImportError:
            return None

    def trace_agent_run(self, question: str, result: dict) -> str | None:
        """手动记录一次 Agent 运行到 LangSmith。"""
        if not self._enabled:
            return None

        try:
            run_id = self._client.create_run(
                name="agent_run",
                inputs={"question": question},
                outputs={"answer": result.get("answer", "")},
                project_name=self.project_name,
                run_type="chain",
            )
            return str(run_id)
        except Exception:
            return None

    def status(self) -> dict:
        return {
            "type": "langsmith",
            "enabled": self._enabled,
            "project": self.project_name,
            "api_key_configured": bool(os.environ.get("LANGCHAIN_API_KEY", "").startswith("ls_")),
        }


# ──────────────────────────────────────────────
# OTelExporter — OpenTelemetry 标准导出
# ──────────────────────────────────────────────

class OTelExporter:
    """
    OpenTelemetry 导出器，将 Agent 追踪数据导出到 OTLP 端点。
    需要 opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp。

    用法:
        exporter = OTelExporter(endpoint="http://localhost:4317")
        with exporter.trace("research") as span:
            await do_work()
    """

    def __init__(self, endpoint: str = "http://localhost:4317", service_name: str = "agent-demo"):
        self.endpoint = endpoint
        self.service_name = service_name
        self._enabled = False
        self._tracer = None

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(
                # 批量导出
                __import__("opentelemetry.sdk.trace.export").sdk.trace.export.BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=endpoint)
                )
            )
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(service_name)
            self._enabled = True
        except ImportError:
            pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    def trace(self, name: str, attributes: dict | None = None) -> "_OTelSpanContext":
        """创建追踪上下文。"""
        return _OTelSpanContext(self._tracer, name, attributes)

    def status(self) -> dict:
        return {
            "type": "opentelemetry",
            "enabled": self._enabled,
            "endpoint": self.endpoint,
            "service": self.service_name,
        }


class _OTelSpanContext:
    """OTel Span 上下文管理器。"""

    def __init__(self, tracer, name: str, attributes: dict | None = None):
        self._tracer = tracer
        self._name = name
        self._attributes = attributes or {}
        self._span = None

    def __enter__(self):
        if self._tracer:
            self._span = self._tracer.start_span(self._name)
            for k, v in self._attributes.items():
                self._span.set_attribute(k, v)
        return self

    def __exit__(self, *args):
        if self._span:
            self._span.end()

    def record_event(self, name: str, attributes: dict | None = None):
        if self._span:
            self._span.add_event(name, attributes)


# ──────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────

def create_tracer(
    backend: str = "simple",
    *,
    output_dir: str = "traces",
    project_name: str = "agent-demo",
    otel_endpoint: str = "http://localhost:4317",
) -> SimpleTracer | LangSmithTracer | OTelExporter:
    """
    创建 Tracer 实例。

    Args:
        backend: "simple" | "langsmith" | "otel"
        output_dir: 本地追踪输出目录
        project_name: LangSmith 项目名称
        otel_endpoint: OTLP 端点

    Returns:
        对应后端的 Tracer 实例
    """
    backends = {
        "simple": lambda: SimpleTracer(output_dir=output_dir),
        "langsmith": lambda: LangSmithTracer(project_name=project_name),
        "otel": lambda: OTelExporter(endpoint=otel_endpoint),
    }

    factory = backends.get(backend)
    if not factory:
        raise ValueError(f"Unknown tracer backend: {backend}. Choose from: {list(backends.keys())}")

    return factory()
