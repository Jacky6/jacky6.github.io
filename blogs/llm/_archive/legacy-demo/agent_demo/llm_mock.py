"""
Mock LLM — 开发用，无需 API key。

替换为真实模型:
  from langchain_openai import ChatOpenAI
  llm = ChatOpenAI(model="gpt-4o")
"""

from __future__ import annotations
import uuid


class MockLLM:
    """
    模拟 LLM 调用。
    
    行为逻辑:
    - 有 tools 且历史中无 Observation → 返回 tool_calls (触发工具)
    - 有 tools 且历史中已有 Observation → 返回最终答案
    - 无 tools → 普通回复
    - response_format=review → 自检评分
    - 内容含 plan/decompose → 返回规划步骤
    """

    def __call__(self, messages: list[dict], tools: list[dict] | None = None,
                 temperature: float = 0, response_format: dict | None = None) -> dict:
        last = messages[-1]["content"] if messages else ""
        has_observation = any("Observation:" in m.get("content", "") for m in messages)

        # 自检模式
        if response_format and response_format.get("type") == "review":
            # 模拟修订轮次限制：第 1 轮需要修订，之后通过
            revision_count = len([m for m in messages if "之前的回答不够好" in m.get("content", "")])
            if revision_count == 0:
                return {"quality_score": 0.55, "needs_revision": True, "feedback": "缺少具体数据支撑"}
            return {"quality_score": 0.85, "needs_revision": False, "feedback": "答案质量良好"}

        # JSON 结构化输出模式（reflection review）
        if response_format and response_format.get("type") == "json_object":
            # 模拟评审结果
            revision_count = len([m for m in messages if "之前的回答不够好" in m.get("content", "")])
            if revision_count == 0:
                return {"content": '{"overall": 0.5, "needs_revision": true, "feedback": "缺少具体数据"}'}
            return {"content": '{"overall": 0.85, "needs_revision": false, "feedback": "答案质量良好"}'}

        # 规划模式
        if any(kw in last.lower() for kw in ["decompose", "plan this", "steps for"]):
            return {"steps": [
                {"id": "step_1", "title": "信息收集", "status": "pending"},
                {"id": "step_2", "title": "数据分析", "status": "pending"},
                {"id": "step_3", "title": "生成报告", "status": "pending"},
            ]}

        # 工具调用模式
        if tools:
            if not has_observation:
                # 第一次调用 → 返回工具调用
                return {"tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": tools[0]["name"],
                    "arguments": {"query": last[:60], "expression": "500*365", "city": "北京"}
                }]}
            else:
                # 已有 Observation → 返回最终答案
                return {"content": "[Mock LLM] 基于工具返回结果，已完成研究分析。"}

        # 普通回复
        return {"content": f"[Mock LLM] 已完成分析。({len(messages)} 轮)"}

    def invoke(self, messages, tools=None, response_format=None, **kwargs):
        """Sync invoke — 兼容 LangChain ChatModel.invoke() 接口。"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list) and messages and not isinstance(messages[0], dict):
            # List of message objects — extract content
            messages = [{"role": m.get("role", "user"), "content": m.get("content", str(m))} for m in messages]
        return self(messages, tools=tools, response_format=response_format, **kwargs)

    async def ainvoke(self, messages, tools=None, response_format=None, **kwargs):
        """Async invoke — 兼容 LangChain ChatModel.ainvoke() 接口。"""
        return self.invoke(messages, tools=tools, response_format=response_format, **kwargs)

    async def astream(self, messages: list[dict], tools: list[dict] | None = None,
                      callback=None, **kwargs):
        """
        异步流式输出，模拟 LLM token-by-token 输出。
        返回格式与 __call__ 一致（dict），但过程中通过 callback 实时推送 token。
        """
        response = self(messages, tools=tools, **kwargs)
        content = response.get("content", "")
        
        # 如果有 tool_calls，直接返回
        if "tool_calls" in response:
            if callback:
                for tc in response["tool_calls"]:
                    if hasattr(callback, 'on_tool_call'):
                        await callback.on_tool_call(tc.get('name', ''), tc.get('arguments', {}))
            return response

        # 逐词模拟流式输出
        chunks = content.split()
        for chunk in chunks:
            text = chunk + " "
            if callback:
                await callback.on_token(text)

        return response

    def with_structured_output(self, schema):
        """
        返回一个包装对象，调用时返回结构化输出。
        兼容 LangChain ChatModel 的 with_structured_output 接口。
        """
        class _StructuredOutputWrapper:
            def __init__(self, llm, schema):
                self.llm = llm
                self.schema = schema

            def invoke(self, messages, **kwargs):
                # Check schema type and return appropriate mock data
                schema_name = getattr(self.schema, '__name__', str(self.schema))
                
                try:
                    # Try to instantiate the schema directly (Pydantic model)
                    if 'PerceptionResult' in schema_name or 'PerceptionResult' in str(self.schema):
                        return self.schema(
                            intent="question",
                            understood="用户询问天气相关信息",
                            key_entities=["北京", "天气"],
                            language="zh",
                            complexity="medium",
                            urgency="low",
                            sentiment="neutral",
                            requires_tool=True,
                            confidence=0.9,
                        )
                    elif 'ReviewResult' in schema_name or 'ReviewResult' in str(self.schema):
                        return self.schema(quality_score=0.85, needs_revision=False, feedback="答案质量良好")
                except Exception:
                    pass
                
                # Generic fallback — return dict
                response = self.llm(messages, **kwargs)
                return response

            async def ainvoke(self, messages, **kwargs):
                return self.invoke(messages, **kwargs)

        return _StructuredOutputWrapper(self, schema)
