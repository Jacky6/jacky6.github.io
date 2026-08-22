"""
真实 LLM 调用 — 通义千问 (DashScope)

通过 OpenAI 兼容接口调用 qwen3-max。
替换 MockLLM 后, Agent 将使用真实推理。

适配层: 将 ChatOpenAI 封装为 graph 节点期望的 callable 接口。
新增: 异步流式输出支持 (astream)。
"""

from __future__ import annotations
import asyncio
import json
import re
from typing import Callable, AsyncIterator, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# ── 流式回调类型 ──

class StreamCallback:
    """流式输出回调接口。"""

    async def on_token(self, text: str) -> None:
        """每收到一段 token 就调用。"""
        print(text, end="", flush=True)

    async def on_tool_call(self, tool_name: str, args: dict) -> None:
        """工具调用时回调。"""
        pass

    async def update_tool_call(self, tool_name: str, args: dict) -> None:
        """更新工具调用参数（在已知工具名后补充完整参数）。"""
        pass

    async def on_tool_result(self, tool_name: str, result: str) -> None:
        """工具结果返回时回调。"""
        pass

    async def on_done(self) -> None:
        """流式输出结束。"""
        pass


DEFAULT_CALLBACK = StreamCallback()


class RealLLM:
    """
    真实 LLM 适配器。
    对外接口与 MockLLM 一致: llm(messages, tools=..., response_format=...)
    新增: astream() 支持异步流式输出。
    """

    def __init__(self):
        self.chat = ChatOpenAI(
            model="qwen3-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-8b293e101d5e41f1b3116d9848b7ac59",
            streaming=True,
            timeout=30,
            max_retries=2,
        )
        # 非流式客户端：用于 astream 结束时获取完整的 request_id
        self.chat_no_stream = ChatOpenAI(
            model="qwen3-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-8b293e101d5e41f1b3116d9848b7ac59",
            streaming=False,
            timeout=30,
            max_retries=1,
        )

    def __call__(self, messages: list[dict], tools: list[dict] | None = None,
                 temperature: float = 0, response_format: dict | None = None) -> dict:
        # 同步版本 — 仅用于非流式场景（review 等）
        return asyncio.get_event_loop().run_until_complete(
            self.__acall(messages, tools, temperature, response_format)
        )

    async def __acall(self, messages: list[dict], tools: list[dict] | None = None,
                      temperature: float = 0, response_format: dict | None = None) -> dict:
        """内部异步调用。"""
        return await self.acall(messages, tools, temperature, response_format)

    async def acall(self, messages: list[dict], tools: list[dict] | None = None,
                    temperature: float = 0, response_format: dict | None = None) -> dict:
        """公开异步调用接口。"""
        lc_messages = self._to_lc_messages(messages)

        # 自检模式
        if response_format and response_format.get("type") == "review":
            return await self._do_review(lc_messages, messages)

        # 有工具 → bind_tools（用非流式客户端保证 response_metadata 包含 id）
        if tools:
            tool_chat = self.chat_no_stream.bind_tools(tools)
            resp = await tool_chat.ainvoke(lc_messages)

            if hasattr(resp, 'tool_calls') and resp.tool_calls:
                return {
                    "tool_calls": [
                        {
                            "id": tc.get("id", ""),
                            "name": tc["name"],
                            "arguments": tc.get("args", {}),
                        }
                        for tc in resp.tool_calls
                    ],
                    "request_id": self._extract_request_id(resp),
                }

            content = resp.content if hasattr(resp, 'content') else str(resp)
            return {"content": content, "request_id": self._extract_request_id(resp)}

        # 普通调用（用非流式客户端）
        resp = await self.chat_no_stream.ainvoke(lc_messages)
        return {"content": resp.content if hasattr(resp, 'content') else str(resp),
                "request_id": self._extract_request_id(resp)}

    async def astream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        callback: StreamCallback = DEFAULT_CALLBACK,
    ) -> dict:
        """
        异步流式生成。

        返回格式与 __call__ 一致，但过程中通过 callback.on_token()
        实时吐出每个 token。
        额外返回 `request_id` 便于排查问题。
        """
        lc_messages = self._to_lc_messages(messages)
        full_content = []
        last_resp = None  # 保存最后一次响应，用于提取 request_id

        if tools:
            # 工具调用场景：DashScope 流式 tool_calls 的 args 为空，必须用非流式获取完整参数
            # 但先用流式看 LLM 是否会生成"思考文本"（如"我来查询..."），如果有就流式打印
            tool_chat = self.chat.bind_tools(tools)
            detected_tool_name = None
            async for chunk in tool_chat.astream(lc_messages):
                last_resp = chunk
                # 检查是否有 tool_call 意图
                chunk_tc = getattr(chunk, 'tool_calls', None)
                if chunk_tc:
                    for tc in chunk_tc:
                        tname = tc.get('name', '')
                        if tname:
                            detected_tool_name = tname
                            break
                # 文本内容（思考文本）
                if hasattr(chunk, 'content') and chunk.content:
                    text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    full_content.append(text)
                    await callback.on_token(text)
                # 检测到 tool_call → 停止流式
                if detected_tool_name:
                    break

            if detected_tool_name:
                # 立即通知用户：检测到工具调用意图（args 等后续补全）
                await callback.on_tool_call(detected_tool_name, {})
                # 用非流式客户端获取完整 tool_calls + request_id
                full_resp = await self.chat_no_stream.bind_tools(tools).ainvoke(lc_messages)
                last_resp = full_resp
                tool_calls = getattr(full_resp, 'tool_calls', None)
                if not full_content:
                    full_content.append(getattr(full_resp, 'content', '') or '')

                if tool_calls:
                    tc_result = {
                        "tool_calls": [
                            {
                                "id": tc.get("id", ""),
                                "name": tc["name"],
                                "arguments": tc.get("args", {}),
                            }
                            for tc in tool_calls
                        ],
                        "request_id": self._extract_request_id(last_resp),
                    }
                    await callback.on_done()
                    return tc_result

        else:
            # 纯文本流式
            async for chunk in self.chat.astream(lc_messages):
                last_resp = chunk
                if hasattr(chunk, 'content') and chunk.content:
                    text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    full_content.append(text)
                    await callback.on_token(text)

        # 流式 chunk 的 response_metadata 通常没有 id，用非流式客户端再 invoke 一次获取 request_id
        if last_resp is None or not self._extract_request_id(last_resp):
            try:
                full_resp = await self.chat_no_stream.ainvoke(lc_messages)
                last_resp = full_resp
            except Exception:
                pass  # 如果 invoke 失败，request_id 留空

        await callback.on_done()
        return {"content": "".join(full_content),
                "request_id": self._extract_request_id(last_resp)}

    # ── 内部辅助 ──

    @staticmethod
    def _extract_request_id(resp) -> str:
        """从 LangChain 响应中提取 DashScope request_id。"""
        if resp is None:
            return ""
        meta = getattr(resp, 'response_metadata', {}) or {}
        # DashScope 通常放在 response_metadata.request_id 或 response_metadata['x-request-id']
        rid = meta.get('request_id', '') or meta.get('x-request-id', '')
        if rid:
            return rid
        return meta.get('id', '') or ""

    def _to_lc_messages(self, messages: list[dict]) -> list:
        """转换为 LangChain 消息格式。"""
        lc_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                lc_messages.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))
        return lc_messages

    async def _do_review(self, lc_messages: list, messages: list[dict]) -> dict:
        """自检评审。"""
        review_prompt = HumanMessage(
            content=f"Please review the following answer and score it 0-1. "
                    f"Return JSON: {{\"score\": float, \"needs_revision\": bool, \"feedback\": str}}\n\n"
                    f"Answer: {messages[-1].get('content', '')}"
        )
        resp = await self.chat_no_stream.ainvoke([review_prompt])
        text = resp.content if hasattr(resp, 'content') else str(resp)
        match = re.search(r'\{[^}]+\}', text)
        if match:
            try:
                data = json.loads(match.group())
                return {
                    "quality_score": data.get("score", 0.8),
                    "needs_revision": data.get("needs_revision", False),
                    "feedback": data.get("feedback", ""),
                    "request_id": self._extract_request_id(resp),
                }
            except Exception:
                pass
        return {"quality_score": 0.8, "needs_revision": False,
                "feedback": text[:200], "request_id": self._extract_request_id(resp)}


# 单例
llm = RealLLM()
