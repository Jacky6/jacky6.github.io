"""
§08 - MCP (Model Context Protocol) 客户端

覆盖:
  - SSE 传输连接 MCP 远程工具服务器
  - 动态发现远端工具 Schema
  - 远端工具调用 + 本地回退
  - MCP vs Function Calling 对比

参考配置:
  {
    "sls_tb_ocr": {
      "type": "sse",
      "url": "https://aliyun-er-k-liv-xlpqubyrhq.cn-beijing.fcapp.run/sse",
      "headers": {
        "Authorization": "***"
      }
    }
  }
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from loguru import logger
from mcp import ClientSession
from mcp.client.sse import sse_client


class MCPToolConfig:
    """MCP 工具服务器配置。"""

    def __init__(
        self,
        name: str,
        url: str,
        auth_token: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.name = name
        self.url = url
        self.headers = headers or {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"


class MCPClient:
    """
    MCP SSE 客户端（异步）。

    生命周期:
      1. connect()   — 建立 SSE 连接 (async context)
      2. list_tools() — 获取远端工具列表
      3. call_tool()  — 执行工具调用
      4. close()     — 清理资源
    """

    def __init__(self, config: MCPToolConfig):
        self.config = config
        self._session: ClientSession | None = None
        self._connected = False
        self._sse_streams = None

    async def connect(self) -> None:
        """建立 SSE 连接到 MCP 服务器。"""
        try:
            sse_ctx = sse_client(
                url=self.config.url,
                headers=self.config.headers,
            )
            read_stream, write_stream = await sse_ctx.__aenter__()
            self._sse_streams = sse_ctx

            session_ctx = ClientSession(read_stream, write_stream)
            self._session = await session_ctx.__aenter__()
            self._session_ctx = session_ctx

            await self._session.initialize()
            self._connected = True
            logger.info(f"✅ MCP 已连接: {self.config.name} ({self.config.url})")
        except Exception as e:
            logger.warning(f"⚠️ MCP 连接失败 ({self.config.name}): {e}")
            self._connected = False

    async def close(self) -> None:
        """断开连接并清理资源。"""
        try:
            if self._session and self._session_ctx:
                await self._session_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            if self._sse_streams:
                await self._sse_streams.__aexit__(None, None, None)
        except Exception:
            pass
        self._connected = False
        logger.info(f"🔌 MCP 已断开: {self.config.name}")

    async def list_tools(self) -> list[dict]:
        """获取 MCP 服务器提供的所有工具 Schema。"""
        if not self._connected or not self._session:
            return []
        try:
            response = await self._session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object"},
                    "_source": "mcp",
                    "_mcp_name": self.config.name,
                }
                for tool in response.tools
            ]
        except Exception as e:
            logger.warning(f"⚠️ MCP list_tools 失败: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """执行远端 MCP 工具调用。"""
        if not self._connected or not self._session:
            return f"[MCP 错误] 未连接: {self.config.name}"
        try:
            result = await self._session.call_tool(tool_name, arguments=arguments)
            texts = []
            for content in result.content:
                if hasattr(content, "text") and content.text:
                    texts.append(content.text)
                elif hasattr(content, "type") and content.type == "text":
                    texts.append(str(content))
                else:
                    texts.append(str(content))
            return "\n".join(texts) if texts else "[MCP 工具无输出]"
        except Exception as e:
            logger.warning(f"⚠️ MCP call_tool 失败 ({tool_name}): {e}")
            return f"[MCP 异常] {tool_name}: {e}"

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── 同步便捷接口（非异步上下文使用）──

class AsyncMCPWrapper:
    """
    MCPClient 的异步包装器。
    直接在当前事件循环中保持 SSE 连接。
    """

    def __init__(self, client: MCPClient):
        self._client = client
        self._tools_cache: list[dict] = []

    async def connect(self) -> None:
        await self._client.connect()

    async def list_tools(self) -> list[dict]:
        if not self._tools_cache:
            self._tools_cache = await self._client.list_tools()
        return self._tools_cache

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        return await self._client.call_tool(tool_name, arguments)

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected


# ── 同步便捷接口（非异步上下文使用）──

class SyncMCPWrapper:
    """
    MCPClient 的同步包装器。

    使用持久化事件循环 + 后台线程保持 SSE 连接存活，
    避免每次 asyncio.run() 创建新循环导致上下文失效的问题。
    注意：不能在已运行的事件循环中调用此包装器。
    """

    def __init__(self, client: MCPClient):
        self._client = client
        self._tools_cache: list[dict] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def _run_loop(self):
        """在后台线程中运行事件循环。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
            self._loop = None

    def _run_coro(self, coro):
        """在持久化事件循环中执行协程。"""
        if not self._loop or self._loop.is_closed():
            raise RuntimeError("Event loop not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=30)

    def connect(self) -> None:
        """同步建立连接（保持 SSE 上下文存活）。"""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        self._run_coro(self._client.connect())

    def close(self) -> None:
        """同步断开连接。"""
        if self._client.is_connected:
            self._run_coro(self._client.close())
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    def list_tools(self) -> list[dict]:
        """同步获取工具列表（带缓存）。"""
        if not self._tools_cache:
            self._tools_cache = self._run_coro(self._client.list_tools())
        return self._tools_cache

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """同步执行工具调用。"""
        return self._run_coro(self._client.call_tool(tool_name, arguments))

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected
