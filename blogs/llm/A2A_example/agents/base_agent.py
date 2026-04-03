"""
Agent 基类
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.message import Message, MessageType, MessageStatus, MessagePayload
from protocol.discovery import ServiceDiscovery, ServiceInfo
from protocol.transport import Transport, InMemoryTransport


class AgentState:
    """Agent 状态"""
    IDLE = "idle"           # 空闲
    BUSY = "busy"           # 忙碌
    PROCESSING = "processing"  # 处理中
    ERROR = "error"         # 错误
    STOPPED = "stopped"     # 已停止


class BaseAgent(ABC):
    """
    Agent 基类
    
    所有 Agent 都应该继承此类
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        host: str = "localhost",
        port: int = 8000,
        capabilities: List[dict] = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.host = host
        self.port = port
        self.capabilities = capabilities or []
        
        self._state = AgentState.IDLE
        self._transport: Optional[Transport] = None
        self._discovery: Optional[ServiceDiscovery] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._tasks: Dict[str, dict] = {}  # task_id -> task_info
        
        # 消息处理器
        self._handlers: Dict[str, Callable[[Message], Awaitable[Message]]] = {}
        self._register_default_handlers()
    
    @property
    def address(self) -> str:
        """获取 Agent 地址"""
        return f"{self.agent_id}@{self.host}:{self.port}"
    
    @property
    def state(self) -> str:
        """获取当前状态"""
        return self._state
    
    @state.setter
    def state(self, value: str):
        self._state = value
    
    def _register_default_handlers(self):
        """注册默认处理器"""
        self._handlers["ping"] = self._handle_ping
        self._handlers["status"] = self._handle_status
    
    async def _handle_ping(self, message: Message) -> Message:
        """处理 Ping 请求"""
        return Message.create_response(
            sender=self.address,
            receiver=message.sender,
            in_reply_to=message.message_id,
            content={"status": "pong", "timestamp": datetime.now().isoformat()},
        )
    
    async def _handle_status(self, message: Message) -> Message:
        """处理状态查询"""
        return Message.create_response(
            sender=self.address,
            receiver=message.sender,
            in_reply_to=message.message_id,
            content={
                "agent_id": self.agent_id,
                "name": self.name,
                "state": self._state,
                "active_tasks": len(self._tasks),
                "timestamp": datetime.now().isoformat(),
            },
        )
    
    async def start(self):
        """启动 Agent"""
        print(f"[Agent] 启动：{self.name} ({self.agent_id})")
        self._running = True
        self._state = AgentState.IDLE
        
        # 初始化传输
        self._transport = InMemoryTransport()
        await self._transport.start()
        
        # 注册到传输层
        self._transport.register_agent(self.agent_id, self._handle_message)
        
        # 注册到服务发现
        if self._discovery:
            self._discovery.register_agent(
                agent_id=self.agent_id,
                name=self.name,
                host=self.host,
                port=self.port,
                capabilities=self.capabilities,
            )
        
        print(f"[Agent] {self.name} 已就绪")
    
    async def stop(self):
        """停止 Agent"""
        print(f"[Agent] 停止：{self.name}")
        self._running = False
        self._state = AgentState.STOPPED
        
        if self._transport:
            await self._transport.stop()
    
    async def _handle_message(self, message: Message) -> Message:
        """
        处理传入消息
        
        这是消息处理的主入口
        """
        intent = message.payload.intent
        
        print(f"[Agent] 收到消息：{intent} (from {message.sender})")
        
        # 查找处理器
        if intent in self._handlers:
            try:
                self._state = AgentState.PROCESSING
                response = await self._handlers[intent](message)
                self._state = AgentState.IDLE
                return response
            except Exception as e:
                self._state = AgentState.ERROR
                return Message.create_error(
                    sender=self.address,
                    receiver=message.sender,
                    in_reply_to=message.message_id,
                    error_message=str(e),
                    error_code="HANDLER_ERROR",
                )
        else:
            return Message.create_error(
                sender=self.address,
                receiver=message.sender,
                in_reply_to=message.message_id,
                error_message=f"未知的意图：{intent}",
                error_code="UNKNOWN_INTENT",
            )
    
    def register_handler(self, intent: str, handler: Callable[[Message], Awaitable[Message]]):
        """注册消息处理器"""
        self._handlers[intent] = handler
        print(f"[Agent] 注册处理器：{intent}")
    
    async def send_message(self, message: Message) -> Message:
        """发送消息"""
        if not self._transport:
            raise RuntimeError("Agent 未启动")
        
        print(f"[Agent] 发送消息：{message.payload.intent} (to {message.receiver})")
        response = await self._transport.send(message)
        return response
    
    async def send_request(
        self,
        receiver: str,
        intent: str,
        content: dict,
        **metadata
    ) -> Message:
        """发送请求"""
        message = Message.create_request(
            sender=self.address,
            receiver=receiver,
            intent=intent,
            content=content,
            **metadata
        )
        return await self.send_message(message)
    
    def set_discovery(self, discovery: ServiceDiscovery):
        """设置服务发现"""
        self._discovery = discovery
    
    def get_service_info(self) -> ServiceInfo:
        """获取服务信息"""
        from protocol.discovery import ServiceInfo, ServiceCapability
        
        caps = [
            ServiceCapability(
                name=c.get("name", ""),
                description=c.get("description", ""),
                input_schema=c.get("input_schema", {}),
                output_schema=c.get("output_schema", {}),
                tags=c.get("tags", []),
            )
            for c in self.capabilities
        ]
        
        return ServiceInfo(
            agent_id=self.agent_id,
            name=self.name,
            host=self.host,
            port=self.port,
            capabilities=caps,
        )
    
    @abstractmethod
    async def process_task(self, task: dict) -> dict:
        """
        处理任务（子类必须实现）
        
        Args:
            task: 任务信息
        
        Returns:
            处理结果
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name}({self.agent_id})"
