"""
传输层 - 处理消息的网络传输
"""

import json
import aiohttp
from abc import ABC, abstractmethod
from typing import Optional, Callable, Awaitable
from .message import Message


class Transport(ABC):
    """传输层抽象基类"""
    
    @abstractmethod
    async def send(self, message: Message) -> Message:
        """发送消息并接收响应"""
        pass
    
    @abstractmethod
    async def receive(self) -> Message:
        """接收消息"""
        pass
    
    @abstractmethod
    async def start(self):
        """启动传输"""
        pass
    
    @abstractmethod
    async def stop(self):
        """停止传输"""
        pass


class HTTPTransport(Transport):
    """
    HTTP 传输层
    
    使用 HTTP/REST 进行消息传输
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        base_path: str = "/a2a",
        timeout: int = 30,
    ):
        self.host = host
        self.port = port
        self.base_path = base_path
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._server: Optional[aiohttp.web.Application] = None
        self._message_handler: Optional[Callable[[Message], Awaitable[Message]]] = None
    
    @property
    def endpoint(self) -> str:
        """获取本地端点"""
        return f"http://{self.host}:{self.port}{self.base_path}"
    
    async def start(self):
        """启动 HTTP 服务器"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        print(f"[Transport] HTTP 传输已启动：{self.endpoint}")
    
    async def stop(self):
        """停止传输"""
        if self._session:
            await self._session.close()
        print("[Transport] HTTP 传输已停止")
    
    def set_handler(self, handler: Callable[[Message], Awaitable[Message]]):
        """设置消息处理器"""
        self._message_handler = handler
    
    async def send(self, message: Message, target_endpoint: str = None) -> Message:
        """
        发送消息到目标端点
        
        Args:
            message: 要发送的消息
            target_endpoint: 目标端点 URL（可选，从 message.receiver 解析）
        
        Returns:
            响应消息
        """
        if not self._session:
            await self.start()
        
        # 确定目标端点
        if not target_endpoint:
            # 从 receiver 解析端点 (格式：agent_id@host:port)
            receiver = message.receiver
            if "@" in receiver:
                _, host_port = receiver.split("@")
                host, port = host_port.rsplit(":", 1)
                target_endpoint = f"http://{host}:{port}{self.base_path}"
            else:
                raise ValueError(f"无法从 receiver 解析端点：{receiver}")
        
        # 发送 HTTP POST 请求
        try:
            async with self._session.post(
                target_endpoint,
                json=message.to_dict(),
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return Message.from_dict(data)
                else:
                    error_text = await response.text()
                    return Message.create_error(
                        sender=message.receiver,
                        receiver=message.sender,
                        in_reply_to=message.message_id,
                        error_message=f"HTTP 错误：{response.status} - {error_text}",
                        error_code="HTTP_ERROR",
                    )
        except aiohttp.ClientError as e:
            return Message.create_error(
                sender=message.receiver,
                receiver=message.sender,
                in_reply_to=message.message_id,
                error_message=f"网络错误：{str(e)}",
                error_code="NETWORK_ERROR",
            )
        except Exception as e:
            return Message.create_error(
                sender=message.receiver,
                receiver=message.sender,
                in_reply_to=message.message_id,
                error_message=f"未知错误：{str(e)}",
                error_code="UNKNOWN_ERROR",
            )
    
    async def receive(self) -> Message:
        """
        接收消息（需要配合服务器使用）
        
        这个方法通常在服务器端被调用
        """
        raise NotImplementedError("请使用 HTTP 服务器接收消息")
    
    def create_server(self):
        """创建 HTTP 服务器"""
        from aiohttp import web
        
        app = web.Application()
        
        async def handle_message(request):
            """处理传入消息"""
            try:
                data = await request.json()
                message = Message.from_dict(data)
                
                if self._message_handler:
                    response = await self._message_handler(message)
                else:
                    response = Message.create_error(
                        sender="transport",
                        receiver=message.sender,
                        in_reply_to=message.message_id,
                        error_message="没有设置消息处理器",
                        error_code="NO_HANDLER",
                    )
                
                return web.json_response(response.to_dict())
            
            except json.JSONDecodeError as e:
                error_msg = Message.create_error(
                    sender="transport",
                    receiver="unknown",
                    in_reply_to="",
                    error_message=f"JSON 解析错误：{str(e)}",
                    error_code="PARSE_ERROR",
                )
                return web.json_response(error_msg.to_dict(), status=400)
            
            except Exception as e:
                error_msg = Message.create_error(
                    sender="transport",
                    receiver="unknown",
                    in_reply_to="",
                    error_message=f"服务器错误：{str(e)}",
                    error_code="SERVER_ERROR",
                )
                return web.json_response(error_msg.to_dict(), status=500)
        
        app.router.add_post(self.base_path, handle_message)
        app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}))
        
        self._server = app
        return app
    
    async def run_server(self):
        """运行 HTTP 服务器"""
        if not self._server:
            self.create_server()
        
        from aiohttp import web
        
        runner = web.AppRunner(self._server)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        print(f"[Transport] 服务器运行在：http://{self.host}:{self.port}")
        
        # 保持运行
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await runner.cleanup()


class InMemoryTransport(Transport):
    """
    内存传输层（用于测试）
    
    消息直接在内存中传递，不经过网络
    """
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._handlers: Dict[str, Callable] = {}
    
    async def start(self):
        print("[Transport] 内存传输已启动")
    
    async def stop(self):
        print("[Transport] 内存传输已停止")
    
    def register_agent(self, agent_id: str, handler: Callable[[Message], Awaitable[Message]]):
        """注册 Agent"""
        self._queues[agent_id] = asyncio.Queue()
        self._handlers[agent_id] = handler
    
    async def send(self, message: Message) -> Message:
        """发送消息"""
        receiver_id = message.receiver.split("@")[0] if "@" in message.receiver else message.receiver
        
        if receiver_id not in self._queues:
            return Message.create_error(
                sender="transport",
                receiver=message.sender,
                in_reply_to=message.message_id,
                error_message=f"未知的接收者：{receiver_id}",
                error_code="UNKNOWN_RECEIVER",
            )
        
        # 将消息放入接收者队列
        await self._queues[receiver_id].put(message)
        
        # 处理消息并获取响应
        if receiver_id in self._handlers:
            response = await self._handlers[receiver_id](message)
            return response
        
        return Message.create_error(
            sender=receiver_id,
            receiver=message.sender,
            in_reply_to=message.message_id,
            error_message="没有处理器",
            error_code="NO_HANDLER",
        )
    
    async def receive(self, agent_id: str) -> Message:
        """接收消息"""
        if agent_id not in self._queues:
            raise ValueError(f"未注册的 Agent: {agent_id}")
        
        return await self._queues[agent_id].get()
