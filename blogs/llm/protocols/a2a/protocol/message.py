"""
A2A 消息格式定义
"""

import uuid
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional, List


class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"           # 请求
    RESPONSE = "response"         # 响应
    NOTIFICATION = "notification" # 通知
    ACK = "acknowledgement"       # 确认
    ERROR = "error"               # 错误
    STATUS = "status"             # 状态更新


class MessageStatus(Enum):
    """消息状态"""
    PENDING = "pending"       # 待处理
    PROCESSING = "processing" # 处理中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    TIMEOUT = "timeout"       # 超时


@dataclass
class MessagePayload:
    """消息负载"""
    intent: str                      # 意图：task_request, query, response 等
    content: Dict[str, Any]          # 内容
    context: Dict[str, Any] = field(default_factory=dict)  # 上下文
    
    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "content": self.content,
            "context": self.context,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MessagePayload":
        return cls(
            intent=data.get("intent", ""),
            content=data.get("content", {}),
            context=data.get("context", {}),
        )


@dataclass
class Message:
    """
    A2A 消息
    
    消息格式遵循 A2A 协议规范 v1.0
    """
    sender: str                      # 发送者标识 (agent_id@host:port)
    receiver: str                    # 接收者标识
    payload: MessagePayload          # 消息负载
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message_type: MessageType = MessageType.REQUEST
    protocol_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    in_reply_to: Optional[str] = None  # 回复的消息 ID
    error: Optional[str] = None        # 错误信息
    
    # 元数据字段
    @property
    def priority(self) -> str:
        """获取优先级"""
        return self.metadata.get("priority", "normal")
    
    @priority.setter
    def priority(self, value: str):
        self.metadata["priority"] = value
    
    @property
    def ttl(self) -> int:
        """获取生存时间（秒）"""
        return self.metadata.get("ttl", 3600)
    
    @ttl.setter
    def ttl(self, value: int):
        self.metadata["ttl"] = value
    
    @property
    def trace_id(self) -> str:
        """获取追踪 ID"""
        return self.metadata.get("trace_id", str(uuid.uuid4()))
    
    @trace_id.setter
    def trace_id(self, value: str):
        self.metadata["trace_id"] = value
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "timestamp": self.timestamp,
            "message_type": self.message_type.value,
            "protocol_version": self.protocol_version,
            "payload": self.payload.to_dict(),
            "metadata": self.metadata,
            "in_reply_to": self.in_reply_to,
            "error": self.error,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """从字典创建"""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            sender=data.get("sender", ""),
            receiver=data.get("receiver", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            message_type=MessageType(data.get("message_type", "request")),
            protocol_version=data.get("protocol_version", "1.0"),
            payload=MessagePayload.from_dict(data.get("payload", {})),
            metadata=data.get("metadata", {}),
            in_reply_to=data.get("in_reply_to"),
            error=data.get("error"),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """从 JSON 字符串创建"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    # ========== 快捷创建方法 ==========
    
    @classmethod
    def create_request(
        cls,
        sender: str,
        receiver: str,
        intent: str,
        content: dict,
        **metadata
    ) -> "Message":
        """创建请求消息"""
        msg = cls(
            sender=sender,
            receiver=receiver,
            payload=MessagePayload(intent=intent, content=content),
            message_type=MessageType.REQUEST,
        )
        msg.metadata.update(metadata)
        return msg
    
    @classmethod
    def create_response(
        cls,
        sender: str,
        receiver: str,
        in_reply_to: str,
        content: dict,
        **metadata
    ) -> "Message":
        """创建响应消息"""
        msg = cls(
            sender=sender,
            receiver=receiver,
            payload=MessagePayload(intent="response", content=content),
            message_type=MessageType.RESPONSE,
            in_reply_to=in_reply_to,
        )
        msg.metadata.update(metadata)
        return msg
    
    @classmethod
    def create_error(
        cls,
        sender: str,
        receiver: str,
        in_reply_to: str,
        error_message: str,
        error_code: str = "UNKNOWN_ERROR",
        **metadata
    ) -> "Message":
        """创建错误消息"""
        msg = cls(
            sender=sender,
            receiver=receiver,
            payload=MessagePayload(
                intent="error",
                content={"error_code": error_code, "error_message": error_message}
            ),
            message_type=MessageType.ERROR,
            in_reply_to=in_reply_to,
            error=error_message,
        )
        msg.metadata.update(metadata)
        return msg
    
    @classmethod
    def create_status_update(
        cls,
        sender: str,
        receiver: str,
        task_id: str,
        status: MessageStatus,
        progress: float = 0.0,
        message: str = "",
        **metadata
    ) -> "Message":
        """创建状态更新消息"""
        msg = cls(
            sender=sender,
            receiver=receiver,
            payload=MessagePayload(
                intent="status_update",
                content={
                    "task_id": task_id,
                    "status": status.value,
                    "progress": progress,
                    "message": message,
                }
            ),
            message_type=MessageType.STATUS,
        )
        msg.metadata.update(metadata)
        return msg
    
    def __str__(self) -> str:
        return f"Message({self.message_type.value}: {self.sender} → {self.receiver})"
    
    def __repr__(self) -> str:
        return self.to_json(indent=None)
