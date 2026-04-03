"""
示例 1: 基础通信

演示两个 Agent 之间的基本消息传递
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.message import Message, MessageType
from protocol.transport import InMemoryTransport


async def basic_communication():
    """基础通信示例"""
    print("=" * 60)
    print("A2A 示例 1: 基础通信")
    print("=" * 60)
    
    # 创建传输层
    transport = InMemoryTransport()
    await transport.start()
    
    # 定义消息处理器
    async def agent_b_handler(message: Message) -> Message:
        """Agent B 的处理器"""
        print(f"\n[Agent B] 收到：{message.payload.intent}")
        print(f"         内容：{message.payload.content}")
        
        # 处理不同类型的消息
        if message.payload.intent == "greeting":
            return Message.create_response(
                sender="agent_b@localhost:8002",
                receiver=message.sender,
                in_reply_to=message.message_id,
                content={"response": "你好！很高兴见到你！"},
            )
        
        elif message.payload.intent == "query":
            question = message.payload.content.get("question", "")
            return Message.create_response(
                sender="agent_b@localhost:8002",
                receiver=message.sender,
                in_reply_to=message.message_id,
                content={
                    "answer": f"关于'{question}'的答案是：这是一个示例回答。",
                    "confidence": 0.9,
                },
            )
        
        elif message.payload.intent == "ping":
            return Message.create_response(
                sender="agent_b@localhost:8002",
                receiver=message.sender,
                in_reply_to=message.message_id,
                content={"status": "pong", "timestamp": "2026-04-01T10:00:00"},
            )
        
        else:
            return Message.create_error(
                sender="agent_b@localhost:8002",
                receiver=message.sender,
                in_reply_to=message.message_id,
                error_message=f"未知意图：{message.payload.intent}",
                error_code="UNKNOWN_INTENT",
            )
    
    # 注册 Agent B
    transport.register_agent("agent_b", agent_b_handler)
    
    # ========== 场景 1: 问候 ==========
    print("\n[场景 1] 问候消息")
    print("-" * 40)
    
    greeting_msg = Message.create_request(
        sender="agent_a@localhost:8001",
        receiver="agent_b@localhost:8002",
        intent="greeting",
        content={"message": "你好"},
    )
    
    print(f"[Agent A] 发送：{greeting_msg.to_json(indent=2)}")
    
    response = await transport.send(greeting_msg)
    
    print(f"[Agent A] 收到响应：{response.payload.content}")
    
    # ========== 场景 2: 查询 ==========
    print("\n[场景 2] 查询消息")
    print("-" * 40)
    
    query_msg = Message.create_request(
        sender="agent_a@localhost:8001",
        receiver="agent_b@localhost:8002",
        intent="query",
        content={"question": "什么是 A2A 协议？"},
        priority="high",
    )
    
    print(f"[Agent A] 发送查询：{query_msg.payload.content['question']}")
    
    response = await transport.send(query_msg)
    
    print(f"[Agent A] 收到答案：{response.payload.content['answer']}")
    print(f"         置信度：{response.payload.content['confidence']}")
    
    # ========== 场景 3: Ping/Pong ==========
    print("\n[场景 3] Ping/Pong 测试")
    print("-" * 40)
    
    ping_msg = Message.create_request(
        sender="agent_a@localhost:8001",
        receiver="agent_b@localhost:8002",
        intent="ping",
        content={},
    )
    
    print("[Agent A] 发送 PING...")
    response = await transport.send(ping_msg)
    print(f"[Agent A] 收到 PONG: {response.payload.content}")
    
    # ========== 场景 4: 错误处理 ==========
    print("\n[场景 4] 错误处理")
    print("-" * 40)
    
    error_msg = Message.create_request(
        sender="agent_a@localhost:8001",
        receiver="agent_b@localhost:8002",
        intent="unknown_intent",
        content={},
    )
    
    print("[Agent A] 发送未知意图消息...")
    response = await transport.send(error_msg)
    
    if response.message_type == MessageType.ERROR:
        print(f"[Agent A] 收到错误：{response.error}")
        print(f"         错误码：{response.payload.content.get('error_code')}")
    
    # 清理
    await transport.stop()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(basic_communication())
