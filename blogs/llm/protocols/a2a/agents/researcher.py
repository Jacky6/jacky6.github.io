"""
研究员 Agent

负责信息搜索和收集
"""

import asyncio
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, AgentState


class ResearcherAgent(BaseAgent):
    """研究员 Agent"""
    
    def __init__(self, **kwargs):
        capabilities = [
            {
                "name": "information_research",
                "description": "信息搜索和研究",
                "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"findings": {"type": "string"}}},
                "tags": ["research", "search"],
            },
            {
                "name": "data_collection",
                "description": "数据收集",
                "input_schema": {"type": "object", "properties": {"source": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"data": {"type": "string"}}},
                "tags": ["collection"],
            },
        ]
        
        super().__init__(
            agent_id="researcher",
            name="研究员 Agent",
            capabilities=capabilities,
            **kwargs
        )
        
        # 注册处理器
        self.register_handler("execute_task", self._handle_execute_task)
        self.register_handler("research", self._handle_research)
    
    async def _handle_execute_task(self, message: Message) -> Message:
        """处理任务执行请求"""
        task_type = message.payload.content.get("task_type", "")
        description = message.payload.content.get("description", "")
        task_id = message.payload.content.get("task_id", "")
        
        print(f"[Researcher] 执行任务：{task_type} - {description[:50]}...")
        
        # 发送状态更新
        await self._send_status_update(
            receiver=message.sender,
            task_id=task_id,
            status=MessageStatus.PROCESSING,
            progress=0.3,
            message="开始研究...",
        )
        
        # 执行研究
        result = await self._perform_research(description)
        
        # 发送完成状态
        await self._send_status_update(
            receiver=message.sender,
            task_id=task_id,
            status=MessageStatus.COMPLETED,
            progress=1.0,
            message="研究完成",
        )
        
        return Message.create_response(
            sender=self.address,
            receiver=message.sender,
            in_reply_to=message.message_id,
            content={
                "task_id": task_id,
                "status": "completed",
                "result": result,
            },
        )
    
    async def _handle_research(self, message: Message) -> Message:
        """处理研究请求"""
        topic = message.payload.content.get("topic", "")
        
        if not topic:
            return Message.create_error(
                sender=self.address,
                receiver=message.sender,
                in_reply_to=message.message_id,
                error_message="缺少研究主题",
                error_code="MISSING_TOPIC",
            )
        
        result = await self._perform_research(topic)
        
        return Message.create_response(
            sender=self.address,
            receiver=message.sender,
            in_reply_to=message.message_id,
            content={"topic": topic, "findings": result},
        )
    
    async def _perform_research(self, topic: str) -> str:
        """执行研究（模拟实现）"""
        # 实际应用中这里会调用搜索 API
        await asyncio.sleep(1)  # 模拟延迟
        
        # 模拟研究结果
        findings = f"""关于"{topic}"的研究结果：

1. 概述
{topic}是一个重要的研究领域，涉及多个方面。

2. 关键点
- 关键点 1：相关技术和方法
- 关键点 2：应用场景
- 关键点 3：发展趋势

3. 数据来源
- 来源 1：学术论文
- 来源 2：行业报告
- 来源 3：专家观点

4. 总结
{topic}正在快速发展，建议持续关注最新动态。

[研究员 Agent - {datetime.now().strftime('%Y-%m-%d %H:%M')}]
"""
        return findings
    
    async def _send_status_update(
        self,
        receiver: str,
        task_id: str,
        status: MessageStatus,
        progress: float,
        message: str,
    ):
        """发送状态更新"""
        from protocol.message import Message
        
        status_msg = Message.create_status_update(
            sender=self.address,
            receiver=receiver,
            task_id=task_id,
            status=status,
            progress=progress,
            message=message,
        )
        
        try:
            await self.send_message(status_msg)
        except:
            pass  # 忽略发送失败
    
    async def process_task(self, task: dict) -> dict:
        """实现抽象方法"""
        topic = task.get("topic", "")
        result = await self._perform_research(topic)
        return {"status": "completed", "findings": result}
