"""
作家 Agent

负责内容撰写和报告生成
"""

import asyncio
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent


class WriterAgent(BaseAgent):
    """作家 Agent"""
    
    def __init__(self, **kwargs):
        capabilities = [
            {
                "name": "content_writing",
                "description": "内容撰写和创作",
                "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}, "style": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"content": {"type": "string"}}},
                "tags": ["writing", "content"],
            },
            {
                "name": "report_generation",
                "description": "报告生成",
                "input_schema": {"type": "object", "properties": {"data": {"type": "string"}, "format": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"report": {"type": "string"}}},
                "tags": ["report"],
            },
        ]
        
        super().__init__(
            agent_id="writer",
            name="作家 Agent",
            capabilities=capabilities,
            **kwargs
        )
        
        self.register_handler("execute_task", self._handle_execute_task)
        self.register_handler("write", self._handle_write)
    
    async def _handle_execute_task(self, message: Message) -> Message:
        """处理任务执行"""
        task_type = message.payload.content.get("task_type", "")
        description = message.payload.content.get("description", "")
        task_id = message.payload.content.get("task_id", "")
        
        print(f"[Writer] 撰写任务：{task_type}")
        
        # 执行撰写
        result = await self._perform_writing(description)
        
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
    
    async def _handle_write(self, message: Message) -> Message:
        """处理撰写请求"""
        topic = message.payload.content.get("topic", "")
        style = message.payload.content.get("style", "professional")
        
        if not topic:
            return Message.create_error(
                sender=self.address,
                receiver=message.sender,
                in_reply_to=message.message_id,
                error_message="缺少主题",
                error_code="MISSING_TOPIC",
            )
        
        result = await self._perform_writing(f"{topic} ({style}风格)")
        
        return Message.create_response(
            sender=self.address,
            receiver=message.sender,
            in_reply_to=message.message_id,
            content={"topic": topic, "content": result},
        )
    
    async def _perform_writing(self, description: str) -> str:
        """执行撰写（模拟）"""
        await asyncio.sleep(0.5)
        
        content = f"""# 专题报告

## 摘要
本报告针对"{description}"进行详细阐述和分析。

## 引言
{description}是当今重要话题之一。随着技术和社会的发展，相关内容变得越来越重要。

## 主体内容

### 1. 背景介绍
该领域的背景和发展历程值得深入了解。

### 2. 核心要点
- 要点一：关键概念和定义
- 要点二：主要特征和属性
- 要点三：影响因素和关系

### 3. 实践应用
在实际应用中，需要注意以下几点：
1. 适用场景的识别
2. 方法选择的合理性
3. 效果评估的客观性

### 4. 案例分析
通过具体案例可以更好地理解相关概念。

## 结论
综上所述，{description}是一个值得深入研究的领域。

## 参考文献
[1] 相关资料 1
[2] 相关资料 2
[3] 相关资料 3

---
_生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
_作家 Agent_
"""
        return content
    
    async def process_task(self, task: dict) -> dict:
        """实现抽象方法"""
        topic = task.get("topic", "")
        result = await self._perform_writing(topic)
        return {"status": "completed", "content": result}
