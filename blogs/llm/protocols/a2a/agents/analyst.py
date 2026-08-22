"""
分析师 Agent

负责数据分析和总结
"""

import asyncio
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent


class AnalystAgent(BaseAgent):
    """分析师 Agent"""
    
    def __init__(self, **kwargs):
        capabilities = [
            {
                "name": "data_analysis",
                "description": "数据分析和洞察",
                "input_schema": {"type": "object", "properties": {"data": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"insights": {"type": "string"}}},
                "tags": ["analysis", "insights"],
            },
            {
                "name": "summarization",
                "description": "内容总结",
                "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"summary": {"type": "string"}}},
                "tags": ["summary"],
            },
        ]
        
        super().__init__(
            agent_id="analyst",
            name="分析师 Agent",
            capabilities=capabilities,
            **kwargs
        )
        
        self.register_handler("execute_task", self._handle_execute_task)
        self.register_handler("analyze", self._handle_analyze)
    
    async def _handle_execute_task(self, message: Message) -> Message:
        """处理任务执行"""
        task_type = message.payload.content.get("task_type", "")
        description = message.payload.content.get("description", "")
        task_id = message.payload.content.get("task_id", "")
        
        print(f"[Analyst] 分析任务：{task_type}")
        
        # 执行分析
        result = await self._perform_analysis(description)
        
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
    
    async def _handle_analyze(self, message: Message) -> Message:
        """处理分析请求"""
        content = message.payload.content.get("content", "")
        
        if not content:
            return Message.create_error(
                sender=self.address,
                receiver=message.sender,
                in_reply_to=message.message_id,
                error_message="缺少分析内容",
                error_code="MISSING_CONTENT",
            )
        
        result = await self._perform_analysis(content)
        
        return Message.create_response(
            sender=self.address,
            receiver=message.sender,
            in_reply_to=message.message_id,
            content={"analysis": result},
        )
    
    async def _perform_analysis(self, content: str) -> str:
        """执行分析（模拟）"""
        await asyncio.sleep(0.5)
        
        analysis = f"""分析报告

1. 内容概述
分析对象：{content[:100]}...

2. 关键发现
- 发现 1：内容结构和组织良好
- 发现 2：信息密度适中
- 发现 3：逻辑清晰

3. 数据洞察
- 关键词频率分析完成
- 主题分类完成
- 情感分析：中性/正面

4. 建议
- 建议 1：可以进一步扩展某些部分
- 建议 2：考虑添加更多数据支持
- 建议 3：注意信息来源的可靠性

5. 总结
内容质量良好，信息有价值。

[分析师 Agent - {datetime.now().strftime('%Y-%m-%d %H:%M')}]
"""
        return analysis
    
    async def process_task(self, task: dict) -> dict:
        """实现抽象方法"""
        content = task.get("content", "")
        result = await self._perform_analysis(content)
        return {"status": "completed", "analysis": result}
