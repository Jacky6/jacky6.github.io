"""
协调者 Agent

负责任务分解和协调其他 Agent
"""

import asyncio
from typing import Dict, List
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, AgentState
from protocol.message import Message, MessageStatus


class CoordinatorAgent(BaseAgent):
    """协调者 Agent"""
    
    def __init__(self, **kwargs):
        capabilities = [
            {
                "name": "task_coordination",
                "description": "任务分解和协调",
                "input_schema": {"type": "object", "properties": {"task": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"result": {"type": "string"}}},
                "tags": ["coordination", "management"],
            },
            {
                "name": "task_delegation",
                "description": "任务分配给其他 Agent",
                "input_schema": {"type": "object", "properties": {"task": {"type": "string"}, "agent": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"status": {"type": "string"}}},
                "tags": ["delegation"],
            },
        ]
        
        super().__init__(
            agent_id="coordinator",
            name="协调者 Agent",
            capabilities=capabilities,
            **kwargs
        )
        
        self._known_agents: Dict[str, str] = {}  # agent_id -> address
        self._task_results: Dict[str, dict] = {}
        
        # 注册处理器
        self.register_handler("coordinate_task", self._handle_coordinate_task)
        self.register_handler("register_agent", self._handle_register_agent)
    
    def register_agent(self, agent_id: str, address: str):
        """注册已知 Agent"""
        self._known_agents[agent_id] = address
        print(f"[Coordinator] 注册 Agent: {agent_id} @ {address}")
    
    async def _handle_register_agent(self, message: Message) -> Message:
        """处理 Agent 注册"""
        agent_id = message.payload.content.get("agent_id")
        address = message.payload.content.get("address")
        
        if agent_id and address:
            self.register_agent(agent_id, address)
            return Message.create_response(
                sender=self.address,
                receiver=message.sender,
                in_reply_to=message.message_id,
                content={"status": "registered", "agent_id": agent_id},
            )
        
        return Message.create_error(
            sender=self.address,
            receiver=message.sender,
            in_reply_to=message.message_id,
            error_message="缺少 agent_id 或 address",
            error_code="MISSING_PARAMS",
        )
    
    async def _handle_coordinate_task(self, message: Message) -> Message:
        """处理任务协调请求"""
        task = message.payload.content.get("task", "")
        task_id = message.payload.content.get("task_id", str(datetime.now().timestamp()))
        
        print(f"[Coordinator] 协调任务：{task_id}")
        
        # 分解任务（简化版）
        subtasks = self._decompose_task(task)
        
        # 分配子任务
        results = await self._delegate_subtasks(task_id, subtasks)
        
        # 汇总结果
        final_result = self._aggregate_results(results)
        
        self._task_results[task_id] = {
            "subtasks": subtasks,
            "results": results,
            "final": final_result,
            "completed_at": datetime.now().isoformat(),
        }
        
        return Message.create_response(
            sender=self.address,
            receiver=message.sender,
            in_reply_to=message.message_id,
            content={
                "task_id": task_id,
                "status": "completed",
                "result": final_result,
            },
        )
    
    def _decompose_task(self, task: str) -> List[dict]:
        """分解任务为子任务"""
        # 简化实现：根据关键词分解
        subtasks = []
        
        if "研究" in task or "搜索" in task:
            subtasks.append({
                "id": f"research_{len(subtasks)}",
                "type": "research",
                "agent": "researcher",
                "description": f"研究：{task}",
            })
        
        if "分析" in task or "总结" in task:
            subtasks.append({
                "id": f"analysis_{len(subtasks)}",
                "type": "analysis",
                "agent": "analyst",
                "description": f"分析：{task}",
            })
        
        if "写" in task or "报告" in task:
            subtasks.append({
                "id": f"writing_{len(subtasks)}",
                "type": "writing",
                "agent": "writer",
                "description": f"撰写：{task}",
            })
        
        # 如果没有匹配，创建一个默认任务
        if not subtasks:
            subtasks.append({
                "id": "default_0",
                "type": "research",
                "agent": "researcher",
                "description": task,
            })
        
        print(f"[Coordinator] 任务分解为 {len(subtasks)} 个子任务")
        return subtasks
    
    async def _delegate_subtasks(self, task_id: str, subtasks: List[dict]) -> Dict[str, dict]:
        """分配子任务"""
        results = {}
        
        for i, subtask in enumerate(subtasks):
            agent_id = subtask.get("agent", "researcher")
            
            # 获取 Agent 地址
            agent_address = self._known_agents.get(agent_id)
            
            if not agent_address:
                print(f"[Coordinator] 未知 Agent: {agent_id}，使用默认地址")
                agent_address = f"{agent_id}@localhost:800{i+2}"
            
            # 发送任务
            try:
                response = await self.send_request(
                    receiver=agent_address,
                    intent="execute_task",
                    content={
                        "task_id": f"{task_id}_{subtask['id']}",
                        "task_type": subtask["type"],
                        "description": subtask["description"],
                    },
                )
                
                if response.message_type.value == "response":
                    results[subtask["id"]] = {
                        "status": "success",
                        "result": response.payload.content.get("result", ""),
                    }
                else:
                    results[subtask["id"]] = {
                        "status": "error",
                        "error": response.error or "未知错误",
                    }
                
            except Exception as e:
                results[subtask["id"]] = {
                    "status": "error",
                    "error": str(e),
                }
        
        return results
    
    def _aggregate_results(self, results: Dict[str, dict]) -> str:
        """汇总结果"""
        successful = sum(1 for r in results.values() if r["status"] == "success")
        total = len(results)
        
        parts = [f"完成任务：{successful}/{total}"]
        
        for task_id, result in results.items():
            if result["status"] == "success":
                parts.append(f"\n- {task_id}: {result['result'][:100]}...")
            else:
                parts.append(f"\n- {task_id}: 失败 - {result.get('error', '未知错误')}")
        
        return "\n".join(parts)
    
    async def process_task(self, task: dict) -> dict:
        """实现抽象方法"""
        return {"status": "coordinated", "task": task}
