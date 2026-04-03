"""
示例 4: 多 Agent 协作

完整演示多 Agent 协作完成研究任务
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.coordinator import CoordinatorAgent
from agents.researcher import ResearcherAgent
from agents.analyst import AnalystAgent
from agents.writer import WriterAgent
from protocol.discovery import ServiceDiscovery


async def multi_agent_collaboration():
    """多 Agent 协作示例"""
    print("=" * 60)
    print("A2A 示例 4: 多 Agent 协作")
    print("=" * 60)
    
    # ========== 步骤 1: 创建 Agent ==========
    print("\n[步骤 1] 创建 Agent")
    print("-" * 40)
    
    coordinator = CoordinatorAgent()
    researcher = ResearcherAgent()
    analyst = AnalystAgent()
    writer = WriterAgent()
    
    agents = [coordinator, researcher, analyst, writer]
    
    # ========== 步骤 2: 启动 Agent ==========
    print("\n[步骤 2] 启动 Agent")
    print("-" * 40)
    
    for agent in agents:
        await agent.start()
    
    # ========== 步骤 3: 注册 Agent 到协调者 ==========
    print("\n[步骤 3] 注册 Agent 到协调者")
    print("-" * 40)
    
    coordinator.register_agent("researcher", researcher.address)
    coordinator.register_agent("analyst", analyst.address)
    coordinator.register_agent("writer", writer.address)
    
    print(f"已注册 Agent:")
    for agent_id, address in coordinator._known_agents.items():
        print(f"  - {agent_id}: {address}")
    
    # ========== 步骤 4: 执行协作任务 ==========
    print("\n[步骤 4] 执行协作任务")
    print("-" * 40)
    
    # 任务 1: 研究 AI 发展趋势
    print("\n任务 1: 研究人工智能的发展趋势")
    print("-" * 40)
    
    task1_msg = await coordinator.send_request(
        receiver=coordinator.address,
        intent="coordinate_task",
        content={
            "task": "研究人工智能的发展趋势",
            "task_id": "task_001",
        },
    )
    
    if task1_msg.message_type.value == "response":
        result = task1_msg.payload.content.get("result", "")
        print(f"\n任务 1 结果:\n{result}")
    
    # 任务 2: 分析并撰写报告
    print("\n\n任务 2: 分析 Python 生态系统并撰写报告")
    print("-" * 40)
    
    task2_msg = await coordinator.send_request(
        receiver=coordinator.address,
        intent="coordinate_task",
        content={
            "task": "分析 Python 生态系统并撰写一份详细报告",
            "task_id": "task_002",
        },
    )
    
    if task2_msg.message_type.value == "response":
        result = task2_msg.payload.content.get("result", "")
        print(f"\n任务 2 结果:\n{result}")
    
    # ========== 步骤 5: 查看任务历史 ==========
    print("\n[步骤 5] 查看任务历史")
    print("-" * 40)
    
    print(f"协调者完成的任务数：{len(coordinator._task_results)}")
    
    for task_id, task_info in coordinator._task_results.items():
        print(f"\n任务：{task_id}")
        print(f"  子任务数：{len(task_info['subtasks'])}")
        print(f"  完成时间：{task_info.get('completed_at', 'N/A')}")
    
    # ========== 步骤 6: 停止 Agent ==========
    print("\n[步骤 6] 停止 Agent")
    print("-" * 40)
    
    for agent in reversed(agents):
        await agent.stop()
    
    print("\n所有 Agent 已停止")
    
    print("\n" + "=" * 60)
    print("多 Agent 协作示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(multi_agent_collaboration())
