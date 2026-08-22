"""
示例 3: 任务分配

演示协调者如何分配任务给多个 Agent
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.message import Message
from protocol.transport import InMemoryTransport
from datetime import datetime


async def task_delegation_demo():
    """任务分配示例"""
    print("=" * 60)
    print("A2A 示例 3: 任务分配")
    print("=" * 60)
    
    # 创建传输层
    transport = InMemoryTransport()
    await transport.start()
    
    # ========== 定义 Worker Agent ==========
    print("\n[步骤 1] 创建 Worker Agent")
    print("-" * 40)
    
    async def create_worker(worker_id: str, specialty: str):
        """创建 Worker 处理器"""
        
        async def handler(message: Message) -> Message:
            intent = message.payload.intent
            
            if intent == "execute_task":
                task = message.payload.content
                print(f"\n[{worker_id}] 接收任务：{task.get('description', '')[:30]}...")
                
                # 模拟任务执行
                await asyncio.sleep(0.5)
                
                return Message.create_response(
                    sender=worker_id,
                    receiver=message.sender,
                    in_reply_to=message.message_id,
                    content={
                        "task_id": task.get("task_id"),
                        "status": "completed",
                        "result": f"[{specialty}] 完成：{task.get('description', '')[:50]}",
                        "worker": worker_id,
                        "completed_at": datetime.now().isoformat(),
                    },
                )
            
            elif intent == "get_status":
                return Message.create_response(
                    sender=worker_id,
                    receiver=message.sender,
                    in_reply_to=message.message_id,
                    content={
                        "worker": worker_id,
                        "status": "idle",
                        "specialty": specialty,
                    },
                )
            
            return Message.create_error(
                sender=worker_id,
                receiver=message.sender,
                in_reply_to=message.message_id,
                error_message=f"未知意图：{intent}",
            )
        
        return handler
    
    # 注册 Worker
    worker1_handler = await create_worker("worker_1", "研究")
    worker2_handler = await create_worker("worker_2", "分析")
    worker3_handler = await create_worker("worker_3", "写作")
    
    transport.register_agent("worker_1", worker1_handler)
    transport.register_agent("worker_2", worker2_handler)
    transport.register_agent("worker_3", worker3_handler)
    
    print("✓ 注册 3 个 Worker Agent")
    print("  - worker_1: 研究专家")
    print("  - worker_2: 分析专家")
    print("  - worker_3: 写作专家")
    
    # ========== 场景 1: 单任务分配 ==========
    print("\n[场景 1] 单任务分配")
    print("-" * 40)
    
    task_msg = Message.create_request(
        sender="coordinator",
        receiver="worker_1@localhost:8002",
        intent="execute_task",
        content={
            "task_id": "task_001",
            "task_type": "research",
            "description": "研究人工智能的最新发展",
        },
    )
    
    print(f"[Coordinator] 分配任务给 worker_1")
    response = await transport.send(task_msg)
    
    print(f"[Coordinator] 收到结果：{response.payload.content['result']}")
    
    # ========== 场景 2: 多任务并行分配 ==========
    print("\n[场景 2] 多任务并行分配")
    print("-" * 40)
    
    tasks = [
        {"worker": "worker_1", "type": "research", "desc": "收集市场数据"},
        {"worker": "worker_2", "type": "analysis", "desc": "分析数据趋势"},
        {"worker": "worker_3", "type": "writing", "desc": "撰写总结报告"},
    ]
    
    async def assign_task(task_info, task_num):
        """分配单个任务"""
        msg = Message.create_request(
            sender="coordinator",
            receiver=f"{task_info['worker']}@localhost:800{task_num + 1}",
            intent="execute_task",
            content={
                "task_id": f"task_{task_num:03d}",
                "task_type": task_info["type"],
                "description": task_info["desc"],
            },
        )
        return await transport.send(msg)
    
    # 并行分配任务
    print("[Coordinator] 并行分配 3 个任务...")
    results = await asyncio.gather(
        assign_task(tasks[0], 1),
        assign_task(tasks[1], 2),
        assign_task(tasks[2], 3),
    )
    
    print("\n[Coordinator] 所有任务完成:")
    for i, result in enumerate(results):
        content = result.payload.content
        print(f"  任务{i+1}: {content.get('result', 'N/A')}")
    
    # ========== 场景 3: 任务链（依赖关系） ==========
    print("\n[场景 3] 任务链（依赖关系）")
    print("-" * 40)
    
    # 任务链：研究 → 分析 → 写作
    chain_result = None
    
    # 步骤 1: 研究
    print("[任务链] 步骤 1: 研究...")
    msg1 = Message.create_request(
        sender="coordinator",
        receiver="worker_1@localhost:8002",
        intent="execute_task",
        content={
            "task_id": "chain_001",
            "task_type": "research",
            "description": "研究主题 X",
        },
    )
    result1 = await transport.send(msg1)
    print(f"         完成：{result1.payload.content['result']}")
    
    # 步骤 2: 分析（依赖研究结果）
    print("[任务链] 步骤 2: 分析...")
    msg2 = Message.create_request(
        sender="coordinator",
        receiver="worker_2@localhost:8003",
        intent="execute_task",
        content={
            "task_id": "chain_002",
            "task_type": "analysis",
            "description": f"分析研究结果：{result1.payload.content['result']}",
            "input_from": "chain_001",
        },
    )
    result2 = await transport.send(msg2)
    print(f"         完成：{result2.payload.content['result']}")
    
    # 步骤 3: 写作（依赖分析结果）
    print("[任务链] 步骤 3: 写作...")
    msg3 = Message.create_request(
        sender="coordinator",
        receiver="worker_3@localhost:8004",
        intent="execute_task",
        content={
            "task_id": "chain_003",
            "task_type": "writing",
            "description": f"基于分析结果撰写报告",
            "input_from": "chain_002",
        },
    )
    result3 = await transport.send(msg3)
    print(f"         完成：{result3.payload.content['result']}")
    
    print("\n[任务链] 完成！最终输出:")
    print(f"  {result3.payload.content['result']}")
    
    # ========== 场景 4: 负载均衡 ==========
    print("\n[场景 4] 负载均衡")
    print("-" * 40)
    
    # 模拟多个相同类型的任务
    workers = ["worker_1", "worker_2", "worker_3"]
    task_count = {w: 0 for w in workers}
    
    for i in range(6):
        # 简单的轮询负载均衡
        worker = workers[i % len(workers)]
        task_count[worker] += 1
        
        msg = Message.create_request(
            sender="coordinator",
            receiver=f"{worker}@localhost:800{(i%3)+2}",
            intent="execute_task",
            content={
                "task_id": f"lb_{i:03d}",
                "task_type": "general",
                "description": f"通用任务 {i+1}",
            },
        )
        await transport.send(msg)
        print(f"[负载均衡] 任务{i+1} → {worker}")
    
    print("\n任务分配统计:")
    for worker, count in task_count.items():
        print(f"  {worker}: {count} 个任务")
    
    # ========== 清理 ==========
    await transport.stop()
    
    print("\n" + "=" * 60)
    print("任务分配示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(task_delegation_demo())
