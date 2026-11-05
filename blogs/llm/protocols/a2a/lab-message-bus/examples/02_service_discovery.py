"""
示例 2: 服务发现

演示 Agent 如何注册和发现服务
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.discovery import ServiceRegistry, ServiceDiscovery, ServiceCapability


async def service_discovery_demo():
    """服务发现示例"""
    print("=" * 60)
    print("A2A 示例 2: 服务发现")
    print("=" * 60)
    
    # 创建注册中心
    registry = ServiceRegistry()
    discovery = ServiceDiscovery(registry)
    
    # ========== 场景 1: 注册服务 ==========
    print("\n[场景 1] 注册 Agent 服务")
    print("-" * 40)
    
    # 注册研究员 Agent
    researcher = discovery.register_agent(
        agent_id="researcher",
        name="研究员 Agent",
        host="localhost",
        port=8002,
        capabilities=[
            {
                "name": "information_research",
                "description": "信息搜索和研究",
                "tags": ["research", "search"],
            },
            {
                "name": "data_collection",
                "description": "数据收集",
                "tags": ["collection"],
            },
        ],
        metadata={"version": "1.0", "author": "A2A Team"},
    )
    
    print(f"✓ 注册：{researcher.name}")
    print(f"  地址：{researcher.address}")
    print(f"  端点：{researcher.endpoint}")
    print(f"  能力：{[c.name for c in researcher.capabilities]}")
    
    # 注册分析师 Agent
    analyst = discovery.register_agent(
        agent_id="analyst",
        name="分析师 Agent",
        host="localhost",
        port=8003,
        capabilities=[
            {
                "name": "data_analysis",
                "description": "数据分析和洞察",
                "tags": ["analysis"],
            },
            {
                "name": "summarization",
                "description": "内容总结",
                "tags": ["summary"],
            },
        ],
    )
    
    print(f"✓ 注册：{analyst.name}")
    
    # 注册作家 Agent
    writer = discovery.register_agent(
        agent_id="writer",
        name="作家 Agent",
        host="localhost",
        port=8004,
        capabilities=[
            {
                "name": "content_writing",
                "description": "内容撰写",
                "tags": ["writing"],
            },
        ],
    )
    
    print(f"✓ 注册：{writer.name}")
    
    # ========== 场景 2: 列出所有服务 ==========
    print("\n[场景 2] 列出所有服务")
    print("-" * 40)
    
    all_services = discovery.list_all_services()
    
    print(f"已注册服务数量：{all_services['count']}")
    print("\n服务列表:")
    for agent_id, service in all_services['services'].items():
        print(f"\n  [{agent_id}]")
        print(f"    名称：{service['name']}")
        print(f"    地址：{service['address']}")
        print(f"    状态：{service['status']}")
        print(f"    能力：{[c['name'] for c in service['capabilities']]}")
    
    # ========== 场景 3: 根据能力查找服务 ==========
    print("\n[场景 3] 根据能力查找服务")
    print("-" * 40)
    
    # 查找有 research 能力的服务
    researchers = discovery.discover(capability="information_research")
    print(f"具有 'information_research' 能力的服务：{len(researchers)}")
    for r in researchers:
        print(f"  - {r.name} @ {r.address}")
    
    # 查找有 analysis 能力的服务
    analysts = discovery.discover(capability="data_analysis")
    print(f"\n具有 'data_analysis' 能力的服务：{len(analysts)}")
    for a in analysts:
        print(f"  - {a.name} @ {a.address}")
    
    # ========== 场景 4: 获取 Agent 端点 ==========
    print("\n[场景 4] 获取 Agent 端点")
    print("-" * 40)
    
    endpoint = discovery.get_agent_endpoint("researcher")
    print(f"Researcher 端点：{endpoint}")
    
    endpoint = discovery.get_agent_endpoint("analyst")
    print(f"Analyst 端点：{endpoint}")
    
    # ========== 场景 5: 心跳和健康检查 ==========
    print("\n[场景 5] 心跳和健康检查")
    print("-" * 40)
    
    # 更新心跳
    registry.update_heartbeat("researcher")
    registry.update_heartbeat("analyst")
    
    # 检查健康状态
    for agent_id in ["researcher", "analyst", "writer"]:
        service = registry.get_service(agent_id)
        if service:
            is_healthy = service.is_healthy(timeout_seconds=60)
            print(f"{agent_id}: {'✓ 健康' if is_healthy else '✗ 不健康'}")
    
    # ========== 场景 6: 服务注销 ==========
    print("\n[场景 6] 服务注销")
    print("-" * 40)
    
    registry.unregister("writer")
    print("✓ 注销 Writer 服务")
    
    # 再次列出服务
    remaining = discovery.list_all_services()
    print(f"剩余服务数量：{remaining['count']}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(service_discovery_demo())
