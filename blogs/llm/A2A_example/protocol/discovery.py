"""
服务发现与注册
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ServiceCapability:
    """服务能力定义"""
    name: str                        # 能力名称
    description: str                 # 描述
    input_schema: Dict[str, Any] = field(default_factory=dict)   # 输入 schema
    output_schema: Dict[str, Any] = field(default_factory=dict)  # 输出 schema
    tags: List[str] = field(default_factory=list)                # 标签
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "tags": self.tags,
        }


@dataclass
class ServiceInfo:
    """服务信息"""
    agent_id: str                    # Agent ID
    name: str                        # 服务名称
    host: str                        # 主机
    port: int                        # 端口
    capabilities: List[ServiceCapability] = field(default_factory=list)
    status: str = "active"           # active, inactive, busy
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def address(self) -> str:
        """获取服务地址"""
        return f"{self.agent_id}@{self.host}:{self.port}"
    
    @property
    def endpoint(self) -> str:
        """获取 HTTP 端点"""
        return f"http://{self.host}:{self.port}/a2a"
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "metadata": self.metadata,
            "address": self.address,
            "endpoint": self.endpoint,
        }
    
    def is_healthy(self, timeout_seconds: int = 60) -> bool:
        """检查服务是否健康"""
        if self.status != "active":
            return False
        
        last = datetime.fromisoformat(self.last_heartbeat)
        elapsed = (datetime.now() - last).total_seconds()
        return elapsed < timeout_seconds


class ServiceRegistry:
    """
    服务注册中心
    
    维护所有可用 Agent 的注册信息
    """
    
    def __init__(self):
        self._services: Dict[str, ServiceInfo] = {}
    
    def register(self, service: ServiceInfo) -> bool:
        """注册服务"""
        self._services[service.agent_id] = service
        print(f"[Registry] 服务注册：{service.address}")
        return True
    
    def unregister(self, agent_id: str) -> bool:
        """注销服务"""
        if agent_id in self._services:
            del self._services[agent_id]
            print(f"[Registry] 服务注销：{agent_id}")
            return True
        return False
    
    def get_service(self, agent_id: str) -> Optional[ServiceInfo]:
        """获取服务信息"""
        return self._services.get(agent_id)
    
    def list_services(self, status: str = None) -> List[ServiceInfo]:
        """列出所有服务"""
        services = list(self._services.values())
        if status:
            services = [s for s in services if s.status == status]
        return services
    
    def find_by_capability(self, capability_name: str) -> List[ServiceInfo]:
        """根据能力查找服务"""
        results = []
        for service in self._services.values():
            for cap in service.capabilities:
                if cap.name == capability_name:
                    results.append(service)
                    break
        return results
    
    def update_heartbeat(self, agent_id: str) -> bool:
        """更新心跳"""
        if agent_id in self._services:
            self._services[agent_id].last_heartbeat = datetime.now().isoformat()
            return True
        return False
    
    def cleanup_stale(self, timeout_seconds: int = 300) -> List[str]:
        """清理过期服务"""
        stale = []
        for agent_id, service in self._services.items():
            if not service.is_healthy(timeout_seconds):
                stale.append(agent_id)
        
        for agent_id in stale:
            self.unregister(agent_id)
        
        return stale
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "services": {k: v.to_dict() for k, v in self._services.items()},
            "count": len(self._services),
            "timestamp": datetime.now().isoformat(),
        }


class ServiceDiscovery:
    """
    服务发现
    
    支持多种发现方式：本地注册中心、DNS、多播等
    """
    
    def __init__(self, registry: ServiceRegistry = None):
        self._registry = registry or ServiceRegistry()
        self._cache: Dict[str, List[ServiceInfo]] = {}
        self._cache_ttl = 60  # 缓存 TTL（秒）
    
    def register_agent(
        self,
        agent_id: str,
        name: str,
        host: str,
        port: int,
        capabilities: List[dict],
        **metadata
    ) -> ServiceInfo:
        """注册 Agent 服务"""
        caps = [
            ServiceCapability(
                name=c.get("name", ""),
                description=c.get("description", ""),
                input_schema=c.get("input_schema", {}),
                output_schema=c.get("output_schema", {}),
                tags=c.get("tags", []),
            )
            for c in capabilities
        ]
        
        service = ServiceInfo(
            agent_id=agent_id,
            name=name,
            host=host,
            port=port,
            capabilities=caps,
            metadata=metadata,
        )
        
        self._registry.register(service)
        self._invalidate_cache()
        return service
    
    def discover(self, capability: str = None) -> List[ServiceInfo]:
        """发现服务"""
        # 检查缓存
        cache_key = f"cap:{capability}" if capability else "all"
        if cache_key in self._cache:
            cached_time, cached_services = self._cache[cache_key]
            if (datetime.now().timestamp() - cached_time) < self._cache_ttl:
                return cached_services
        
        # 查询注册中心
        if capability:
            services = self._registry.find_by_capability(capability)
        else:
            services = self._registry.list_services(status="active")
        
        # 更新缓存
        self._cache[cache_key] = (datetime.now().timestamp(), services)
        
        return services
    
    def get_agent_endpoint(self, agent_id: str) -> Optional[str]:
        """获取 Agent 端点"""
        service = self._registry.get_service(agent_id)
        if service:
            return service.endpoint
        return None
    
    def _invalidate_cache(self):
        """使缓存失效"""
        self._cache.clear()
    
    def list_all_services(self) -> dict:
        """列出所有服务"""
        return self._registry.to_dict()
