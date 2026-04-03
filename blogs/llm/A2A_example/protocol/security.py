"""
安全认证模块
"""

import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict
from dataclasses import dataclass, field


@dataclass
class AuthToken:
    """认证令牌"""
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    issued_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: str = field(default_factory=lambda: (datetime.now() + timedelta(hours=24)).isoformat())
    permissions: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """检查令牌是否有效"""
        expiry = datetime.fromisoformat(self.expires_at)
        return datetime.now() < expiry
    
    def has_permission(self, permission: str) -> bool:
        """检查是否有权限"""
        return permission in self.permissions
    
    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "agent_id": self.agent_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "permissions": self.permissions,
        }


class SecurityManager:
    """
    安全管理器
    
    处理 Agent 认证和授权
    """
    
    def __init__(self, secret_key: str = None):
        import secrets
        self._secret_key = secret_key or secrets.token_hex(32)
        self._tokens: Dict[str, AuthToken] = {}
        self._registered_agents: Dict[str, dict] = {}
    
    def register_agent(
        self,
        agent_id: str,
        api_key: str = None,
        permissions: list = None,
    ) -> str:
        """
        注册 Agent
        
        Returns:
            API Key
        """
        import secrets
        
        api_key = api_key or secrets.token_urlsafe(32)
        api_key_hash = self._hash_key(api_key)
        
        self._registered_agents[agent_id] = {
            "api_key_hash": api_key_hash,
            "permissions": permissions or ["send", "receive"],
            "created_at": datetime.now().isoformat(),
        }
        
        print(f"[Security] Agent 注册：{agent_id}")
        return api_key
    
    def authenticate(self, agent_id: str, api_key: str) -> bool:
        """认证 Agent"""
        if agent_id not in self._registered_agents:
            return False
        
        stored_hash = self._registered_agents[agent_id]["api_key_hash"]
        provided_hash = self._hash_key(api_key)
        
        return stored_hash == provided_hash
    
    def issue_token(self, agent_id: str, api_key: str, ttl_hours: int = 24) -> Optional[AuthToken]:
        """颁发令牌"""
        if not self.authenticate(agent_id, api_key):
            return None
        
        token = AuthToken(
            agent_id=agent_id,
            expires_at=(datetime.now() + timedelta(hours=ttl_hours)).isoformat(),
            permissions=self._registered_agents[agent_id]["permissions"],
        )
        
        self._tokens[token.token_id] = token
        return token
    
    def validate_token(self, token_id: str) -> bool:
        """验证令牌"""
        if token_id not in self._tokens:
            return False
        
        token = self._tokens[token_id]
        return token.is_valid()
    
    def get_token(self, token_id: str) -> Optional[AuthToken]:
        """获取令牌"""
        token = self._tokens.get(token_id)
        if token and token.is_valid():
            return token
        return None
    
    def revoke_token(self, token_id: str) -> bool:
        """撤销令牌"""
        if token_id in self._tokens:
            del self._tokens[token_id]
            return True
        return False
    
    def _hash_key(self, api_key: str) -> str:
        """哈希 API Key"""
        return hashlib.sha256(
            f"{api_key}{self._secret_key}".encode()
        ).hexdigest()
    
    def verify_message_signature(
        self,
        message_content: str,
        signature: str,
        agent_id: str,
    ) -> bool:
        """验证消息签名"""
        if agent_id not in self._registered_agents:
            return False
        
        # 简化实现：实际应该使用非对称加密
        expected = hashlib.sha256(
            f"{message_content}{self._secret_key}".encode()
        ).hexdigest()
        
        return expected == signature
    
    def sign_message(self, message_content: str, agent_id: str) -> str:
        """签名消息"""
        if agent_id not in self._registered_agents:
            raise ValueError(f"未注册的 Agent: {agent_id}")
        
        return hashlib.sha256(
            f"{message_content}{self._secret_key}".encode()
        ).hexdigest()
