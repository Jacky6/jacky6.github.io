"""
§09 - Safety Guardrails 安全护栏

Layers:
  1. Input validation (dangerous pattern detection, prompt injection)
  2. Output review (harmful content, PII leak)
  3. Token budget enforcement
  4. Rate limiting
  5. Permission system (tool-level access control)
  6. Human-in-the-Loop (approval workflow)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────
# 1. Data Models
# ──────────────────────────────────────────────

class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolPermission(str, Enum):
    PUBLIC = "public"       # 无需审批
    APPROVAL = "approval"   # 需要审批
    BLOCKED = "blocked"     # 禁止使用


@dataclass
class SafetyCheck:
    """安全检查结果。"""

    passed: bool
    risk: RiskLevel = RiskLevel.SAFE
    reason: str = ""
    suggestions: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# 2. Input Validator
# ──────────────────────────────────────────────

class InputValidator:
    """输入安全检查。"""

    DANGEROUS_PATTERNS = [
        # Prompt injection
        (r"(?i)ignore\s+(all\s+)?(previous\s+)?(instructions?|rules?|prompts?)", "Prompt 注入攻击"),
        (r"(?i)you\s+are\s+(now\s+)?(no\s+longer\s+)?(bound\s+by|restricted\s+by)", "角色覆盖攻击"),
        (r"(?i)(system\s*|admin\s*)?\s*override", "系统覆盖指令"),
        (r"(?i)dan\s*mode|jailbreak|developer\s*mode", "越狱尝试"),

        # Harmful requests
        (r"(?i)(how\s+to|tell\s+me\s+how|教我)\s*(make|build|create).*(bomb|weapon|poison|explosive)", "危险品制造请求"),
        (r"(?i)(hack|破解|入侵).*(system|account|server|network)", "入侵请求"),
        (r"(?i)( steal|盗取).*(password|credit.?card|identity)", "盗窃请求"),
    ]

    MAX_INPUT_LENGTH = 10000

    def check(self, text: str) -> SafetyCheck:
        """检查输入安全性。"""
        if not text or not text.strip():
            return SafetyCheck(passed=False, risk=RiskLevel.LOW, reason="空输入")

        if len(text) > self.MAX_INPUT_LENGTH:
            return SafetyCheck(
                passed=False,
                risk=RiskLevel.MEDIUM,
                reason=f"输入过长 ({len(text)} > {self.MAX_INPUT_LENGTH})",
                suggestions=["截断输入或分段处理"],
            )

        # Pattern matching
        for pattern, description in self.DANGEROUS_PATTERNS:
            if re.search(pattern, text):
                return SafetyCheck(
                    passed=False,
                    risk=RiskLevel.CRITICAL,
                    reason=f"检测到危险模式: {description}",
                    suggestions=["拒绝执行并记录日志"],
                )

        return SafetyCheck(passed=True, risk=RiskLevel.SAFE)


# ──────────────────────────────────────────────
# 3. Output Reviewer
# ──────────────────────────────────────────────

class OutputReviewer:
    """输出安全检查。"""

    # PII patterns to detect potential data leaks
    PII_PATTERNS = [
        (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "疑似信用卡号"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "疑似 SSN"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "疑似邮箱泄露"),
    ]

    def check(self, text: str) -> SafetyCheck:
        """检查输出安全性。"""
        if not text:
            return SafetyCheck(passed=True, risk=RiskLevel.SAFE)

        # Check for PII leaks
        for pattern, description in self.PII_PATTERNS:
            if re.search(pattern, text):
                return SafetyCheck(
                    passed=False,
                    risk=RiskLevel.HIGH,
                    reason=f"输出包含敏感信息: {description}",
                    suggestions=["脱敏处理后重新输出"],
                )

        return SafetyCheck(passed=True, risk=RiskLevel.SAFE)


# ──────────────────────────────────────────────
# 4. Rate Limiter
# ──────────────────────────────────────────────

class RateLimiter:
    """速率限制器。"""

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    def allow(self) -> bool:
        """检查是否允许下一次请求。"""
        now = time.time()
        cutoff = now - self.window_seconds

        # Remove old timestamps
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        if len(self._timestamps) >= self.max_requests:
            return False

        self._timestamps.append(now)
        return True

    def wait_time(self) -> float:
        """返回需要等待的秒数。"""
        if self.allow():
            return 0.0
        if not self._timestamps:
            return 0.0
        oldest = min(self._timestamps)
        return max(0.0, (oldest + self.window_seconds) - time.time())


# ──────────────────────────────────────────────
# 5. Permission Manager
# ──────────────────────────────────────────────

class PermissionManager:
    """工具权限管理器。"""

    def __init__(self):
        self._permissions: dict[str, ToolPermission] = {}
        self._default = ToolPermission.PUBLIC

    def set_permission(self, tool_name: str, level: ToolPermission):
        self._permissions[tool_name] = level

    def check(self, tool_name: str) -> tuple[bool, str]:
        """检查工具调用权限。"""
        level = self._permissions.get(tool_name, self._default)

        if level == ToolPermission.BLOCKED:
            return False, f"工具 '{tool_name}' 已被禁止使用"

        if level == ToolPermission.APPROVAL:
            return False, f"工具 '{tool_name}' 需要人工审批"

        return True, ""

    def requires_approval(self, tool_name: str) -> bool:
        return self._permissions.get(tool_name, self._default) == ToolPermission.APPROVAL


# ──────────────────────────────────────────────
# 6. HITL (Human-in-the-Loop) Manager
# ──────────────────────────────────────────────

class HITLManager:
    """人工审批管理器。"""

    def __init__(self):
        self._pending_approvals: dict[str, dict] = {}
        self._approval_timeout: float = 300.0  # 5 min

    def request_approval(
        self,
        action_id: str,
        action_type: str,
        description: str,
        risk_reason: str = "",
    ) -> str:
        """请求人工审批，返回审批 ID。"""
        self._pending_approvals[action_id] = {
            "type": action_type,
            "description": description,
            "risk_reason": risk_reason,
            "status": "pending",
            "requested_at": time.time(),
            "decision": None,
        }
        return action_id

    def approve(self, action_id: str) -> bool:
        """审批通过。"""
        if action_id in self._pending_approvals:
            self._pending_approvals[action_id]["status"] = "approved"
            self._pending_approvals[action_id]["decision"] = "approved"
            return True
        return False

    def reject(self, action_id: str, reason: str = "") -> bool:
        """审批拒绝。"""
        if action_id in self._pending_approvals:
            self._pending_approvals[action_id]["status"] = "rejected"
            self._pending_approvals[action_id]["decision"] = f"rejected: {reason}"
            return True
        return False

    def get_status(self, action_id: str) -> dict | None:
        return self._pending_approvals.get(action_id)

    def check_timeout(self, action_id: str) -> bool:
        """检查审批是否超时。"""
        approval = self._pending_approvals.get(action_id)
        if not approval:
            return True
        elapsed = time.time() - approval["requested_at"]
        return elapsed > self._approval_timeout

    def cleanup_expired(self) -> int:
        """清理超时的审批请求。"""
        expired = [
            aid for aid, a in self._pending_approvals.items()
            if (a["status"] == "pending" and self.check_timeout(aid))
        ]
        for aid in expired:
            self._pending_approvals[aid]["status"] = "expired"
        return len(expired)


# ──────────────────────────────────────────────
# 7. SafetyGuard (unified entry point, backward compatible)
# ──────────────────────────────────────────────

class SafetyGuard:
    """统一安全护栏——向后兼容。"""

    def __init__(self, max_tokens: int = 10000):
        self.max_tokens = max_tokens
        self.input_validator = InputValidator()
        self.output_reviewer = OutputReviewer()
        self.rate_limiter = RateLimiter()
        self.permissions = PermissionManager()
        self.hitl = HITLManager()

    def check_input(self, text: str) -> tuple[bool, str]:
        """检查输入安全性。"""
        result = self.input_validator.check(text)
        return result.passed, result.reason

    def check_output(self, text: str) -> tuple[bool, str]:
        """检查输出安全性。"""
        result = self.output_reviewer.check(text)
        return result.passed, result.reason

    def check_rate_limit(self) -> bool:
        """检查速率限制。"""
        return self.rate_limiter.allow()

    def check_tool_permission(self, tool_name: str) -> tuple[bool, str]:
        """检查工具权限。"""
        return self.permissions.check(tool_name)

    def check(self, state: dict | None = None) -> tuple[bool, str]:
        """综合检查（向后兼容 control_loop 调用）。"""
        if state is None:
            return True, "OK"

        # Token budget check
        token_usage = state.get("token_usage", 0)
        if token_usage > self.max_tokens:
            return False, f"超出 Token 预算 ({token_usage}/{self.max_tokens})"

        return True, "OK"
