"""
§11 - A2A Protocol Client (Agent-to-Agent)

Concept verification module demonstrating:
  1. A2A core concepts (Agent Card, Message, Task)
  2. Agent Card discovery
  3. Task lifecycle management
  4. Multi-agent collaboration patterns

References:
  - https://a2a-protocol.org/latest/
  - A2A != MCP: Agent↔Agent vs Agent↔Tool

Note: This is a conceptual implementation. For production,
use the official A2A SDK (Python/TypeScript).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ──────────────────────────────────────────────
# 1. Data Models (A2A Spec)
# ──────────────────────────────────────────────

class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ArtifactType(str, Enum):
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    DATA = "data"


@dataclass
class AgentSkill:
    """Agent 能力描述。"""

    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text"])
    output_modes: list[str] = field(default_factory=lambda: ["text"])


@dataclass
class AgentCard:
    """
    A2A Agent Card — Agent 的身份和能力声明。

    Equivalent to MCP's server info, but for Agent↔Agent communication.
    """

    name: str
    description: str
    url: str
    version: str = "1.0.0"
    skills: list[AgentSkill] = field(default_factory=list)
    default_input_modes: list[str] = field(default_factory=lambda: ["text"])
    default_output_modes: list[str] = field(default_factory=lambda: ["text"])
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "skills": [
                {"id": s.id, "name": s.name, "description": s.description}
                for s in self.skills
            ],
            "capabilities": self.capabilities,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentCard":
        skills = [
            AgentSkill(
                id=s.get("id", ""),
                name=s.get("name", ""),
                description=s.get("description", ""),
                tags=s.get("tags", []),
            )
            for s in data.get("skills", [])
        ]
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            url=data.get("url", ""),
            version=data.get("version", "1.0.0"),
            skills=skills,
            capabilities=data.get("capabilities", {}),
        )


@dataclass
class A2AMessage:
    """A2A 消息。"""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"  # user | agent
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    """任务产出物。"""

    type: ArtifactType = ArtifactType.TEXT
    content: str = ""
    name: str = ""


@dataclass
class A2ATask:
    """
    A2A Task — 跨 Agent 协作的基本单元。

    Lifecycle: SUBMITTED → WORKING → COMPLETED/FAILED
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.SUBMITTED
    messages: list[A2AMessage] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    context_id: str = ""  # session/group identifier
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str, metadata: dict = None):
        self.messages.append(A2AMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        ))
        self.updated_at = time.time()

    def add_artifact(self, content: str, artifact_type: ArtifactType = ArtifactType.TEXT, name: str = ""):
        self.artifacts.append(Artifact(
            type=artifact_type,
            content=content,
            name=name,
        ))
        self.status = TaskStatus.COMPLETED
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "messages": [
                {"message_id": m.message_id, "role": m.role, "content": m.content}
                for m in self.messages
            ],
            "artifacts": [
                {"type": a.type.value, "content": a.content[:200], "name": a.name}
                for a in self.artifacts
            ],
            "context_id": self.context_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ──────────────────────────────────────────────
# 2. A2A Client
# ──────────────────────────────────────────────

class A2AClient:
    """
    A2A 协议客户端 — 概念验证实现。

    In production, this would make HTTP/gRPC calls to remote agents.
    Here we simulate the protocol for learning purposes.
    """

    def __init__(self, agent_card: AgentCard):
        self.agent_card = agent_card
        self._tasks: dict[str, A2ATask] = {}
        self._known_agents: dict[str, AgentCard] = {}

    # ── Agent Discovery ──

    def register_agent(self, card: AgentCard):
        """注册已知 Agent。"""
        self._known_agents[card.name] = card

    def discover(self, agent_name: str) -> Optional[AgentCard]:
        """发现 Agent 能力。"""
        return self._known_agents.get(agent_name)

    def list_known_agents(self) -> list[dict]:
        """列出所有已知 Agent。"""
        return [
            {"name": name, "skills": [s.name for s in card.skills]}
            for name, card in self._known_agents.items()
        ]

    # ── Task Management ──

    def send_task(
        self,
        agent_name: str,
        message: str,
        context_id: str = "",
    ) -> A2ATask:
        """
        向另一个 Agent 发送任务请求。

        Returns the created task (simulated response).
        """
        target = self._known_agents.get(agent_name)
        if not target:
            raise ValueError(f"Unknown agent: {agent_name}")

        task = A2ATask(
            context_id=context_id or str(uuid.uuid4()),
        )
        task.add_message("user", message, {"target_agent": agent_name})

        # Simulate agent working on the task
        task.status = TaskStatus.WORKING
        task.add_message("agent", f"[{agent_name}] Processing your request...")

        # Simulate completion
        response = self._simulate_response(agent_name, message)
        task.add_artifact(response)
        task.updated_at = time.time()

        self._tasks[task.task_id] = task
        return task

    async def send_task_async(
        self,
        agent_name: str,
        message: str,
        context_id: str = "",
    ) -> A2ATask:
        """异步版本（适配 async agent pipeline）。"""
        import asyncio
        await asyncio.sleep(0.1)  # Simulate network
        return self.send_task(agent_name, message, context_id)

    def get_task(self, task_id: str) -> Optional[A2ATask]:
        """查询任务状态。"""
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务。"""
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.SUBMITTED, TaskStatus.WORKING):
            task.status = TaskStatus.CANCELED
            return True
        return False

    def list_tasks(self, context_id: str = None) -> list[dict]:
        """列出任务。"""
        tasks = list(self._tasks.values())
        if context_id:
            tasks = [t for t in tasks if t.context_id == context_id]
        return [t.to_dict() for t in tasks]

    # ── Collaboration Patterns ──

    def chain_agents(
        self,
        agent_sequence: list[str],
        initial_message: str,
    ) -> list[A2ATask]:
        """
        Agent Chain — 串行多 Agent 协作。

        Example: Researcher → Analyst → Writer
        """
        tasks = []
        current_context = str(uuid.uuid4())
        current_message = initial_message

        for agent_name in agent_sequence:
            if agent_name not in self._known_agents:
                raise ValueError(f"Unknown agent in chain: {agent_name}")

            task = self.send_task(agent_name, current_message, current_context)
            tasks.append(task)

            # Pass the result to the next agent
            if task.artifacts:
                current_message = f"Previous result:\n{task.artifacts[-1].content}\n\nPlease continue with: {current_message}"

        return tasks

    def fan_out(
        self,
        agents: list[str],
        message: str,
    ) -> list[A2ATask]:
        """
        Fan-Out — 并行多 Agent 协作（Map pattern）。

        Example: Send same query to multiple specialists, collect all responses.
        """
        context_id = str(uuid.uuid4())
        tasks = []
        for agent_name in agents:
            if agent_name in self._known_agents:
                task = self.send_task(agent_name, message, context_id)
                tasks.append(task)
        return tasks

    def fan_in(self, tasks: list[A2ATask], synthesis_agent: str) -> A2ATask:
        """
        Fan-In — 汇总多个 Agent 的结果（Reduce pattern）。

        Example: Synthesizer combines results from multiple specialists.
        """
        if synthesis_agent not in self._known_agents:
            raise ValueError(f"Unknown synthesis agent: {synthesis_agent}")

        # Collect all artifacts
        combined = "\n\n---\n\n".join(
            f"[{t.messages[0].metadata.get('target_agent', 'unknown')}]\n"
            + (t.artifacts[-1].content if t.artifacts else "No result")
            for t in tasks
        )

        synthesis_task = self.send_task(
            synthesis_agent,
            f"Synthesize the following results from multiple agents:\n\n{combined}",
            context_id=tasks[0].context_id if tasks else "",
        )
        return synthesis_task

    # ── Helpers ──

    @staticmethod
    def _simulate_response(agent_name: str, message: str) -> str:
        """模拟 Agent 响应（用于概念验证）。"""
        return (
            f"[{agent_name} response]\n"
            f"Task received: {message[:100]}...\n"
            f"This is a simulated A2A response for learning purposes."
        )


# ──────────────────────────────────────────────
# 3. A2A vs MCP Comparison (learning reference)
# ──────────────────────────────────────────────

A2A_VS_MCP = """
┌─────────────────────────────────────────────────────────────────┐
│  A2A vs MCP — Key Differences                                    │
├───────────────────────┬───────────────────┬─────────────────────┤
│  Aspect               │  MCP              │  A2A                │
├───────────────────────┼───────────────────┼─────────────────────┤
│  Communication        │  Agent ↔ Tool     │  Agent ↔ Agent      │
│  Protocol             │  JSON-RPC over    │  HTTP/gRPC +        │
│                       │  stdio/SSE        │  streaming          │
│  Discovery            │  Server lists     │  Agent Card         │
│                       │  tools            │  (skills, modes)    │
│  State                │  Stateless calls   │  Task lifecycle     │
│  Autonomy             │  Tool is passive   │  Agent is active    │
│  Streaming            │  SSE for progress  │  Streaming + events │
│  Use Case             │  "Use this API"    │  "Collaborate with  │
│                       │                   │   this agent"        │
└───────────────────────┴───────────────────┴─────────────────────┘

Key Insight: "Agents Are Not Tools"
  - MCP: Agent uses a tool (one-directional, stateless)
  - A2A: Agent collaborates with another agent (bidirectional, stateful)
  - They are complementary, not competitive!

Ecosystem:
  - Google ADK (Agent Development Kit) — A2A reference implementation
  - IBM ACP (Agent Communication Protocol)
  - Cisco agntcy — Open source A2A framework
"""
