"""A2A 1.0 数据模型验证 — a2a-sdk 官方类型示例。

对应教程：protocols/a2a/11-A2A_Protocol.md

验证 A2A 1.0 的核心数据模型（全部基于 a2a-sdk 1.3.0 实测）：
- Message（必填 message_id、Role 枚举、Part oneof）
- AgentCard（supported_interfaces 而非顶层 url）
- Task / TaskStatus / TaskState（proto 风格枚举）
- SendMessageRequest / StreamResponse

运行：python3 data_models.py
依赖：pip install a2a-sdk
"""

from a2a.types import (
    AgentCard, AgentInterface, AgentSkill,
    Artifact, Message, Part, Role,
    Task, TaskStatus, TaskState,
)
from a2a.types.a2a_pb2 import SendMessageRequest, SendMessageResponse, StreamResponse


def demo_message():
    """Message：必填 message_id + Role 枚举 + Part oneof。"""
    msg = Message(
        message_id="msg-001",              # 1.0 必填
        role=Role.ROLE_USER,               # 枚举，不是 "user" 字符串
        parts=[
            Part(text="帮我分析这份报告"),   # oneof: text
            Part(url="https://example.com/report.pdf"),  # oneof: url
        ],
        context_id="ctx-001",              # 多轮会话关联
    )
    print(f"Message: id={msg.message_id}, role={msg.role}")
    print(f"  parts: text={msg.parts[0].text!r}, url={msg.parts[1].url!r}")
    return msg


def demo_agent_card():
    """AgentCard：接口在 supported_interfaces，不是顶层 url。"""
    card = AgentCard(
        name="research-agent",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url="https://agent.example.com/a2a",
                protocol_binding="jsonrpc",
            )
        ],
        skills=[
            AgentSkill(
                id="deep-research",
                name="深度研究",
                description="多源检索与综合分析",
            )
        ],
    )
    print(f"\nAgentCard: name={card.name}, version={card.version}")
    print(f"  接口: {list(card.supported_interfaces)[0].url}")
    print(f"  技能: {[s.name for s in card.skills]}")
    return card


def demo_task_lifecycle():
    """Task 生命周期：TaskState 枚举（proto 风格带前缀）。"""
    states = [
        "TASK_STATE_SUBMITTED",
        "TASK_STATE_WORKING",
        "TASK_STATE_COMPLETED",
    ]
    task = Task(id="task-001", context_id="ctx-001")
    for s in states:
        task.status.state = TaskState.Value(s)
        print(f"  Task {task.id}: {s}")
    return task


def demo_artifact():
    """Artifact：Task 的交付物。"""
    artifact = Artifact(
        artifact_id="art-001",
        name="分析报告",
        parts=[Part(text="结论：市场增长 15%")],
    )
    print(f"\nArtifact: {artifact.name}, 内容={artifact.parts[0].text!r}")
    return artifact


def demo_request_response():
    """请求/响应：SendMessageRequest 的字段是 message（不是 params）。"""
    req = SendMessageRequest(
        message=Message(
            message_id="msg-002",
            role=Role.ROLE_USER,
            parts=[Part(text="开始任务")],
        )
    )
    print(f"\nSendMessageRequest: message_id={req.message.message_id}")

    # StreamResponse 是 oneof：task / message / status_update / artifact_update
    print(f"StreamResponse oneof 字段: {[f.name for f in StreamResponse().DESCRIPTOR.fields]}")
    print(f"SendMessageResponse oneof 字段: {[f.name for f in SendMessageResponse().DESCRIPTOR.fields]}")
    return req


if __name__ == "__main__":
    print("=" * 60)
    print("A2A 1.0 数据模型验证（a2a-sdk 1.3.0）")
    print("=" * 60)

    demo_message()
    demo_agent_card()

    print("\nTask 生命周期:")
    demo_task_lifecycle()

    demo_artifact()
    demo_request_response()

    print("\n✅ 全部数据模型构造成功")
