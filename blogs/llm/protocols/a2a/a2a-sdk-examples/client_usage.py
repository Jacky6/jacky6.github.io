"""A2A Client 用法示例 — a2a-sdk 官方客户端。

对应教程：protocols/a2a/11-A2A_Protocol.md 第 4 节

演示 Client 的正确用法（基于 a2a-sdk 1.3.0 实测的 API 签名）：
- create_client 工厂（URL 或 AgentCard）
- SendMessageRequest 构造（字段是 message，不是 params）
- send_message 流式消费（StreamResponse oneof）

⚠️ 本示例需要真实运行的 A2A Server 才能完整执行。
   运行方式见 README.md；数据模型验证请先跑 data_models.py（无需网络）。
"""

import asyncio

from a2a.client import create_client, ClientConfig
from a2a.types import Message, Part, Role
from a2a.types.a2a_pb2 import SendMessageRequest, GetTaskRequest


async def send_and_stream(agent_url: str, text: str) -> None:
    """连接远程 Agent，发送消息并消费流式响应。"""

    # ① 创建 Client：传 URL（自动发现 AgentCard）或直接传 AgentCard
    client = await create_client(
        agent_url,
        client_config=ClientConfig(streaming=True),
    )

    # ② 构造请求（注意：字段是 message，不是 params）
    request = SendMessageRequest(
        message=Message(
            message_id="msg-001",          # 1.0 必填
            role=Role.ROLE_USER,
            parts=[Part(text=text)],
        )
    )

    # ③ send_message 返回 AsyncIterator[StreamResponse]
    async for response in client.send_message(request):
        # StreamResponse 是 oneof，用 HasField 判断
        if response.HasField("task"):
            task = response.task
            print(f"[Task] {task.id} state={task.status.state}")
        elif response.HasField("message"):
            for part in response.message.parts:
                if part.text:
                    print(f"[回复] {part.text}")
        elif response.HasField("status_update"):
            print(f"[状态] {response.status_update.status.state}")
        elif response.HasField("artifact_update"):
            print(f"[产物] {response.artifact_update.artifact.name}")


async def poll_task(agent_url: str, task_id: str) -> None:
    """轮询任务状态（非流式备选方案）。"""
    client = await create_client(agent_url)
    task = await client.get_task(GetTaskRequest(task_id=task_id))
    print(f"Task {task.id}: state={task.status.state}")
    for artifact in task.artifacts:
        for part in artifact.parts:
            if part.text:
                print(f"  产物: {part.text}")


if __name__ == "__main__":
    # 需要真实 Server：python3 server_minimal.py（另开终端）
    AGENT_URL = "http://localhost:9999"

    print("本示例需要运行中的 A2A Server。")
    print("先运行: python3 server_minimal.py")
    print(f"然后: 修改 AGENT_URL={AGENT_URL} 后取消下方注释\n")

    # asyncio.run(send_and_stream(AGENT_URL, "帮我分析这份报告"))
    # asyncio.run(poll_task(AGENT_URL, "task-001"))
