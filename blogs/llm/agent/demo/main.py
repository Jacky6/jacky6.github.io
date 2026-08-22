"""运行 Demo — 真实 LLM (qwen3-max)，异步流式输出。"""
import asyncio
from agent_demo.graph import ResearchAgent
from agent_demo.mcp_client import MCPToolConfig

async def main():
    # SLS MCP 远端工具配置
    mcp_configs = [
        MCPToolConfig(
            name="sls_tb_ocr",
            url="https://aliyun-er-k-liv-xlpqubyrhq.cn-beijing.fcapp.run/sse",
            auth_token="A5390750-3581-4701-A9F9-77418B494709",
        ),
    ]

    agent = ResearchAgent(mcp_configs=mcp_configs)

    # 全链路验证: 只调用一次工具
    await agent.arun(
        "请调用 sls_list_projects 工具，参数 regionId='cn-beijing'，列出北京区域的 SLS 项目。"
        "收到工具结果后，用中文总结你看到的项目列表，每个项目一行。"
    )

if __name__ == "__main__":
    asyncio.run(main())
