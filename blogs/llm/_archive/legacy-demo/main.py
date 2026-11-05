"""运行 Demo — 真实 LLM (qwen3-max)，异步流式输出。"""
import asyncio
import os
from agent_demo.graph import ResearchAgent
from agent_demo.mcp_client import MCPToolConfig

async def main():
    # SLS MCP 远端工具配置
    # 凭据从环境变量读取，禁止硬编码：
    #   MCP_SLS_URL         — MCP Server 地址
    #   MCP_SLS_TOKEN       — MCP Bearer Token
    mcp_url = os.environ.get("MCP_SLS_URL")
    mcp_token = os.environ.get("MCP_SLS_TOKEN")
    mcp_configs = []
    if mcp_url and mcp_token:
        mcp_configs = [
            MCPToolConfig(
                name="sls_tb_ocr",
                url=mcp_url,
                auth_token=mcp_token,
            ),
        ]
    else:
        print("提示：未设置 MCP_SLS_URL / MCP_SLS_TOKEN，将仅使用本地工具运行。")

    agent = ResearchAgent(mcp_configs=mcp_configs)

    # 全链路验证: 只调用一次工具
    await agent.arun(
        "请调用 sls_list_projects 工具，参数 regionId='cn-beijing'，列出北京区域的 SLS 项目。"
        "收到工具结果后，用中文总结你看到的项目列表，每个项目一行。"
    )

if __name__ == "__main__":
    asyncio.run(main())
