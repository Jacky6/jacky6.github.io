"""
MCP 工具使用示例

演示如何将 SSE-based MCP 服务器集成到 Agent 中。

⚠️ 历史遗留：本示例使用已弃用的 SSE 传输，仅作参考。
   新实现请参考 protocols/mcp/ 的 Streamable HTTP 主线。
"""
import os
from agent_demo.graph import ResearchAgent
from agent_demo.mcp_client import MCPToolConfig

# ── 示例 1: 纯本地工具（无 MCP）──
# agent = ResearchAgent()
# agent.run("北京天气怎么样？")

# ── 示例 2: 添加 MCP 远端工具 ──
# 凭据从环境变量读取，禁止硬编码：
#   MCP_SLS_URL   — MCP Server 地址
#   MCP_SLS_TOKEN — MCP Bearer Token
mcp_url = os.environ.get("MCP_SLS_URL")
mcp_token = os.environ.get("MCP_SLS_TOKEN")

if not (mcp_url and mcp_token):
    raise SystemExit(
        "未设置 MCP_SLS_URL / MCP_SLS_TOKEN 环境变量。"
        "请复制 .env.example 为 .env 并填入真实值。"
    )

mcp_configs = [
    MCPToolConfig(
        name="sls_tb_ocr",
        url=mcp_url,
        auth_token=mcp_token,
    ),
    # 可以添加更多 MCP 服务器...
    # MCPToolConfig(
    #     name="another_mcp",
    #     url=os.environ["MCP_OTHER_URL"],
    #     auth_token=os.environ["MCP_OTHER_TOKEN"],
    # ),
]

agent = ResearchAgent(mcp_configs=mcp_configs)
agent.run("使用 OCR 工具识别以下表格...")
