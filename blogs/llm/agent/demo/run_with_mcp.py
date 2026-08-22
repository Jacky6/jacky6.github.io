"""
MCP 工具使用示例

演示如何将 SSE-based MCP 服务器集成到 Agent 中。
"""
from agent_demo.graph import ResearchAgent
from agent_demo.mcp_client import MCPToolConfig

# ── 示例 1: 纯本地工具（无 MCP）──
# agent = ResearchAgent()
# agent.run("北京天气怎么样？")

# ── 示例 2: 添加 MCP 远端工具 ──
mcp_configs = [
    MCPToolConfig(
        name="sls_tb_ocr",
        url="https://aliyun-er-k-liv-xlpqubyrhq.cn-beijing.fcapp.run/sse",
        auth_token="A5390750-3581-4701-A9F9-77418B494709",
    ),
    # 可以添加更多 MCP 服务器...
    # MCPToolConfig(
    #     name="another_mcp",
    #     url="https://example.com/mcp/sse",
    #     auth_token="your-token-here",
    # ),
]

agent = ResearchAgent(mcp_configs=mcp_configs)
agent.run("使用 OCR 工具识别以下表格...")
