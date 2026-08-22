"""
05: 完整集成演示 - LangGraph Agent 调用 Tool + MCP + Skill
展示:
  1. 三种能力如何集成到一个 Agent
  2. Agent 如何自动选择合适的工具/Skill
  3. 完整的调用流程和结果展示

运行前:
  export OPENAI_API_KEY="your-key-here"
  pip install -r requirements.txt
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json
import asyncio
from typing import Literal

# ============================================================
# 1. 定义基础工具 (Plugin Tool 概念)
# ============================================================

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a given city.
    
    Args:
        city: City name, e.g. Shanghai, Tokyo
        unit: Temperature unit, 'celsius' or 'fahrenheit'
    """
    mock_data = {
        "shanghai": {"temp": 28, "condition": "Sunny ☀️", "humidity": 65},
        "tokyo": {"temp": 25, "condition": "Rainy 🌧️", "humidity": 80},
    }
    data = mock_data.get(city.lower(), {"temp": 20, "condition": "Unknown", "humidity": 50})
    if unit == "fahrenheit":
        data["temp"] = round(data["temp"] * 9 / 5 + 32)
    return json.dumps(data, ensure_ascii=False)


@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a given symbol."""
    mock_prices = {
        "AAPL": {"price": 215.48, "currency": "USD"},
        "TSLA": {"price": 248.42, "currency": "USD"},
        "BABA": {"price": 85.30, "currency": "USD"},
    }
    info = mock_prices.get(symbol.upper(), {"price": 0, "currency": "USD"})
    return json.dumps({symbol.upper(): info}, ensure_ascii=False)


# ============================================================
# 2. 模拟 MCP Server (MCP Tool 概念)
# ============================================================

class MockMCPClient:
    def __init__(self):
        self.tools = [
            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "query_database",
                "description": "Execute SQL query against the database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL query"}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    async def call_tool(self, name, arguments):
        print(f"  📡 [MCP] 调用: {name}({arguments})")
        mock_results = {
            "read_file": {"content": [{"type": "text", "text": "Hello World\nSample file content."}]},
            "query_database": {"content": [{"type": "text", "text": "id | name\n1  | Alice\n2  | Bob"}]},
        }
        return mock_results.get(name, {"content": [{"type": "text", "text": "Unknown"}]})["content"][0]["text"]


def create_mcp_tools(client: MockMCPClient):
    """将 MCP Server 的工具注册为 LangChain Tool"""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field
    
    langchain_tools = []
    for tool_def in client.tools:
        schema = tool_def["inputSchema"]
        fields = {}
        for prop_name, prop_def in schema.get("properties", {}).items():
            fields[prop_name] = (str, Field(description=prop_def.get("description", "")))
        
        params = type(
            f"{tool_def['name']}Params",
            (BaseModel,),
            {"__annotations__": {k: v[0] for k, v in fields.items()}}
        )
        
        async def execute_tool(**kwargs, tool_name=tool_def["name"], c=client):
            return await c.call_tool(tool_name, kwargs)
        
        tool = StructuredTool.from_function(
            coroutine=execute_tool,
            name=tool_def["name"],
            description=tool_def["description"],
            args_schema=params,
        )
        langchain_tools.append(tool)
        print(f"  ✅ [MCP] 注册: {tool_def['name']}")
    
    return langchain_tools


# ============================================================
# 3. Skill 机制 (注入系统 Prompt)
# ============================================================

SKILLS_XML = """
<available_skills>
  <skill>
    <name>weather</name>
    <description>Check current weather using wttr.in API</description>
    <location>skills/weather/SKILL.md</location>
  </skill>
  <skill>
    <name>backup</name>
    <description>Backup PostgreSQL database</description>
    <location>skills/backup/SKILL.md</location>
  </skill>
</available_skills>
"""


# ============================================================
# 4. 构建完整系统 Prompt
# ============================================================

SYSTEM_PROMPT = f"""You are a helpful assistant with access to various tools and skills.

## Skills
Scan <available_skills>. If one clearly applies, read its SKILL.md with `read_file`, then follow it.
If several apply, choose the most specific. One skill up front max.

{SKILLS_XML}

Available tools:
- get_weather: Get weather info for any city
- get_stock_price: Get stock price by ticker symbol
- read_file: Read file contents (MCP)
- query_database: Execute SQL queries (MCP)

Use tools when appropriate. Answer in Chinese if user asks in Chinese.
"""


# ============================================================
# 5. 创建完整 Agent
# ============================================================

def create_full_agent():
    """创建集成 Tool + MCP + Skill 的 Agent"""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
    
    # 1. Plugin Tool
    plugin_tools = [get_weather, get_stock_price]
    
    # 2. MCP Tool
    mcp_client = MockMCPClient()
    mcp_tools = create_mcp_tools(mcp_client)
    
    # 3. 所有工具合并
    all_tools = plugin_tools + mcp_tools
    
    agent = create_react_agent(
        llm,
        tools=all_tools,
        prompt=SYSTEM_PROMPT,
    )
    
    return agent


# ============================================================
# 6. 测试用例
# ============================================================

async def run_test(query: str, expected_tools: list):
    """运行单个测试"""
    print(f"\n{'='*60}")
    print(f"测试: {query}")
    print(f"预期工具: {', '.join(expected_tools)}")
    print(f"{'='*60}")
    
    agent = create_full_agent()
    
    response = await agent.ainvoke({
        "messages": [("user", query)]
    })
    
    print(f"\n✅ Agent 回复: {response['messages'][-1].content}")
    print(f"---")


async def main():
    print("🤖 Agent + Tool + MCP + Skill 完整演示\n")
    
    # 测试 1: 天气查询 (Tool)
    await run_test(
        "上海今天天气怎么样？",
        ["get_weather"]
    )
    
    # 测试 2: 股票查询 (Tool)
    await run_test(
        "苹果和特斯拉的股票价格？",
        ["get_stock_price"]
    )
    
    # 测试 3: 文件操作 (MCP)
    await run_test(
        "帮我读取 readme.md 文件的内容",
        ["read_file"]
    )
    
    # 测试 4: 数据库查询 (MCP)
    await run_test(
        "查询数据库中的所有用户",
        ["query_database"]
    )
    
    # 测试 5: 组合查询 (多个工具)
    await run_test(
        "对比上海和东京的天气，同时查下阿里巴巴股票价格",
        ["get_weather", "get_stock_price"]
    )
    
    print("\n" + "="*60)
    print("✅ 所有测试完成!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
