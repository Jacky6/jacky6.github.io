"""
03: Agent + MCP Server
展示:
  1. 如何连接 MCP Server
  2. MCP 工具如何注册为 LangChain Tool
  3. Agent 如何调用 MCP 工具

运行前需要:
  - 启动一个 MCP Server (或使用内置的 filesystem MCP)
  - pip install langchain-mcp-adapters
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json
import asyncio


# ============================================================
# 模拟 MCP Client (真实场景用 langchain-mcp-adapters)
# ============================================================

class MockMCPClient:
    """模拟 MCP Client - 展示 MCP 工具如何工作"""
    
    def __init__(self):
        # 模拟 MCP Server 返回的工具列表
        self.tools = [
            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "list_directory",
                "description": "List contents of a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"}
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
                        "query": {"type": "string", "description": "SQL query to execute"}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    async def call_tool(self, name, arguments):
        """模拟 MCP 工具调用"""
        print(f"  📡 MCP Server 调用: {name}({arguments})")
        
        mock_results = {
            "read_file": {
                "content": [{"type": "text", "text": "Hello World\nThis is a sample file."}]
            },
            "list_directory": {
                "content": [{"type": "text", "text": "Documents/\nDownloads/\nProjects/\nreadme.md"}]
            },
            "query_database": {
                "content": [{"type": "text", "text": "id | name    | age\n1  | Alice   | 30\n2  | Bob     | 25"}]
            }
        }
        
        result = mock_results.get(name, {"content": [{"type": "text", "text": "Unknown tool"}]})
        return result["content"][0]["text"]


# ============================================================
# 创建 MCP 工具 (从 MCP Server 注册)
# ============================================================

def create_mcp_tools(client: MockMCPClient):
    """将 MCP Server 的工具注册为 LangChain Tool"""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field
    
    langchain_tools = []
    
    for tool_def in client.tools:
        # 创建动态 Pydantic 模型
        schema = tool_def["inputSchema"]
        fields = {}
        for prop_name, prop_def in schema.get("properties", {}).items():
            fields[prop_name] = (
                str,
                Field(description=prop_def.get("description", ""))
            )
        
        params = type(
            f"{tool_def['name']}Params",
            (BaseModel,),
            {"__annotations__": {k: v[0] for k, v in fields.items()}}
        )
        
        # 创建 Tool
        async def execute_tool(**kwargs, tool_name=tool_def["name"], c=client):
            return await c.call_tool(tool_name, kwargs)
        
        tool = StructuredTool.from_function(
            coroutine=execute_tool,
            name=tool_def["name"],
            description=tool_def["description"],
            args_schema=params,
        )
        
        langchain_tools.append(tool)
        print(f"  ✅ MCP Tool 注册: {tool_def['name']}")
    
    return langchain_tools


# ============================================================
# 创建 Agent
# ============================================================

def create_mcp_agent():
    """创建带 MCP 工具的 Agent"""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
    
    # 连接 MCP Server 并获取工具
    mcp_client = MockMCPClient()
    mcp_tools = create_mcp_tools(mcp_client)
    
    # 创建 Agent
    agent = create_react_agent(
        llm,
        tools=mcp_tools,
        prompt="""You are a helpful assistant with access to a file system and database.

Available MCP tools:
- read_file: Read file contents
- list_directory: List directory contents
- query_database: Execute SQL queries

Use tools to help the user explore the file system and database.
Answer in Chinese.
""",
    )
    
    return agent


# ============================================================
# 测试用例
# ============================================================

async def test_mcp():
    print("\n" + "="*60)
    print("测试: MCP 工具调用")
    print("="*60)
    
    agent = create_mcp_agent()
    
    print("\n用户: 帮我看看 Documents 目录下有什么文件")
    response = await agent.ainvoke({
        "messages": [("user", "帮我看看 Documents 目录下有什么文件")]
    })
    print(f"Agent: {response['messages'][-1].content}")
    
    print("\n用户: 读取 readme.md 文件的内容")
    response = await agent.ainvoke({
        "messages": [("user", "读取 readme.md 文件的内容")]
    })
    print(f"Agent: {response['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(test_mcp())
