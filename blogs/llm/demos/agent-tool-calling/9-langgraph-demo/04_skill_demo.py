"""
04: Agent + Skill 机制
展示:
  1. Skill 如何注入系统 Prompt
  2. Agent 如何"读取" SKILL.md
  3. Skill 如何指导 Agent 使用工具

核心概念: Skill 不注册工具！它只是系统 Prompt 里的指南。
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
import json
import os

# ============================================================
# 工具定义 (Agent 可用的基本工具)
# ============================================================

@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def web_fetch(url: str) -> str:
    """Fetch content from a URL."""
    # 模拟 web_fetch
    print(f"  🌐 访问: {url}")
    
    if "wttr.in" in url:
        city = url.split("/")[-1].split("?")[0]
        return json.dumps({
            "city": city,
            "temperature": "28°C",
            "condition": "Sunny ☀️",
            "humidity": "65%",
            "wind": "10 km/h"
        }, ensure_ascii=False)
    
    return f"<html>Content from {url}</html>"


@tool
def run_shell_command(command: str) -> str:
    """Execute a shell command."""
    print(f"  💻 执行: {command}")
    
    if "backup" in command.lower() or "pg_dump" in command:
        return "Backup successful: /backup/mydb_20250528.sql (15MB)"
    
    return "Command executed successfully."


# ============================================================
# Skill 定义 (注入系统 Prompt 的 XML 格式)
# ============================================================

SKILLS_XML = """
<available_skills>
  <skill>
    <name>weather</name>
    <description>Check current weather and forecasts using wttr.in API</description>
    <location>skills/weather/SKILL.md</location>
  </skill>
  <skill>
    <name>backup</name>
    <description>Backup PostgreSQL database to local directory</description>
    <location>skills/backup/SKILL.md</location>
  </skill>
</available_skills>
"""

# Skill 的详细指南 (模拟读取 SKILL.md)
SKILL_GUIDES = {
    "weather": """# Weather Skill

## How to use
1. Use the `web_fetch` tool to call wttr.in API:
   - Quick: `web_fetch(url="https://wttr.in/{city}?format=2")`
   - Detailed: `web_fetch(url="https://wttr.in/{city}?format=j1")`

2. Parse the JSON response and summarize:
   - Temperature
   - Weather condition
   - Humidity/Wind

## Tips
- City names in English work best
- Add `&lang=zh` for Chinese output
""",
    "backup": """# Backup Skill

## How to use
1. Run backup command:
   `run_shell_command(command="pg_dump -U postgres mydb > /backup/mydb_$(date +%Y%m%d).sql")`

2. Verify backup was created:
   `run_shell_command(command="ls -lh /backup/mydb_*.sql")`

3. Report success with file size and location
"""
}


# ============================================================
# 构建系统 Prompt (包含 Skills XML)
# ============================================================

def build_system_prompt():
    """构建包含 Skills 的系统 Prompt"""
    
    return f"""You are a helpful assistant.

## Skills
Scan <available_skills>. If one clearly applies, read its SKILL.md at exact <location> with `read_file`, then follow it.
If several apply, choose the most specific. If none clearly apply, read none.
One skill up front max. Never guess/fabricate skill paths.

{SKILLS_XML}

Available tools:
- read_file: Read file contents
- web_fetch: Fetch content from URL
- run_shell_command: Execute shell commands

Answer in Chinese when user asks in Chinese.
"""


# ============================================================
# 创建带 Skill 的 Agent
# ============================================================

def create_skill_agent():
    """创建带 Skill 机制的 Agent"""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
    
    tools = [read_file, web_fetch, run_shell_command]
    
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=build_system_prompt(),
    )
    
    return agent


# ============================================================
# 测试用例
# ============================================================

def test_skill():
    print("\n" + "="*60)
    print("测试: Skill 机制")
    print("="*60)
    
    agent = create_skill_agent()
    
    print("\n用户: 上海今天天气怎么样？")
    print("预期: Agent 看到 weather skill → 调用 web_fetch")
    response = agent.invoke({
        "messages": [("user", "上海今天天气怎么样？")]
    })
    print(f"Agent: {response['messages'][-1].content}")
    
    print("\n" + "-"*40)
    print("用户: 帮我备份数据库")
    print("预期: Agent 看到 backup skill → 调用 run_shell_command")
    response = agent.invoke({
        "messages": [("user", "帮我备份数据库")]
    })
    print(f"Agent: {response['messages'][-1].content}")


if __name__ == "__main__":
    test_skill()
