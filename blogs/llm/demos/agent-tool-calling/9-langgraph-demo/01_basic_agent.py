"""
01: LangGraph Agent 基础
展示: 如何创建一个最简单的 Agent
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 创建 LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# 创建 Agent
agent = create_react_agent(
    llm,
    tools=[],  # 暂时没有工具
    prompt="You are a helpful assistant.",
)

# 调用 Agent
response = agent.invoke({
    "messages": [("user", "你好，请简单介绍一下你自己。")]
})

print("=== 回复 ===")
print(response["messages"][-1].content)
