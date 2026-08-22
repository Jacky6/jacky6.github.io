"""
02: Agent + 自定义 Tool
展示: 
  1. 如何用 @tool 装饰器注册工具
  2. Agent 如何自动调用工具
  3. 完整的 tool_call 流程
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import json

# ============================================================
# 注册 Tool
# ============================================================

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a given city.
    
    Args:
        city: City name, e.g. Shanghai, Tokyo, New York
        unit: Temperature unit, 'celsius' or 'fahrenheit'
    """
    # 模拟 API 调用
    mock_data = {
        "shanghai": {"temp": 28, "condition": "Sunny ☀️", "humidity": 65},
        "tokyo": {"temp": 25, "condition": "Rainy 🌧️", "humidity": 80},
        "new york": {"temp": 22, "condition": "Cloudy ⛅", "humidity": 55},
    }
    
    city_lower = city.lower()
    data = mock_data.get(city_lower, {"temp": 20, "condition": "Unknown", "humidity": 50})
    
    if unit == "fahrenheit":
        data["temp"] = round(data["temp"] * 9 / 5 + 32)
    
    return json.dumps(data, ensure_ascii=False)


@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a given symbol.
    
    Args:
        symbol: Stock ticker symbol, e.g. AAPL, TSLA, BABA
    """
    mock_prices = {
        "AAPL": {"price": 215.48, "currency": "USD", "change": "+1.2%"},
        "TSLA": {"price": 248.42, "currency": "USD", "change": "-0.5%"},
        "BABA": {"price": 85.30, "currency": "USD", "change": "+2.1%"},
    }
    
    info = mock_prices.get(symbol.upper(), {"price": 0, "currency": "USD", "change": "N/A"})
    return json.dumps({symbol.upper(): info}, ensure_ascii=False)


# ============================================================
# 创建 Agent
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# 注册工具
tools = [get_weather, get_stock_price]

# 创建 Agent (React 模式)
agent = create_react_agent(
    llm,
    tools=tools,
    prompt="""You are a helpful assistant.

Available tools:
- get_weather: Get weather info for any city
- get_stock_price: Get stock price by ticker symbol

Use tools when appropriate. Answer in Chinese if the user asks in Chinese.
""",
)


# ============================================================
# 测试用例
# ============================================================

def test_weather():
    print("\n" + "="*60)
    print("测试 1: 天气查询")
    print("="*60)
    
    response = agent.invoke({
        "messages": [("user", "上海今天天气怎么样？")]
    })
    
    print(f"\n用户: 上海今天天气怎么样？")
    print(f"Agent: {response['messages'][-1].content}")


def test_stock():
    print("\n" + "="*60)
    print("测试 2: 股票查询")
    print("="*60)
    
    response = agent.invoke({
        "messages": [("user", "苹果和特斯拉的股票价格？")]
    })
    
    print(f"\n用户: 苹果和特斯拉的股票价格？")
    print(f"Agent: {response['messages'][-1].content}")


def test_combined():
    print("\n" + "="*60)
    print("测试 3: 组合查询 (天气 + 股票)")
    print("="*60)
    
    response = agent.invoke({
        "messages": [("user", "帮我对比一下上海和东京的天气，同时查下阿里巴巴的股票价格")]
    })
    
    print(f"\n用户: 帮我对比一下上海和东京的天气，同时查下阿里巴巴的股票价格")
    print(f"Agent: {response['messages'][-1].content}")


if __name__ == "__main__":
    test_weather()
    test_stock()
    test_combined()
    print("\n✅ 所有测试完成!")
