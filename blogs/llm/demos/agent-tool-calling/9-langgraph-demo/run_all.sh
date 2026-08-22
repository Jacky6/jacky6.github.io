#!/bin/bash
# run_all.sh - 一键运行所有 demo

set -e

export OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"

echo "🤖 Agent Tool/MCP/Skill Demo - 运行所有测试"
echo "================================================"

cd "$(dirname "$0")"

echo ""
echo "📦 安装依赖..."
pip install -r requirements.txt -q

echo ""
echo "▶️ 运行 01: 基础 Agent"
python 01_basic_agent.py || echo "⚠️  需要设置 OPENAI_API_KEY"

echo ""
echo "▶️ 运行 02: Tool 演示"
python 02_tool_demo.py || echo "⚠️  需要设置 OPENAI_API_KEY"

echo ""
echo "▶️ 运行 03: MCP 演示"
python 03_mcp_demo.py || echo "⚠️  需要设置 OPENAI_API_KEY"

echo ""
echo "▶️ 运行 04: Skill 演示"
python 04_skill_demo.py || echo "⚠️  需要设置 OPENAI_API_KEY"

echo ""
echo "▶️ 运行 05: 完整集成演示"
python 05_full_demo.py || echo "⚠️  需要设置 OPENAI_API_KEY"

echo ""
echo "✅ 所有 demo 运行完成!"
