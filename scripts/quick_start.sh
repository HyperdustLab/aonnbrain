#!/bin/bash
# 快速启动脚本 - 最简单的运行方式

echo "=========================================="
echo "AONN 长时间演化实验 - 快速启动"
echo "=========================================="
echo ""
echo "这将运行 15000 步演化实验（约 8 小时）"
echo "使用 Ollama cogito:32b 模型"
echo "按 Ctrl+C 可以随时中断"
echo ""
read -p "按 Enter 继续，或 Ctrl+C 取消..."

python scripts/run_long_evolution.py --ollama-model cogito:32b

