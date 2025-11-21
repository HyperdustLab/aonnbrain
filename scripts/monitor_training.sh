#!/bin/bash
# 监控训练进度

LOG_FILE="data/mnist_training_5000steps.log"

echo "=========================================="
echo "MNIST 训练监控 (5000步)"
echo "=========================================="
echo ""

# 检查进程是否运行
if ps aux | grep -q "[r]un_mnist_active_inference_v2.py"; then
    echo "✅ 训练进程正在运行"
    PROGRESS=$(tail -20 "$LOG_FILE" | grep "MNIST Active" | tail -1 | sed -n 's/.*\([0-9]*\)\/5000.*/\1/p')
    if [ ! -z "$PROGRESS" ]; then
        PERCENT=$(echo "scale=1; $PROGRESS * 100 / 5000" | bc)
        echo "   当前进度: $PROGRESS/5000 步 ($PERCENT%)"
    fi
else
    echo "❌ 训练进程未运行（可能已完成）"
fi

echo ""
echo "📊 最新进度:"
tail -3 "$LOG_FILE" | grep -E "(Step|MNIST Active)" | tail -2

echo ""
echo "📈 自由能 (最近):"
tail -50 "$LOG_FILE" | grep "F=" | tail -5 | sed 's/.*F=\([0-9.]*\).*/  \1/'

echo ""
echo "🎯 准确率 (最近):"
tail -50 "$LOG_FILE" | grep "Acc=" | tail -5 | sed 's/.*Acc=\([0-9.]*\)%.*/  \1%/' 

echo ""
echo "🔄 演化事件 (最近):"
tail -200 "$LOG_FILE" | grep -E "(演化决策|evolution_option|add_pipeline|prune|最佳选项)" | tail -3

echo ""
echo "=========================================="
