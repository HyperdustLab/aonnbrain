#!/bin/bash
# 监控300000步实验并检查模型权重保存

LOG_FILE="data/pure_fep_mnist_improved_300000steps.log"
OUTPUT_FILE="data/pure_fep_mnist_improved_300000steps.json"
MODEL_FILE="data/pure_fep_mnist_improved_300000steps_model.pth"

echo "等待实验完成..."
while ps aux | grep -q "[r]un_pure_fep_mnist_improved.py.*300000"; do
    PROGRESS=$(tail -5 "$LOG_FILE" | grep "Pure FEP MNIST" | tail -1 | sed -n 's/.*\([0-9]*\)\/300000.*/\1/p' 2>/dev/null)
    if [ ! -z "$PROGRESS" ]; then
        PERCENT=$(echo "scale=2; $PROGRESS * 100 / 300000" | bc 2>/dev/null)
        F=$(tail -5 "$LOG_FILE" | grep "Pure FEP MNIST" | tail -1 | sed -n 's/.*F=\([0-9.]*\).*/\1/p' 2>/dev/null)
        ACC=$(tail -5 "$LOG_FILE" | grep "Pure FEP MNIST" | tail -1 | sed -n 's/.*Acc=\([0-9.]*\)%.*/\1/p' 2>/dev/null)
        echo "  进度: $PROGRESS/300000 ($PERCENT%) | F=$F | Acc=$ACC%"
    fi
    sleep 60
done

echo ""
echo "✅ 实验完成！"
echo ""

# 等待文件保存
sleep 5

# 检查文件是否存在
if [ -f "$OUTPUT_FILE" ]; then
    echo "📊 结果文件已生成: $OUTPUT_FILE"
else
    echo "❌ 结果文件尚未生成"
fi

if [ -f "$MODEL_FILE" ]; then
    echo "✅ 模型权重文件已生成: $MODEL_FILE"
    ls -lh "$MODEL_FILE"
else
    echo "❌ 模型权重文件尚未生成"
fi

echo ""
python3 << 'PYEOF'
import json
import os
import numpy as np

filepath = 'data/pure_fep_mnist_improved_300000steps.json'
if os.path.exists(filepath):
    data = json.load(open(filepath))
    
    print("=" * 80)
    print("纯 FEP MNIST 改进版实验结果（300000步，带权重）")
    print("=" * 80)
    
    print(f"\n📊 最终结果:")
    print(f"  最终自由能: {data.get('final_free_energy', 0):.4f}")
    print(f"  训练准确率: {data.get('final_accuracy', 0)*100:.2f}%")
    print(f"  验证准确率: {data.get('val_accuracy', 0)*100:.2f}%")
    
    if 'model_path' in data:
        print(f"  模型权重: {data['model_path']}")
        if os.path.exists(data['model_path']):
            import os
            size = os.path.getsize(data['model_path']) / (1024 * 1024)
            print(f"  权重文件大小: {size:.2f} MB")
    
    free_energy_history = data.get('free_energy_history', [])
    if free_energy_history:
        print(f"\n⚡ 自由能变化:")
        print(f"  初始: {free_energy_history[0]:.4f}")
        print(f"  最终: {free_energy_history[-1]:.4f}")
        reduction = (free_energy_history[0] - free_energy_history[-1]) / free_energy_history[0] * 100 if free_energy_history[0] > 0 else 0
        print(f"  降低: {reduction:.2f}%")
    
    accuracy_history = data.get('accuracy_history', [])
    if accuracy_history:
        print(f"\n🎯 准确率变化:")
        print(f"  初始: {accuracy_history[0]*100:.2f}%")
        print(f"  最终: {accuracy_history[-1]*100:.2f}%")
        print(f"  最高: {max(accuracy_history)*100:.2f}%")
    
    print("\n" + "=" * 80)
else:
    print("❌ 结果文件尚未生成")
PYEOF

