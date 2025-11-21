#!/bin/bash
# 监控卷积编码器/解码器实验

LOG_FILE="data/mnist_conv_1000steps.log"
OUTPUT_FILE="data/mnist_conv_1000steps.json"

echo "等待实验完成..."
while ps aux | grep -q "[r]un_mnist_active_inference_v2.py"; do
    PROGRESS=$(tail -5 "$LOG_FILE" | grep "MNIST Active" | tail -1 | sed -n 's/.*\([0-9]*\)\/1000.*/\1/p' 2>/dev/null)
    if [ ! -z "$PROGRESS" ]; then
        PERCENT=$(echo "scale=1; $PROGRESS * 100 / 1000" | bc 2>/dev/null)
        F=$(tail -5 "$LOG_FILE" | grep "MNIST Active" | tail -1 | sed -n 's/.*F=\([0-9.]*\).*/\1/p' 2>/dev/null)
        ACC=$(tail -5 "$LOG_FILE" | grep "MNIST Active" | tail -1 | sed -n 's/.*Acc=\([0-9.]*\)%.*/\1/p' 2>/dev/null)
        echo "  进度: $PROGRESS/1000 ($PERCENT%) | F=$F | Acc=$ACC%"
    fi
    sleep 30
done

echo ""
echo "✅ 实验完成！"
echo ""

# 等待文件保存
sleep 5

# 检查文件是否存在
if [ -f "$OUTPUT_FILE" ]; then
    echo "📊 结果文件已生成: $OUTPUT_FILE"
    echo ""
    python3 << 'PYEOF'
import json
import os

filepath = 'data/mnist_conv_1000steps.json'
if os.path.exists(filepath):
    data = json.load(open(filepath))
    
    print("=" * 80)
    print("MNIST 卷积编码器/解码器实验结果（1000步）")
    print("=" * 80)
    
    print(f"\n📊 最终结果:")
    print(f"  最终自由能: {data.get('final_free_energy', 0):.4f}")
    print(f"  训练准确率: {data.get('final_accuracy', 0)*100:.2f}%")
    print(f"  验证准确率: {data.get('val_accuracy', 0)*100:.2f}%")
    
    structure = data.get('final_structure', {})
    print(f"\n🏗️  最终网络结构:")
    print(f"  Objects: {structure.get('num_objects', 0)}")
    print(f"  Aspects: {structure.get('num_aspects', 0)}")
    print(f"  Pipelines: {structure.get('num_pipelines', 0)}")
    
    free_energy_history = data.get('free_energy_history', [])
    if free_energy_history:
        print(f"\n⚡ 自由能变化:")
        print(f"  初始: {free_energy_history[0]:.4f}")
        print(f"  最终: {free_energy_history[-1]:.4f}")
        reduction = (free_energy_history[0] - free_energy_history[-1]) / free_energy_history[0] * 100 if free_energy_history[0] > 0 else 0
        print(f"  降低: {reduction:.2f}%")
        print(f"  最低: {min(free_energy_history):.4f}")
    
    accuracy_history = data.get('accuracy_history', [])
    if accuracy_history:
        print(f"\n🎯 准确率变化:")
        print(f"  初始: {accuracy_history[0]*100:.2f}%")
        print(f"  最终: {accuracy_history[-1]*100:.2f}%")
        print(f"  最高: {max(accuracy_history)*100:.2f}%")
    
    F_obs_history = data.get('F_obs_history', [])
    F_dyn_history = data.get('F_dyn_history', [])
    F_class_history = data.get('F_class_history', [])
    if F_obs_history and F_dyn_history and F_class_history:
        F_total_final = F_obs_history[-1] + F_dyn_history[-1] + F_class_history[-1]
        print(f"\n📈 自由能组件（最终值）:")
        print(f"  F_obs: {F_obs_history[-1]:.4f} ({F_obs_history[-1]/F_total_final*100:.1f}%)")
        print(f"  F_dyn: {F_dyn_history[-1]:.4f} ({F_dyn_history[-1]/F_total_final*100:.1f}%)")
        print(f"  F_class: {F_class_history[-1]:.4f} ({F_class_history[-1]/F_total_final*100:.1f}%)")
    
    # 与之前的线性网络对比
    print(f"\n📊 与线性网络对比（1000步）:")
    print(f"  线性网络: F=172.22, Acc=15.0%, F_obs=166.73 (96.8%)")
    print(f"  卷积网络: F={data.get('final_free_energy', 0):.2f}, Acc={data.get('final_accuracy', 0)*100:.1f}%, F_obs={F_obs_history[-1] if F_obs_history else 0:.2f}")
    
    print("\n" + "=" * 80)
else:
    print("❌ 结果文件尚未生成")
PYEOF
else
    echo "❌ 结果文件尚未生成，请检查日志: $LOG_FILE"
    tail -20 "$LOG_FILE"
fi
