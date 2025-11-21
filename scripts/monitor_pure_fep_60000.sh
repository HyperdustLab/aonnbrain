#!/bin/bash
# 监控纯 FEP MNIST 60000 步实验

LOG_FILE="data/pure_fep_mnist_60000steps.log"
OUTPUT_FILE="data/pure_fep_mnist_60000steps.json"

echo "等待实验完成..."
while ps aux | grep -q "[r]un_pure_fep_mnist.py"; do
    PROGRESS=$(tail -5 "$LOG_FILE" | grep "Pure FEP MNIST" | tail -1 | sed -n 's/.*\([0-9]*\)\/60000.*/\1/p' 2>/dev/null)
    if [ ! -z "$PROGRESS" ]; then
        PERCENT=$(echo "scale=1; $PROGRESS * 100 / 60000" | bc 2>/dev/null)
        F=$(tail -5 "$LOG_FILE" | grep "Pure FEP MNIST" | tail -1 | sed -n 's/.*F=\([0-9.]*\).*/\1/p' 2>/dev/null)
        ACC=$(tail -5 "$LOG_FILE" | grep "Pure FEP MNIST" | tail -1 | sed -n 's/.*Acc=\([0-9.]*\)%.*/\1/p' 2>/dev/null)
        echo "  进度: $PROGRESS/60000 ($PERCENT%) | F=$F | Acc=$ACC%"
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
    echo ""
    python3 << 'PYEOF'
import json
import os

filepath = 'data/pure_fep_mnist_60000steps.json'
if os.path.exists(filepath):
    data = json.load(open(filepath))
    
    print("=" * 80)
    print("纯 FEP MNIST 实验结果（60000步）")
    print("=" * 80)
    
    print(f"\n📊 最终结果:")
    print(f"  最终自由能: {data.get('final_free_energy', 0):.4f}")
    print(f"  训练准确率: {data.get('final_accuracy', 0)*100:.2f}%")
    print(f"  验证准确率: {data.get('val_accuracy', 0)*100:.2f}%")
    
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
        print(f"  平均: {sum(accuracy_history)/len(accuracy_history)*100:.2f}%")
    
    F_obs_history = data.get('F_obs_history', [])
    F_encoder_history = data.get('F_encoder_history', [])
    F_dyn_history = data.get('F_dyn_history', [])
    F_pref_history = data.get('F_pref_history', [])
    if F_obs_history and F_encoder_history and F_dyn_history and F_pref_history:
        F_total_final = F_obs_history[-1] + F_encoder_history[-1] + F_dyn_history[-1] + F_pref_history[-1]
        print(f"\n📈 自由能组件（最终值）:")
        print(f"  F_obs: {F_obs_history[-1]:.4f} ({F_obs_history[-1]/F_total_final*100:.1f}%)")
        print(f"  F_encoder: {F_encoder_history[-1]:.4f} ({F_encoder_history[-1]/F_total_final*100:.1f}%)")
        print(f"  F_dyn: {F_dyn_history[-1]:.4f} ({F_dyn_history[-1]/F_total_final*100:.1f}%)")
        print(f"  F_pref: {F_pref_history[-1]:.4f} ({F_pref_history[-1]/F_total_final*100:.1f}%)")
    
    print("\n" + "=" * 80)
else:
    print("❌ 结果文件尚未生成")
PYEOF
else
    echo "❌ 结果文件尚未生成，请检查日志: $LOG_FILE"
    tail -20 "$LOG_FILE"
fi

