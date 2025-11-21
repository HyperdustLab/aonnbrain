#!/usr/bin/env python3
"""
分析改进版纯 FEP MNIST 60000步实验结果
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results(json_path):
    """加载实验结果"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def plot_results(data, output_dir="data/plots"):
    """绘制分析图表"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    free_energy_history = data.get('free_energy_history', [])
    accuracy_history = data.get('accuracy_history', [])
    F_obs_history = data.get('F_obs_history', [])
    F_encoder_history = data.get('F_encoder_history', [])
    F_dyn_history = data.get('F_dyn_history', [])
    F_pref_history = data.get('F_pref_history', [])
    
    config = data.get('config', {})
    obs_weight = config.get('obs_weight', 0.1)
    encoder_weight = config.get('encoder_weight', 1.0)
    pref_weight = config.get('pref_weight', 10.0)
    
    steps = np.arange(len(free_energy_history))
    
    # 1. 综合分析图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1.1 总自由能（加权后）
    ax = axes[0, 0]
    ax.plot(steps, free_energy_history, alpha=0.3, linewidth=0.5, label='Total Free Energy (Weighted)', color='blue')
    window = 1000
    if len(free_energy_history) > window:
        moving_avg = np.convolve(free_energy_history, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], moving_avg, linewidth=2, label=f'Moving Average ({window} steps)', color='red')
    ax.set_xlabel('Step')
    ax.set_ylabel('Free Energy (Weighted)')
    ax.set_title('Total Free Energy Evolution (Improved)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.2 准确率变化
    ax = axes[0, 1]
    if accuracy_history:
        eval_steps = np.arange(len(accuracy_history)) * config.get('eval_interval', 100)
        ax.plot(eval_steps, [a*100 for a in accuracy_history], 'o-', alpha=0.6, linewidth=1, markersize=3, label='Accuracy')
        ax.axhline(y=10, color='gray', linestyle='--', label='Random Guess (10%)')
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Target (90%)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Classification Accuracy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
    
    # 1.3 自由能组件（加权前）
    ax = axes[1, 0]
    if F_obs_history and F_encoder_history and F_dyn_history and F_pref_history:
        # 使用移动平均平滑
        if len(F_obs_history) > window:
            F_obs_smooth = np.convolve(F_obs_history, np.ones(window)/window, mode='valid')
            F_encoder_smooth = np.convolve(F_encoder_history, np.ones(window)/window, mode='valid')
            F_dyn_smooth = np.convolve(F_dyn_history, np.ones(window)/window, mode='valid')
            F_pref_smooth = np.convolve(F_pref_history, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            
            ax.plot(smooth_steps, F_obs_smooth, label='F_obs (raw)', linewidth=2, alpha=0.7)
            ax.plot(smooth_steps, F_encoder_smooth, label='F_encoder (raw)', linewidth=2, alpha=0.7)
            ax.plot(smooth_steps, F_dyn_smooth, label='F_dyn (raw)', linewidth=2, alpha=0.7)
            ax.plot(smooth_steps, F_pref_smooth, label='F_pref (raw)', linewidth=2, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Free Energy Component (Raw)')
    ax.set_title('Free Energy Components Evolution (Raw Values)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # 使用对数刻度
    
    # 1.4 自由能组件占比（加权后）
    ax = axes[1, 1]
    if F_obs_history and F_encoder_history and F_dyn_history and F_pref_history:
        # 计算最后1000步的加权平均
        last_n = min(1000, len(F_obs_history))
        F_total_weighted = (
            obs_weight * np.mean(F_obs_history[-last_n:]) +
            encoder_weight * np.mean(F_encoder_history[-last_n:]) +
            np.mean(F_dyn_history[-last_n:]) +
            pref_weight * np.mean(F_pref_history[-last_n:])
        )
        F_obs_weighted = obs_weight * np.mean(F_obs_history[-last_n:])
        F_encoder_weighted = encoder_weight * np.mean(F_encoder_history[-last_n:])
        F_dyn_weighted = np.mean(F_dyn_history[-last_n:])
        F_pref_weighted = pref_weight * np.mean(F_pref_history[-last_n:])
        
        ax.bar(['F_obs\n(weight=0.1)', 'F_encoder\n(weight=1.0)', 'F_dyn', 'F_pref\n(weight=10.0)'],
               [F_obs_weighted/F_total_weighted*100, F_encoder_weighted/F_total_weighted*100, 
                F_dyn_weighted/F_total_weighted*100, F_pref_weighted/F_total_weighted*100],
               color=['blue', 'green', 'orange', 'red'])
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Free Energy Components Distribution (Weighted, Last 1000 steps)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pure_fep_mnist_improved_60000steps_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_dir}/pure_fep_mnist_improved_60000steps_analysis.png")
    plt.close()
    
    # 2. 对比原版和改进版（如果原版结果存在）
    original_path = "data/pure_fep_mnist_60000steps.json"
    if Path(original_path).exists():
        original_data = load_results(original_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 2.1 自由能对比
        ax = axes[0]
        orig_F = original_data.get('free_energy_history', [])
        imp_F = free_energy_history
        
        if len(orig_F) > window:
            orig_smooth = np.convolve(orig_F, np.ones(window)/window, mode='valid')
            imp_smooth = np.convolve(imp_F, np.ones(window)/window, mode='valid')
            orig_steps = np.arange(len(orig_F))[window-1:]
            imp_steps = np.arange(len(imp_F))[window-1:]
            
            ax.plot(orig_steps, orig_smooth, label='Original', linewidth=2, alpha=0.7)
            ax.plot(imp_steps, imp_smooth, label='Improved', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Free Energy (Moving Average)')
        ax.set_title('Free Energy Comparison: Original vs Improved')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2.2 准确率对比
        ax = axes[1]
        orig_acc = original_data.get('accuracy_history', [])
        imp_acc = accuracy_history
        
        if orig_acc and imp_acc:
            orig_eval_steps = np.arange(len(orig_acc)) * 100
            imp_eval_steps = np.arange(len(imp_acc)) * config.get('eval_interval', 100)
            
            ax.plot(orig_eval_steps, [a*100 for a in orig_acc], 'o-', label='Original', linewidth=1, markersize=3, alpha=0.7)
            ax.plot(imp_eval_steps, [a*100 for a in imp_acc], 's-', label='Improved', linewidth=1, markersize=3, alpha=0.7)
            ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Random (10%)')
            ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Target (90%)')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Comparison: Original vs Improved')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pure_fep_mnist_original_vs_improved.png', dpi=150, bbox_inches='tight')
        print(f"✅ 对比图表已保存: {output_dir}/pure_fep_mnist_original_vs_improved.png")
        plt.close()

if __name__ == "__main__":
    json_path = "data/pure_fep_mnist_improved_60000steps.json"
    data = load_results(json_path)
    plot_results(data)
    print("\n✅ 分析完成！")

