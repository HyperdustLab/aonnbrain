#!/usr/bin/env python3
"""
分析纯 FEP MNIST 60000步实验结果
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
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
    
    steps = np.arange(len(free_energy_history))
    
    # 1. 自由能变化曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1.1 总自由能
    ax = axes[0, 0]
    ax.plot(steps, free_energy_history, alpha=0.6, linewidth=0.5, label='Total Free Energy')
    # 移动平均
    window = 1000
    if len(free_energy_history) > window:
        moving_avg = np.convolve(free_energy_history, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], moving_avg, linewidth=2, label=f'Moving Average ({window} steps)', color='red')
    ax.set_xlabel('Step')
    ax.set_ylabel('Free Energy')
    ax.set_title('Total Free Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.2 准确率变化
    ax = axes[0, 1]
    ax.plot(steps, [a*100 for a in accuracy_history], alpha=0.6, linewidth=0.5, label='Accuracy')
    if len(accuracy_history) > window:
        moving_avg = np.convolve([a*100 for a in accuracy_history], np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], moving_avg, linewidth=2, label=f'Moving Average ({window} steps)', color='red')
    ax.axhline(y=10, color='gray', linestyle='--', label='Random Guess (10%)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Accuracy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.3 自由能组件
    ax = axes[1, 0]
    if F_obs_history and F_encoder_history and F_dyn_history and F_pref_history:
        # 使用移动平均平滑
        if len(F_obs_history) > window:
            F_obs_smooth = np.convolve(F_obs_history, np.ones(window)/window, mode='valid')
            F_encoder_smooth = np.convolve(F_encoder_history, np.ones(window)/window, mode='valid')
            F_dyn_smooth = np.convolve(F_dyn_history, np.ones(window)/window, mode='valid')
            F_pref_smooth = np.convolve(F_pref_history, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            
            ax.plot(smooth_steps, F_obs_smooth, label='F_obs', linewidth=2)
            ax.plot(smooth_steps, F_encoder_smooth, label='F_encoder', linewidth=2)
            ax.plot(smooth_steps, F_dyn_smooth, label='F_dyn', linewidth=2)
            ax.plot(smooth_steps, F_pref_smooth, label='F_pref', linewidth=2)
        else:
            ax.plot(steps, F_obs_history, label='F_obs', alpha=0.6)
            ax.plot(steps, F_encoder_history, label='F_encoder', alpha=0.6)
            ax.plot(steps, F_dyn_history, label='F_dyn', alpha=0.6)
            ax.plot(steps, F_pref_history, label='F_pref', alpha=0.6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Free Energy Component')
    ax.set_title('Free Energy Components Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.4 自由能组件占比
    ax = axes[1, 1]
    if F_obs_history and F_encoder_history and F_dyn_history and F_pref_history:
        # 计算最后1000步的平均占比
        last_n = min(1000, len(F_obs_history))
        F_total_avg = (
            np.mean(F_obs_history[-last_n:]) +
            np.mean(F_encoder_history[-last_n:]) +
            np.mean(F_dyn_history[-last_n:]) +
            np.mean(F_pref_history[-last_n:])
        )
        F_obs_pct = np.mean(F_obs_history[-last_n:]) / F_total_avg * 100
        F_encoder_pct = np.mean(F_encoder_history[-last_n:]) / F_total_avg * 100
        F_dyn_pct = np.mean(F_dyn_history[-last_n:]) / F_total_avg * 100
        F_pref_pct = np.mean(F_pref_history[-last_n:]) / F_total_avg * 100
        
        ax.bar(['F_obs', 'F_encoder', 'F_dyn', 'F_pref'],
               [F_obs_pct, F_encoder_pct, F_dyn_pct, F_pref_pct],
               color=['blue', 'green', 'orange', 'red'])
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Free Energy Components Distribution (Last 1000 steps)')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pure_fep_mnist_60000steps_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_dir}/pure_fep_mnist_60000steps_analysis.png")
    plt.close()
    
    # 2. 详细学习曲线（前10000步和后10000步对比）
    if len(free_energy_history) > 20000:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 前10000步
        ax = axes[0]
        early_steps = steps[:10000]
        early_F = free_energy_history[:10000]
        ax.plot(early_steps, early_F, alpha=0.6, linewidth=0.5, label='Free Energy')
        window = 500
        if len(early_F) > window:
            moving_avg = np.convolve(early_F, np.ones(window)/window, mode='valid')
            ax.plot(early_steps[window-1:], moving_avg, linewidth=2, label=f'Moving Avg ({window})', color='red')
        ax.set_xlabel('Step')
        ax.set_ylabel('Free Energy')
        ax.set_title('Early Learning (First 10000 steps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 后10000步
        ax = axes[1]
        late_steps = steps[-10000:]
        late_F = free_energy_history[-10000:]
        ax.plot(late_steps, late_F, alpha=0.6, linewidth=0.5, label='Free Energy')
        if len(late_F) > window:
            moving_avg = np.convolve(late_F, np.ones(window)/window, mode='valid')
            ax.plot(late_steps[window-1:], moving_avg, linewidth=2, label=f'Moving Avg ({window})', color='red')
        ax.set_xlabel('Step')
        ax.set_ylabel('Free Energy')
        ax.set_title('Late Learning (Last 10000 steps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pure_fep_mnist_60000steps_early_vs_late.png', dpi=150, bbox_inches='tight')
        print(f"✅ 图表已保存: {output_dir}/pure_fep_mnist_60000steps_early_vs_late.png")
        plt.close()

if __name__ == "__main__":
    json_path = "data/pure_fep_mnist_60000steps.json"
    data = load_results(json_path)
    plot_results(data)
    print("\n✅ 分析完成！")

