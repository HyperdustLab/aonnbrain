#!/usr/bin/env python3
"""
MNIST AONN 可视化工具
用于观察和监控 MNIST AONN 的工作情况
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import Dict, List, Optional


def load_results(filepath: str) -> Dict:
    """加载实验结果"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_free_energy_evolution(data: Dict, output_dir: str = "data/plots"):
    """绘制自由能演化曲线"""
    free_energy_history = data.get('free_energy_history', [])
    if not free_energy_history:
        print("⚠️ 没有自由能历史数据")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 1. 总自由能曲线
    axes[0].plot(free_energy_history, label='Total Free Energy', color='blue', linewidth=1.5)
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('Free Energy', fontsize=12)
    axes[0].set_title('Free Energy Evolution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 添加统计信息
    if len(free_energy_history) > 0:
        initial_F = free_energy_history[0]
        final_F = free_energy_history[-1]
        min_F = min(free_energy_history)
        reduction = (initial_F - final_F) / initial_F * 100 if initial_F > 0 else 0
        
        stats_text = f'Initial: {initial_F:.2f}\nFinal: {final_F:.2f}\nMin: {min_F:.2f}\nReduction: {reduction:.1f}%'
        axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 自由能组件（如果有）
    F_obs_history = data.get('F_obs_history', [])
    F_dyn_history = data.get('F_dyn_history', [])
    F_class_history = data.get('F_class_history', [])
    
    if F_obs_history or F_dyn_history or F_class_history:
        if F_obs_history:
            axes[1].plot(F_obs_history, label='F_obs (Observation)', color='green', alpha=0.7)
        if F_dyn_history:
            axes[1].plot(F_dyn_history, label='F_dyn (Dynamics)', color='orange', alpha=0.7)
        if F_class_history:
            axes[1].plot(F_class_history, label='F_class (Classification)', color='red', alpha=0.7)
        
        axes[1].set_xlabel('Step', fontsize=12)
        axes[1].set_ylabel('Free Energy Component', fontsize=12)
        axes[1].set_title('Free Energy Components', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        # 如果没有组件数据，显示移动平均
        if len(free_energy_history) > 10:
            window = min(50, len(free_energy_history) // 10)
            moving_avg = np.convolve(free_energy_history, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(free_energy_history)), moving_avg, 
                        label=f'Moving Average (window={window})', color='purple', linewidth=2)
            axes[1].plot(free_energy_history, alpha=0.3, color='blue', label='Raw')
            axes[1].set_xlabel('Step', fontsize=12)
            axes[1].set_ylabel('Free Energy', fontsize=12)
            axes[1].set_title('Free Energy (with Moving Average)', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
    
    plt.tight_layout()
    output_path = f"{output_dir}/free_energy_evolution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 自由能演化图已保存: {output_path}")
    plt.close()


def plot_accuracy_evolution(data: Dict, output_dir: str = "data/plots"):
    """绘制准确率演化曲线"""
    accuracy_history = data.get('accuracy_history', [])
    if not accuracy_history:
        print("⚠️ 没有准确率历史数据")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 转换为百分比
    accuracy_pct = [a * 100 for a in accuracy_history]
    
    ax.plot(accuracy_pct, label='Accuracy', color='green', linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Classification Accuracy Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 添加统计信息
    if len(accuracy_pct) > 0:
        initial_acc = accuracy_pct[0]
        final_acc = accuracy_pct[-1]
        max_acc = max(accuracy_pct)
        
        stats_text = f'Initial: {initial_acc:.1f}%\nFinal: {final_acc:.1f}%\nMax: {max_acc:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    output_path = f"{output_dir}/accuracy_evolution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 准确率演化图已保存: {output_path}")
    plt.close()


def plot_network_structure_evolution(data: Dict, output_dir: str = "data/plots"):
    """绘制网络结构演化"""
    snapshots = data.get('snapshots', [])
    if not snapshots:
        print("⚠️ 没有快照数据")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    steps = [s.get('step', 0) for s in snapshots]
    num_aspects = [s.get('structure', {}).get('num_aspects', 0) for s in snapshots]
    num_objects = [s.get('structure', {}).get('num_objects', 0) for s in snapshots]
    num_pipelines = [s.get('structure', {}).get('num_pipelines', 0) for s in snapshots]
    
    # 1. Aspects 和 Objects 数量
    axes[0].plot(steps, num_aspects, label='Aspects', color='blue', marker='o', markersize=3)
    axes[0].plot(steps, num_objects, label='Objects', color='red', marker='s', markersize=3)
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Network Structure Evolution (Aspects & Objects)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. Pipelines 数量
    axes[1].plot(steps, num_pipelines, label='Pipelines', color='green', marker='^', markersize=3)
    axes[1].set_xlabel('Step', fontsize=12)
    axes[1].set_ylabel('Pipeline Count', fontsize=12)
    axes[1].set_title('Pipeline Evolution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    output_path = f"{output_dir}/network_structure_evolution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 网络结构演化图已保存: {output_path}")
    plt.close()


def plot_evolution_events(data: Dict, output_dir: str = "data/plots"):
    """绘制演化事件时间线"""
    evolution_events = data.get('evolution_events', [])
    evolution_decisions = data.get('evolution_decisions', [])
    
    if not evolution_events and not evolution_decisions:
        print("⚠️ 没有演化事件数据")
        return
    
    # 如果没有显式事件，从演化决策中提取
    if not evolution_events and evolution_decisions:
        evolution_events = []
        for dec in evolution_decisions:
            step = dec.get('step', 0)
            option = dec.get('option', 'unknown')
            evolution_events.append({
                'step': step,
                'type': option,
                'details': dec
            })
    
    if not evolution_events:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 统计事件类型
    event_types = Counter([e.get('type', 'unknown') for e in evolution_events])
    
    # 为每种事件类型分配颜色和y位置
    type_colors = {
        'add_pipeline': 'green',
        'prune': 'red',
        'prune_forced': 'darkred',
        'add_sensory_aspect': 'blue',
        'no_change': 'gray',
    }
    
    y_positions = {}
    y_offset = 0
    for event_type in event_types.keys():
        y_positions[event_type] = y_offset
        y_offset += 1
    
    # 绘制事件点
    plotted_types = set()
    for event in evolution_events:
        step = event.get('step', 0)
        event_type = event.get('type', 'unknown')
        color = type_colors.get(event_type, 'black')
        y_pos = y_positions.get(event_type, 0)
        
        label = event_type if event_type not in plotted_types else ''
        ax.scatter(step, y_pos, c=color, s=50, alpha=0.6, label=label)
        plotted_types.add(event_type)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Event Type', fontsize=12)
    ax.set_title('Evolution Events Timeline', fontsize=14, fontweight='bold')
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加图例
    handles = []
    for event_type, color in type_colors.items():
        if event_type in event_types:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                     markersize=8, label=f'{event_type} ({event_types[event_type]})'))
    ax.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    output_path = f"{output_dir}/evolution_events_timeline.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 演化事件时间线已保存: {output_path}")
    plt.close()


def print_summary(data: Dict):
    """打印实验摘要"""
    print("\n" + "=" * 80)
    print("MNIST AONN 实验摘要")
    print("=" * 80)
    
    # 最终结构
    structure = data.get('final_structure', {})
    print(f"\n📊 最终网络结构:")
    print(f"   Objects: {structure.get('num_objects', 0)}")
    print(f"   Aspects: {structure.get('num_aspects', 0)}")
    print(f"   Pipelines: {structure.get('num_pipelines', 0)}")
    
    # Pipeline 详情
    pipelines = structure.get('pipelines', [])
    if pipelines:
        print(f"\n🔗 Pipeline 详情:")
        for i, p in enumerate(pipelines, 1):
            print(f"   Pipeline {i}:")
            print(f"     深度: {p.get('depth', 0)} 层")
            print(f"     宽度: {p.get('num_aspects', 0)} aspects/层")
            print(f"     输入/输出: {p.get('input_dim', 0)} -> {p.get('output_dim', 0)}")
    
    # 自由能统计
    free_energy_history = data.get('free_energy_history', [])
    if free_energy_history:
        print(f"\n⚡ 自由能统计:")
        print(f"   初始: {free_energy_history[0]:.4f}")
        print(f"   最终: {free_energy_history[-1]:.4f}")
        print(f"   最低: {min(free_energy_history):.4f}")
        reduction = (free_energy_history[0] - free_energy_history[-1]) / free_energy_history[0] * 100
        print(f"   降低: {reduction:.2f}%")
    
    # 准确率统计
    accuracy_history = data.get('accuracy_history', [])
    if accuracy_history:
        print(f"\n🎯 准确率统计:")
        print(f"   初始: {accuracy_history[0]*100:.2f}%")
        print(f"   最终: {accuracy_history[-1]*100:.2f}%")
        print(f"   最高: {max(accuracy_history)*100:.2f}%")
    
    # 演化决策统计
    evolution_decisions = data.get('evolution_decisions', [])
    if evolution_decisions:
        from collections import Counter
        types = Counter([e.get('option', 'unknown') for e in evolution_decisions])
        print(f"\n🔄 演化决策统计:")
        for opt, count in types.most_common():
            print(f"   {opt}: {count} 次 ({count/len(evolution_decisions)*100:.1f}%)")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="MNIST AONN 可视化工具")
    parser.add_argument("--input", type=str, required=True, help="实验结果JSON文件路径")
    parser.add_argument("--output-dir", type=str, default="data/plots", help="输出目录")
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--free-energy", action="store_true", help="生成自由能演化图")
    parser.add_argument("--accuracy", action="store_true", help="生成准确率演化图")
    parser.add_argument("--structure", action="store_true", help="生成网络结构演化图")
    parser.add_argument("--events", action="store_true", help="生成演化事件时间线")
    parser.add_argument("--summary", action="store_true", help="打印实验摘要")
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"📂 加载实验结果: {args.input}")
    data = load_results(args.input)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 默认生成所有图表
    if args.all or not any([args.free_energy, args.accuracy, args.structure, args.events]):
        args.free_energy = True
        args.accuracy = True
        args.structure = True
        args.events = True
        args.summary = True
    
    # 生成图表
    if args.summary:
        print_summary(data)
    
    if args.free_energy:
        plot_free_energy_evolution(data, args.output_dir)
    
    if args.accuracy:
        plot_accuracy_evolution(data, args.output_dir)
    
    if args.structure:
        plot_network_structure_evolution(data, args.output_dir)
    
    if args.events:
        plot_evolution_events(data, args.output_dir)
    
    print(f"\n✅ 可视化完成！图表保存在: {args.output_dir}")


if __name__ == "__main__":
    main()

