#!/usr/bin/env python3
"""
分析 MNIST 主动推理实验 V2 的结果
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


def analyze_results(result_file: str, output_dir: str = "data/plots"):
    """分析实验结果并生成可视化"""
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MNIST 主动推理实验 V2 - 结果分析")
    print("=" * 80)
    print(f"结果文件: {result_file}")
    print()
    
    # 1. 基本统计
    print("基本统计:")
    print(f"  总步数: {data.get('num_steps', 'N/A')}")
    print(f"  最终自由能: {data.get('final_free_energy', 0):.4f}")
    print(f"  最终训练准确率: {data.get('final_accuracy', 0)*100:.2f}%")
    print(f"  验证准确率: {data.get('val_accuracy', 0)*100:.2f}%")
    print()
    
    # 2. 自由能分析
    if 'free_energy_history' in data and len(data['free_energy_history']) > 0:
        F_history = np.array(data['free_energy_history'])
        
        print("自由能分析:")
        print(f"  初始: {F_history[0]:.4f}")
        print(f"  最终: {F_history[-1]:.4f}")
        print(f"  下降: {F_history[0] - F_history[-1]:.4f} ({((F_history[0] - F_history[-1]) / F_history[0] * 100):.2f}%)")
        print(f"  最小值: {F_history.min():.4f} (步数 {F_history.argmin()})")
        print(f"  最大值: {F_history.max():.4f} (步数 {F_history.argmax()})")
        print(f"  平均值: {F_history.mean():.4f}")
        print()
        
        # 绘制自由能曲线
        plt.figure(figsize=(12, 6))
        plt.plot(F_history, alpha=0.7, linewidth=1)
        plt.xlabel('Step')
        plt.ylabel('Free Energy')
        plt.title('Free Energy Evolution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'free_energy_curve.png', dpi=150)
        plt.close()
        print(f"  ✓ 保存自由能曲线: {output_dir / 'free_energy_curve.png'}")
    
    # 3. 自由能组成分析
    if 'F_obs_history' in data and len(data['F_obs_history']) > 0:
        F_obs = np.array(data['F_obs_history'])
        F_dyn = np.array(data.get('F_dyn_history', [0] * len(F_obs)))
        F_class = np.array(data.get('F_class_history', [0] * len(F_obs)))
        
        print("自由能组成分析:")
        print(f"  F_obs (观察生成):")
        print(f"    平均值: {F_obs.mean():.4f}")
        print(f"    占比: {F_obs.mean() / (F_obs.mean() + F_dyn.mean() + F_class.mean()) * 100:.2f}%")
        print(f"  F_dyn (状态转移):")
        print(f"    平均值: {F_dyn.mean():.4f}")
        print(f"    占比: {F_dyn.mean() / (F_obs.mean() + F_dyn.mean() + F_class.mean()) * 100:.2f}%")
        print(f"  F_class (分类):")
        print(f"    平均值: {F_class.mean():.4f}")
        print(f"    占比: {F_class.mean() / (F_obs.mean() + F_dyn.mean() + F_class.mean()) * 100:.2f}%")
        print()
        
        # 绘制自由能组成
        plt.figure(figsize=(12, 6))
        steps = np.arange(len(F_obs))
        plt.plot(steps, F_obs, label='F_obs (Observation)', alpha=0.7)
        plt.plot(steps, F_dyn, label='F_dyn (Dynamics)', alpha=0.7)
        plt.plot(steps, F_class, label='F_class (Classification)', alpha=0.7)
        plt.xlabel('Step')
        plt.ylabel('Free Energy Contribution')
        plt.title('Free Energy Components Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'free_energy_components.png', dpi=150)
        plt.close()
        print(f"  ✓ 保存自由能组成图: {output_dir / 'free_energy_components.png'}")
    
    # 4. 准确率分析
    if 'accuracy_history' in data and len(data['accuracy_history']) > 0:
        acc_history = np.array(data['accuracy_history'])
        
        print("准确率分析:")
        print(f"  初始 (前100步平均): {acc_history[:100].mean()*100:.2f}%")
        print(f"  最终 (后100步平均): {acc_history[-100:].mean()*100:.2f}%")
        print(f"  最高: {acc_history.max()*100:.2f}%")
        print(f"  平均值: {acc_history.mean()*100:.2f}%")
        print()
        
        # 绘制准确率曲线（移动平均）
        window = 50
        if len(acc_history) > window:
            acc_smooth = np.convolve(acc_history, np.ones(window)/window, mode='valid')
            steps_smooth = np.arange(window-1, len(acc_history))
        else:
            acc_smooth = acc_history
            steps_smooth = np.arange(len(acc_history))
        
        plt.figure(figsize=(12, 6))
        plt.plot(acc_history, alpha=0.3, label='Raw', linewidth=0.5)
        plt.plot(steps_smooth, acc_smooth, label=f'Moving Average (window={window})', linewidth=2)
        plt.axhline(y=0.1, color='r', linestyle='--', label='Random Guess (10%)', alpha=0.5)
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_curve.png', dpi=150)
        plt.close()
        print(f"  ✓ 保存准确率曲线: {output_dir / 'accuracy_curve.png'}")
    
    # 5. 快照分析
    if 'snapshots' in data and len(data['snapshots']) > 0:
        snapshots = data['snapshots']
        print(f"快照分析 (共 {len(snapshots)} 个快照):")
        print()
        
        # 提取关键指标
        steps = [s['step'] for s in snapshots]
        F_total = [s['free_energy'] for s in snapshots]
        F_obs = [s.get('free_energy_obs', 0) for s in snapshots]
        F_dyn = [s.get('free_energy_dyn', 0) for s in snapshots]
        F_class = [s.get('free_energy_class', 0) for s in snapshots]
        acc = [s['accuracy'] for s in snapshots]
        
        # 绘制快照对比
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 自由能
        axes[0, 0].plot(steps, F_total, 'o-', label='F_total', linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Free Energy')
        axes[0, 0].set_title('Total Free Energy Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 自由能组成
        axes[0, 1].plot(steps, F_obs, 'o-', label='F_obs', alpha=0.7)
        axes[0, 1].plot(steps, F_dyn, 'o-', label='F_dyn', alpha=0.7)
        axes[0, 1].plot(steps, F_class, 'o-', label='F_class', alpha=0.7)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Free Energy Contribution')
        axes[0, 1].set_title('Free Energy Components Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 准确率
        axes[1, 0].plot(steps, acc, 'o-', label='Accuracy', linewidth=2, color='green')
        axes[1, 0].axhline(y=0.1, color='r', linestyle='--', label='Random Guess', alpha=0.5)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 自由能 vs 准确率
        axes[1, 1].scatter(F_total, acc, alpha=0.6, s=50)
        axes[1, 1].set_xlabel('Free Energy')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Free Energy vs Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'snapshots_analysis.png', dpi=150)
        plt.close()
        print(f"  ✓ 保存快照分析图: {output_dir / 'snapshots_analysis.png'}")
    
    # 6. 演化事件分析（从快照中提取结构变化）
    evolution_events = []
    
    # 如果有演化事件记录，直接使用
    if 'evolution_events' in data and data['evolution_events'] and len(data['evolution_events']) > 0:
        evolution_events = data['evolution_events']
    # 否则从快照中提取结构变化
    elif 'snapshots' in data and len(data['snapshots']) > 1:
        snapshots = data['snapshots']
        prev_structure = None
        prev_snap = None
        for snap in snapshots:
            step = snap['step']
            structure = snap.get('structure', {})
            num_aspects = structure.get('num_aspects', 0)
            num_objects = structure.get('num_objects', 0)
            free_energy = snap.get('free_energy', 0)
            
            if prev_structure is not None and prev_snap is not None:
                prev_aspects = prev_structure.get('num_aspects', 0)
                prev_objects = prev_structure.get('num_objects', 0)
                prev_F = prev_snap.get('free_energy', 0)
                
                if num_aspects != prev_aspects or num_objects != prev_objects:
                    # 检测到结构变化
                    if num_objects > prev_objects:
                        evolution_events.append({
                            "step": step,
                            "event_type": "create_object",
                            "details": {"num_objects": num_objects, "prev_objects": prev_objects},
                            "trigger_condition": "structure_change",
                            "free_energy_before": prev_F,
                            "free_energy_after": free_energy,
                        })
                    if num_aspects > prev_aspects:
                        evolution_events.append({
                            "step": step,
                            "event_type": "create_aspect",
                            "details": {"num_aspects": num_aspects, "prev_aspects": prev_aspects},
                            "trigger_condition": "structure_change",
                            "free_energy_before": prev_F,
                            "free_energy_after": free_energy,
                        })
            
            prev_structure = structure
            prev_snap = snap
    
    if len(evolution_events) > 0:
        events = evolution_events
        print(f"演化事件分析 (共 {len(events)} 个事件):")
        print()
        
        # 按类型统计
        event_types = {}
        for event in events:
            event_type = event.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print("事件类型统计:")
        for event_type, count in event_types.items():
            print(f"  {event_type}: {count}")
        print()
        
        # 显示所有事件
        print("所有演化事件:")
        for i, event in enumerate(events):
            print(f"  事件 {i+1}:")
            print(f"    步数: {event.get('step', 'N/A')}")
            print(f"    类型: {event.get('event_type', 'N/A')}")
            print(f"    触发条件: {event.get('trigger_condition', 'N/A')}")
            print(f"    详情: {event.get('details', {})}")
            print(f"    自由能变化: {event.get('free_energy_before', 0):.4f} -> {event.get('free_energy_after', 0):.4f}")
            print()
        
        # 绘制演化事件时间线
        if len(events) > 0:
            steps = [e.get('step', 0) for e in events]
            event_types_list = [e.get('event_type', 'unknown') for e in events]
            F_before = [e.get('free_energy_before', 0) for e in events]
            F_after = [e.get('free_energy_after', 0) for e in events]
            
            # 创建事件类型到颜色的映射
            unique_types = list(set(event_types_list))
            colors_map = plt.cm.Set3(range(len(unique_types)))
            type_to_color = {t: colors_map[i] for i, t in enumerate(unique_types)}
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # 上图：事件时间线
            for i, (step, event_type) in enumerate(zip(steps, event_types_list)):
                axes[0].scatter(step, i, c=[type_to_color[event_type]], s=100, alpha=0.7, label=event_type if i == 0 or event_types_list[i-1] != event_type else "")
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('Event Index')
            axes[0].set_title('Evolution Events Timeline')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 下图：自由能变化
            axes[1].plot(steps, F_before, 'o-', label='F Before', alpha=0.7, linewidth=2)
            axes[1].plot(steps, F_after, 's-', label='F After', alpha=0.7, linewidth=2)
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Free Energy')
            axes[1].set_title('Free Energy Before/After Evolution Events')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'evolution_events.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 保存演化事件图: {output_dir / 'evolution_events.png'}")
    
    # 演化摘要
    if 'evolution_summary' in data and data['evolution_summary']:
        summary = data['evolution_summary']
        print("演化摘要:")
        print(f"  总步数: {summary.get('total_steps', 'N/A')}")
        print(f"  总事件数: {summary.get('total_events', 0)}")
        if 'stats' in summary:
            stats = summary['stats']
            print(f"  Objects 创建: {stats.get('objects_created', 0)}")
            print(f"  Aspects 创建: {stats.get('aspects_created', 0)}")
            print(f"  Pipelines 创建: {stats.get('pipelines_created', 0)}")
            print(f"  剪枝次数: {stats.get('pruned_count', 0)}")
        print()
    
    print()
    print("=" * 80)
    print("分析完成！所有图表已保存到:", output_dir)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="分析 MNIST 主动推理实验 V2 结果")
    parser.add_argument("--input", type=str, default="data/mnist_active_inference_v2_1000steps.json", help="结果文件路径")
    parser.add_argument("--output-dir", type=str, default="data/plots", help="输出目录")
    
    args = parser.parse_args()
    
    analyze_results(args.input, args.output_dir)


if __name__ == "__main__":
    main()

