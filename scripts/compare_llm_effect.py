#!/usr/bin/env python3
"""
对比有无 LLM 对自由能降低和学习的影响
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_experiment_data(json_path: Path) -> Dict:
    """加载实验数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_free_energy_trajectory(data: Dict) -> Tuple[List[int], List[float]]:
    """提取自由能轨迹"""
    steps = []
    free_energies = []
    
    if 'snapshots' in data:
        for snapshot in data['snapshots']:
            steps.append(snapshot.get('step', len(steps)))
            free_energies.append(snapshot.get('free_energy', 0.0))
    elif 'free_energy_history' in data:
        for step, fe in enumerate(data['free_energy_history']):
            steps.append(step)
            free_energies.append(fe)
    else:
        # 尝试从其他字段提取
        if 'final_free_energy' in data:
            free_energies.append(data['final_free_energy'])
            steps.append(data.get('num_steps', 0) - 1)
    
    return steps, free_energies


def extract_network_structure(data: Dict) -> Dict[str, List[int]]:
    """提取网络结构变化"""
    structures = {
        'num_objects': [],
        'num_aspects': [],
        'num_pipelines': [],
    }
    
    if 'snapshots' in data:
        for snapshot in data['snapshots']:
            structure = snapshot.get('structure', {})
            structures['num_objects'].append(structure.get('num_objects', 0))
            structures['num_aspects'].append(structure.get('num_aspects', 0))
            structures['num_pipelines'].append(structure.get('num_pipelines', 0))
    
    return structures


def analyze_learning_effect(steps: List[int], free_energies: List[float]) -> Dict:
    """分析学习效果"""
    if len(free_energies) < 2:
        return {}
    
    fe_array = np.array(free_energies)
    
    # 初始和最终自由能
    initial_fe = free_energies[0] if free_energies else 0.0
    final_fe = free_energies[-1] if free_energies else 0.0
    
    # 自由能降低
    fe_reduction = initial_fe - final_fe
    fe_reduction_percent = (fe_reduction / initial_fe * 100) if initial_fe > 0 else 0
    
    # 平均自由能
    mean_fe = np.mean(fe_array)
    
    # 自由能下降率（每步平均下降）
    if len(free_energies) > 1:
        fe_slope = (final_fe - initial_fe) / (len(free_energies) - 1)
    else:
        fe_slope = 0
    
    # 最低自由能
    min_fe = np.min(fe_array)
    min_fe_step = steps[np.argmin(fe_array)]
    
    # 自由能稳定性（标准差）
    fe_std = np.std(fe_array)
    
    # 自由能变化趋势（线性拟合斜率）
    if len(steps) > 1:
        coeffs = np.polyfit(steps, fe_array, 1)
        trend_slope = coeffs[0]
    else:
        trend_slope = 0
    
    return {
        'initial_fe': initial_fe,
        'final_fe': final_fe,
        'fe_reduction': fe_reduction,
        'fe_reduction_percent': fe_reduction_percent,
        'mean_fe': mean_fe,
        'min_fe': min_fe,
        'min_fe_step': min_fe_step,
        'fe_std': fe_std,
        'fe_slope': fe_slope,
        'trend_slope': trend_slope,
    }


def plot_comparison(
    data_with_llm: Dict,
    data_without_llm: Dict,
    output_path: Path,
):
    """绘制对比图"""
    # 提取数据
    steps_llm, fe_llm = extract_free_energy_trajectory(data_with_llm)
    steps_no_llm, fe_no_llm = extract_free_energy_trajectory(data_without_llm)
    
    struct_llm = extract_network_structure(data_with_llm)
    struct_no_llm = extract_network_structure(data_without_llm)
    
    # 分析学习效果
    analysis_llm = analyze_learning_effect(steps_llm, fe_llm)
    analysis_no_llm = analyze_learning_effect(steps_no_llm, fe_no_llm)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 自由能轨迹对比
    ax1 = axes[0, 0]
    if steps_llm and fe_llm:
        ax1.plot(steps_llm, fe_llm, 'b-o', label='有 LLM', linewidth=2, markersize=4)
    if steps_no_llm and fe_no_llm:
        ax1.plot(steps_no_llm, fe_no_llm, 'r-s', label='无 LLM', linewidth=2, markersize=4)
    ax1.set_xlabel('步数', fontsize=12)
    ax1.set_ylabel('自由能', fontsize=12)
    ax1.set_title('自由能变化轨迹对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. 网络结构对比 - Aspects
    ax2 = axes[0, 1]
    if struct_llm['num_aspects']:
        ax2.plot(range(len(struct_llm['num_aspects'])), struct_llm['num_aspects'], 
                'b-o', label='有 LLM', linewidth=2, markersize=4)
    if struct_no_llm['num_aspects']:
        ax2.plot(range(len(struct_no_llm['num_aspects'])), struct_no_llm['num_aspects'], 
                'r-s', label='无 LLM', linewidth=2, markersize=4)
    ax2.set_xlabel('步数', fontsize=12)
    ax2.set_ylabel('Aspects 数量', fontsize=12)
    ax2.set_title('网络结构演化 - Aspects', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. 学习效果对比（柱状图）
    ax3 = axes[1, 0]
    metrics = ['初始自由能', '最终自由能', '自由能降低', '平均自由能']
    if analysis_llm and analysis_no_llm:
        values_llm = [
            analysis_llm.get('initial_fe', 0),
            analysis_llm.get('final_fe', 0),
            analysis_llm.get('fe_reduction', 0),
            analysis_llm.get('mean_fe', 0),
        ]
        values_no_llm = [
            analysis_no_llm.get('initial_fe', 0),
            analysis_no_llm.get('final_fe', 0),
            analysis_no_llm.get('fe_reduction', 0),
            analysis_no_llm.get('mean_fe', 0),
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax3.bar(x - width/2, values_llm, width, label='有 LLM', color='blue', alpha=0.7)
        ax3.bar(x + width/2, values_no_llm, width, label='无 LLM', color='red', alpha=0.7)
        ax3.set_ylabel('自由能值', fontsize=12)
        ax3.set_title('学习效果对比', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics, rotation=15, ha='right')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 学习效率对比（文本摘要）
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "学习效果分析\n" + "=" * 50 + "\n\n"
    
    if analysis_llm:
        summary_text += "【有 LLM 版本】\n"
        summary_text += f"初始自由能: {analysis_llm.get('initial_fe', 0):.4f}\n"
        summary_text += f"最终自由能: {analysis_llm.get('final_fe', 0):.4f}\n"
        summary_text += f"自由能降低: {analysis_llm.get('fe_reduction', 0):.4f} "
        summary_text += f"({analysis_llm.get('fe_reduction_percent', 0):.2f}%)\n"
        summary_text += f"平均自由能: {analysis_llm.get('mean_fe', 0):.4f}\n"
        summary_text += f"最低自由能: {analysis_llm.get('min_fe', 0):.4f} "
        summary_text += f"(第 {analysis_llm.get('min_fe_step', 0)} 步)\n"
        summary_text += f"自由能稳定性 (std): {analysis_llm.get('fe_std', 0):.4f}\n"
        summary_text += f"趋势斜率: {analysis_llm.get('trend_slope', 0):.6f}\n"
        summary_text += "\n"
    
    if analysis_no_llm:
        summary_text += "【无 LLM 版本】\n"
        summary_text += f"初始自由能: {analysis_no_llm.get('initial_fe', 0):.4f}\n"
        summary_text += f"最终自由能: {analysis_no_llm.get('final_fe', 0):.4f}\n"
        summary_text += f"自由能降低: {analysis_no_llm.get('fe_reduction', 0):.4f} "
        summary_text += f"({analysis_no_llm.get('fe_reduction_percent', 0):.2f}%)\n"
        summary_text += f"平均自由能: {analysis_no_llm.get('mean_fe', 0):.4f}\n"
        summary_text += f"最低自由能: {analysis_no_llm.get('min_fe', 0):.4f} "
        summary_text += f"(第 {analysis_no_llm.get('min_fe_step', 0)} 步)\n"
        summary_text += f"自由能稳定性 (std): {analysis_no_llm.get('fe_std', 0):.4f}\n"
        summary_text += f"趋势斜率: {analysis_no_llm.get('trend_slope', 0):.6f}\n"
        summary_text += "\n"
    
    if analysis_llm and analysis_no_llm:
        summary_text += "【对比结论】\n"
        fe_reduction_diff = analysis_llm.get('fe_reduction', 0) - analysis_no_llm.get('fe_reduction', 0)
        if fe_reduction_diff > 0:
            summary_text += f"✓ 有 LLM 版本自由能降低更多: +{fe_reduction_diff:.4f}\n"
        elif fe_reduction_diff < 0:
            summary_text += f"✗ 无 LLM 版本自由能降低更多: {abs(fe_reduction_diff):.4f}\n"
        else:
            summary_text += "≈ 两者自由能降低相同\n"
        
        mean_fe_diff = analysis_llm.get('mean_fe', 0) - analysis_no_llm.get('mean_fe', 0)
        if mean_fe_diff < 0:
            summary_text += f"✓ 有 LLM 版本平均自由能更低: {mean_fe_diff:.4f}\n"
        elif mean_fe_diff > 0:
            summary_text += f"✗ 无 LLM 版本平均自由能更低: {abs(mean_fe_diff):.4f}\n"
        else:
            summary_text += "≈ 两者平均自由能相同\n"
        
        trend_diff = analysis_llm.get('trend_slope', 0) - analysis_no_llm.get('trend_slope', 0)
        if trend_diff < 0:
            summary_text += f"✓ 有 LLM 版本下降趋势更明显: {trend_diff:.6f}\n"
        elif trend_diff > 0:
            summary_text += f"✗ 无 LLM 版本下降趋势更明显: {abs(trend_diff):.6f}\n"
        else:
            summary_text += "≈ 两者下降趋势相同\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="对比有无 LLM 对学习的影响")
    parser.add_argument("--with-llm", type=str, required=True, help="有 LLM 的实验结果 JSON 文件")
    parser.add_argument("--without-llm", type=str, required=True, help="无 LLM 的实验结果 JSON 文件")
    parser.add_argument("--output", type=str, default="data/llm_comparison.png", help="输出图片路径")
    
    args = parser.parse_args()
    
    # 加载数据
    data_with_llm = load_experiment_data(Path(args.with_llm))
    data_without_llm = load_experiment_data(Path(args.without_llm))
    
    # 绘制对比图
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(data_with_llm, data_without_llm, output_path)
    
    # 打印详细分析
    steps_llm, fe_llm = extract_free_energy_trajectory(data_with_llm)
    steps_no_llm, fe_no_llm = extract_free_energy_trajectory(data_without_llm)
    
    analysis_llm = analyze_learning_effect(steps_llm, fe_llm)
    analysis_no_llm = analyze_learning_effect(steps_no_llm, fe_no_llm)
    
    print("\n" + "=" * 80)
    print("详细分析结果")
    print("=" * 80)
    print(f"\n有 LLM 版本: {args.with_llm}")
    print(f"无 LLM 版本: {args.without_llm}")
    print()
    
    if analysis_llm:
        print("【有 LLM 版本】")
        for key, value in analysis_llm.items():
            print(f"  {key}: {value:.6f}")
        print()
    
    if analysis_no_llm:
        print("【无 LLM 版本】")
        for key, value in analysis_no_llm.items():
            print(f"  {key}: {value:.6f}")
        print()


if __name__ == "__main__":
    main()

