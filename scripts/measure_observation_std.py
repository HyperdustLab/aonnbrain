#!/usr/bin/env python3
"""
测量世界模型观察值的标准差

用于验证自由能公式中 σ_obs ≈ 0.4 的假设是否合理
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aonn.models.general_ai_world_model import GeneralAIWorldModel
from aonn.models.lineworm_world_model import LineWormWorldModel
import argparse


def measure_observation_std(world_model, num_steps=100, verbose=True):
    """
    测量世界模型观察值的标准差
    
    Args:
        world_model: 世界模型实例
        num_steps: 运行的步数
        verbose: 是否输出详细信息
    
    Returns:
        dict: 包含每个感官的标准差统计
    """
    # 重置并获取初始观察
    obs_initial = world_model.reset()
    
    if verbose:
        print('【初始观察值统计】')
        print('-' * 80)
        for sense, value in obs_initial.items():
            std_val = value.std().item()
            mean_val = value.mean().item()
            dim = value.shape[0]
            print(f'  {sense:12s} ({dim:3d} 维): 均值={mean_val:7.4f}, 标准差={std_val:.4f}')
        print()
    
    # 收集多步观察值
    all_observations = {sense: [] for sense in obs_initial.keys()}
    
    for step in range(num_steps):
        # 生成随机动作
        if hasattr(world_model, 'action_dim'):
            action = torch.randn(world_model.action_dim, device=world_model.device) * 0.1
        else:
            action = torch.randn(32, device=world_model.device) * 0.1
        
        # 执行一步
        if isinstance(world_model, GeneralAIWorldModel):
            obs, reward, done = world_model.step(action)
        else:
            # LineWormWorldModel
            obs, reward, done = world_model.step(action)
        
        # 收集观察值
        for sense, value in obs.items():
            all_observations[sense].append(value.detach().clone())
    
    # 计算统计信息
    stats = {}
    for sense, obs_list in all_observations.items():
        # 将所有观察值堆叠成矩阵 [num_steps, obs_dim]
        obs_matrix = torch.stack(obs_list, dim=0)
        
        # 方法 1: 每步标准差的平均值（推荐）
        std_per_step = [obs.std().item() for obs in obs_list]
        mean_std_per_step = sum(std_per_step) / len(std_per_step)
        
        # 方法 2: 总体标准差（所有维度、所有时间步）
        overall_std = obs_matrix.std().item()
        
        # 方法 3: 每个维度的标准差（跨时间步），然后取平均
        std_per_dim = obs_matrix.std(dim=0)  # [obs_dim]
        mean_std_per_dim = std_per_dim.mean().item()
        std_of_std = std_per_dim.std().item()
        
        stats[sense] = {
            'mean_std_per_step': mean_std_per_step,  # 推荐使用这个
            'overall_std': overall_std,
            'mean_std_per_dim': mean_std_per_dim,
            'std_of_std': std_of_std,
            'obs_dim': obs_list[0].shape[0],
        }
    
    if verbose:
        print(f'【运行 {num_steps} 步后的统计】')
        print('-' * 80)
        for sense, s in stats.items():
            obs_dim = s['obs_dim']
            mean_std = s['mean_std_per_step']
            overall_std_val = s['overall_std']
            mean_std_dim = s['mean_std_per_dim']
            std_of_std = s['std_of_std']
            print(f'  {sense:12s} ({obs_dim:3d} 维):')
            print(f'    每步标准差的平均值（推荐）: {mean_std:.4f}')
            print(f'    总体标准差: {overall_std_val:.4f}')
            print(f'    每维标准差的平均值: {mean_std_dim:.4f}')
            print(f'    每维标准差的标准差: {std_of_std:.4f}')
            print()
    
    # 计算总体平均标准差
    overall_avg_std = sum(s['mean_std_per_step'] for s in stats.values()) / len(stats)
    
    if verbose:
        print('【总体平均标准差】')
        print('-' * 80)
        print(f'  每步标准差的平均值: {overall_avg_std:.4f}')
        print()
        print('【结论】')
        print('-' * 80)
        if abs(overall_avg_std - 0.4) < 0.1:
            print(f'  ✓ 假设 σ_obs ≈ 0.4 非常合理')
            print(f'    实际值 {overall_avg_std:.4f} 与假设值 0.4 非常接近（差异 < 0.1）')
        else:
            print(f'  ✗ 假设 σ_obs ≈ 0.4 不太合理')
            print(f'    实际值 {overall_avg_std:.4f} 与假设值 0.4 差异较大（≥ 0.1）')
        print()
    
    return stats, overall_avg_std


def main():
    parser = argparse.ArgumentParser(description="测量世界模型观察值的标准差")
    parser.add_argument("--model", type=str, choices=["general_ai", "lineworm"], default="general_ai",
                       help="世界模型类型")
    parser.add_argument("--steps", type=int, default=100, help="运行的步数")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print('=' * 80)
    print(f'测量 {args.model} 世界模型的观察值标准差')
    print('=' * 80)
    print()
    
    if args.model == "general_ai":
        world_model = GeneralAIWorldModel(
            semantic_dim=1024,
            memory_dim=512,
            context_dim=256,
            physical_dim=64,
            goal_dim=256,
            vision_dim=512,
            language_dim=512,
            audio_dim=128,
            multimodal_dim=256,
            action_dim=256,
            device=device,
            state_noise_std=0.01,
            observation_noise_std=0.01,
            enable_tools=True,
        )
    else:
        world_model = LineWormWorldModel(
            state_dim=256,
            action_dim=32,
            chemo_dim=128,
            thermo_dim=32,
            touch_dim=64,
            device=device,
        )
    
    stats, overall_avg_std = measure_observation_std(world_model, num_steps=args.steps, verbose=True)
    
    print('【自由能贡献的重新计算】')
    print('-' * 80)
    print(f'如果 σ_obs = {overall_avg_std:.4f}，那么：')
    sigma_sq = overall_avg_std ** 2
    print(f'  error^2 的期望 = {overall_avg_std:.4f}^2 = {sigma_sq:.4f}')
    
    # 计算平均观察维度
    avg_obs_dim = sum(s['obs_dim'] for s in stats.values()) / len(stats)
    avg_obs_dim_int = int(avg_obs_dim)
    print(f'  对于平均 obs_dim = {avg_obs_dim_int}:')
    sum_error_sq = avg_obs_dim * sigma_sq
    F_expected = 0.5 * sum_error_sq
    print(f'    sum(error^2) 的期望 = {avg_obs_dim_int} × {sigma_sq:.4f} = {sum_error_sq:.2f}')
    print(f'    F 的期望 = 0.5 × {sum_error_sq:.2f} = {F_expected:.2f}')
    print()
    print(f'之前假设 σ_obs = 0.4 时，F ≈ 28')
    print(f'实际 σ_obs = {overall_avg_std:.4f} 时，F ≈ {F_expected:.2f}')
    print()


if __name__ == "__main__":
    main()
