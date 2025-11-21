#!/usr/bin/env python3
"""
检查 AONN Brain 中所有 Object 的状态值

用于诊断 Object 状态值是否过大，是否导致自由能过高
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aonn.models.general_ai_world_model import GeneralAIWorldModel, GeneralAIWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.aspects.mock_llm_client import MockLLMClient
import argparse


def check_object_states(config: dict, device: torch.device, num_steps: int = 10):
    """
    检查 Object 状态值
    
    Args:
        config: 配置字典
        device: 设备
        num_steps: 检查的步数
    """
    print("=" * 80)
    print("Object 状态值检查")
    print("=" * 80)
    print()
    
    # 创建世界模型
    world_model = GeneralAIWorldModel(
        semantic_dim=config["world_model"]["semantic_dim"],
        memory_dim=config["world_model"]["memory_dim"],
        context_dim=config["world_model"]["context_dim"],
        physical_dim=config["world_model"]["physical_dim"],
        goal_dim=config["world_model"]["goal_dim"],
        vision_dim=config["world_model"]["vision_dim"],
        language_dim=config["world_model"]["language_dim"],
        audio_dim=config["world_model"]["audio_dim"],
        multimodal_dim=config["world_model"]["multimodal_dim"],
        action_dim=config["act_dim"],
        device=device,
        state_noise_std=config["world_model"].get("state_noise_std", 0.01),
        observation_noise_std=config["world_model"].get("observation_noise_std", 0.01),
        enable_tools=config["world_model"].get("enable_tools", True),
    )
    world_interface = GeneralAIWorldInterface(world_model)
    
    # 创建 LLM 客户端
    llm_client = MockLLMClient(
        input_dim=config["sem_dim"],
        output_dim=config["sem_dim"],
        hidden_dims=config.get("llm", {}).get("hidden_dims", [256, 512, 256]),
        device=device,
    )
    
    # 创建 AONN Brain
    brain = AONNBrainV3(
        config=config,
        llm_client=llm_client,
        device=device,
        enable_evolution=True,
    )
    
    # 获取 state_clip_value
    state_clip_value = config.get("state_clip_value", 5.0)
    
    print(f"状态裁剪值 (state_clip_value): {state_clip_value}")
    print(f"检查步数: {num_steps}")
    print()
    
    obs = world_interface.reset()
    prev_obs = None
    prev_action = None
    action = torch.randn(config["act_dim"], device=device) * 0.1
    
    # 统计信息
    all_stats = []
    
    for step in range(num_steps):
        if step > 0:
            obs, reward = world_interface.step(action)
        
        # 设置观察到 brain
        for sense, value in obs.items():
            if sense in brain.objects:
                brain.objects[sense].set_state(value)
        
        # 网络演化
        full_state = world_model.get_true_state()
        if full_state.shape[-1] >= config["state_dim"]:
            target_state = full_state[:config["state_dim"]]
        else:
            padding = torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)
            target_state = torch.cat([full_state, padding], dim=-1)
        
        brain.evolve_network(obs, target=target_state)
        
        # 主动推理
        if len(brain.aspects) > 0:
            loop = ActiveInferenceLoop(
                brain.objects,
                brain.aspects,
                infer_lr=config.get("infer_lr", 0.005),
                max_grad_norm=config.get("max_grad_norm", 100.0),
                device=device,
            )
            num_iters = config.get("num_infer_iters", 3)
            loop.infer_states(target_objects=("internal",), num_iters=num_iters, sanitize_callback=brain.sanitize_states)
            brain.sanitize_states()
        
        # 生成动作
        if "action" in brain.objects and len(brain.aspect_pipelines) > 0:
            action = brain.objects["internal"].state
            for pipeline in brain.aspect_pipelines:
                action = pipeline(action)
            brain.objects["action"].set_state(action)
        else:
            action = torch.randn(config["act_dim"], device=device) * 0.1
            if "action" in brain.objects:
                brain.objects["action"].set_state(action)
        
        # 学习世界模型
        if prev_obs is not None and prev_action is not None:
            full_state = world_model.get_true_state()
            if full_state.shape[-1] >= config["state_dim"]:
                target_state = full_state[:config["state_dim"]]
            else:
                padding = torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)
                target_state = torch.cat([full_state, padding], dim=-1)
            brain.learn_world_model(
                observation=prev_obs,
                action=prev_action,
                next_observation=obs,
                target_state=target_state,
                learning_rate=config.get("learning_rate", 0.005),
            )
            brain.sanitize_states()
        
        prev_obs = {sense: value.clone() for sense, value in obs.items()}
        prev_action = action.clone()
        
        # 检查所有 Object 的状态值
        step_stats = {
            "step": step,
            "objects": {},
            "warnings": [],
        }
        
        for name, obj in brain.objects.items():
            state = obj.state
            if state is None:
                continue
            
            # 计算统计信息
            state_norm = state.norm().item()
            state_max = state.max().item()
            state_min = state.min().item()
            state_mean = state.mean().item()
            state_std = state.std().item()
            
            # 检查是否超过裁剪值
            exceeds_clip = abs(state_max) > state_clip_value or abs(state_min) > state_clip_value
            has_nan = not torch.isfinite(state).all()
            has_inf = torch.isinf(state).any()
            
            step_stats["objects"][name] = {
                "norm": state_norm,
                "max": state_max,
                "min": state_min,
                "mean": state_mean,
                "std": state_std,
                "dim": obj.dim,
                "exceeds_clip": exceeds_clip,
                "has_nan": has_nan,
                "has_inf": has_inf,
            }
            
            # 收集警告
            if exceeds_clip:
                step_stats["warnings"].append(
                    f"{name}: 状态值超过裁剪值 (max={state_max:.2f}, min={state_min:.2f}, clip={state_clip_value})"
                )
            if has_nan:
                step_stats["warnings"].append(f"{name}: 包含 NaN 值")
            if has_inf:
                step_stats["warnings"].append(f"{name}: 包含 Inf 值")
        
        all_stats.append(step_stats)
        
        # 输出当前步的统计信息
        if step == 0 or step == num_steps - 1 or len(step_stats["warnings"]) > 0:
            print(f"Step {step}:")
            print(f"  Object 数量: {len(brain.objects)}")
            print(f"  Aspect 数量: {len(brain.aspects)}")
            print(f"  自由能: {brain.compute_free_energy().item():.4f}")
            
            if len(step_stats["warnings"]) > 0:
                print(f"  ⚠️  警告 ({len(step_stats['warnings'])} 个):")
                for warning in step_stats["warnings"]:
                    print(f"    - {warning}")
            
            # 显示状态值最大的几个 Object
            sorted_objects = sorted(
                step_stats["objects"].items(),
                key=lambda x: x[1]["norm"],
                reverse=True
            )[:5]
            print(f"  状态值最大的 5 个 Object:")
            for name, stats in sorted_objects:
                print(f"    {name:20s} (dim={stats['dim']:3d}): "
                      f"norm={stats['norm']:7.2f}, "
                      f"max={stats['max']:7.2f}, "
                      f"min={stats['min']:7.2f}, "
                      f"mean={stats['mean']:7.2f}, "
                      f"std={stats['std']:7.2f}")
            print()
    
    # 汇总统计
    print("=" * 80)
    print("汇总统计")
    print("=" * 80)
    print()
    
    # 统计每个 Object 的平均状态值
    object_avg_stats = {}
    for step_stats in all_stats:
        for name, stats in step_stats["objects"].items():
            if name not in object_avg_stats:
                object_avg_stats[name] = {
                    "norms": [],
                    "maxs": [],
                    "mins": [],
                    "means": [],
                    "stds": [],
                    "exceeds_clip_count": 0,
                    "has_nan_count": 0,
                    "has_inf_count": 0,
                }
            object_avg_stats[name]["norms"].append(stats["norm"])
            object_avg_stats[name]["maxs"].append(stats["max"])
            object_avg_stats[name]["mins"].append(stats["min"])
            object_avg_stats[name]["means"].append(stats["mean"])
            object_avg_stats[name]["stds"].append(stats["std"])
            if stats["exceeds_clip"]:
                object_avg_stats[name]["exceeds_clip_count"] += 1
            if stats["has_nan"]:
                object_avg_stats[name]["has_nan_count"] += 1
            if stats["has_inf"]:
                object_avg_stats[name]["has_inf_count"] += 1
    
    print(f"Object 状态值统计 (共 {len(all_stats)} 步):")
    print()
    print(f"{'Object 名称':<20} {'维度':<6} {'平均范数':<10} {'最大范数':<10} {'最小范数':<10} {'超过裁剪':<10} {'NaN':<6} {'Inf':<6}")
    print("-" * 80)
    
    for name, stats in sorted(object_avg_stats.items(), key=lambda x: sum(x[1]["norms"]) / len(x[1]["norms"]), reverse=True):
        avg_norm = sum(stats["norms"]) / len(stats["norms"])
        max_norm = max(stats["norms"])
        min_norm = min(stats["norms"])
        # 从任意一步获取维度（因为维度不会改变）
        dim = None
        for step_stats in all_stats:
            if name in step_stats["objects"]:
                dim = step_stats["objects"][name]["dim"]
                break
        if dim is None:
            continue
        exceeds_clip_pct = stats["exceeds_clip_count"] / len(all_stats) * 100
        has_nan_pct = stats["has_nan_count"] / len(all_stats) * 100
        has_inf_pct = stats["has_inf_count"] / len(all_stats) * 100
        
        print(f"{name:<20} {dim:<6} {avg_norm:<10.2f} {max_norm:<10.2f} {min_norm:<10.2f} "
              f"{exceeds_clip_pct:>6.1f}%  {has_nan_pct:>4.1f}%  {has_inf_pct:>4.1f}%")
    
    print()
    print("=" * 80)
    print("诊断结论")
    print("=" * 80)
    print()
    
    # 检查是否有问题
    total_warnings = sum(len(s["warnings"]) for s in all_stats)
    if total_warnings == 0:
        print("✓ 所有 Object 状态值正常")
        print(f"  - 所有状态值都在裁剪范围内 ([-{state_clip_value}, {state_clip_value}])")
        print(f"  - 没有 NaN 或 Inf 值")
    else:
        print(f"⚠️  发现 {total_warnings} 个警告")
        print()
        print("建议:")
        print("1. 如果状态值超过裁剪值，检查 sanitize_states() 是否被正确调用")
        print("2. 如果有 NaN/Inf，检查梯度是否爆炸，考虑降低学习率或增加梯度裁剪")
        print("3. 如果状态值持续过大，考虑降低 infer_lr 或增加 state_clip_value")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="检查 AONN Brain 中所有 Object 的状态值")
    parser.add_argument("--steps", type=int, default=10, help="检查的步数")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    config = {
        "state_dim": 1024,
        "act_dim": 256,
        "sem_dim": 512,
        "sense_dims": {"vision": 512, "language": 512, "audio": 128, "multimodal": 256},
        "world_model": {
            "semantic_dim": 1024, "memory_dim": 512, "context_dim": 256, "physical_dim": 64, "goal_dim": 256,
            "vision_dim": 512, "language_dim": 512, "audio_dim": 128, "multimodal_dim": 256,
            "state_noise_std": 0.01, "observation_noise_std": 0.01, "enable_tools": True,
        },
        "evolution": {
            "free_energy_threshold": 0.1,
            "max_aspects": 10000,
            "batch_growth": {
                "base": 2, "max_per_step": 8, "max_total": 2500, "min_per_sense": 4,
                "error_threshold": 0.15, "error_multiplier": 0.5,
            }
        },
        "pipeline_growth": {
            "enable": True, "initial_depth": 3, "initial_width": 1,
            "depth_increment": 1, "width_increment": 0, "max_stages": 1000,
            "min_interval": 1, "free_energy_trigger": 0.1, "max_depth": 50,
        },
        "state_clip_value": 5.0,
        "infer_lr": 0.005,
        "learning_rate": 0.005,
        "num_infer_iters": 3,
        "max_grad_norm": 100.0,
        "llm": {"model": "mock", "embedding_model": "mock", "summary_size": 8, "max_tokens": 120, "temperature": 0.7, "timeout": 120.0},
    }
    
    check_object_states(config, device, num_steps=args.steps)


if __name__ == "__main__":
    main()

