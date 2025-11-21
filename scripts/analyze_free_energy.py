#!/usr/bin/env python3
"""
深入分析自由能居高不下的原因
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from aonn.models.general_ai_world_model import GeneralAIWorldModel, GeneralAIWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.aspects.mock_llm_client import MockLLMClient
from aonn.core.active_inference_loop import ActiveInferenceLoop
from typing import Dict

def analyze_free_energy(config: Dict, device: torch.device, num_steps: int = 5):
    """深入分析自由能组成"""
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
    )
    world_interface = GeneralAIWorldInterface(world_model)
    
    # 创建 LLM 客户端
    llm_client = MockLLMClient(
        input_dim=config.get("sem_dim", 128),
        output_dim=config.get("sem_dim", 128),
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
    
    # 运行几步
    obs = world_interface.reset()
    action = torch.randn(config["act_dim"], device=device) * 0.1
    prev_obs = None
    prev_action = None
    
    print("=" * 80)
    print("自由能深入分析")
    print("=" * 80)
    print()
    
    for step in range(num_steps):
        # 设置观察
        for sense, value in obs.items():
            if sense in brain.objects:
                brain.objects[sense].set_state(value)
        
        # 演化网络
        full_state = world_model.get_true_state()
        target_state = full_state[:config["state_dim"]] if full_state.shape[-1] >= config["state_dim"] else torch.cat([full_state, torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)], dim=-1)
        brain.evolve_network(obs, target=target_state)
        
        # 计算自由能组成
        total_F = brain.compute_free_energy().item()
        
        print(f"Step {step}:")
        print(f"  总自由能: {total_F:.4f}")
        print()
        
        # 详细分析各个 Aspect 的贡献
        aspect_contributions = {}
        aspect_details = []
        
        for aspect in brain.aspects:
            contrib = aspect.free_energy_contrib(brain.objects).item()
            aspect_type = type(aspect).__name__
            
            if aspect_type not in aspect_contributions:
                aspect_contributions[aspect_type] = []
            aspect_contributions[aspect_type].append(contrib)
            
            # 记录详细信息
            aspect_details.append({
                "type": aspect_type,
                "name": getattr(aspect, "name", "unknown"),
                "contrib": contrib,
            })
        
        # 按类型汇总
        print("  各类型 Aspect 贡献:")
        total_by_type = {}
        count_by_type = {}
        for aspect_type, contribs in aspect_contributions.items():
            total = sum(contribs)
            count = len(contribs)
            total_by_type[aspect_type] = total
            count_by_type[aspect_type] = count
            avg = total / count if count > 0 else 0
            max_contrib = max(contribs) if contribs else 0
            min_contrib = min(contribs) if contribs else 0
            print(f"    {aspect_type}:")
            print(f"      总数={total:.4f}, 数量={count}, 平均={avg:.4f}, 最大={max_contrib:.4f}, 最小={min_contrib:.4f}")
        
        print()
        
        # 分析 LinearGenerativeAspect 的详细情况
        linear_aspects = [a for a in aspect_details if a["type"] == "LinearGenerativeAspect"]
        if linear_aspects:
            contribs = [a["contrib"] for a in linear_aspects]
            print(f"  LinearGenerativeAspect 详细分析 (共 {len(linear_aspects)} 个):")
            print(f"    平均贡献: {np.mean(contribs):.4f}")
            print(f"    中位数: {np.median(contribs):.4f}")
            print(f"    最大值: {np.max(contribs):.4f}")
            print(f"    最小值: {np.min(contribs):.4f}")
            print(f"    标准差: {np.std(contribs):.4f}")
            
            # 分析贡献分布
            large_contribs = [c for c in contribs if c > 100]
            medium_contribs = [c for c in contribs if 10 < c <= 100]
            small_contribs = [c for c in contribs if c <= 10]
            print(f"    贡献分布:")
            print(f"      大 (>100): {len(large_contribs)} 个")
            print(f"      中 (10-100): {len(medium_contribs)} 个")
            print(f"      小 (<=10): {len(small_contribs)} 个")
            
            # 检查状态值范围
            print()
            print("  状态值分析:")
            for sense_name in brain.senses:
                if sense_name in brain.objects:
                    state = brain.objects[sense_name].state
                    state_norm = torch.norm(state).item()
                    state_max = state.max().item()
                    state_min = state.min().item()
                    state_mean = state.mean().item()
                    print(f"    {sense_name}:")
                    print(f"      范数={state_norm:.4f}, 最大值={state_max:.4f}, 最小值={state_min:.4f}, 均值={state_mean:.4f}")
            
            # 检查 internal 状态
            if "internal" in brain.objects:
                internal_state = brain.objects["internal"].state
                internal_norm = torch.norm(internal_state).item()
                internal_max = internal_state.max().item()
                internal_min = internal_state.min().item()
                internal_mean = internal_state.mean().item()
                print(f"    internal:")
                print(f"      范数={internal_norm:.4f}, 最大值={internal_max:.4f}, 最小值={internal_min:.4f}, 均值={internal_mean:.4f}")
        
        print()
        print(f"  网络规模: Objects={len(brain.objects)}, Aspects={len(brain.aspects)}, Pipelines={len(brain.aspect_pipelines)}")
        print()
        
        # 执行一步
        prev_obs = {sense: value.clone() for sense, value in obs.items()}
        prev_action = action.clone()
        obs, reward = world_interface.step(action)
        action = torch.randn(config["act_dim"], device=device) * 0.1
        
        # 学习世界模型
        if prev_obs is not None and prev_action is not None:
            full_state = world_model.get_true_state()
            target_state = full_state[:config["state_dim"]] if full_state.shape[-1] >= config["state_dim"] else torch.cat([full_state, torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)], dim=-1)
            brain.learn_world_model(
                observation=prev_obs,
                action=prev_action,
                next_observation=obs,
                target_state=target_state,
                learning_rate=config.get("learning_rate", 0.001),
            )
    
    print("=" * 80)
    print("分析结论")
    print("=" * 80)
    print()
    print("可能的原因：")
    print("1. 如果 LinearGenerativeAspect 贡献很大且分布不均：")
    print("   - 说明某些感官预测误差特别大")
    print("   - 可能是状态值本身很大，导致预测误差很大")
    print("   - 或者新创建的 Aspects 还没有学习")
    print()
    print("2. 如果状态值很大（范数 > 10）：")
    print("   - 说明状态值可能被裁剪或爆炸")
    print("   - 需要检查 state_clip_value 和学习率")
    print()
    print("3. 如果大部分 Aspects 贡献都很小，但总自由能很大：")
    print("   - 说明 Aspects 数量太多，累积误差大")
    print("   - 需要提高 free_energy_threshold 或降低 max_aspects")
    print()

if __name__ == "__main__":
    device = torch.device("cpu")
    config = {
        "state_dim": 1024,
        "act_dim": 256,
        "sem_dim": 512,
        "sense_dims": {
            "vision": 512,
            "language": 512,
            "audio": 128,
            "multimodal": 256,
        },
        "enable_world_model_learning": True,
        "world_model": {
            "semantic_dim": 1024,
            "memory_dim": 512,
            "context_dim": 256,
            "physical_dim": 64,
            "goal_dim": 256,
            "vision_dim": 512,
            "language_dim": 512,
            "audio_dim": 128,
            "multimodal_dim": 256,
            "state_noise_std": 0.01,
            "observation_noise_std": 0.01,
            "enable_tools": True,
        },
        "evolution": {
            "free_energy_threshold": 0.15,
            "prune_threshold": 0.01,
            "max_objects": 100,
            "max_aspects": 800,
            "error_ema_alpha": 0.4,
            "batch_growth": {
                "base": 16,
                "max_per_step": 64,
                "max_total": 400,
                "min_per_sense": 8,
                "error_threshold": 0.15,
                "error_multiplier": 0.7,
            },
        },
        "pipeline_growth": {
            "enable": True,
            "initial_depth": 3,
            "initial_width": 1,
            "depth_increment": 1,
            "width_increment": 0,
            "max_stages": 1000,
            "min_interval": 1,
            "free_energy_trigger": 0.1,
            "max_depth": 50,
        },
        "state_clip_value": 5.0,
        "infer_lr": 0.05,
        "learning_rate": 0.001,
        "num_infer_iters": 3,
        "llm": {
            "hidden_dims": [256, 512, 256],
        },
    }
    
    analyze_free_energy(config, device, num_steps=3)

