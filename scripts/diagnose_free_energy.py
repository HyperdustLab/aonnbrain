#!/usr/bin/env python3
"""
诊断自由能居高不下的原因
分析各个 Aspect 对自由能的贡献
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aonn.models.general_ai_world_model import GeneralAIWorldModel, GeneralAIWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.aspects.mock_llm_client import MockLLMClient
from typing import Dict

def diagnose_free_energy(config: Dict, device: torch.device):
    """诊断自由能组成"""
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
    
    print("=" * 70)
    print("自由能诊断分析")
    print("=" * 70)
    print()
    
    for step in range(3):
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
        
        # 分析各个 Aspect 的贡献
        aspect_contributions = {}
        for aspect in brain.aspects:
            contrib = aspect.free_energy_contrib(brain.objects).item()
            aspect_type = type(aspect).__name__
            if aspect_type not in aspect_contributions:
                aspect_contributions[aspect_type] = []
            aspect_contributions[aspect_type].append(contrib)
        
        print("  各类型 Aspect 贡献:")
        total_by_type = {}
        count_by_type = {}
        for aspect_type, contribs in aspect_contributions.items():
            total = sum(contribs)
            count = len(contribs)
            total_by_type[aspect_type] = total
            count_by_type[aspect_type] = count
            avg = total / count if count > 0 else 0
            print(f"    {aspect_type}: 总数={total:.4f}, 数量={count}, 平均={avg:.4f}")
        
        print()
        print("  贡献占比:")
        for aspect_type, total in sorted(total_by_type.items(), key=lambda x: x[1], reverse=True):
            percentage = (total / total_F * 100) if total_F > 0 else 0
            print(f"    {aspect_type}: {percentage:.2f}% ({total:.4f})")
        
        print()
        print(f"  网络规模: Objects={len(brain.objects)}, Aspects={len(brain.aspects)}, Pipelines={len(brain.aspect_pipelines)}")
        print()
        
        # 执行一步
        obs, reward = world_interface.step(action)
        action = torch.randn(config["act_dim"], device=device) * 0.1
    
    print("=" * 70)
    print("诊断结论")
    print("=" * 70)
    print()
    print("主要问题:")
    print("1. 如果 LinearGenerativeAspect 贡献最大：")
    print("   - 说明感官预测误差大，需要更多学习时间或降低复杂度")
    print("2. 如果 LLMAspect 贡献大：")
    print("   - 说明 LLM 预测误差大，可能需要调整 LLM 权重或使用更好的模型")
    print("3. 如果 DynamicsAspect/ObservationAspect 贡献大：")
    print("   - 说明世界模型学习不够，需要提高学习率或增加学习时间")
    print("4. 如果 Aspects 数量过多：")
    print("   - 说明网络演化太快，需要提高 free_energy_threshold 或限制 max_aspects")
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
            "free_energy_threshold": 0.05,
            "prune_threshold": 0.01,
            "max_objects": 100,
            "max_aspects": 2000,
            "error_ema_alpha": 0.4,
            "batch_growth": {
                "base": 16,
                "max_per_step": 64,
                "max_total": 400,
                "min_per_sense": 8,
                "error_threshold": 0.05,
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
        "infer_lr": 0.01,
        "learning_rate": 0.0001,
        "llm": {
            "hidden_dims": [256, 512, 256],
        },
    }
    
    diagnose_free_energy(config, device)

