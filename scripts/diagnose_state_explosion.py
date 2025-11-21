#!/usr/bin/env python3
"""
诊断状态值爆炸的原因
检查梯度大小、状态更新幅度等
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aonn.models.general_ai_world_model import GeneralAIWorldModel, GeneralAIWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.aspects.mock_llm_client import MockLLMClient
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.core.free_energy import compute_total_free_energy
from typing import Dict

def diagnose_state_explosion(config: Dict, device: torch.device):
    """诊断状态值爆炸"""
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
    
    print("=" * 80)
    print("状态值爆炸诊断")
    print("=" * 80)
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
        
        print(f"Step {step}:")
        print(f"  Aspects 数量: {len(brain.aspects)}")
        
        # 检查推理前的状态
        internal_before = brain.objects["internal"].state.clone()
        print(f"  推理前 internal 状态:")
        print(f"    范数: {torch.norm(internal_before).item():.4f}")
        print(f"    范围: [{internal_before.min().item():.4f}, {internal_before.max().item():.4f}]")
        
        # 执行一次推理迭代，检查梯度
        if len(brain.aspects) > 0:
            # 重置为可微
            mu = brain.objects["internal"].clone_detached(requires_grad=True)
            brain.objects["internal"].state = mu
            
            # 计算自由能
            F = compute_total_free_energy(brain.objects, brain.aspects)
            print(f"  自由能: {F.item():.4f}")
            
            # 反向传播（只执行一次）
            try:
                F.backward(retain_graph=False)
            except RuntimeError as e:
                if "backward through the graph" in str(e):
                    print(f"  ⚠️  梯度图已释放，跳过梯度检查")
                    continue
                else:
                    raise
            
            # 检查梯度
            grad = brain.objects["internal"].state.grad
            if grad is not None:
                grad_norm = torch.norm(grad).item()
                grad_max = grad.abs().max().item()
                grad_mean = grad.abs().mean().item()
                print(f"  梯度统计:")
                print(f"    范数: {grad_norm:.4f}")
                print(f"    最大值: {grad_max:.4f}")
                print(f"    平均值: {grad_mean:.4f}")
                
                # 计算更新幅度
                infer_lr = config.get("infer_lr", 0.05)
                update = infer_lr * grad
                update_norm = torch.norm(update).item()
                update_max = update.abs().max().item()
                print(f"  更新幅度 (infer_lr={infer_lr}):")
                print(f"    范数: {update_norm:.4f}")
                print(f"    最大值: {update_max:.4f}")
                
                # 预测更新后的状态
                predicted_state = internal_before - update
                predicted_norm = torch.norm(predicted_state).item()
                predicted_max = predicted_state.abs().max().item()
                print(f"  预测更新后状态:")
                print(f"    范数: {predicted_norm:.4f}")
                print(f"    最大值: {predicted_max:.4f}")
                
                # 检查是否会超过裁剪值
                clip_value = config.get("state_clip_value", 5.0)
                if predicted_max > clip_value:
                    print(f"  ⚠️  预测状态会超过裁剪值 {clip_value}!")
                    print(f"     需要裁剪 {predicted_max - clip_value:.4f}")
        
        # 执行完整推理
        if len(brain.aspects) > 0:
            loop = ActiveInferenceLoop(
                brain.objects,
                brain.aspects,
                infer_lr=config.get("infer_lr", 0.05),
                device=device,
            )
            num_iters = config.get("num_infer_iters", 3)
            loop.infer_states(target_objects=("internal",), num_iters=num_iters)
            brain.sanitize_states()
        
        # 检查推理后的状态
        internal_after = brain.objects["internal"].state
        print(f"  推理后 internal 状态:")
        print(f"    范数: {torch.norm(internal_after).item():.4f}")
        print(f"    范围: [{internal_after.min().item():.4f}, {internal_after.max().item():.4f}]")
        
        state_change = torch.norm(internal_after - internal_before).item()
        print(f"  状态变化范数: {state_change:.4f}")
        print()
        
        # 执行一步
        obs, reward = world_interface.step(action)
        action = torch.randn(config["act_dim"], device=device) * 0.1
    
    print("=" * 80)
    print("诊断结论")
    print("=" * 80)
    print()
    print("如果梯度很大 (>100):")
    print("  - 说明自由能对状态的敏感度很高")
    print("  - 需要降低 infer_lr 或增加梯度裁剪")
    print()
    print("如果更新幅度很大 (>5.0):")
    print("  - 说明单次更新就会超过裁剪值")
    print("  - 需要降低 infer_lr 或增加梯度裁剪")
    print()
    print("如果状态变化很大 (>10.0):")
    print("  - 说明多次迭代累积导致状态爆炸")
    print("  - 需要减少 num_iters 或在每次迭代后裁剪")
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
            "free_energy_threshold": 0.5,
            "prune_threshold": 0.01,
            "max_objects": 100,
            "max_aspects": 400,
            "error_ema_alpha": 0.4,
            "batch_growth": {
                "base": 8,
                "max_per_step": 64,
                "max_total": 400,
                "min_per_sense": 8,
                "error_threshold": 0.5,
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
        "learning_rate": 0.005,
        "num_infer_iters": 3,
        "llm": {
            "hidden_dims": [256, 512, 256],
        },
    }
    
    diagnose_state_explosion(config, device)

