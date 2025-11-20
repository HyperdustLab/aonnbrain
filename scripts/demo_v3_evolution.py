#!/usr/bin/env python3
"""
AONN Brain V3 动态演化演示

演示：
1. 从最小初始网络开始
2. 与世界模型交互
3. 观察网络结构的动态演化
4. 记录自我模型的变化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from typing import List, Dict
import json

from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.models.world_model import SimpleWorldModel, WorldModelInterface
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.core.free_energy import compute_total_free_energy
import torch.optim as optim


def demo_v3_evolution():
    print("=" * 70)
    print("AONN Brain V3 动态演化演示")
    print("=" * 70)
    
    # 配置
    config = {
        "obs_dim": 16,
        "state_dim": 32,
        "act_dim": 8,
        "enable_world_model_learning": True,  # 启用世界模型学习
        "evolution": {
            "free_energy_threshold": 0.5,
            "prune_threshold": 0.01,
            "max_objects": 20,
            "max_aspects": 100,
        }
    }
    
    device = torch.device("cpu")
    
    # ========== 1. 创建最小初始网络 ==========
    print("\n[步骤 1] 创建最小初始网络...")
    brain = AONNBrainV3(config=config, device=device, enable_evolution=True)
    print("初始网络结构:")
    print(brain.visualize_network())
    
    # 记录初始自我模型
    initial_snapshot = brain.observe_self_model()
    print(f"\n初始自由能: {initial_snapshot['free_energy']:.4f}")
    
    # ========== 2. 创建世界模型 ==========
    print("\n[步骤 2] 创建世界模型...")
    world_model = SimpleWorldModel(
        state_dim=config["state_dim"],
        action_dim=config["act_dim"],
        obs_dim=config["obs_dim"],
        device=device
    )
    world_interface = WorldModelInterface(world_model)
    
    # 重置环境
    initial_obs = world_interface.reset()
    print(f"初始观察: shape={initial_obs.shape}, norm={torch.norm(initial_obs).item():.4f}")
    
    # ========== 3. 交互和演化循环 ==========
    print("\n[步骤 3] 开始交互和演化循环...")
    num_steps = 30  # 减少步数以加快演示
    
    evolution_log: List[Dict] = []
    
    # 创建优化器（用于世界模型参数学习）
    if brain.world_model_aspects is not None:
        world_model_optimizer = optim.Adam(
            brain.world_model_aspects.get_all_parameters(),
            lr=0.001
        )
        print("✓ 世界模型学习已启用")
    
    # 存储上一观察（用于学习）
    prev_obs = None
    prev_action = None
    
    for step in range(num_steps):
        print(f"\n--- 步骤 {step + 1}/{num_steps} ---")
        
        # 获取观察
        if step == 0:
            obs = initial_obs
        else:
            obs, reward = world_interface.step(action)
            print(f"观察: norm={torch.norm(obs).item():.4f}, 奖励: {reward.item():.4f}")
        
        # 设置观察到 brain
        brain.objects["sensory"].set_state(obs)
        
        # 网络演化
        brain.evolve_network(obs)
        
        # 主动推理（更新 internal 状态）
        if len(brain.aspects) > 0:
            # 只在有 Aspect 时进行推理，且只推理一次避免梯度图问题
            try:
                infer_loop = ActiveInferenceLoop(
                    objects=brain.objects,
                    aspects=brain.aspects,
                    infer_lr=0.1,
                    device=device
                )
                infer_loop.infer_states(target_objects=("internal",), num_iters=1)
            except Exception:
                # 如果推理失败，跳过（可能是梯度图问题）
                pass
        
        # 计算自由能
        F = brain.compute_free_energy().item()
        print(f"自由能: {F:.4f}")
        
        # 生成动作（如果有 action Object）
        if "action" in brain.objects:
            # 如果有 Pipeline，使用 Pipeline
            if len(brain.aspect_pipelines) > 0:
                internal_state = brain.objects["internal"].state
                action = brain.aspect_pipelines[0](internal_state)
                brain.objects["action"].set_state(action)
            else:
                # 否则随机动作
                action = torch.randn(config["act_dim"], device=device) * 0.1
                brain.objects["action"].set_state(action)
        else:
            action = torch.randn(config["act_dim"], device=device) * 0.1
        
        # 学习世界模型（如果有上一观察和动作）
        if prev_obs is not None and prev_action is not None and brain.world_model_aspects is not None:
            # 获取目标状态（从世界模型）
            target_state = world_model.get_true_state()
            
            # 学习世界模型参数
            brain.learn_world_model(
                observation=prev_obs,
                action=prev_action,
                next_observation=obs,
                target_state=target_state,
                learning_rate=0.001,
            )
        
        # 保存当前观察和动作（用于下一步学习）
        prev_obs = obs.clone()
        prev_action = action.clone()
        
        # 观察自我模型（每5步）
        if (step + 1) % 5 == 0:
            snapshot = brain.observe_self_model()
            evolution_log.append(snapshot)
            
            print(f"\n自我模型快照 (步骤 {step + 1}):")
            print(f"  - 自由能: {snapshot['free_energy']:.4f}")
            print(f"  - Objects: {snapshot['structure']['num_objects']}")
            print(f"  - Aspects: {snapshot['structure']['num_aspects']}")
            print(f"  - Pipelines: {snapshot['structure']['num_pipelines']}")
            
            # 显示世界模型学习状态
            if brain.world_model_aspects is not None:
                print(f"  - 世界模型 Aspect: {len(brain.world_model_aspects.aspects)} 个")
                # 计算世界模型的自由能贡献
                world_F = sum(
                    aspect.free_energy_contrib(brain.objects).item()
                    for aspect in brain.world_model_aspects.aspects
                )
                print(f"  - 世界模型自由能贡献: {world_F:.4f}")
    
    # ========== 4. 最终网络结构 ==========
    print("\n" + "=" * 70)
    print("[步骤 4] 最终网络结构")
    print("=" * 70)
    print(brain.visualize_network())
    
    # ========== 5. 演化历史分析 ==========
    print("\n" + "=" * 70)
    print("[步骤 5] 演化历史分析")
    print("=" * 70)
    
    evolution_history = brain.get_evolution_history()
    print(f"\n总演化事件: {len(evolution_history)}")
    
    event_types = {}
    for event in evolution_history:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    
    print("\n事件类型统计:")
    for event_type, count in event_types.items():
        print(f"  - {event_type}: {count}")
    
    print("\n最近10个演化事件:")
    for event in evolution_history[-10:]:
        print(f"  步骤 {event.step}: {event.event_type} - {event.trigger_condition}")
        print(f"    自由能: {event.free_energy_before:.4f} → {event.free_energy_after:.4f}")
    
    # ========== 6. 自我模型变化可视化 ==========
    print("\n" + "=" * 70)
    print("[步骤 6] 自我模型变化趋势")
    print("=" * 70)
    
    if len(evolution_log) > 0:
        steps = [s["step"] for s in evolution_log]
        free_energies = [s["free_energy"] for s in evolution_log]
        num_objects = [s["structure"]["num_objects"] for s in evolution_log]
        num_aspects = [s["structure"]["num_aspects"] for s in evolution_log]
        num_pipelines = [s["structure"]["num_pipelines"] for s in evolution_log]
        
        print("\n自由能变化:")
        for i, (step, F) in enumerate(zip(steps, free_energies)):
            print(f"  步骤 {step}: {F:.4f}")
        
        print("\n网络规模变化:")
        print(f"  Objects: {num_objects[0]} → {num_objects[-1]}")
        print(f"  Aspects: {num_aspects[0]} → {num_aspects[-1]}")
        print(f"  Pipelines: {num_pipelines[0]} → {num_pipelines[-1]}")
        
        # 保存演化日志
        log_file = Path(__file__).parent.parent / "data" / "evolution_log.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            json.dump(evolution_log, f, indent=2, default=str)
        print(f"\n演化日志已保存到: {log_file}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo_v3_evolution()

