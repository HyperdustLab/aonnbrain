#!/usr/bin/env python3
"""
测试不同 Aspect 数量对网络演化的影响

用于验证：如果减少 Aspect 数量，网络能否在复杂的世界模型中正常演化？
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aonn.models.general_ai_world_model import GeneralAIWorldModel, GeneralAIWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.aspects.mock_llm_client import MockLLMClient
from aonn.core.free_energy import compute_total_free_energy
import json
from tqdm import tqdm


def test_aspect_count_impact(max_aspects: int, num_steps: int = 100, verbose: bool = False):
    """
    测试指定 Aspect 数量下的网络演化情况
    
    Args:
        max_aspects: 最大 Aspect 数量
        num_steps: 演化步数
        verbose: 是否输出详细信息
    """
    device = torch.device("cpu")
    
    # 配置
    config = {
        "state_dim": 1024,
        "act_dim": 256,
        "sem_dim": 512,
        "sense_dims": {"vision": 512, "language": 512, "audio": 128, "multimodal": 256},
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
            "free_energy_threshold": 0.1,
            "prune_threshold": 0.01,
            "max_objects": 100,
            "max_aspects": max_aspects,  # 测试不同的 Aspect 数量
            "error_ema_alpha": 0.4,
            "batch_growth": {
                "base": 8,
                "max_per_step": 32,
                "max_total": max_aspects // 3,  # 每个感官约 max_aspects/12
                "min_per_sense": 8,
                "error_threshold": 0.1,
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
        "learning_rate": 0.01,
        "num_infer_iters": 2,
        "max_grad_norm": 268.0,
    }
    
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
        state_noise_std=config["world_model"]["state_noise_std"],
        observation_noise_std=config["world_model"]["observation_noise_std"],
        enable_tools=config["world_model"]["enable_tools"],
    )
    world_interface = GeneralAIWorldInterface(world_model)
    
    # 创建 LLM 客户端（Mock）
    llm_client = MockLLMClient(
        input_dim=config["sem_dim"],
        output_dim=config["sem_dim"],
        hidden_dims=[256, 512, 256],
        device=device,
    )
    
    # 创建 AONN Brain
    brain = AONNBrainV3(
        config=config,
        llm_client=llm_client,
        device=device,
        enable_evolution=True,
    )
    
    # 运行演化
    obs = world_interface.reset()
    prev_obs = None
    prev_action = None
    action = torch.randn(config["act_dim"], device=device) * 0.1
    
    results = {
        "max_aspects": max_aspects,
        "num_steps": num_steps,
        "free_energy_history": [],
        "aspect_count_history": [],
        "pipeline_count_history": [],
        "final_stats": {},
    }
    
    for step in tqdm(range(num_steps), desc=f"max_aspects={max_aspects}"):
        if step > 0:
            obs, reward = world_interface.step(action)
        
        # 设置观察
        for sense, value in obs.items():
            if sense in brain.objects:
                brain.objects[sense].set_state(value)
        
        # 获取目标状态
        full_state = world_model.get_true_state()
        if full_state.shape[-1] >= config["state_dim"]:
            target_state = full_state[:config["state_dim"]]
        else:
            padding = torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)
            target_state = torch.cat([full_state, padding], dim=-1)
        
        # 演化网络
        brain.evolve_network(obs, target=target_state)
        
        # 推理状态
        if len(brain.aspects) > 0:
            try:
                loop = ActiveInferenceLoop(
                    brain.objects,
                    brain.aspects,
                    infer_lr=config["infer_lr"],
                    max_grad_norm=config["max_grad_norm"],
                    device=device,
                )
                loop.infer_states(
                    target_objects=("internal",),
                    num_iters=config["num_infer_iters"],
                    sanitize_callback=brain.sanitize_states,
                )
                brain.sanitize_states()
            except Exception as e:
                if verbose:
                    print(f"Step {step}: Inference error: {e}")
                pass
        
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
            try:
                brain.learn_world_model(
                    observation=prev_obs,
                    action=prev_action,
                    next_observation=obs,
                    target_state=target_state,
                    learning_rate=config["learning_rate"],
                )
                brain.sanitize_states()
            except Exception as e:
                if verbose:
                    print(f"Step {step}: Learning error: {e}")
                pass
        
        # 记录统计信息
        F = brain.compute_free_energy().item()
        structure = brain.observe_self_model().get("structure", {})
        num_aspects = structure.get("num_aspects", 0)
        num_pipelines = structure.get("num_pipelines", 0)
        
        results["free_energy_history"].append(F)
        results["aspect_count_history"].append(num_aspects)
        results["pipeline_count_history"].append(num_pipelines)
        
        if verbose and step % 10 == 0:
            print(f"Step {step}: F={F:.4f}, Aspects={num_aspects}, Pipelines={num_pipelines}")
        
        prev_obs = {sense: value.clone() for sense, value in obs.items()}
        prev_action = action.clone()
    
    # 记录最终统计
    structure = brain.observe_self_model().get("structure", {})
    results["final_stats"] = {
        "num_aspects": structure.get("num_aspects", 0),
        "num_pipelines": structure.get("num_pipelines", 0),
        "num_objects": structure.get("num_objects", 0),
        "final_free_energy": results["free_energy_history"][-1] if results["free_energy_history"] else float('inf'),
        "avg_free_energy": sum(results["free_energy_history"]) / len(results["free_energy_history"]) if results["free_energy_history"] else float('inf'),
        "min_free_energy": min(results["free_energy_history"]) if results["free_energy_history"] else float('inf'),
        "max_free_energy": max(results["free_energy_history"]) if results["free_energy_history"] else float('inf'),
    }
    
    return results


def main():
    """测试不同 Aspect 数量的演化情况"""
    print("=" * 80)
    print("测试不同 Aspect 数量对网络演化的影响")
    print("=" * 80)
    print()
    
    # 测试不同的 Aspect 数量
    aspect_counts = [50, 100, 200, 400]
    num_steps = 100
    
    all_results = {}
    
    for max_aspects in aspect_counts:
        print(f"\n测试 max_aspects = {max_aspects}")
        print("-" * 80)
        try:
            results = test_aspect_count_impact(max_aspects, num_steps=num_steps, verbose=False)
            all_results[max_aspects] = results
            
            # 输出结果摘要
            stats = results["final_stats"]
            print(f"  最终 Aspect 数量: {stats['num_aspects']}")
            print(f"  最终 Pipeline 数量: {stats['num_pipelines']}")
            print(f"  最终自由能: {stats['final_free_energy']:.4f}")
            print(f"  平均自由能: {stats['avg_free_energy']:.4f}")
            print(f"  最小自由能: {stats['min_free_energy']:.4f}")
            print(f"  最大自由能: {stats['max_free_energy']:.4f}")
            
            # 判断演化状态
            if stats['final_free_energy'] < 30:
                status = "✓ 可以正常演化"
            elif stats['final_free_energy'] < 60:
                status = "⚠ 演化较慢"
            elif stats['final_free_energy'] < 100:
                status = "⚠ 演化困难"
            else:
                status = "✗ 难以正常演化"
            
            print(f"  演化状态: {status}")
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    output_file = Path(__file__).parent.parent / "data" / "aspect_count_impact_test.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 输出对比分析
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)
    print()
    
    for max_aspects in sorted(all_results.keys()):
        stats = all_results[max_aspects]["final_stats"]
        print(f"max_aspects = {max_aspects:3d}:")
        print(f"  最终 Aspect: {stats['num_aspects']:3d}, 最终 F: {stats['final_free_energy']:7.2f}, 平均 F: {stats['avg_free_energy']:7.2f}")


if __name__ == "__main__":
    main()

