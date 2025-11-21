#!/usr/bin/env python3
"""
MNIST 演化实验脚本
演示如何在统一架构下通过动态网络演化解决 MNIST 分类问题
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import argparse
import math
import time
from typing import Dict, Optional

import torch
from tqdm import tqdm

from aonn.models.mnist_world_model import MNISTWorldModel, MNISTWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.aspects.classification_aspect import ClassificationAspect
from aonn.aspects.pipeline_aspect import PipelineAspect


def run_experiment(
    num_steps: int,
    config: Dict,
    device: torch.device,
    *,
    verbose: bool = False,
    output: str = "data/mnist_results.json",
    save_interval: int = 100,
):
    """运行 MNIST 演化实验"""
    
    # 创建 MNIST 世界模型
    world_model = MNISTWorldModel(
        state_dim=config.get("state_dim", 256),
        action_dim=config.get("act_dim", 10),
        obs_dim=config.get("obs_dim", 784),
        device=device,
    )
    world_interface = MNISTWorldInterface(world_model)
    
    # 创建 AONN Brain
    brain = AONNBrainV3(config=config, device=device, enable_evolution=True)
    
    # 创建分类 Aspect（初始）
    classification_aspect = ClassificationAspect(
        internal_name="internal",
        target_name="target",
        state_dim=config.get("state_dim", 256),
        num_classes=10,
        name="classification",
    )
    brain.aspects.append(classification_aspect)
    brain.aspect_modules.append(classification_aspect)
    
    # 创建 target Object
    brain.create_object("target", dim=10)
    
    # 初始化环境
    obs = world_interface.reset()
    brain.objects["vision"].set_state(obs["vision"])
    
    prev_obs = None
    snapshots = []
    accuracy_history = []
    
    progress = tqdm(range(num_steps), desc=f"MNIST {num_steps}")
    
    try:
        for step in progress:
            step_start_time = time.perf_counter()
            
            # 1. 获取观察和目标
            if prev_obs is None:
                obs = world_interface.reset()
            else:
                # 使用上一步的 action 执行 step
                action = brain.objects.get("action", None)
                if action is not None:
                    action_logits = classification_aspect.predict(brain.objects)
                    obs, reward = world_interface.step(action_logits)
                else:
                    obs = world_interface.reset()
            
            target = world_interface.get_target()
            
            # 2. 设置观察和目标
            brain.objects["vision"].set_state(obs["vision"])
            brain.objects["target"].set_state(target)
            
            # 3. 网络演化
            brain.evolve_network(obs, target=target)
            
            # 4. 主动推理（更新 internal 状态）
            if len(brain.aspects) > 0:
                try:
                    # 确保所有 Object 状态都是 detached
                    for obj_name, obj in brain.objects.items():
                        state = obj.state
                        if state.requires_grad and state.is_leaf and state.grad is not None:
                            obj.set_state(state.detach())
                        elif not state.is_leaf:
                            obj.set_state(state.detach())
                    
                    loop = ActiveInferenceLoop(
                        brain.objects,
                        brain.aspects,
                        infer_lr=config.get("infer_lr", 0.01),
                        max_grad_norm=config.get("max_grad_norm", None),
                        device=device,
                    )
                    num_iters = config.get("num_infer_iters", 3)
                    loop.infer_states(
                        target_objects=("internal",), 
                        num_iters=num_iters, 
                        sanitize_callback=brain.sanitize_states
                    )
                    brain.sanitize_states()
                except Exception as e:
                    if verbose:
                        print(f"  [Step {step}] 推理错误: {e}")
                    pass
            
            # 5. 世界模型学习（学习分类器和 Pipeline 参数）
            if prev_obs is not None:
                try:
                    brain.learn_world_model(
                        observation=prev_obs,
                        action=None,  # 分类任务不需要动作
                        next_observation=obs,
                        target_state=world_model.get_true_state(),
                        learning_rate=config.get("learning_rate", 0.001),
                    )
                except Exception as e:
                    if verbose:
                        print(f"  [Step {step}] 学习错误: {e}")
                    pass
            
            prev_obs = {sense: value.clone() for sense, value in obs.items()}
            
            # 6. 计算并记录自由能
            F = brain.compute_free_energy().item()
            if not math.isfinite(F):
                brain.sanitize_states()
                F = 1e-6
            
            # 7. 获取预测并评估
            action_logits = classification_aspect.predict(brain.objects)
            pred_class = action_logits.argmax().item()
            true_class = target.argmax().item()
            accuracy = 1.0 if pred_class == true_class else 0.0
            accuracy_history.append(accuracy)
            
            # 8. 记录快照
            if step % save_interval == 0 or step == num_steps - 1:
                structure = brain.get_network_structure()
                snapshot = {
                    "step": step,
                    "free_energy": F,
                    "accuracy": accuracy,
                    "predicted_class": pred_class,
                    "true_class": true_class,
                    "structure": structure,
                }
                snapshots.append(snapshot)
            
            # 9. 更新进度条
            avg_accuracy = sum(accuracy_history[-100:]) / min(100, len(accuracy_history))
            structure = brain.get_network_structure()
            progress.set_postfix(
                F=f"{F:.2f}",
                Acc=f"{avg_accuracy*100:.1f}%",
                Asp=structure.get('num_aspects', 0),
                Pipe=structure.get('num_pipelines', 0),
            )
            
            if verbose and step % 10 == 0:
                print(f"Step {step}: F={F:.4f}, Acc={accuracy*100:.1f}%, "
                      f"Pred={pred_class}, True={true_class}, "
                      f"Aspects={structure.get('num_aspects', 0)}, "
                      f"Pipelines={structure.get('num_pipelines', 0)}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  实验被用户中断")
    except Exception as e:
        print(f"\n\n❌ 实验发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 最终结果
    final_snapshot = brain.observe_self_model()
    final_F = brain.compute_free_energy().item()
    final_accuracy = sum(accuracy_history[-100:]) / min(100, len(accuracy_history)) if accuracy_history else 0.0
    
    result = {
        "num_steps": num_steps,
        "final_free_energy": final_F,
        "final_accuracy": final_accuracy,
        "final_structure": final_snapshot.get("structure", {}),
        "snapshots": snapshots,
        "accuracy_history": accuracy_history[-1000:],  # 只保存最后1000步
        "evolution_summary": brain.evolution.get_evolution_summary() if brain.evolution else {},
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="MNIST 演化实验")
    parser.add_argument("--steps", type=int, default=100, help="实验步数")
    parser.add_argument("--state-dim", type=int, default=256, help="状态维度")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--output", type=str, default="data/mnist_results.json", help="输出文件路径")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--save-interval", type=int, default=50, help="快照保存间隔")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 配置
    config = {
        "obs_dim": 784,
        "state_dim": args.state_dim,
        "act_dim": 10,
        "sense_dims": {"vision": 784},
        "enable_world_model_learning": True,
        "evolution": {
            "free_energy_threshold": 2.0,  # 分类任务需要更高的阈值
            "prune_threshold": 0.01,
            "max_objects": 20,
            "max_aspects": 500,
            "error_ema_alpha": 0.5,
            "batch_growth": {
                "base": 8,
                "max_per_step": 32,
                "max_total": 200,
                "min_per_sense": 1,
                "error_threshold": 1.5,
                "error_multiplier": 0.7,
            },
            "pipeline_growth": {
                "enable": True,
                "free_energy_trigger": 1.0,
                "min_interval": 10,
                "max_stages": 5,
                "max_depth": 6,
                "initial_width": 64,
                "width_increment": 0,
                "initial_depth": 3,
                "depth_increment": 1,
            },
        },
        "infer_lr": 0.01,
        "learning_rate": 0.001,
        "num_infer_iters": 3,
        "max_grad_norm": 100.0,
        "state_clip_value": 5.0,
    }
    
    print("=" * 80)
    print("MNIST 演化实验")
    print("=" * 80)
    print(f"实验步数: {args.steps}")
    print(f"状态维度: {config['state_dim']}")
    print(f"观察维度: {config['obs_dim']}")
    print(f"动作维度: {config['act_dim']}")
    print("=" * 80)
    print()
    
    result = run_experiment(
        num_steps=args.steps,
        config=config,
        device=device,
        verbose=args.verbose,
        output=args.output,
        save_interval=args.save_interval,
    )
    
    # 保存结果
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print()
    print("=" * 80)
    print("实验完成！")
    print("=" * 80)
    print(f"结果保存到: {output_path}")
    print(f"最终自由能: {result['final_free_energy']:.4f}")
    print(f"最终准确率: {result['final_accuracy']*100:.2f}%")
    print(f"最终结构: {result['final_structure'].get('num_objects', 0)} Objects, "
          f"{result['final_structure'].get('num_aspects', 0)} Aspects, "
          f"{result['final_structure'].get('num_pipelines', 0)} Pipelines")
    print("=" * 80)


if __name__ == "__main__":
    main()

