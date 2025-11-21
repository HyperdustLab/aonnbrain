#!/usr/bin/env python3
"""
MNIST 自动演化实验脚本
测试 vision_pipeline 通过自动演化创建（而非硬编码）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import argparse
from typing import Dict, Optional

import torch
from torch.optim import Adam
from tqdm import tqdm

from aonn.models.mnist_world_model import MNISTWorldModel, MNISTWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.aspects.classification_aspect import ClassificationAspect


def run_experiment(
    num_steps: int,
    config: Dict,
    device: torch.device,
    *,
    verbose: bool = False,
    output: str = "data/mnist_auto_evolution.json",
    save_interval: int = 100,
):
    """运行 MNIST 自动演化实验"""
    
    # 创建训练和验证世界模型
    train_world = MNISTWorldModel(
        state_dim=config.get("state_dim", 128),
        action_dim=config.get("act_dim", 10),
        obs_dim=config.get("obs_dim", 784),
        device=device,
        train=True,
    )
    train_interface = MNISTWorldInterface(train_world)
    
    val_world = MNISTWorldModel(
        state_dim=config.get("state_dim", 128),
        action_dim=config.get("act_dim", 10),
        obs_dim=config.get("obs_dim", 784),
        device=device,
        train=False,  # 使用测试集
    )
    val_interface = MNISTWorldInterface(val_world)
    
    # 创建 AONN Brain（最小初始架构，不包含 vision_pipeline）
    brain = AONNBrainV3(config=config, device=device, enable_evolution=True)
    
    # 创建 target Object
    brain.create_object("target", dim=10)
    
    print("初始化 AONN Brain（自动演化模式）...")
    print("  ✓ 初始网络：最小架构（无 vision_pipeline）")
    print("  ✓ 等待自动演化创建 vision_pipeline...")
    print()
    
    # 创建分类器（这个可以预先创建，因为分类器不是自动演化的目标）
    classification_aspect = brain.create_unified_aspect(
        aspect_type="classification",
        src_names=["internal"],
        dst_names=["target"],
        name="mnist_classifier",
        state_dim=config.get("state_dim", 128),
        num_classes=10,
        hidden_dim=config.get("state_dim", 128),
        loss_weight=config.get("classification_loss_weight", 1.0),
    )
    print(f"  ✓ 创建 mnist_classifier (state_dim={config.get('state_dim', 128)}, num_classes=10)")
    print()
    
    # 创建 Adam 优化器（初始只包含分类器，vision_pipeline 会在演化后添加）
    aspect_params = list(classification_aspect.parameters())
    aspect_optimizer = Adam(
        aspect_params,
        lr=config.get("learning_rate", 0.001),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    print(f"  ✓ 初始化 Adam 优化器: lr={config.get('learning_rate', 0.001)}")
    print()
    
    snapshots = []
    accuracy_history = []
    free_energy_history = []
    evolution_events = []  # 记录演化事件
    
    # 初始化观察
    obs = train_interface.reset()
    action = None
    prev_obs = None
    
    # 检查 vision_pipeline 是否已创建
    vision_pipeline_created = False
    
    progress = tqdm(range(num_steps), desc=f"MNIST Auto Evolution {num_steps}")
    
    try:
        for step in progress:
            # 1. 获取观察和目标
            if step > 0:
                obs, reward = train_interface.step(action)
            else:
                obs = train_interface.reset()
            
            # 设置观察
            for sense, value in obs.items():
                if sense in brain.objects:
                    brain.objects[sense].set_state(value)
            
            # 获取目标标签
            target = train_interface.get_target()
            brain.objects["target"].set_state(target)
            
            # 2. 网络演化（这里会自动创建 vision_pipeline）
            num_aspects_before = len(brain.aspects)
            brain.evolve_network(obs)
            num_aspects_after = len(brain.aspects)
            
            # 检查是否创建了 vision_pipeline
            if not vision_pipeline_created:
                vision_pipeline = None
                for asp in brain.aspects:
                    if (hasattr(asp, 'name') and 
                        ('vision' in asp.name.lower() and 'pipeline' in asp.name.lower())):
                        vision_pipeline = asp
                        vision_pipeline_created = True
                        evolution_events.append({
                            "step": step + 1,
                            "event": "vision_pipeline_created",
                            "name": asp.name,
                            "free_energy": brain.compute_free_energy().item(),
                        })
                        print(f"\n🎉 [Step {step+1}] 自动演化创建了 vision_pipeline: {asp.name}")
                        # 更新优化器，添加 vision_pipeline 的参数
                        aspect_params = list(asp.parameters()) + list(classification_aspect.parameters())
                        aspect_optimizer = Adam(
                            aspect_params,
                            lr=config.get("learning_rate", 0.001),
                            betas=(0.9, 0.999),
                            eps=1e-8,
                        )
                        print(f"  ✓ 更新优化器，添加 vision_pipeline 参数")
                        break
            
            # 如果 vision_pipeline 已创建，获取它
            if vision_pipeline_created and vision_pipeline is None:
                for asp in brain.aspects:
                    if (hasattr(asp, 'name') and 
                        ('vision' in asp.name.lower() and 'pipeline' in asp.name.lower())):
                        vision_pipeline = asp
                        break
            
            # 3. 主动推理（状态推理）
            if len(brain.aspects) > 0:
                try:
                    # 确保状态是 detached 的
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
                        max_grad_norm=config.get("max_grad_norm", 100.0),
                        device=device,
                    )
                    num_iters = config.get("num_infer_iters", 5)
                    loop.infer_states(
                        target_objects=("internal",),
                        num_iters=num_iters,
                        sanitize_callback=brain.sanitize_states
                    )
                    brain.sanitize_states()
                except Exception as e:
                    if verbose:
                        print(f"Step {step}: Inference error: {e}")
                    pass
            
            # 4. 参数学习（使用 Adam 优化器）
            if step > 0 and vision_pipeline_created:  # 只有 vision_pipeline 创建后才学习
                try:
                    aspect_optimizer.zero_grad()
                    F = brain.compute_free_energy()
                    
                    if torch.isfinite(F) and F.requires_grad:
                        F.backward()
                        
                        # 梯度裁剪
                        max_grad_norm = config.get("max_grad_norm", None)
                        if max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(aspect_params, max_grad_norm)
                        
                        aspect_optimizer.step()
                        brain.sanitize_states()
                except Exception as e:
                    if verbose:
                        print(f"Step {step}: Learning error: {e}")
                    pass
            
            # 5. 生成动作（分类预测）
            with torch.no_grad():
                logits = classification_aspect.predict(brain.objects)
                action = torch.softmax(logits, dim=-1)
                
                # 评估准确率
                pred_label = logits.argmax().item()
                true_label = train_interface.world_model.get_label()
                correct = (pred_label == true_label)
                accuracy_history.append(1.0 if correct else 0.0)
            
            # 6. 记录自由能
            with torch.no_grad():
                F = brain.compute_free_energy()
                free_energy_history.append(F.item())
            
            # 7. 保存快照
            if (step + 1) % save_interval == 0 or step == num_steps - 1:
                structure = brain.get_network_structure()
                avg_acc = sum(accuracy_history[-100:]) / min(100, len(accuracy_history))
                avg_F = sum(free_energy_history[-100:]) / min(100, len(free_energy_history))
                
                snapshot = {
                    "step": step + 1,
                    "free_energy": avg_F,
                    "accuracy": avg_acc,
                    "structure": structure,
                    "vision_pipeline_created": vision_pipeline_created,
                }
                snapshots.append(snapshot)
            
            # 8. 更新进度条
            avg_acc = sum(accuracy_history[-100:]) / min(100, len(accuracy_history))
            avg_F = sum(free_energy_history[-100:]) / min(100, len(free_energy_history))
            structure = brain.get_network_structure()
            
            vision_status = "✓" if vision_pipeline_created else "✗"
            progress.set_postfix(
                F=f"{avg_F:.3f}",
                Acc=f"{avg_acc*100:.1f}%",
                Asp=structure.get('num_aspects', 0),
                Pipe=structure.get('num_pipelines', 0),
                Vision=vision_status,
            )
            
            if verbose and (step + 1) % 50 == 0:
                print(f"[Step {step+1}] F={avg_F:.4f}, Acc={avg_acc*100:.2f}%, "
                      f"Aspects={structure.get('num_aspects', 0)}, "
                      f"Vision Pipeline={'Created' if vision_pipeline_created else 'Not Created'}")
            
            prev_obs = {sense: value.clone() for sense, value in obs.items()}
    
    except KeyboardInterrupt:
        print("\n\n⚠️  实验被用户中断")
    except Exception as e:
        print(f"\n\n❌ 实验发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 验证
    print("\n开始验证...")
    val_accuracy = validate_with_world_model(
        brain=brain,
        val_interface=val_interface,
        classification_aspect=classification_aspect,
        num_samples=min(1000, len(val_world.dataset)),
        config=config,
        device=device,
    )
    
    # 最终结果
    final_snapshot = brain.observe_self_model()
    final_F = free_energy_history[-1] if free_energy_history else 0.0
    final_accuracy = sum(accuracy_history[-100:]) / min(100, len(accuracy_history)) if accuracy_history else 0.0
    
    result = {
        "num_steps": num_steps,
        "vision_pipeline_created": vision_pipeline_created,
        "evolution_events": evolution_events,
        "final_free_energy": final_F,
        "final_train_accuracy": final_accuracy,
        "final_val_accuracy": val_accuracy,
        "final_structure": final_snapshot.get("structure", {}),
        "snapshots": snapshots,
        "accuracy_history": accuracy_history[-1000:],
        "free_energy_history": free_energy_history[-1000:],
        "evolution_summary": brain.evolution.get_evolution_summary() if brain.evolution else {},
    }
    
    return result


def validate_with_world_model(
    brain,
    val_interface,
    classification_aspect,
    num_samples: int,
    config: Dict,
    device: torch.device,
):
    """使用世界模型验证"""
    correct = 0
    
    with torch.no_grad():
        obs = val_interface.reset()
        
        for i in range(num_samples):
            # 设置观察
            for sense, value in obs.items():
                if sense in brain.objects:
                    brain.objects[sense].set_state(value)
            
            # 获取目标（用于推理）
            target = val_interface.get_target()
            brain.objects["target"].set_state(target)
            
            # 主动推理（不更新参数）
            if len(brain.aspects) > 0:
                try:
                    loop = ActiveInferenceLoop(
                        brain.objects,
                        brain.aspects,
                        infer_lr=config.get("infer_lr", 0.01),
                        max_grad_norm=config.get("max_grad_norm", 100.0),
                        device=device,
                    )
                    loop.infer_states(
                        target_objects=("internal",),
                        num_iters=config.get("num_infer_iters", 3),
                        sanitize_callback=brain.sanitize_states
                    )
                except Exception:
                    pass
            
            # 预测
            logits = classification_aspect.predict(brain.objects)
            pred_label = logits.argmax().item()
            true_label = val_interface.world_model.get_label()
            
            if pred_label == true_label:
                correct += 1
            
            # 移动到下一个样本
            action = torch.softmax(logits, dim=-1)
            obs, _ = val_interface.step(action)
    
    accuracy = correct / num_samples
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="MNIST 自动演化实验")
    parser.add_argument("--steps", type=int, default=500, help="训练步数")
    parser.add_argument("--state-dim", type=int, default=128, help="状态维度")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--output", type=str, default="data/mnist_auto_evolution.json", help="输出文件路径")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--save-interval", type=int, default=50, help="快照保存间隔")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 配置（启用自动演化创建 vision_pipeline）
    config = {
        "obs_dim": 784,
        "state_dim": args.state_dim,
        "act_dim": 10,
        "sense_dims": {"vision": 784},
        "enable_world_model_learning": False,
        "evolution": {
            "free_energy_threshold": 2.0,  # 自由能阈值，超过此值会触发自动创建
            "prune_threshold": 0.01,
            "max_objects": 20,
            "max_aspects": 500,
            "error_ema_alpha": 0.5,
            "batch_growth": {
                "base": 0,
                "max_per_step": 0,
                "max_total": 0,
                "min_per_sense": 0,
            },
            "pipeline_growth": {
                "enable": False,  # 禁用自动创建 internal->internal pipeline
            },
        },
        # 关键配置：启用自动创建 vision_pipeline
        "pipeline_growth": {
            "use_pipeline_for_encoder": True,  # 使用 Pipeline 而不是简单的 Encoder
            "initial_width": 32,  # Pipeline 宽度
            "initial_depth": 4,   # Pipeline 深度
        },
        "infer_lr": 0.01,
        "learning_rate": 0.001,
        "classification_loss_weight": 1.0,
        "num_infer_iters": 5,
        "max_grad_norm": 100.0,
        "state_clip_value": 5.0,
    }
    
    print("=" * 80)
    print("MNIST 自动演化实验")
    print("=" * 80)
    print(f"训练步数: {args.steps}")
    print(f"状态维度: {config['state_dim']}")
    print(f"观察维度: {config['obs_dim']}")
    print(f"动作维度: {config['act_dim']}")
    print(f"自由能阈值: {config['evolution']['free_energy_threshold']}")
    print(f"自动创建 vision_pipeline: 启用")
    print(f"Pipeline 配置: depth={config['pipeline_growth']['initial_depth']}, "
          f"width={config['pipeline_growth']['initial_width']}")
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
    print(f"vision_pipeline 是否自动创建: {'是' if result['vision_pipeline_created'] else '否'}")
    if result['evolution_events']:
        for event in result['evolution_events']:
            print(f"  步骤 {event['step']}: {event['event']} ({event['name']}), "
                  f"自由能={event['free_energy']:.4f}")
    print(f"最终自由能: {result['final_free_energy']:.4f}")
    print(f"最终训练准确率: {result['final_train_accuracy']*100:.2f}%")
    print(f"最终验证准确率: {result['final_val_accuracy']*100:.2f}%")
    print(f"最终结构: {result['final_structure'].get('num_objects', 0)} Objects, "
          f"{result['final_structure'].get('num_aspects', 0)} Aspects, "
          f"{result['final_structure'].get('num_pipelines', 0)} Pipelines")
    print("=" * 80)


if __name__ == "__main__":
    main()

