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
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from aonn.models.mnist_world_model import MNISTWorldModel
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.aspects.classification_aspect import ClassificationAspect
from aonn.aspects.pipeline_aspect import PipelineAspect


def run_experiment(
    num_epochs: int,
    config: Dict,
    device: torch.device,
    *,
    max_steps: Optional[int] = None,
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
    # DataLoader for independent MNIST training
    batch_size = config.get("batch_size", 128)
    train_loader = DataLoader(
        world_model.dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    
    # 创建 AONN Brain
    brain = AONNBrainV3(config=config, device=device, enable_evolution=True)
    
    # 创建 target Object（必须在创建 Aspect 之前）
    brain.create_object("target", dim=10)
    
    # 显式创建参考网络的 Pipeline 与分类器（保持与参考实现对齐）
    print("初始化 AONN Brain（显式创建参考 Pipeline + 分类器）...")
    pipeline_spec = config.get("pipeline_spec", {})
    pipeline_depth = pipeline_spec.get("depth", 4)
    pipeline_width = pipeline_spec.get("width", 32)
    
    vision_pipeline = brain.create_unified_aspect(
        aspect_type="pipeline",
        src_names=["vision"],
        dst_names=["internal"],
        name="vision_pipeline",
        num_aspects=pipeline_width,
        depth=pipeline_depth,
        input_dim=config.get("obs_dim", 784),
        output_dim=config.get("state_dim", 128),
    )
    
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
    
    # 初始化监督学习优化器（Pipeline + 分类器共同优化，交叉熵）
    classifier_lr = config.get("classifier_learning_rate", 0.001)
    supervised_params = list(vision_pipeline.parameters()) + list(classification_aspect.parameters())
    supervised_optimizer = Adam(
        supervised_params,
        lr=classifier_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    print(f"  ✓ 创建 vision_pipeline (depth={pipeline_depth}, width={pipeline_width})")
    print(f"  ✓ 创建 mnist_classifier (state_dim={config.get('state_dim', 128)}, num_classes=10)")
    print(f"  ✓ 初始化监督优化器: lr={classifier_lr}, 参数数={len(supervised_params)}")
    print()
    
    snapshots = []
    accuracy_history = []
    
    total_batches = len(train_loader)
    if total_batches == 0:
        raise RuntimeError("MNIST 数据集为空，无法训练。")
    
    planned_steps = num_epochs * total_batches
    if max_steps is not None:
        planned_steps = min(planned_steps, max_steps)
    progress = tqdm(total=planned_steps, desc=f"MNIST {num_epochs}e")
    global_step = 0
    
    try:
        for epoch in range(num_epochs):
            if max_steps is not None and global_step >= max_steps:
                break
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                if max_steps is not None and global_step >= max_steps:
                    break
            
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                
                target_onehot = torch.zeros(images.size(0), brain.objects["target"].dim, device=device)
                target_onehot.scatter_(1, labels.view(-1, 1), 1.0)
                
                brain.objects["vision"].set_state(images)
                brain.objects["target"].set_state(target_onehot)
                
                internal_pred = vision_pipeline.pipeline(images)
                brain.objects["internal"].set_state(internal_pred)
                
                supervised_optimizer.zero_grad()
                logits = classification_aspect.predict(brain.objects)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                
                max_grad_norm = config.get("max_grad_norm", None)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(supervised_params, max_grad_norm)
                supervised_optimizer.step()
                
                preds = logits.argmax(dim=-1)
                batch_acc = (preds == labels).float().mean().item()
                accuracy_history.append(batch_acc)
                
                global_step += 1
                
                with torch.no_grad():
                    classification_F = classification_aspect.free_energy_contrib(brain.objects).item()
                aspect_contributions = {"ClassificationAspect": classification_F}
                
                if global_step % save_interval == 0 or global_step == planned_steps:
                    structure = brain.get_network_structure()
                    snapshot = {
                        "epoch": epoch,
                        "batch_index": batch_idx,
                        "step": global_step,
                        "free_energy": classification_F,
                        "free_energy_contributions": aspect_contributions,
                        "classification_free_energy": classification_F,
                        "accuracy": batch_acc,
                        "structure": structure,
                    }
                    snapshots.append(snapshot)
                
                avg_accuracy = sum(accuracy_history[-100:]) / min(100, len(accuracy_history))
                structure = brain.get_network_structure()
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss.item():.2f}",
                    Acc=f"{avg_accuracy*100:.1f}%",
                    Asp=structure.get('num_aspects', 0),
                    Pipe=structure.get('num_pipelines', 0),
                )
                
                if verbose and global_step % 50 == 0:
                    print(f"[Epoch {epoch+1}/{num_epochs}] Step {global_step}: "
                          f"Loss={loss.item():.4f}, Acc={batch_acc*100:.2f}%, "
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
    final_F = snapshots[-1]["free_energy"] if snapshots else 0.0
    final_accuracy = sum(accuracy_history[-100:]) / min(100, len(accuracy_history)) if accuracy_history else 0.0
    
    result = {
        "num_epochs": num_epochs,
        "trained_steps": global_step,
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
    parser.add_argument("--epochs", type=int, default=5, help="训练 epoch 数")
    parser.add_argument("--steps", type=int, default=None, help="最大训练步数（可选）")
    parser.add_argument("--state-dim", type=int, default=128, help="状态维度（默认128，与参考网络对齐）")
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
        "enable_world_model_learning": False,
        "evolution": {
            "free_energy_threshold": 2.0,  # 分类任务需要更高的阈值
            "prune_threshold": 0.01,
            "max_objects": 20,
            "max_aspects": 500,
            "error_ema_alpha": 0.5,
            "batch_growth": {
                "base": 0,  # 禁用批量创建独立 Aspects，优先使用 Pipeline
                "max_per_step": 0,  # 禁用
                "max_total": 0,  # 禁用
                "min_per_sense": 0,  # 禁用，优先使用 Pipeline
                "error_threshold": 1.5,
                "error_multiplier": 0.7,
            },
            "pipeline_growth": {
                "enable": True,
                "free_energy_trigger": 1.0,
                "min_interval": 5,  # 降低间隔，更快扩展
                "max_stages": 1,  # 只创建一个 pipeline（与参考网络对齐）
                "max_depth": 4,  # 最大深度 4（与参考网络对齐）
                "initial_width": 32,  # 32 aspects/层（与参考网络对齐）
                "width_increment": 0,  # 不增加宽度
                "initial_depth": 4,  # 初始深度 4（与参考网络对齐）
                "depth_increment": 0,  # 不增加深度
            },
        },
        "infer_lr": 0.01,
        "learning_rate": 0.001,  # 其他 Aspect 的学习率
        "classifier_learning_rate": 0.001,  # Pipeline+分类器监督学习率
        "classification_loss_weight": 1.0,
        "num_epochs": args.epochs,
        "num_infer_iters": 3,
        "max_grad_norm": 100.0,
        "state_clip_value": 5.0,
        "batch_size": 128,  # 批量大小（梯度累积，与参考网络对齐）
        "pipeline_spec": {
            "depth": 4,
            "width": 32,
        },
    }
    
    print("=" * 80)
    print("MNIST 演化实验")
    print("=" * 80)
    print(f"训练 epoch: {args.epochs}")
    print(f"状态维度: {config['state_dim']}")
    print(f"观察维度: {config['obs_dim']}")
    print(f"动作维度: {config['act_dim']}")
    print("=" * 80)
    print()
    
    result = run_experiment(
        num_epochs=args.epochs,
        config=config,
        device=device,
        max_steps=args.steps,
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

