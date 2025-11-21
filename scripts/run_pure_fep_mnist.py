#!/usr/bin/env python3
"""
纯 FEP MNIST 实验：不使用 AONN 框架，只使用生成模型和主动推理

核心组件：
1. MNIST 世界模型
2. EncoderAspect: p(state | obs)
3. ObservationAspect: p(obs | state)
4. DynamicsAspect: p(state_{t+1} | state_t, action)
5. PreferenceAspect: p(target | state) - 将分类目标转化为先验约束
6. 主动推理学习循环
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import argparse
from typing import Dict
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from aonn.models.mnist_world_model import MNISTWorldModel, MNISTWorldInterface
from aonn.aspects.encoder_aspect import EncoderAspect
from aonn.aspects.world_model_aspects import ObservationAspect, DynamicsAspect, PreferenceAspect
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.core.object import ObjectNode
from aonn.core.free_energy import compute_total_free_energy


class PureFEPMNISTClassifier:
    """
    纯 FEP MNIST 分类器：使用生成模型和主动推理
    不使用 AONN Brain，直接管理 Objects 和 Aspects
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        obs_dim: int = 784,
        action_dim: int = 10,
        device=None,
        use_conv: bool = True,
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cpu")
        self.use_conv = use_conv
        
        # 创建 Objects（不使用 AONN Brain）
        self.objects = {
            "vision": ObjectNode("vision", obs_dim, device=device),
            "internal": ObjectNode("internal", state_dim, device=device, init="normal"),
            "action": ObjectNode("action", action_dim, device=device),
            "target": ObjectNode("target", action_dim, device=device),  # one-hot 标签
        }
        
        # 创建生成模型 Aspects
        # 1. Encoder: vision -> internal
        self.encoder = EncoderAspect(
            sensory_name="vision",
            internal_name="internal",
            input_dim=obs_dim,
            output_dim=state_dim,
            use_conv=use_conv,
            image_size=28 if use_conv else None,
        ).to(device)
        
        # 2. Observation: internal -> vision
        self.observation = ObservationAspect(
            internal_name="internal",
            sensory_name="vision",
            state_dim=state_dim,
            obs_dim=obs_dim,
            use_conv=use_conv,
            image_size=28 if use_conv else None,
        ).to(device)
        
        # 3. Dynamics: internal + action -> internal
        self.dynamics = DynamicsAspect(
            internal_name="internal",
            action_name="action",
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)
        
        # 4. Preference: internal -> target (将分类目标转化为先验约束)
        # 在纯 FEP 中，分类目标通过 PreferenceAspect 作为先验约束
        self.preference = PreferenceAspect(
            internal_name="internal",
            target_name="target",
            state_dim=state_dim,
            weight=1.0,  # 先验权重
        ).to(device)
        
        # 收集所有 Aspects
        self.aspects = [
            self.encoder,
            self.observation,
            self.dynamics,
            self.preference,
        ]
        
        # 创建主动推理循环
        self.infer_loop = ActiveInferenceLoop(
            objects=self.objects,
            aspects=self.aspects,
            infer_lr=0.01,
            max_grad_norm=100.0,
            device=device,
        )
        
        # 创建分类器（仅用于评估，不参与自由能计算）
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        ).to(device)
    
    def compute_free_energy(self) -> torch.Tensor:
        """计算总自由能"""
        return compute_total_free_energy(self.objects, self.aspects)
    
    def sanitize_states(self):
        """清理状态（防止 NaN/Inf）"""
        for obj in self.objects.values():
            state = obj.state
            state = torch.clamp(state, -10.0, 10.0)
            state = torch.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
            obj.state = state
    
    def predict_class(self, num_infer_iters: int = 5) -> int:
        """
        预测类别：通过主动推理推断 internal，然后使用分类器预测
        """
        # 1. 状态推理：推断 internal（如果需要）
        if num_infer_iters > 0:
            self.infer_loop.infer_states(
                target_objects=("internal",),
                num_iters=num_infer_iters,
                sanitize_callback=self.sanitize_states,
            )
        
        # 2. 使用分类器预测
        with torch.no_grad():
            internal = self.objects["internal"].state
            logits = self.classifier(internal)
            pred_class = logits.argmax().item()
        
        return pred_class


def run_pure_fep_experiment(
    num_steps: int,
    config: Dict,
    device: torch.device,
    *,
    verbose: bool = False,
    output: str = "data/pure_fep_mnist.json",
    save_interval: int = 100,
):
    """运行纯 FEP MNIST 实验"""
    
    # 创建世界模型
    train_world = MNISTWorldModel(
        state_dim=config.get("state_dim", 128),
        action_dim=config.get("action_dim", 10),
        obs_dim=config.get("obs_dim", 784),
        device=device,
        train=True,
    )
    train_interface = MNISTWorldInterface(train_world)
    
    val_world = MNISTWorldModel(
        state_dim=config.get("state_dim", 128),
        action_dim=config.get("action_dim", 10),
        obs_dim=config.get("obs_dim", 784),
        device=device,
        train=False,
    )
    val_interface = MNISTWorldInterface(val_world)
    
    # 创建纯 FEP 分类器
    fep_system = PureFEPMNISTClassifier(
        state_dim=config.get("state_dim", 128),
        obs_dim=config.get("obs_dim", 784),
        action_dim=config.get("action_dim", 10),
        device=device,
        use_conv=config.get("use_conv", True),
    )
    
    # 创建优化器（所有生成模型参数）
    all_params = (
        list(fep_system.encoder.parameters()) +
        list(fep_system.observation.parameters()) +
        list(fep_system.dynamics.parameters()) +
        list(fep_system.preference.parameters()) +
        list(fep_system.classifier.parameters())  # 分类器也参与训练
    )
    optimizer = Adam(
        all_params,
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    
    # 实验记录
    snapshots = []
    accuracy_history = []
    free_energy_history = []
    F_obs_history = []
    F_dyn_history = []
    F_encoder_history = []
    F_pref_history = []
    
    # 初始化观察
    obs = train_interface.reset()
    prev_state = None
    prev_action = None
    
    progress = tqdm(range(num_steps), desc="Pure FEP MNIST")
    
    try:
        for step in progress:
            # 1. 设置当前观察
            fep_system.objects["vision"].set_state(obs["vision"])
            
            # 2. 设置目标标签
            target = train_interface.get_target()
            fep_system.objects["target"].set_state(target)
            
            # 3. 状态推理：推断 internal（最小化自由能）
            fep_system.infer_loop.infer_states(
                target_objects=("internal",),
                num_iters=config.get("num_infer_iters", 5),
                sanitize_callback=fep_system.sanitize_states,
            )
            
            current_state = fep_system.objects["internal"].state.clone()
            
            # 4. 行动选择：通过优化自由能选择 action
            # 在纯 FEP 中，action 可以是分类预测
            # 这里我们通过最小化自由能来优化 action
            # 简化：直接使用分类器预测作为 action
            with torch.no_grad():
                internal = fep_system.objects["internal"].state
                action_logits = fep_system.classifier(internal)
                action = torch.softmax(action_logits, dim=-1)
                fep_system.objects["action"].set_state(action)
            
            # 5. 执行行动，获取新观察
            if step > 0:
                obs, reward = train_interface.step(action)
            else:
                obs = train_interface.reset()
            
            # 6. 计算自由能组件（用于记录）
            with torch.no_grad():
                F_obs = fep_system.observation.free_energy_contrib(fep_system.objects)
                F_encoder = fep_system.encoder.free_energy_contrib(fep_system.objects)
                
                # Dynamics 需要下一状态
                if prev_state is not None and prev_action is not None:
                    temp_internal_next = ObjectNode("internal_next", fep_system.state_dim, device=device)
                    temp_internal_next.set_state(current_state)
                    temp_objects = fep_system.objects.copy()
                    temp_objects["internal_next"] = temp_internal_next
                    temp_objects["internal"] = ObjectNode("internal", fep_system.state_dim, device=device)
                    temp_objects["internal"].set_state(prev_state)
                    temp_objects["action"] = ObjectNode("action", fep_system.action_dim, device=device)
                    temp_objects["action"].set_state(prev_action)
                    F_dyn = fep_system.dynamics.free_energy_contrib(temp_objects)
                else:
                    F_dyn = torch.tensor(0.0, device=device)
                
                F_pref = fep_system.preference.free_energy_contrib(fep_system.objects)
                F_total = F_obs + F_encoder + F_dyn + F_pref
                
                F_obs_history.append(F_obs.item())
                F_encoder_history.append(F_encoder.item())
                F_dyn_history.append(F_dyn.item())
                F_pref_history.append(F_pref.item())
                free_energy_history.append(F_total.item())
            
            # 7. 参数学习（更新生成模型）
            if step > 0:
                try:
                    optimizer.zero_grad()
                    
                    # 设置下一状态（用于 Dynamics）
                    if prev_state is not None and prev_action is not None:
                        if "internal_next" not in fep_system.objects:
                            fep_system.objects["internal_next"] = ObjectNode(
                                "internal_next", fep_system.state_dim, device=device
                            )
                        fep_system.objects["internal_next"].set_state(current_state)
                        temp_prev_internal = ObjectNode("internal_prev", fep_system.state_dim, device=device)
                        temp_prev_internal.set_state(prev_state)
                        temp_prev_action = ObjectNode("action_prev", fep_system.action_dim, device=device)
                        temp_prev_action.set_state(prev_action)
                        temp_objects = fep_system.objects.copy()
                        temp_objects["internal"] = temp_prev_internal
                        temp_objects["action"] = temp_prev_action
                        F_dyn = fep_system.dynamics.free_energy_contrib(temp_objects)
                    else:
                        F_dyn = torch.tensor(0.0, device=device)
                    
                    # 计算完整自由能
                    F_obs = fep_system.observation.free_energy_contrib(fep_system.objects)
                    F_encoder = fep_system.encoder.free_energy_contrib(fep_system.objects)
                    F_pref = fep_system.preference.free_energy_contrib(fep_system.objects)
                    F_total = F_obs + F_encoder + F_dyn + F_pref
                    
                    # 分类器损失（用于训练分类器）
                    internal = fep_system.objects["internal"].state
                    logits = fep_system.classifier(internal)
                    target_class = target.argmax().item()
                    F_class = nn.functional.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([target_class], device=device),
                    )
                    
                    # 总损失 = 自由能 + 分类损失
                    total_loss = F_total + config.get("classification_weight", 1.0) * F_class
                    
                    if torch.isfinite(total_loss) and total_loss.requires_grad:
                        total_loss.backward()
                        
                        # 梯度裁剪
                        max_grad_norm = config.get("max_grad_norm", None)
                        if max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                        
                        optimizer.step()
                        fep_system.sanitize_states()
                except Exception as e:
                    if verbose:
                        print(f"Step {step}: Learning error: {e}")
                    pass
            
            # 8. 评估准确率
            with torch.no_grad():
                # 在评估时，不需要再次推理（已经在步骤3中推理过了）
                internal = fep_system.objects["internal"].state
                logits = fep_system.classifier(internal)
                pred_class = logits.argmax().item()
                true_label = train_interface.world_model.get_label()
                correct = (pred_class == true_label)
                accuracy_history.append(1.0 if correct else 0.0)
            
            # 9. 保存快照
            if step % save_interval == 0 or step == num_steps - 1:
                snapshot = {
                    "step": step,
                    "free_energy": F_total.item() if 'F_total' in locals() else 0.0,
                    "free_energy_obs": F_obs.item() if 'F_obs' in locals() else 0.0,
                    "free_energy_encoder": F_encoder.item() if 'F_encoder' in locals() else 0.0,
                    "free_energy_dyn": F_dyn.item() if 'F_dyn' in locals() else 0.0,
                    "free_energy_pref": F_pref.item() if 'F_pref' in locals() else 0.0,
                    "accuracy": sum(accuracy_history[-100:]) / min(100, len(accuracy_history)),
                }
                snapshots.append(snapshot)
                
                if verbose:
                    print(f"\n[Step {step}] F={snapshot['free_energy']:.4f} "
                          f"(obs={snapshot['free_energy_obs']:.4f}, "
                          f"enc={snapshot['free_energy_encoder']:.4f}, "
                          f"dyn={snapshot['free_energy_dyn']:.4f}, "
                          f"pref={snapshot['free_energy_pref']:.4f}), "
                          f"Acc={snapshot['accuracy']*100:.2f}%")
            
            # 更新历史状态
            prev_state = current_state.clone()
            prev_action = action.clone()
            
            # 更新进度条
            avg_acc = sum(accuracy_history[-100:]) / min(100, len(accuracy_history))
            progress.set_postfix({
                "F": f"{F_total.item():.3f}" if 'F_total' in locals() else "N/A",
                "Acc": f"{avg_acc*100:.1f}%",
            })
    
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    
    # 最终评估
    print("\n开始最终评估...")
    val_accuracy = evaluate_accuracy(
        fep_system,
        val_interface,
        num_samples=min(1000, len(val_world.dataset)),
        config=config,
        device=device,
    )
    
    # 保存结果
    result = {
        "config": config,
        "num_steps": num_steps,
        "final_free_energy": free_energy_history[-1] if free_energy_history else 0.0,
        "final_accuracy": sum(accuracy_history[-100:]) / min(100, len(accuracy_history)),
        "val_accuracy": val_accuracy,
        "snapshots": snapshots,
        "free_energy_history": free_energy_history,
        "F_obs_history": F_obs_history,
        "F_encoder_history": F_encoder_history,
        "F_dyn_history": F_dyn_history,
        "F_pref_history": F_pref_history,
        "accuracy_history": accuracy_history,
    }
    
    with open(output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 实验完成！")
    print(f"最终训练准确率: {result['final_accuracy']*100:.2f}%")
    print(f"验证准确率: {result['val_accuracy']*100:.2f}%")
    print(f"结果已保存到: {output}")
    
    return result


def evaluate_accuracy(
    fep_system: PureFEPMNISTClassifier,
    world_interface: MNISTWorldInterface,
    num_samples: int,
    config: Dict,
    device: torch.device,
) -> float:
    """评估准确率"""
    correct = 0
    
    with torch.no_grad():
        obs = world_interface.reset()
        
        for i in range(num_samples):
            # 设置观察
            fep_system.objects["vision"].set_state(obs["vision"])
            
            # 设置目标
            target = world_interface.get_target()
            fep_system.objects["target"].set_state(target)
            
            # 状态推理（在 no_grad 上下文中，但需要手动推理）
            # 简化：直接使用编码器编码，不进行主动推理
            vision_state = fep_system.objects["vision"].state
            if fep_system.use_conv:
                if vision_state.dim() == 1:
                    vision_state = vision_state.view(1, 1, 28, 28)
                elif vision_state.dim() == 2:
                    batch_size = vision_state.shape[0]
                    vision_state = vision_state.view(batch_size, 1, 28, 28)
            # 直接调用编码器的 encoder 网络
            internal = fep_system.encoder.encoder(vision_state)
            if internal.dim() > 1:
                internal = internal.squeeze(0)
            fep_system.objects["internal"].set_state(internal)
            
            # 预测
            logits = fep_system.classifier(internal)
            pred_class = logits.argmax().item()
            true_label = world_interface.world_model.get_label()
            
            if pred_class == true_label:
                correct += 1
            
            # 移动到下一个样本
            action = torch.softmax(logits, dim=-1)
            obs, _ = world_interface.step(action)
    
    accuracy = correct / num_samples
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="纯 FEP MNIST 实验")
    parser.add_argument("--steps", type=int, default=1000, help="训练步数")
    parser.add_argument("--state-dim", type=int, default=128, help="状态维度")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--output", type=str, default="data/pure_fep_mnist.json", help="输出文件路径")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--save-interval", type=int, default=100, help="快照保存间隔")
    parser.add_argument("--no-conv", action="store_true", help="不使用卷积编码器/解码器")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    config = {
        "obs_dim": 784,
        "state_dim": args.state_dim,
        "action_dim": 10,
        "use_conv": not args.no_conv,
        "infer_lr": 0.01,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "classification_weight": 1.0,  # 分类损失权重
        "num_infer_iters": 5,
        "eval_infer_iters": 1,
        "num_action_iters": 3,
        "action_lr": 0.1,
        "max_grad_norm": 100.0,
    }
    
    print("=" * 80)
    print("纯 FEP MNIST 实验")
    print("=" * 80)
    print(f"训练步数: {args.steps}")
    print(f"状态维度: {config['state_dim']}")
    print(f"使用卷积: {config['use_conv']}")
    print(f"设备: {device}")
    print()
    
    result = run_pure_fep_experiment(
        num_steps=args.steps,
        config=config,
        device=device,
        verbose=args.verbose,
        output=args.output,
        save_interval=args.save_interval,
    )
    
    return result


if __name__ == "__main__":
    main()

