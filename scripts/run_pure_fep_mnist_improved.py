#!/usr/bin/env python3
"""
纯 FEP MNIST 实验（改进版）：解决编码器和解码器收敛问题

核心改进：
1. 方案2：使用编码器输出作为 internal 的初始值，减少状态推理迭代次数
2. 方案5：对于静态分类任务，提供选项直接使用编码器输出（跳过状态推理）
3. 调整自由能权重，平衡编码器和解码器的学习
4. 使用分离优化器，为不同组件设置不同学习率
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
    纯 FEP MNIST 分类器（改进版）
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
        
        # 创建 Objects
        self.objects = {
            "vision": ObjectNode("vision", obs_dim, device=device),
            "internal": ObjectNode("internal", state_dim, device=device, init="normal"),
            "action": ObjectNode("action", action_dim, device=device),
            "target": ObjectNode("target", action_dim, device=device),
        }
        
        # 创建生成模型 Aspects
        self.encoder = EncoderAspect(
            sensory_name="vision",
            internal_name="internal",
            input_dim=obs_dim,
            output_dim=state_dim,
            use_conv=use_conv,
            image_size=28 if use_conv else None,
        ).to(device)
        
        self.observation = ObservationAspect(
            internal_name="internal",
            sensory_name="vision",
            state_dim=state_dim,
            obs_dim=obs_dim,
            use_conv=use_conv,
            image_size=28 if use_conv else None,
        ).to(device)
        
        self.dynamics = DynamicsAspect(
            internal_name="internal",
            action_name="action",
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)
        
        self.preference = PreferenceAspect(
            internal_name="internal",
            target_name="target",
            state_dim=state_dim,
            weight=1.0,
        ).to(device)
        
        self.aspects = [
            self.encoder,
            self.observation,
            self.dynamics,
            self.preference,
        ]
        
        # 主动推理循环
        self.infer_loop = ActiveInferenceLoop(
            objects=self.objects,
            aspects=self.aspects,
            infer_lr=0.01,
            max_grad_norm=10.0,
        )
        
        # 独立分类器（用于评估，不参与自由能计算）
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).to(device)
    
    def compute_free_energy(self):
        """计算总自由能"""
        return compute_total_free_energy(self.objects, self.aspects)
    
    def sanitize_states(self):
        """清理状态（防止 NaN/Inf）"""
        for obj in self.objects.values():
            state = obj.state
            if torch.isnan(state).any() or torch.isinf(state).any():
                obj.state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            obj.state = torch.clamp(obj.state, -10.0, 10.0)
    
    def predict_class(self, vision_state: torch.Tensor) -> int:
        """预测类别"""
        # 使用编码器直接输出
        with torch.no_grad():
            if vision_state.dim() == 1:
                vision_state = vision_state.unsqueeze(0)
            if self.use_conv:
                vision_reshaped = vision_state.view(-1, 1, 28, 28)
            else:
                vision_reshaped = vision_state
            
            internal = self.encoder.encoder(vision_reshaped)
            if internal.dim() > 1:
                internal = internal.squeeze(0)
            
            logits = self.classifier(internal)
            return logits.argmax(dim=-1).item()


def evaluate_accuracy(
    fep_system: PureFEPMNISTClassifier,
    world_interface: MNISTWorldInterface,
    num_samples: int = 1000,
    device=None,
):
    """评估准确率"""
    # 设置为评估模式（禁用dropout等）
    for aspect in fep_system.aspects:
        if isinstance(aspect, nn.Module):
            aspect.eval()
    fep_system.classifier.eval()
    
    correct = 0
    
    with torch.no_grad():
        for i in range(num_samples):
            obs = world_interface.reset()
            target = world_interface.get_target()
            true_label = target.argmax().item()
            
            # 使用编码器直接输出（不进行状态推理）
            vision_state = obs["vision"]
            pred_class = fep_system.predict_class(vision_state)
            
            if pred_class == true_label:
                correct += 1
    
    # 恢复训练模式
    for aspect in fep_system.aspects:
        if isinstance(aspect, nn.Module):
            aspect.train()
    fep_system.classifier.train()
    
    accuracy = correct / num_samples
    return accuracy


def run_pure_fep_experiment(
    num_steps: int,
    config: Dict,
    device: torch.device,
    *,
    verbose: bool = False,
    output: str = "data/pure_fep_mnist_improved.json",
    save_interval: int = 100,
):
    """运行改进版纯 FEP MNIST 实验"""
    
    # 配置参数
    use_encoder_init = config.get("use_encoder_init", True)  # 方案2：使用编码器初始化
    skip_inference = config.get("skip_inference", False)  # 方案5：跳过状态推理
    num_infer_iters = config.get("num_infer_iters", 2 if use_encoder_init else 5)  # 减少迭代次数
    
    # 自由能权重
    obs_weight = config.get("obs_weight", 0.1)  # 降低观察重建权重
    encoder_weight = config.get("encoder_weight", 1.0)  # 保持编码器权重
    pref_weight = config.get("pref_weight", 10.0)  # 提高分类先验权重
    
    print("=" * 80)
    print("纯 FEP MNIST 实验（改进版）")
    print("=" * 80)
    print(f"\n📋 配置:")
    print(f"  使用编码器初始化: {use_encoder_init}")
    print(f"  跳过状态推理: {skip_inference}")
    print(f"  状态推理迭代次数: {num_infer_iters}")
    print(f"  自由能权重: F_obs={obs_weight}, F_encoder={encoder_weight}, F_pref={pref_weight}")
    print()
    
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
    
    # 创建 FEP 系统
    fep_system = PureFEPMNISTClassifier(
        state_dim=config.get("state_dim", 128),
        obs_dim=config.get("obs_dim", 784),
        action_dim=config.get("action_dim", 10),
        device=device,
        use_conv=config.get("use_conv", True),
    )
    
    # 创建分离优化器（方案4）
    encoder_optimizer = Adam(
        list(fep_system.encoder.parameters()),
        lr=config.get("encoder_lr", 0.001),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    
    observation_optimizer = Adam(
        list(fep_system.observation.parameters()),
        lr=config.get("observation_lr", 0.0001),  # 更低的学习率
        weight_decay=config.get("weight_decay", 1e-4),
    )
    
    preference_optimizer = Adam(
        list(fep_system.preference.parameters()),
        lr=config.get("preference_lr", 0.01),  # 更高的学习率
        weight_decay=config.get("weight_decay", 1e-4),
    )
    
    classifier_optimizer = Adam(
        list(fep_system.classifier.parameters()),
        lr=config.get("classifier_lr", 0.001),
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
    
    progress = tqdm(range(num_steps), desc="Pure FEP MNIST (Improved)")
    
    try:
        for step in progress:
            # 1. 设置当前观察
            fep_system.objects["vision"].set_state(obs["vision"])
            
            # 2. 设置目标标签
            target = train_interface.get_target()
            fep_system.objects["target"].set_state(target)
            
            # 3. 状态推理：推断 internal
            if skip_inference:
                # 方案5：直接使用编码器输出，跳过状态推理
                with torch.no_grad():
                    vision_state = fep_system.objects["vision"].state
                    if vision_state.dim() == 1:
                        vision_state = vision_state.unsqueeze(0)
                    if fep_system.use_conv:
                        vision_reshaped = vision_state.view(-1, 1, 28, 28)
                    else:
                        vision_reshaped = vision_state
                    
                    internal_pred = fep_system.encoder.encoder(vision_reshaped)
                    if internal_pred.dim() > 1:
                        internal_pred = internal_pred.squeeze(0)
                    
                    fep_system.objects["internal"].set_state(internal_pred)
            else:
                # 方案2：使用编码器输出作为初始值
                if use_encoder_init:
                    with torch.no_grad():
                        vision_state = fep_system.objects["vision"].state
                        if vision_state.dim() == 1:
                            vision_state = vision_state.unsqueeze(0)
                        if fep_system.use_conv:
                            vision_reshaped = vision_state.view(-1, 1, 28, 28)
                        else:
                            vision_reshaped = vision_state
                        
                        internal_init = fep_system.encoder.encoder(vision_reshaped)
                        if internal_init.dim() > 1:
                            internal_init = internal_init.squeeze(0)
                        
                        # 设置为需要梯度的叶子张量
                        fep_system.objects["internal"].set_state(
                            internal_init.detach().requires_grad_(True)
                        )
                
                # 进行少量迭代优化
                fep_system.infer_loop.infer_states(
                    target_objects=("internal",),
                    num_iters=num_infer_iters,
                    sanitize_callback=fep_system.sanitize_states,
                )
            
            current_state = fep_system.objects["internal"].state.clone()
            
            # 4. 行动选择（简化：直接使用分类器预测）
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
                F_total = obs_weight * F_obs + encoder_weight * F_encoder + F_dyn + pref_weight * F_pref
                
                F_obs_history.append(F_obs.item())
                F_encoder_history.append(F_encoder.item())
                F_dyn_history.append(F_dyn.item())
                F_pref_history.append(F_pref.item())
                free_energy_history.append(F_total.item())
            
            # 7. 参数学习（使用分离优化器）
            if step > 0:
                try:
                    # 编码器学习
                    encoder_optimizer.zero_grad()
                    F_encoder = encoder_weight * fep_system.encoder.free_energy_contrib(fep_system.objects)
                    if torch.isfinite(F_encoder) and F_encoder.requires_grad:
                        F_encoder.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(fep_system.encoder.parameters(), config.get("max_grad_norm", 100.0))
                        encoder_optimizer.step()
                    
                    # 解码器学习
                    observation_optimizer.zero_grad()
                    F_obs = obs_weight * fep_system.observation.free_energy_contrib(fep_system.objects)
                    if torch.isfinite(F_obs) and F_obs.requires_grad:
                        F_obs.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(fep_system.observation.parameters(), config.get("max_grad_norm", 100.0))
                        observation_optimizer.step()
                    
                    # 先验学习
                    preference_optimizer.zero_grad()
                    F_pref = pref_weight * fep_system.preference.free_energy_contrib(fep_system.objects)
                    if torch.isfinite(F_pref) and F_pref.requires_grad:
                        F_pref.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(fep_system.preference.parameters(), config.get("max_grad_norm", 100.0))
                        preference_optimizer.step()
                    
                    # 分类器学习
                    classifier_optimizer.zero_grad()
                    internal = fep_system.objects["internal"].state
                    logits = fep_system.classifier(internal)
                    target_class = target.argmax().item()
                    F_class = nn.functional.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([target_class], device=device),
                    )
                    if torch.isfinite(F_class) and F_class.requires_grad:
                        F_class.backward()
                        torch.nn.utils.clip_grad_norm_(fep_system.classifier.parameters(), config.get("max_grad_norm", 100.0))
                        classifier_optimizer.step()
                    
                    fep_system.sanitize_states()
                except Exception as e:
                    if verbose:
                        print(f"步骤 {step} 学习错误: {e}")
            
            # 8. 评估准确率
            if (step + 1) % config.get("eval_interval", 100) == 0:
                acc = evaluate_accuracy(fep_system, train_interface, num_samples=100, device=device)
                accuracy_history.append(acc)
                
                # 更新进度条
                F = free_energy_history[-1] if free_energy_history else 0.0
                progress.set_postfix({"F": f"{F:.3f}", "Acc": f"{acc*100:.1f}%"})
            
            # 9. 保存快照
            if (step + 1) % save_interval == 0:
                snapshots.append({
                    "step": step + 1,
                    "free_energy": free_energy_history[-1] if free_energy_history else 0.0,
                    "free_energy_obs": F_obs_history[-1] if F_obs_history else 0.0,
                    "free_energy_encoder": F_encoder_history[-1] if F_encoder_history else 0.0,
                    "free_energy_dyn": F_dyn_history[-1] if F_dyn_history else 0.0,
                    "free_energy_pref": F_pref_history[-1] if F_pref_history else 0.0,
                    "accuracy": accuracy_history[-1] if accuracy_history else 0.0,
                })
            
            prev_state = current_state.clone()
            prev_action = action.clone()
    
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    
    # 最终评估
    print("\n评估最终准确率...")
    final_acc = evaluate_accuracy(fep_system, train_interface, num_samples=1000, device=device)
    val_acc = evaluate_accuracy(fep_system, val_interface, num_samples=1000, device=device)
    
    # 保存模型权重
    model_output = output.replace('.json', '_model.pth')
    checkpoint = {
        "config": config,
        "encoder": fep_system.encoder.state_dict(),
        "observation": fep_system.observation.state_dict(),
        "dynamics": fep_system.dynamics.state_dict(),
        "preference": fep_system.preference.state_dict(),
        "classifier": fep_system.classifier.state_dict(),
    }
    torch.save(checkpoint, model_output)
    print(f"✅ 模型权重已保存到: {model_output}")
    
    # 保存结果
    results = {
        "config": config,
        "num_steps": num_steps,
        "final_free_energy": free_energy_history[-1] if free_energy_history else 0.0,
        "final_accuracy": final_acc,
        "val_accuracy": val_acc,
        "model_path": model_output,
        "snapshots": snapshots,
        "free_energy_history": free_energy_history,
        "accuracy_history": accuracy_history,
        "F_obs_history": F_obs_history,
        "F_encoder_history": F_encoder_history,
        "F_dyn_history": F_dyn_history,
        "F_pref_history": F_pref_history,
    }
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output}")
    print(f"   最终自由能: {results['final_free_energy']:.4f}")
    print(f"   最终准确率: {results['final_accuracy']*100:.2f}%")
    print(f"   验证准确率: {results['val_accuracy']*100:.2f}%")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="纯 FEP MNIST 实验（改进版）")
    parser.add_argument("--steps", type=int, default=1000, help="训练步数")
    parser.add_argument("--output", type=str, default="data/pure_fep_mnist_improved.json", help="输出文件")
    parser.add_argument("--save-interval", type=int, default=100, help="保存间隔")
    parser.add_argument("--use-encoder-init", action="store_true", default=True, help="使用编码器初始化")
    parser.add_argument("--skip-inference", action="store_true", default=False, help="跳过状态推理（直接使用编码器输出）")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    config = {
        "obs_dim": 784,
        "state_dim": 128,
        "action_dim": 10,
        "use_conv": True,
        "infer_lr": 0.01,
        "learning_rate": 0.001,
        "encoder_lr": 0.001,
        "observation_lr": 0.0001,  # 更低的学习率
        "preference_lr": 0.01,  # 更高的学习率
        "classifier_lr": 0.001,
        "weight_decay": 1e-4,
        "classification_weight": 1.0,
        "num_infer_iters": 2,  # 减少迭代次数
        "eval_infer_iters": 1,
        "num_action_iters": 3,
        "action_lr": 0.1,
        "max_grad_norm": 100.0,
        "eval_interval": 100,
        # 改进参数
        "use_encoder_init": args.use_encoder_init,
        "skip_inference": args.skip_inference,
        "obs_weight": 0.1,  # 降低观察重建权重
        "encoder_weight": 1.0,  # 保持编码器权重
        "pref_weight": 10.0,  # 提高分类先验权重
    }
    
    run_pure_fep_experiment(
        num_steps=args.steps,
        config=config,
        device=device,
        verbose=args.verbose,
        output=args.output,
        save_interval=args.save_interval,
    )

