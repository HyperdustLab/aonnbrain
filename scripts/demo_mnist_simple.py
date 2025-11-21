#!/usr/bin/env python3
"""
MNIST AONN 简单识别演示
直接使用训练好的模型进行识别演示
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from aonn.models.mnist_world_model import MNISTWorldModel
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.aspects.classification_aspect import ClassificationAspect
from aonn.aspects.encoder_aspect import EncoderAspect
from aonn.aspects.world_model_aspects import ObservationAspect, DynamicsAspect
from aonn.core.active_inference_loop import ActiveInferenceLoop


def setup_brain(config: dict, device: torch.device):
    """设置 brain（与训练时一致）"""
    brain = AONNBrainV3(config=config, device=device, enable_evolution=False)
    
    # 创建必要的 Objects
    if "target" not in brain.objects:
        brain.create_object("target", dim=10)
    if "action" not in brain.objects:
        brain.create_object("action", dim=10)
    
    state_dim = config.get("state_dim", 128)
    
    # 创建核心 Aspects
    encoder = EncoderAspect(
        sensory_name="vision",
        internal_name="internal",
        input_dim=784,
        output_dim=state_dim,
    )
    brain.aspects.append(encoder)
    brain.add_module("encoder", encoder)
    
    observation_aspect = ObservationAspect(
        internal_name="internal",
        sensory_name="vision",
        state_dim=state_dim,
        obs_dim=784,
    )
    brain.aspects.append(observation_aspect)
    brain.add_module("observation", observation_aspect)
    
    dynamics_aspect = DynamicsAspect(
        internal_name="internal",
        action_name="action",
        state_dim=state_dim,
        action_dim=10,
    )
    brain.aspects.append(dynamics_aspect)
    brain.add_module("dynamics", dynamics_aspect)
    
    classification_aspect = ClassificationAspect(
        internal_name="internal",
        target_name="target",
        state_dim=state_dim,
        num_classes=10,
        hidden_dim=state_dim,
    )
    brain.aspects.append(classification_aspect)
    brain.add_module("classification", classification_aspect)
    
    return brain, encoder, observation_aspect, dynamics_aspect, classification_aspect


def recognize_with_action_tracking(
    brain,
    encoder,
    observation_aspect,
    dynamics_aspect,
    classification_aspect,
    image: torch.Tensor,
    true_label: int,
    device: torch.device,
    num_infer_iters: int = 10,
    num_action_iters: int = 5,
):
    """识别数字并跟踪 action 选择过程"""
    
    # 设置观察和目标
    brain.objects["vision"].set_state(image.flatten())
    target = torch.zeros(10, device=device)
    target[true_label] = 1.0
    # 确保target维度匹配
    target_dim = brain.objects["target"].dim
    if target_dim != 10:
        if target_dim > 10:
            # 填充零
            padding = torch.zeros(target_dim - 10, device=device)
            target = torch.cat([target, padding], dim=-1)
        else:
            # 截断
            target = target[:target_dim]
    brain.objects["target"].set_state(target)
    
    # 记录过程
    action_history = []
    free_energy_history = []
    prediction_history = []
    action_selection_details = []
    
    # 步骤1: 状态推理
    loop = ActiveInferenceLoop(
        brain.objects,
        brain.aspects,
        infer_lr=0.01,
        max_grad_norm=100.0,
        device=device,
    )
    
    for iter_idx in range(num_infer_iters):
        loop.infer_states(
            target_objects=("internal",),
            num_iters=1,
            sanitize_callback=brain.sanitize_states
        )
        
        # 记录预测
        with torch.no_grad():
            logits = classification_aspect.predict(brain.objects)
            pred_probs = torch.softmax(logits, dim=-1)
            pred_class = logits.argmax().item()
            prediction_history.append({
                'iteration': iter_idx,
                'predicted_class': pred_class,
                'confidence': pred_probs[pred_class].item(),
                'all_probs': pred_probs.cpu().numpy(),
            })
        
        F = brain.compute_free_energy().item()
        free_energy_history.append(F)
    
    # 步骤2: Action 选择（通过优化自由能）
    action_logits = torch.zeros(10, device=device, requires_grad=True)
    
    for iter_idx in range(num_action_iters):
        # 设置当前 action
        current_action = torch.softmax(action_logits, dim=-1)
        brain.objects["action"].set_state(current_action)
        
        # 计算预期自由能
        current_internal = brain.objects["internal"].state
        current_obs = brain.objects["vision"].state
        
        from aonn.core.object import ObjectNode
        temp_internal = ObjectNode("internal", dim=current_internal.shape[-1], device=device)
        temp_internal.set_state(current_internal)
        
        temp_vision = ObjectNode("vision", dim=current_obs.shape[-1], device=device)
        temp_vision.set_state(current_obs)
        
        temp_target = ObjectNode("target", dim=10, device=device)
        # 确保target维度匹配
        target_to_set = target.clone()
        if target_to_set.shape[-1] != 10:
            if target_to_set.shape[-1] < 10:
                padding = torch.zeros(10 - target_to_set.shape[-1], device=device)
                target_to_set = torch.cat([target_to_set, padding], dim=-1)
            else:
                target_to_set = target_to_set[:10]
        temp_target.set_state(target_to_set)
        
        temp_action = ObjectNode("action", dim=10, device=device)
        temp_action.set_state(current_action)
        
        temp_objects = {
            "internal": temp_internal,
            "vision": temp_vision,
            "target": temp_target,
            "action": temp_action,
        }
        
        # 计算各组件自由能（保持为tensor以便反向传播）
        F_class = classification_aspect.free_energy_contrib(temp_objects)
        F_obs = observation_aspect.free_energy_contrib({
            "internal": temp_internal,
            "vision": temp_vision,
        })
        F_dyn = dynamics_aspect.free_energy_contrib({
            "internal": temp_internal,
            "action": temp_action,
        })
        
        F_total = F_class + 0.1 * F_obs + 0.1 * F_dyn
        
        # 记录 action 选择详情
        action_selection_details.append({
            'iteration': iter_idx,
            'action_probs': current_action.detach().cpu().numpy(),
            'F_total': F_total.item(),
            'F_class': F_class.item(),
            'F_obs': F_obs.item(),
            'F_dyn': F_dyn.item(),
            'selected_digit': current_action.argmax().item(),
        })
        
        # 反向传播优化 action
        if action_logits.grad is not None:
            action_logits.grad.zero_()
        
        F_total.backward(retain_graph=(iter_idx < num_action_iters - 1))
        
        if action_logits.grad is not None:
            with torch.no_grad():
                action_logits = action_logits - 0.1 * action_logits.grad
                action_logits = action_logits.detach().requires_grad_(True)
        
        action_history.append(current_action.detach().cpu().numpy())
        free_energy_history.append(F_total.item())
    
    # 最终预测
    with torch.no_grad():
        final_logits = classification_aspect.predict(brain.objects)
        final_probs = torch.softmax(final_logits, dim=-1)
        final_pred = final_logits.argmax().item()
        final_confidence = final_probs[final_pred].item()
    
    final_action = torch.softmax(action_logits.detach(), dim=-1)
    
    return {
        "true_label": true_label,
        "predicted_label": final_pred,
        "confidence": final_confidence,
        "prediction_probs": final_probs.cpu().numpy(),
        "action_probs": final_action.cpu().numpy(),
        "action_history": action_history,
        "action_selection_details": action_selection_details,
        "free_energy_history": free_energy_history,
        "prediction_history": prediction_history,
        "correct": final_pred == true_label,
    }


def visualize_recognition(image: np.ndarray, result: dict, output_path: str):
    """可视化识别过程和 action"""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4)
    
    # 1. Input Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Input Image\nTrue Label: {result["true_label"]}', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Prediction Results
    ax2 = fig.add_subplot(gs[0, 1])
    pred_probs = result["prediction_probs"]
    bars = ax2.bar(range(10), pred_probs, 
                   color=['red' if i == result["predicted_label"] else 'blue' for i in range(10)])
    ax2.set_xlabel('Digit', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title(f'Prediction\nPredicted: {result["predicted_label"]} (Conf: {result["confidence"]*100:.1f}%)', 
                   fontsize=14, fontweight='bold')
    ax2.set_xticks(range(10))
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    color = 'green' if result["correct"] else 'red'
    ax2.text(0.5, 0.95, '✓ Correct' if result["correct"] else '✗ Wrong', 
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3), fontsize=14)
    
    # 3. Action Probability Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    action_probs = result["action_probs"]
    bars = ax3.bar(range(10), action_probs, color='orange')
    ax3.set_xlabel('Action (Digit)', fontsize=12)
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title(f'Final Action\nSelected: {action_probs.argmax()}', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(10))
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)
    
    # 4. Action Evolution History
    ax4 = fig.add_subplot(gs[0, 3])
    action_history = result["action_history"]
    if action_history:
        action_array = np.array(action_history)
        for digit in range(10):
            ax4.plot(action_array[:, digit], label=f'{digit}', alpha=0.7, linewidth=2, marker='o', markersize=4)
        ax4.set_xlabel('Action Iteration', fontsize=12)
        ax4.set_ylabel('Action Probability', fontsize=12)
        ax4.set_title('Action Evolution', fontsize=14, fontweight='bold')
        ax4.legend(ncol=2, fontsize=9)
        ax4.grid(True, alpha=0.3)
    
    # 5. Free Energy Evolution
    ax5 = fig.add_subplot(gs[1, :2])
    free_energy_history = result["free_energy_history"]
    ax5.plot(free_energy_history, 'b-', linewidth=2, label='Total Free Energy')
    ax5.axvline(x=len(result["prediction_history"])-1, color='red', linestyle='--', 
                linewidth=2, label='State Inference End')
    ax5.set_xlabel('Iteration', fontsize=12)
    ax5.set_ylabel('Free Energy', fontsize=12)
    ax5.set_title('Free Energy Evolution (State Inference + Action Selection)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    if len(free_energy_history) > 0:
        ax5.text(0.02, 0.98, f'Initial: {free_energy_history[0]:.2f}\nFinal: {free_energy_history[-1]:.2f}',
                 transform=ax5.transAxes, va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Action Selection Details (Free Energy Components)
    ax6 = fig.add_subplot(gs[1, 2:])
    action_details = result["action_selection_details"]
    if action_details:
        iterations = [d['iteration'] for d in action_details]
        F_total = [d['F_total'] for d in action_details]
        F_class = [d['F_class'] for d in action_details]
        F_obs = [d['F_obs'] for d in action_details]
        F_dyn = [d['F_dyn'] for d in action_details]
        
        ax6.plot(iterations, F_total, 'b-', linewidth=2, label='F_total', marker='o')
        ax6.plot(iterations, F_class, 'r--', linewidth=1.5, label='F_class', alpha=0.7)
        ax6.plot(iterations, F_obs, 'g--', linewidth=1.5, label='F_obs', alpha=0.7)
        ax6.plot(iterations, F_dyn, 'orange', linestyle='--', linewidth=1.5, label='F_dyn', alpha=0.7)
        ax6.set_xlabel('Action Iteration', fontsize=12)
        ax6.set_ylabel('Free Energy', fontsize=12)
        ax6.set_title('Free Energy Components During Action Selection', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    # 7. Prediction Evolution
    ax7 = fig.add_subplot(gs[2, :2])
    prediction_history = result["prediction_history"]
    if prediction_history:
        pred_classes = [p['predicted_class'] for p in prediction_history]
        pred_confs = [p['confidence'] for p in prediction_history]
        ax7.plot(pred_classes, 'o-', linewidth=2, markersize=8, label='Predicted Class')
        ax7.axhline(y=result["true_label"], color='green', linestyle='--', linewidth=2, label='True Label')
        ax7.plot(pred_confs, 's-', linewidth=2, markersize=6, label='Confidence', alpha=0.7)
        ax7.set_xlabel('State Inference Iteration', fontsize=12)
        ax7.set_ylabel('Class / Confidence', fontsize=12)
        ax7.set_title('Prediction Evolution', fontsize=14, fontweight='bold')
        ax7.set_yticks(range(10))
        ax7.set_ylim([-0.5, 9.5])
        ax7.grid(True, alpha=0.3)
        ax7.legend()
    
    # 8. Action vs Prediction Comparison
    ax8 = fig.add_subplot(gs[2, 2:])
    x = np.arange(10)
    width = 0.35
    ax8.bar(x - width/2, result["prediction_probs"], width, label='Prediction', alpha=0.8, color='blue')
    ax8.bar(x + width/2, result["action_probs"], width, label='Action', alpha=0.8, color='orange')
    ax8.set_xlabel('Digit', fontsize=12)
    ax8.set_ylabel('Probability', fontsize=12)
    ax8.set_title('Prediction vs Action', fontsize=14, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_ylim([0, 1])
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'MNIST AONN Recognition Process - True: {result["true_label"]}, Predicted: {result["predicted_label"]}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Recognition visualization saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MNIST AONN 简单识别演示")
    parser.add_argument("--num-samples", type=int, default=10, help="识别样本数")
    parser.add_argument("--output-dir", type=str, default="data/recognition_demos", help="输出目录")
    parser.add_argument("--state-dim", type=int, default=128, help="状态维度")
    parser.add_argument("--num-infer-iters", type=int, default=10, help="状态推理迭代次数")
    parser.add_argument("--num-action-iters", type=int, default=5, help="Action选择迭代次数")
    
    args = parser.parse_args()
    
    device = torch.device("cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 配置
    config = {
        "obs_dim": 784,
        "state_dim": args.state_dim,
        "act_dim": 10,
        "sense_dims": {"vision": 784},
    }
    
    print("=" * 80)
    print("MNIST AONN Digit Recognition Demo")
    print("=" * 80)
    
    # 设置 brain
    brain, encoder, observation_aspect, dynamics_aspect, classification_aspect = \
        setup_brain(config, device)
    
    print("✓ Brain initialized")
    print("⚠️  Note: This is an untrained model, for demonstration only")
    print("   To use a trained model, load weights from experiment results")
    print("=" * 80)
    
    # 创建 MNIST 世界模型
    world_model = MNISTWorldModel(
        state_dim=args.state_dim,
        action_dim=10,
        obs_dim=784,
        device=device,
    )
    world_interface = world_model
    
    print(f"✓ Starting recognition of {args.num_samples} samples\n")
    
    correct_count = 0
    
    for i in range(args.num_samples):
        obs = world_interface.reset()
        true_label = world_interface.get_label()
        image = obs["vision"].reshape(28, 28).cpu().numpy()
        
        print(f"Sample {i+1}/{args.num_samples}: True Label = {true_label}")
        
        result = recognize_with_action_tracking(
            brain, encoder, observation_aspect, dynamics_aspect, classification_aspect,
            obs["vision"], true_label, device,
            num_infer_iters=args.num_infer_iters,
            num_action_iters=args.num_action_iters,
        )
        
        if result["correct"]:
            correct_count += 1
        
        print(f"  Predicted: {result['predicted_label']}, Confidence: {result['confidence']*100:.1f}%")
        print(f"  Action Selected: {result['action_probs'].argmax()}")
        print(f"  Result: {'✓ Correct' if result['correct'] else '✗ Wrong'}\n")
        
        # 可视化
        output_path = f"{args.output_dir}/demo_{i+1:03d}_true{true_label}_pred{result['predicted_label']}.png"
        visualize_recognition(image, result, output_path)
    
    print("=" * 80)
    print("Recognition Statistics")
    print("=" * 80)
    print(f"Total Samples: {args.num_samples}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count/args.num_samples*100:.1f}%")
    print(f"Visualization saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

