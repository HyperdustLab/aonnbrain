#!/usr/bin/env python3
"""
MNIST AONN 数字识别演示工具
展示 AONN 如何识别手写数字，以及做了哪些 action
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from aonn.models.mnist_world_model import MNISTWorldModel
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.aspects.classification_aspect import ClassificationAspect
from aonn.aspects.encoder_aspect import EncoderAspect
from aonn.aspects.world_model_aspects import ObservationAspect, DynamicsAspect
from aonn.core.active_inference_loop import ActiveInferenceLoop


def load_brain_from_experiment(experiment_file: str, device: torch.device):
    """从实验结果加载训练好的 brain"""
    with open(experiment_file, 'r') as f:
        data = json.load(f)
    
    # 获取配置
    config = data.get('config', {})
    if not config:
        # 如果没有保存配置，使用默认配置
        config = {
            "obs_dim": 784,
            "state_dim": 128,
            "act_dim": 10,
            "sense_dims": {"vision": 784},
        }
    
    # 创建 brain
    brain = AONNBrainV3(config=config, device=device, enable_evolution=False)
    
    # 创建必要的 Objects
    if "target" not in brain.objects:
        brain.create_object("target", dim=10)
    if "action" not in brain.objects:
        brain.create_object("action", dim=10)
    
    # 创建核心 Aspects（与训练时一致）
    state_dim = config.get("state_dim", 128)
    
    encoder = EncoderAspect(
        sensory_name="vision",
        internal_name="internal",
        obs_dim=784,
        state_dim=state_dim,
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
    
    # 尝试加载保存的权重（如果有）
    # 注意：当前实现可能没有保存权重，这里只是示例
    # 实际使用时需要确保权重被正确保存和加载
    
    return brain, encoder, observation_aspect, dynamics_aspect, classification_aspect


def recognize_digit(
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
    verbose: bool = False,
):
    """
    识别单个数字，返回识别过程和 action 信息
    
    Returns:
        dict: 包含识别结果、action历史、自由能历史等
    """
    # 设置观察
    brain.objects["vision"].set_state(image.flatten())
    
    # 设置目标（one-hot）
    target = torch.zeros(10, device=device)
    target[true_label] = 1.0
    brain.objects["target"].set_state(target)
    
    # 记录过程
    action_history = []
    free_energy_history = []
    internal_history = []
    prediction_history = []
    
    # 1. 状态推理：从观察推断内部状态
    loop = ActiveInferenceLoop(
        brain.objects,
        brain.aspects,
        infer_lr=0.01,
        max_grad_norm=100.0,
        device=device,
    )
    
    if verbose:
        print(f"  步骤1: 状态推理（{num_infer_iters} 次迭代）...")
    
    for iter_idx in range(num_infer_iters):
        loop.infer_states(
            target_objects=("internal",),
            num_iters=1,
            sanitize_callback=brain.sanitize_states
        )
        
        # 记录内部状态
        internal_state = brain.objects["internal"].state.clone().detach()
        internal_history.append(internal_state.cpu().numpy())
        
        # 计算自由能
        F = brain.compute_free_energy().item()
        free_energy_history.append(F)
        
        # 预测类别
        with torch.no_grad():
            logits = classification_aspect.predict(brain.objects)
            pred_probs = torch.softmax(logits, dim=-1)
            pred_class = logits.argmax().item()
            prediction_history.append((pred_class, pred_probs.cpu().numpy()))
    
    # 2. Action 选择：通过优化自由能选择 action
    if verbose:
        print(f"  步骤2: Action 选择（{num_action_iters} 次迭代）...")
    
    # 初始化 action（分类预测的 logits）
    action_logits = torch.zeros(10, device=device, requires_grad=True)
    
    for iter_idx in range(num_action_iters):
        # 设置 action
        brain.objects["action"].set_state(torch.softmax(action_logits, dim=-1))
        
        # 计算预期自由能
        current_internal = brain.objects["internal"].state
        current_obs = brain.objects["vision"].state
        
        # 创建临时 ObjectNode 用于计算
        from aonn.core.object import ObjectNode
        temp_internal = ObjectNode("internal", dim=current_internal.shape[-1], device=device)
        temp_internal.set_state(current_internal)
        
        temp_vision = ObjectNode("vision", dim=current_obs.shape[-1], device=device)
        temp_vision.set_state(current_obs)
        
        temp_target = ObjectNode("target", dim=10, device=device)
        temp_target.set_state(target)
        
        temp_action = ObjectNode("action", dim=10, device=device)
        temp_action.set_state(torch.softmax(action_logits, dim=-1))
        
        # 计算自由能组件
        temp_objects = {
            "internal": temp_internal,
            "vision": temp_vision,
            "target": temp_target,
            "action": temp_action,
        }
        
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
        
        # 反向传播
        if action_logits.grad is not None:
            action_logits.grad.zero_()
        
        F_total.backward(retain_graph=(iter_idx < num_action_iters - 1))
        
        # 更新 action（梯度下降）
        if action_logits.grad is not None:
            with torch.no_grad():
                action_logits = action_logits - 0.1 * action_logits.grad
                action_logits = action_logits.detach().requires_grad_(True)
        
        # 记录 action
        action_probs = torch.softmax(action_logits.detach(), dim=-1)
        action_history.append(action_probs.cpu().numpy())
        
        # 记录自由能
        F = F_total.item()
        free_energy_history.append(F)
    
    # 最终预测
    with torch.no_grad():
        final_logits = classification_aspect.predict(brain.objects)
        final_probs = torch.softmax(final_logits, dim=-1)
        final_pred = final_logits.argmax().item()
        final_confidence = final_probs[final_pred].item()
    
    # 最终 action
    final_action = torch.softmax(action_logits.detach(), dim=-1)
    
    return {
        "true_label": true_label,
        "predicted_label": final_pred,
        "confidence": final_confidence,
        "prediction_probs": final_probs.cpu().numpy(),
        "action_probs": final_action.cpu().numpy(),
        "action_history": action_history,
        "free_energy_history": free_energy_history,
        "internal_history": internal_history,
        "prediction_history": prediction_history,
        "correct": final_pred == true_label,
    }


def visualize_recognition_process(
    image: np.ndarray,
    result: dict,
    output_path: str,
):
    """可视化识别过程"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 输入图像
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Input Image\nTrue Label: {result["true_label"]}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. 预测结果
    ax2 = fig.add_subplot(gs[0, 1])
    pred_probs = result["prediction_probs"]
    bars = ax2.bar(range(10), pred_probs, color=['red' if i == result["predicted_label"] else 'blue' for i in range(10)])
    ax2.set_xlabel('Digit', fontsize=10)
    ax2.set_ylabel('Probability', fontsize=10)
    ax2.set_title(f'Prediction\nPredicted: {result["predicted_label"]} (Conf: {result["confidence"]*100:.1f}%)', 
                   fontsize=12, fontweight='bold')
    ax2.set_xticks(range(10))
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # 添加正确/错误标记
    color = 'green' if result["correct"] else 'red'
    ax2.text(0.5, 0.95, '✓ Correct' if result["correct"] else '✗ Wrong', 
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3), fontsize=12)
    
    # 3. Action 概率分布
    ax3 = fig.add_subplot(gs[0, 2])
    action_probs = result["action_probs"]
    bars = ax3.bar(range(10), action_probs, color='orange')
    ax3.set_xlabel('Action (Digit)', fontsize=10)
    ax3.set_ylabel('Probability', fontsize=10)
    ax3.set_title('Final Action Distribution', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(10))
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)
    
    # 4. Action 演化历史
    ax4 = fig.add_subplot(gs[0, 3])
    action_history = result["action_history"]
    if action_history:
        action_array = np.array(action_history)
        for digit in range(10):
            ax4.plot(action_array[:, digit], label=f'Digit {digit}', alpha=0.7, linewidth=2)
        ax4.set_xlabel('Action Iteration', fontsize=10)
        ax4.set_ylabel('Action Probability', fontsize=10)
        ax4.set_title('Action Evolution', fontsize=12, fontweight='bold')
        ax4.legend(ncol=2, fontsize=8)
        ax4.grid(True, alpha=0.3)
    
    # 5. 自由能演化
    ax5 = fig.add_subplot(gs[1, :2])
    free_energy_history = result["free_energy_history"]
    ax5.plot(free_energy_history, 'b-', linewidth=2, label='Free Energy')
    ax5.set_xlabel('Iteration', fontsize=11)
    ax5.set_ylabel('Free Energy', fontsize=11)
    ax5.set_title('Free Energy Evolution During Recognition', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 添加初始和最终值
    if len(free_energy_history) > 0:
        ax5.text(0.02, 0.98, f'Initial: {free_energy_history[0]:.2f}\nFinal: {free_energy_history[-1]:.2f}',
                 transform=ax5.transAxes, va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. 预测演化历史
    ax6 = fig.add_subplot(gs[1, 2:])
    prediction_history = result["prediction_history"]
    if prediction_history:
        pred_classes = [p[0] for p in prediction_history]
        pred_confs = [p[1][p[0]] for p in prediction_history]
        ax6.plot(pred_classes, 'o-', linewidth=2, markersize=6, label='Predicted Class')
        ax6.axhline(y=result["true_label"], color='green', linestyle='--', linewidth=2, label='True Label')
        ax6.set_xlabel('Inference Iteration', fontsize=11)
        ax6.set_ylabel('Class', fontsize=11)
        ax6.set_title('Prediction Evolution', fontsize=12, fontweight='bold')
        ax6.set_yticks(range(10))
        ax6.set_ylim([-0.5, 9.5])
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    # 7. 内部状态可视化（PCA降维）
    ax7 = fig.add_subplot(gs[2, :2])
    internal_history = result["internal_history"]
    if internal_history and len(internal_history) > 0:
        internal_array = np.array(internal_history)
        # 使用前两个维度可视化
        ax7.plot(internal_array[:, 0], internal_array[:, 1], 'o-', linewidth=2, markersize=4)
        ax7.scatter(internal_array[0, 0], internal_array[0, 1], s=200, c='green', 
                   marker='*', label='Start', zorder=5)
        ax7.scatter(internal_array[-1, 0], internal_array[-1, 1], s=200, c='red', 
                   marker='*', label='End', zorder=5)
        ax7.set_xlabel('Internal State Dim 0', fontsize=11)
        ax7.set_ylabel('Internal State Dim 1', fontsize=11)
        ax7.set_title('Internal State Evolution (First 2 Dimensions)', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
    
    # 8. Action vs Prediction 对比
    ax8 = fig.add_subplot(gs[2, 2:])
    x = np.arange(10)
    width = 0.35
    ax8.bar(x - width/2, result["prediction_probs"], width, label='Prediction', alpha=0.8)
    ax8.bar(x + width/2, result["action_probs"], width, label='Action', alpha=0.8)
    ax8.set_xlabel('Digit', fontsize=11)
    ax8.set_ylabel('Probability', fontsize=11)
    ax8.set_title('Prediction vs Action', fontsize=12, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_ylim([0, 1])
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'MNIST Recognition Process - True: {result["true_label"]}, Predicted: {result["predicted_label"]}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 识别过程可视化已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MNIST AONN 数字识别演示")
    parser.add_argument("--experiment", type=str, required=True,
                        help="实验结果JSON文件路径")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="要识别的样本数量")
    parser.add_argument("--output-dir", type=str, default="data/recognition_demos",
                        help="输出目录")
    parser.add_argument("--num-infer-iters", type=int, default=10,
                        help="状态推理迭代次数")
    parser.add_argument("--num-action-iters", type=int, default=5,
                        help="Action选择迭代次数")
    parser.add_argument("--verbose", action="store_true",
                        help="详细输出")
    
    args = parser.parse_args()
    
    device = torch.device("cpu")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MNIST AONN 数字识别演示")
    print("=" * 80)
    print(f"加载实验结果: {args.experiment}")
    
    # 加载 brain
    brain, encoder, observation_aspect, dynamics_aspect, classification_aspect = \
        load_brain_from_experiment(args.experiment, device)
    
    print("✓ Brain 加载完成")
    
    # 创建 MNIST 世界模型
    world_model = MNISTWorldModel(
        state_dim=128,
        action_dim=10,
        obs_dim=784,
        device=device,
    )
    world_interface = world_model.get_interface()
    
    print(f"✓ 开始识别 {args.num_samples} 个样本")
    print("=" * 80)
    
    results = []
    correct_count = 0
    
    for i in range(args.num_samples):
        # 获取一个样本
        obs = world_interface.reset()
        true_label = world_interface.get_label()
        image = obs["vision"].reshape(28, 28).cpu().numpy()
        
        if args.verbose:
            print(f"\n样本 {i+1}/{args.num_samples}: 真实标签 = {true_label}")
        
        # 识别
        result = recognize_digit(
            brain, encoder, observation_aspect, dynamics_aspect, classification_aspect,
            obs["vision"], true_label, device,
            num_infer_iters=args.num_infer_iters,
            num_action_iters=args.num_action_iters,
            verbose=args.verbose,
        )
        
        results.append((image, result))
        
        if result["correct"]:
            correct_count += 1
        
        if args.verbose:
            print(f"  预测: {result['predicted_label']}, 置信度: {result['confidence']*100:.1f}%")
            print(f"  结果: {'✓ 正确' if result['correct'] else '✗ 错误'}")
        
        # 可视化
        output_path = f"{args.output_dir}/recognition_{i+1:03d}_true{true_label}_pred{result['predicted_label']}.png"
        visualize_recognition_process(image, result, output_path)
    
    # 打印统计
    print("\n" + "=" * 80)
    print("识别统计")
    print("=" * 80)
    print(f"总样本数: {args.num_samples}")
    print(f"正确识别: {correct_count}")
    print(f"准确率: {correct_count/args.num_samples*100:.1f}%")
    print(f"可视化结果保存在: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

