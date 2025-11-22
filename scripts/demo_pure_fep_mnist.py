#!/usr/bin/env python3
"""
纯 FEP MNIST 数字识别演示工具（改进版）
使用改进版的纯 FEP 系统进行数字识别演示
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

from aonn.models.mnist_world_model import MNISTWorldModel, MNISTWorldInterface
from aonn.aspects.encoder_aspect import EncoderAspect
from aonn.aspects.world_model_aspects import ObservationAspect, DynamicsAspect, PreferenceAspect
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.core.object import ObjectNode
from aonn.core.free_energy import compute_total_free_energy


class PureFEPMNISTClassifier:
    """纯 FEP MNIST 分类器（与改进版实验一致）"""
    
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
        
        # 独立分类器
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).to(device)
    
    def compute_free_energy(self):
        """计算总自由能"""
        return compute_total_free_energy(self.objects, self.aspects)
    
    def sanitize_states(self):
        """清理状态"""
        for obj in self.objects.values():
            state = obj.state
            if torch.isnan(state).any() or torch.isinf(state).any():
                obj.state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            obj.state = torch.clamp(obj.state, -10.0, 10.0)
    
    def predict_class(self, vision_state: torch.Tensor) -> int:
        """预测类别"""
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


def load_model_from_experiment(experiment_file: str, device: torch.device):
    """从实验结果加载模型"""
    with open(experiment_file, 'r') as f:
        data = json.load(f)
    
    config = data.get('config', {})
    
    # 创建 FEP 系统
    fep_system = PureFEPMNISTClassifier(
        state_dim=config.get("state_dim", 128),
        obs_dim=config.get("obs_dim", 784),
        action_dim=config.get("action_dim", 10),
        device=device,
        use_conv=config.get("use_conv", True),
    )
    
    # 尝试加载保存的模型权重
    model_path = data.get('model_path', None)
    if model_path and Path(model_path).exists():
        print(f"  加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        fep_system.encoder.load_state_dict(checkpoint['encoder'])
        fep_system.observation.load_state_dict(checkpoint['observation'])
        fep_system.dynamics.load_state_dict(checkpoint['dynamics'])
        fep_system.preference.load_state_dict(checkpoint['preference'])
        fep_system.classifier.load_state_dict(checkpoint['classifier'])
        print("  ✓ 模型权重加载成功")
    else:
        # 尝试从实验结果文件名推断模型路径
        model_path = experiment_file.replace('.json', '_model.pth')
        if Path(model_path).exists():
            print(f"  加载模型权重: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            fep_system.encoder.load_state_dict(checkpoint['encoder'])
            fep_system.observation.load_state_dict(checkpoint['observation'])
            fep_system.dynamics.load_state_dict(checkpoint['dynamics'])
            fep_system.preference.load_state_dict(checkpoint['preference'])
            fep_system.classifier.load_state_dict(checkpoint['classifier'])
            print("  ✓ 模型权重加载成功")
        else:
            print("  ⚠️  未找到模型权重文件，使用随机初始化的模型")
            print("     要使用训练好的模型，请先运行实验脚本保存权重")
    
    return fep_system


def recognize_digit(
    fep_system: PureFEPMNISTClassifier,
    image: torch.Tensor,
    true_label: int,
    device: torch.device,
    num_infer_iters: int = 5,
    use_encoder_init: bool = True,
    verbose: bool = False,
):
    """识别单个数字"""
    # 设置观察
    fep_system.objects["vision"].set_state(image.flatten())
    
    # 设置目标（one-hot）
    target = torch.zeros(10, device=device)
    target[true_label] = 1.0
    fep_system.objects["target"].set_state(target.unsqueeze(0))
    
    # 记录过程
    free_energy_history = []
    prediction_history = []
    internal_history = []
    F_obs_history = []
    F_encoder_history = []
    F_pref_history = []
    
    # 使用编码器初始化（改进版策略）
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
            
            fep_system.objects["internal"].set_state(
                internal_init.detach().requires_grad_(True)
            )
    
    # 状态推理
    if verbose:
        print(f"  状态推理（{num_infer_iters} 次迭代）...")
    
    for iter_idx in range(num_infer_iters):
        fep_system.infer_loop.infer_states(
            target_objects=("internal",),
            num_iters=1,
            sanitize_callback=fep_system.sanitize_states,
        )
        
        # 记录内部状态
        internal_state = fep_system.objects["internal"].state.clone().detach()
        internal_history.append(internal_state.cpu().numpy())
        
        # 计算自由能
        with torch.no_grad():
            F_total = fep_system.compute_free_energy()
            F_obs = fep_system.observation.free_energy_contrib(fep_system.objects)
            F_encoder = fep_system.encoder.free_energy_contrib(fep_system.objects)
            F_pref = fep_system.preference.free_energy_contrib(fep_system.objects)
            
            free_energy_history.append(F_total.item())
            F_obs_history.append(F_obs.item())
            F_encoder_history.append(F_encoder.item())
            F_pref_history.append(F_pref.item())
        
        # 预测类别
        with torch.no_grad():
            internal = fep_system.objects["internal"].state
            logits = fep_system.classifier(internal)
            pred_probs = torch.softmax(logits, dim=-1)
            pred_class = logits.argmax().item()
            confidence = pred_probs[pred_class].item()
            prediction_history.append((pred_class, pred_probs.cpu().numpy(), confidence))
    
    # 最终预测
    with torch.no_grad():
        internal = fep_system.objects["internal"].state
        logits = fep_system.classifier(internal)
        pred_probs = torch.softmax(logits, dim=-1)
        pred_class = logits.argmax().item()
        confidence = pred_probs[pred_class].item()
    
    return {
        "predicted_label": pred_class,
        "confidence": confidence,
        "pred_probs": pred_probs.cpu().numpy(),
        "correct": pred_class == true_label,
        "free_energy_history": free_energy_history,
        "prediction_history": prediction_history,
        "internal_history": internal_history,
        "F_obs_history": F_obs_history,
        "F_encoder_history": F_encoder_history,
        "F_pref_history": F_pref_history,
    }


def visualize_recognition(image: np.ndarray, result: dict, output_path: str):
    """可视化识别过程"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 输入图像
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Input Image\nTrue Label: {result.get("true_label", "?")}', fontsize=12)
    ax.axis('off')
    
    # 2. 预测概率分布
    ax = fig.add_subplot(gs[0, 1])
    pred_probs = result["pred_probs"]
    colors = ['red' if i == result["predicted_label"] else 'blue' for i in range(10)]
    ax.bar(range(10), pred_probs, color=colors, alpha=0.7)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Probability')
    ax.set_title(f'Prediction\nPredicted: {result["predicted_label"]}\nConfidence: {result["confidence"]*100:.1f}%', fontsize=12)
    ax.set_xticks(range(10))
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # 3. 自由能演化
    ax = fig.add_subplot(gs[0, 2])
    F_history = result["free_energy_history"]
    ax.plot(F_history, 'b-', linewidth=2, label='Total Free Energy')
    ax.set_xlabel('Inference Iteration')
    ax.set_ylabel('Free Energy')
    ax.set_title('Free Energy Evolution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 预测演化
    ax = fig.add_subplot(gs[1, 0])
    pred_history = result["prediction_history"]
    pred_classes = [p[0] for p in pred_history]
    confidences = [p[2] for p in pred_history]
    ax.plot(pred_classes, 'o-', linewidth=2, markersize=6, label='Predicted Class')
    ax.set_xlabel('Inference Iteration')
    ax.set_ylabel('Predicted Class')
    ax.set_title('Prediction Evolution', fontsize=12)
    ax.set_yticks(range(10))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 置信度演化
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(confidences, 'g-', linewidth=2, label='Confidence')
    ax.set_xlabel('Inference Iteration')
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence Evolution', fontsize=12)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 自由能组件
    ax = fig.add_subplot(gs[1, 2])
    F_obs = result["F_obs_history"]
    F_encoder = result["F_encoder_history"]
    F_pref = result["F_pref_history"]
    ax.plot(F_obs, label='F_obs', linewidth=2, alpha=0.7)
    ax.plot(F_encoder, label='F_encoder', linewidth=2, alpha=0.7)
    ax.plot(F_pref, label='F_pref', linewidth=2, alpha=0.7)
    ax.set_xlabel('Inference Iteration')
    ax.set_ylabel('Free Energy Component')
    ax.set_title('Free Energy Components', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 7. 内部状态演化（使用前两个维度）
    ax = fig.add_subplot(gs[2, 0])
    internal_history = result["internal_history"]
    if len(internal_history) > 0:
        # 使用前两个维度进行可视化
        internal_array = np.array(internal_history)
        if internal_array.shape[1] >= 2:
            ax.plot(internal_array[:, 0], internal_array[:, 1], 'o-', linewidth=2, markersize=6)
            ax.scatter(internal_array[0, 0], internal_array[0, 1], color='green', s=100, marker='s', label='Start', zorder=5)
            ax.scatter(internal_array[-1, 0], internal_array[-1, 1], color='red', s=100, marker='*', label='End', zorder=5)
            ax.set_xlabel('Dimension 0')
            ax.set_ylabel('Dimension 1')
            ax.set_title('Internal State Evolution', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # 如果维度不足，显示状态范数
            norms = [np.linalg.norm(s) for s in internal_history]
            ax.plot(norms, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Inference Iteration')
            ax.set_ylabel('State Norm')
            ax.set_title('Internal State Norm Evolution', fontsize=12)
            ax.grid(True, alpha=0.3)
    
    # 8. 预测概率热图
    ax = fig.add_subplot(gs[2, 1])
    pred_probs_history = np.array([p[1] for p in pred_history])
    im = ax.imshow(pred_probs_history.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Inference Iteration')
    ax.set_ylabel('Digit Class')
    ax.set_title('Prediction Probability Heatmap', fontsize=12)
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax)
    
    # 9. 结果总结
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    result_text = f"""
Recognition Result

True Label: {result.get('true_label', '?')}
Predicted: {result['predicted_label']}
Confidence: {result['confidence']*100:.1f}%

Status: {'✓ CORRECT' if result['correct'] else '✗ WRONG'}

Initial Free Energy: {result['free_energy_history'][0]:.4f}
Final Free Energy: {result['free_energy_history'][-1]:.4f}
Reduction: {(result['free_energy_history'][0] - result['free_energy_history'][-1]) / result['free_energy_history'][0] * 100:.1f}%

Initial Prediction: {pred_history[0][0]}
Final Prediction: {pred_history[-1][0]}
    """
    ax.text(0.1, 0.5, result_text, fontsize=11, family='monospace', verticalalignment='center')
    
    plt.suptitle(f'Pure FEP MNIST Recognition Demo\nTrue: {result.get("true_label", "?")}, Predicted: {result["predicted_label"]}, {"✓ Correct" if result["correct"] else "✗ Wrong"}', 
                 fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="纯 FEP MNIST 数字识别演示（改进版）")
    parser.add_argument("--experiment", type=str, 
                        default="data/pure_fep_mnist_improved_60000steps.json",
                        help="实验结果JSON文件路径")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="要识别的样本数量")
    parser.add_argument("--output-dir", type=str, default="data/recognition_demos",
                        help="输出目录")
    parser.add_argument("--num-infer-iters", type=int, default=5,
                        help="状态推理迭代次数")
    parser.add_argument("--use-encoder-init", action="store_true", default=True,
                        help="使用编码器初始化")
    parser.add_argument("--verbose", action="store_true",
                        help="详细输出")
    parser.add_argument("--target-labels", type=int, nargs='+', default=None,
                        help="指定要识别的真实标签列表（用于重现特定样本）")
    
    args = parser.parse_args()
    
    device = torch.device("cpu")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("纯 FEP MNIST 数字识别演示（改进版）")
    print("=" * 80)
    print(f"加载实验结果: {args.experiment}")
    
    # 加载模型
    fep_system = load_model_from_experiment(args.experiment, device)
    print("✓ 模型加载完成")
    
    # 创建 MNIST 世界模型
    world_model = MNISTWorldModel(
        state_dim=128,
        action_dim=10,
        obs_dim=784,
        device=device,
        train=False,  # 使用测试集
    )
    world_interface = MNISTWorldInterface(world_model)
    
    print(f"✓ 开始识别 {args.num_samples} 个样本")
    print("=" * 80)
    
    results = []
    correct_count = 0
    
    # 如果指定了特定样本，使用固定种子
    target_labels = getattr(args, 'target_labels', None)
    if target_labels:
        # 使用指定的真实标签
        import random
        random.seed(42)  # 固定种子以确保可重复
        torch.manual_seed(42)
    
    for i in range(args.num_samples):
        # 获取一个样本
        if target_labels and i < len(target_labels):
            # 如果指定了目标标签，尝试找到匹配的样本
            target_label = target_labels[i]
            max_attempts = 1000
            for attempt in range(max_attempts):
                obs = world_interface.reset()
                target = world_interface.get_target()
                true_label = target.argmax().item()
                if true_label == target_label:
                    break
            if attempt >= max_attempts - 1:
                print(f"  ⚠️  无法找到标签 {target_label} 的样本，使用随机样本")
        else:
            obs = world_interface.reset()
            target = world_interface.get_target()
            true_label = target.argmax().item()
        
        image = obs["vision"].reshape(28, 28).cpu().numpy()
        
        if args.verbose:
            print(f"\n样本 {i+1}/{args.num_samples}: 真实标签 = {true_label}")
        
        # 识别
        result = recognize_digit(
            fep_system,
            obs["vision"],
            true_label,
            device,
            num_infer_iters=args.num_infer_iters,
            use_encoder_init=args.use_encoder_init,
            verbose=args.verbose,
        )
        
        result["true_label"] = true_label
        results.append((image, result))
        
        if result["correct"]:
            correct_count += 1
        
        if args.verbose:
            print(f"  预测: {result['predicted_label']}, 置信度: {result['confidence']*100:.1f}%")
            print(f"  结果: {'✓ 正确' if result['correct'] else '✗ 错误'}")
        
        # 可视化
        output_path = f"{args.output_dir}/demo_{i+1:03d}_true{true_label}_pred{result['predicted_label']}.png"
        visualize_recognition(image, result, output_path)
        print(f"  ✓ 已保存: {output_path}")
    
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

