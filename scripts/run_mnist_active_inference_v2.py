#!/usr/bin/env python3
"""
MNIST 主动推理实验 V2：实现完整的生成模型和行动选择

核心改进：
1. 生成模型学习：
   - ObservationAspect: p(obs | state) - 从状态生成观察
   - DynamicsAspect: p(state_{t+1} | state_t, action) - 状态转移
   - EncoderAspect/PipelineAspect: p(state | obs) - 从观察推断状态

2. 行动选择：
   - 通过优化自由能选择行动
   - 行动影响观察和状态转移

3. 完整自由能：
   - F = F_obs + F_dyn + F_classification + F_prior
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import argparse
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from aonn.models.mnist_world_model import MNISTWorldModel, MNISTWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.aspects.classification_aspect import ClassificationAspect
from aonn.aspects.pipeline_aspect import PipelineAspect
from aonn.aspects.encoder_aspect import EncoderAspect
from aonn.aspects.world_model_aspects import ObservationAspect, DynamicsAspect
from aonn.core.active_inference_loop import ActiveInferenceLoop
import torch.nn as nn


def evaluate_evolution_options(
    brain,
    observation_aspect: ObservationAspect,
    dynamics_aspect: DynamicsAspect,
    classification_aspect: ClassificationAspect,
    current_state: torch.Tensor,
    current_obs: torch.Tensor,
    target: torch.Tensor,
    config: Dict,
    device=None,
) -> Dict:
    """
    评估不同演化选项的预期自由能（实际测试）
    
    演化选项：
    1. no_change: 不改变结构
    2. add_sensory_aspect: 添加感官 Aspect（实际创建并测试）
    3. add_pipeline: 添加 Pipeline（实际创建并测试）
    4. prune: 剪枝无用 Aspect（暂不实现）
    
    Returns:
        包含各选项预期自由能的字典
    """
    device = device or torch.device("cpu")
    options = {}
    
    # 导入必要的类（在函数开头，避免作用域问题）
    from aonn.aspects.sensory_aspect import LinearGenerativeAspect
    from aonn.aspects.pipeline_aspect import PipelineAspect
    
    # 使用 brain 的实际 objects 和 aspects 来计算自由能
    # 保存当前状态
    original_internal = brain.objects["internal"].state.clone()
    original_vision = brain.objects["vision"].state.clone()
    original_target = brain.objects["target"].state.clone()
    
    # 设置当前状态
    brain.objects["internal"].set_state(current_state)
    brain.objects["vision"].set_state(current_obs)
    
    # 处理 target 维度：target 可能是 10 维（one-hot），但 brain.objects["target"] 可能是 128 维
    # 如果维度不匹配，需要适配
    target_to_set = target.clone()
    if target_to_set.shape[-1] != brain.objects["target"].dim:
        if target_to_set.shape[-1] < brain.objects["target"].dim:
            # 如果 target 维度小于 brain.objects["target"].dim，填充零
            padding = torch.zeros(brain.objects["target"].dim - target_to_set.shape[-1], device=target_to_set.device)
            target_to_set = torch.cat([target_to_set, padding], dim=-1)
        else:
            # 如果 target 维度大于 brain.objects["target"].dim，截断
            target_to_set = target_to_set[..., :brain.objects["target"].dim]
    brain.objects["target"].set_state(target_to_set)
    
    # 计算当前自由能（作为基准）
    current_aspects = list(brain.aspects)  # 保存当前 aspects
    F_current = brain.compute_free_energy().item()
    options["no_change"] = F_current
    
    # 1. 测试添加感官 Aspect（vision -> internal）
    # 检查是否已存在 vision -> internal 的 aspect
    has_vision_to_internal = any(
        "vision" in asp.src_names and "internal" in asp.dst_names
        for asp in brain.aspects
    )
    
    if not has_vision_to_internal:
        try:
            # 查找相似的现有 aspect 作为参考（internal -> vision 的 ObservationAspect）
            reference_aspect = None
            for asp in brain.aspects:
                if isinstance(asp, ObservationAspect) and "internal" in asp.src_names and "vision" in asp.dst_names:
                    # 尝试从 ObservationAspect 提取权重（如果可能）
                    reference_aspect = asp
                    break
            
            # 创建临时感官 Aspect（使用参考 aspect 的权重初始化）
            vision_dim = current_obs.shape[-1]
            state_dim = current_state.shape[-1]
            
            # 如果找到参考 aspect，尝试提取权重
            init_weight = None
            if reference_aspect is not None and hasattr(reference_aspect, 'observation_model'):
                try:
                    # ObservationAspect 使用 Linear，尝试提取权重
                    if hasattr(reference_aspect.observation_model, 'weight'):
                        ref_weight = reference_aspect.observation_model.weight.detach().clone()
                        if ref_weight.shape == (vision_dim, state_dim):
                            # 添加小噪声
                            noise_scale = config.get("evolution_noise_scale", 0.01)
                            init_weight = ref_weight + noise_scale * torch.randn_like(ref_weight)
                except:
                    pass
            
            temp_sensory_aspect = LinearGenerativeAspect(
                internal_name="internal",
                sensory_name="vision",
                state_dim=state_dim,
                obs_dim=vision_dim,
                name="temp_sensory_aspect",
                init_weight=init_weight,  # 使用参考权重初始化
                init_scale=0.01,  # 降低初始缩放因子
            )
            temp_sensory_aspect = temp_sensory_aspect.to(device)
            
            # 临时添加到 aspects
            brain.aspects.append(temp_sensory_aspect)
            
            # 计算添加后的自由能
            F_with_sensory = brain.compute_free_energy().item()
            
            # 考虑新 aspect 的"潜力"：即使初始自由能高，经过训练后可能降低
            # 使用乐观估计：假设新 aspect 能减少 10-20% 的观察生成误差
            F_obs_current = observation_aspect.free_energy_contrib(brain.objects).item()
            F_potential_reduction = F_obs_current * 0.15  # 假设减少 15%
            # 使用初始自由能和潜力估计的加权平均
            F_optimistic = F_with_sensory - F_potential_reduction
            options["add_sensory_aspect"] = min(F_with_sensory, F_optimistic)  # 取较小值
            
            # 移除临时 aspect
            brain.aspects.remove(temp_sensory_aspect)
            del temp_sensory_aspect
        except Exception as e:
            # 如果创建失败，使用估算值
            F_obs_current = observation_aspect.free_energy_contrib(brain.objects).item()
            F_class_current = classification_aspect.free_energy_contrib(brain.objects).item()
            options["add_sensory_aspect"] = F_obs_current * 0.9 + F_class_current + 0.05
    else:
        # 如果已存在，不评估
        options["add_sensory_aspect"] = F_current + 1.0  # 设置一个较高的值，表示不需要
    
    # 2. 测试添加 Pipeline（internal -> internal，用于增强表示）
    # 检查是否已存在 internal -> internal 的 pipeline
    has_internal_pipeline = any(
        isinstance(asp, PipelineAspect) and
        "internal" in asp.src_names and "internal" in asp.dst_names
        for asp in brain.aspects
    )
    
    if not has_internal_pipeline:
        try:
            
            # 创建临时 Pipeline
            state_dim = current_state.shape[-1]
            pipeline_cfg = config.get("pipeline_growth", {})
            num_aspects = pipeline_cfg.get("initial_width", 16)
            depth = pipeline_cfg.get("initial_depth", 2)
            
            temp_pipeline = PipelineAspect(
                src_names=["internal"],
                dst_names=["internal"],
                input_dim=state_dim,
                output_dim=state_dim,
                num_aspects=num_aspects,
                depth=depth,
                name="temp_internal_pipeline",
            )
            temp_pipeline = temp_pipeline.to(device)
            
            # 临时添加到 aspects
            brain.aspects.append(temp_pipeline)
            
            # 计算添加后的自由能
            F_with_pipeline = brain.compute_free_energy().item()
            
            # 考虑新 pipeline 的"潜力"：即使初始自由能高，经过训练后可能降低
            # 使用乐观估计：假设新 pipeline 能减少 15-25% 的分类误差
            F_class_current = classification_aspect.free_energy_contrib(brain.objects).item()
            F_obs_current = observation_aspect.free_energy_contrib(brain.objects).item()
            # Pipeline 主要用于增强表示，可能同时降低观察和分类误差
            F_potential_reduction = F_class_current * 0.20 + F_obs_current * 0.05  # 假设减少 20% 分类误差 + 5% 观察误差
            # 使用初始自由能和潜力估计的加权平均
            F_optimistic = F_with_pipeline - F_potential_reduction
            options["add_pipeline"] = min(F_with_pipeline, F_optimistic)  # 取较小值
            
            # 移除临时 pipeline
            brain.aspects.remove(temp_pipeline)
            del temp_pipeline
        except Exception as e:
            # 如果创建失败，使用估算值
            F_obs_current = observation_aspect.free_energy_contrib(brain.objects).item()
            F_class_current = classification_aspect.free_energy_contrib(brain.objects).item()
            options["add_pipeline"] = F_obs_current + F_class_current * 0.75 + 0.2
    else:
        # 如果已存在，不评估
        options["add_pipeline"] = F_current + 1.0  # 设置一个较高的值，表示不需要
    
    # 3. 评估剪枝：实际测试移除不重要的 aspect 后的自由能
    # 找出贡献最小的 aspect，测试移除它后的自由能
    # 放宽条件：要求更大的自由能降低才剪枝
    prune_min_reduction_ratio = config.get("prune_min_reduction_ratio", 0.10)  # 至少降低 10% 才剪枝
    aspect_contributions = []
    for asp in brain.aspects:
        try:
            contrib = asp.free_energy_contrib(brain.objects).item()
            # 只考虑可剪枝的 aspect（排除核心生成模型）
            aspect_type = type(asp).__name__
            if aspect_type not in ["ObservationAspect", "DynamicsAspect", "ClassificationAspect", "EncoderAspect"]:
                aspect_contributions.append((asp, contrib))
        except:
            continue
    
    if len(aspect_contributions) > 0:
        # 找出贡献最小的 aspect
        aspect_contributions.sort(key=lambda x: x[1])
        weakest_aspect, weakest_contrib = aspect_contributions[0]
        
        # 临时移除最弱的 aspect
        brain.aspects.remove(weakest_aspect)
        F_without_weakest = brain.compute_free_energy().item()
        brain.aspects.append(weakest_aspect)  # 恢复
        
        # 放宽条件：要求移除后自由能降低至少 prune_min_reduction_ratio 比例才剪枝
        reduction_ratio = (F_current - F_without_weakest) / F_current if F_current > 0 else 0
        if F_without_weakest < F_current and reduction_ratio >= prune_min_reduction_ratio:
            options["prune"] = F_without_weakest
        else:
            # 如果移除后自由能增加，或者降低幅度不够，不剪枝
            options["prune"] = F_current + 1.0  # 设置较高值，表示不应该剪枝
    else:
        # 如果没有可剪枝的 aspect，不评估
        options["prune"] = F_current + 1.0
    
    # 恢复原始状态
    brain.objects["internal"].set_state(original_internal)
    brain.objects["vision"].set_state(original_vision)
    brain.objects["target"].set_state(original_target)
    
    return options


def optimize_action_with_evolution(
    brain,
    observation_aspect: ObservationAspect,
    dynamics_aspect: DynamicsAspect,
    classification_aspect: ClassificationAspect,
    current_state: torch.Tensor,
    current_obs: torch.Tensor,
    target: torch.Tensor,
    config: Dict,
    num_action_iters: int = 5,
    action_lr: float = 0.1,
    device=None,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    通过优化自由能选择行动（包括演化决策）
    
    行动 = 分类预测（连续） + 演化决策（离散）
    
    预期自由能：
    E[F] = ||obs_t - ObservationAspect(state_t)||²  (观察生成误差)
         + ||target - ClassificationAspect(state_t)||²  (分类误差)
         + E[||state_{t+1} - DynamicsAspect(state_t, action)||²]  (状态转移误差)
         + E[F_evolution]  (演化成本/收益)
    
    Args:
        brain: AONN Brain
        observation_aspect: 观察生成模型
        dynamics_aspect: 状态转移模型
        classification_aspect: 分类模型
        current_state: 当前状态 state_t
        current_obs: 当前观察 obs_t
        target: 目标标签（one-hot）
        config: 配置字典
        num_action_iters: 行动优化迭代次数
        action_lr: 行动学习率
        device: 设备
    
    Returns:
        (优化后的行动 action_t, 演化决策)
    """
    device = device or torch.device("cpu")
    
    # 1. 评估演化选项的预期自由能
    try:
        evolution_options = evaluate_evolution_options(
            brain,
            observation_aspect,
            dynamics_aspect,
            classification_aspect,
            current_state,
            current_obs,
            target,
            config,
            device,
        )
        
        # 2. 选择最优演化选项（最小预期自由能）
        best_evolution_option = min(evolution_options.items(), key=lambda x: x[1])
        evolution_decision = None
        
        # 如果最优选项不是"不改变"，且预期收益足够大，执行演化
        evolution_threshold = config.get("evolution_action_threshold", 0.01)  # 至少降低 1%（降低阈值）
        F_current = evolution_options["no_change"]
        F_best = best_evolution_option[1]
        expected_reduction = F_current - F_best
        reduction_ratio = expected_reduction / F_current if F_current > 0 else 0
        
        # 调试输出（可选）
        verbose_evolution = config.get("verbose_evolution", False)
        if verbose_evolution:
            print(f"  [演化评估] 当前F={F_current:.4f}")
            for opt_name, opt_F in evolution_options.items():
                if opt_name != "no_change":
                    opt_reduction = F_current - opt_F
                    opt_ratio = opt_reduction / F_current if F_current > 0 else 0
                    print(f"    {opt_name}: F={opt_F:.4f}, 降低={opt_reduction:.4f} ({opt_ratio*100:.2f}%)")
            print(f"  最佳选项: {best_evolution_option[0]}, 预期F={F_best:.4f}, 预期降低={expected_reduction:.4f} ({reduction_ratio*100:.2f}%)")
        
        # 判断是否执行演化：取消阈值限制，只要不是"不改变"就执行
        # prune 现在已实现，可以执行
        should_evolve = (
            best_evolution_option[0] != "no_change" and 
            F_best < F_current  # 只要预期自由能更低就执行
        )
        
        if should_evolve:
            evolution_decision = {
                "option": best_evolution_option[0],
                "expected_F": F_best,
                "current_F": F_current,
                "expected_reduction": expected_reduction,
                "reduction_ratio": reduction_ratio,
            }
    except Exception as e:
        # 如果评估失败，不执行演化
        verbose_evolution = config.get("verbose_evolution", False)
        if verbose_evolution:
            print(f"  [演化评估错误] {e}")
        evolution_options = {"no_change": float('inf')}
        evolution_decision = None
    
    # 3. 优化分类预测（连续行动）
    action_logits = torch.zeros(10, device=device, requires_grad=True)
    
    # 创建临时 ObjectNode 用于计算自由能
    from aonn.core.object import ObjectNode
    temp_internal = ObjectNode("internal", dim=current_state.shape[-1], device=device)
    temp_internal.set_state(current_state)
    
    temp_vision = ObjectNode("vision", dim=current_obs.shape[-1], device=device)
    temp_vision.set_state(current_obs)
    
    # 处理 target 维度
    target_dim = target.shape[-1] if target.dim() > 0 else 10
    temp_target = ObjectNode("target", dim=target_dim, device=device)
    # 确保 target 维度匹配
    target_to_set = target.clone()
    if target_to_set.shape[-1] != target_dim:
        if target_to_set.shape[-1] < target_dim:
            padding = torch.zeros(target_dim - target_to_set.shape[-1], device=device)
            target_to_set = torch.cat([target_to_set, padding], dim=-1)
        else:
            target_to_set = target_to_set[..., :target_dim]
    temp_target.set_state(target_to_set)
    
    temp_action = ObjectNode("action", dim=10, device=device)
    temp_action.set_state(action_logits)
    
    for iter_idx in range(num_action_iters):
        # 更新 action state
        temp_action.set_state(action_logits)
        
        # 计算分类误差（主要目标）
        temp_objects = {
            "internal": temp_internal,
            "target": temp_target,
        }
        F_class = classification_aspect.free_energy_contrib(temp_objects)
        
        # 计算观察生成误差
        temp_objects_obs = {
            "internal": temp_internal,
            "vision": temp_vision,
        }
        F_obs = observation_aspect.free_energy_contrib(temp_objects_obs)
        
        # 总自由能（主要关注分类，观察生成作为正则化）
        F_total = F_class + 0.1 * F_obs  # 观察生成误差权重较小
        
        # 反向传播（只对 action_logits 求梯度）
        if action_logits.grad is not None:
            action_logits.grad.zero_()
        
        F_total.backward(retain_graph=(iter_idx < num_action_iters - 1))
        
        # 更新行动（梯度下降）
        if action_logits.grad is not None:
            with torch.no_grad():
                action_logits = action_logits - action_lr * action_logits.grad
                action_logits = action_logits.detach().requires_grad_(True)
    
    # 返回 softmax 后的行动（概率分布）和演化决策
    return torch.softmax(action_logits.detach(), dim=-1), evolution_decision


def optimize_action(
    brain,
    observation_aspect: ObservationAspect,
    dynamics_aspect: DynamicsAspect,
    classification_aspect: ClassificationAspect,
    current_state: torch.Tensor,
    current_obs: torch.Tensor,
    target: torch.Tensor,
    num_action_iters: int = 5,
    action_lr: float = 0.1,
    device=None,
) -> torch.Tensor:
    """
    通过优化自由能选择行动
    
    预期自由能：
    E[F] = ||obs_t - ObservationAspect(state_t)||²  (观察生成误差)
         + ||target - ClassificationAspect(state_t)||²  (分类误差)
         + E[||state_{t+1} - DynamicsAspect(state_t, action)||²]  (状态转移误差)
    
    简化：在当前步骤，我们主要优化分类误差，因为这是 MNIST 的主要目标。
    
    Args:
        brain: AONN Brain
        observation_aspect: 观察生成模型
        dynamics_aspect: 状态转移模型
        classification_aspect: 分类模型
        current_state: 当前状态 state_t
        current_obs: 当前观察 obs_t
        target: 目标标签（one-hot）
        num_action_iters: 行动优化迭代次数
        action_lr: 行动学习率
        device: 设备
    
    Returns:
        优化后的行动 action_t (分类预测概率)
    """
    device = device or torch.device("cpu")
    
    # 初始化行动（分类预测的 logits，需要梯度）
    action_logits = torch.zeros(10, device=device, requires_grad=True)
    
    # 创建临时 ObjectNode 用于计算自由能
    from aonn.core.object import ObjectNode
    temp_internal = ObjectNode("internal", dim=current_state.shape[-1], device=device)
    temp_internal.set_state(current_state)
    
    temp_vision = ObjectNode("vision", dim=current_obs.shape[-1], device=device)
    temp_vision.set_state(current_obs)
    
    # 处理 target 维度
    target_dim = target.shape[-1] if target.dim() > 0 else 10
    temp_target = ObjectNode("target", dim=target_dim, device=device)
    # 确保 target 维度匹配
    target_to_set = target.clone()
    if target_to_set.shape[-1] != target_dim:
        if target_to_set.shape[-1] < target_dim:
            padding = torch.zeros(target_dim - target_to_set.shape[-1], device=device)
            target_to_set = torch.cat([target_to_set, padding], dim=-1)
        else:
            target_to_set = target_to_set[..., :target_dim]
    temp_target.set_state(target_to_set)
    
    temp_action = ObjectNode("action", dim=10, device=device)
    temp_action.set_state(action_logits)
    
    for iter_idx in range(num_action_iters):
        # 更新 action state
        temp_action.set_state(action_logits)
        
        # 计算分类误差（主要目标）
        temp_objects = {
            "internal": temp_internal,
            "target": temp_target,
        }
        F_class = classification_aspect.free_energy_contrib(temp_objects)
        
        # 计算观察生成误差
        temp_objects_obs = {
            "internal": temp_internal,
            "vision": temp_vision,
        }
        F_obs = observation_aspect.free_energy_contrib(temp_objects_obs)
        
        # 总自由能（主要关注分类，观察生成作为正则化）
        F_total = F_class + 0.1 * F_obs  # 观察生成误差权重较小
        
        # 反向传播（只对 action_logits 求梯度）
        if action_logits.grad is not None:
            action_logits.grad.zero_()
        
        F_total.backward(retain_graph=(iter_idx < num_action_iters - 1))
        
        # 更新行动（梯度下降）
        if action_logits.grad is not None:
            with torch.no_grad():
                action_logits = action_logits - action_lr * action_logits.grad
                action_logits = action_logits.detach().requires_grad_(True)
    
    # 返回 softmax 后的行动（概率分布）
    return torch.softmax(action_logits.detach(), dim=-1)


def run_experiment(
    num_steps: int,
    config: Dict,
    device: torch.device,
    *,
    verbose: bool = False,
    output: str = "data/mnist_active_inference_v2.json",
    save_interval: int = 100,
):
    """运行 MNIST 主动推理实验 V2"""
    
    # 创建 MNIST 世界模型
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
        train=False,
    )
    val_interface = MNISTWorldInterface(val_world)
    
    # 创建 AONN Brain
    brain = AONNBrainV3(config=config, device=device, enable_evolution=True)
    
    # 创建必要的 Objects（如果不存在）
    if "target" not in brain.objects:
        brain.create_object("target", dim=10)
    if "action" not in brain.objects:
        brain.create_object("action", dim=10)
    
    print("=" * 80)
    print("MNIST 主动推理实验 V2：完整生成模型 + 行动选择")
    print("=" * 80)
    print(f"状态维度: {config.get('state_dim', 128)}")
    print(f"观察维度: {config.get('obs_dim', 784)}")
    print(f"动作维度: {config.get('act_dim', 10)}")
    print()
    
    # 创建生成模型 Aspects
    state_dim = config.get("state_dim", 128)
    obs_dim = config.get("obs_dim", 784)
    act_dim = config.get("act_dim", 10)
    
    # 1. Encoder: vision -> internal (p(state | obs))
    # 使用卷积编码器（MNIST 是 28x28 图像）
    encoder = EncoderAspect(
        sensory_name="vision",
        internal_name="internal",
        input_dim=obs_dim,
        output_dim=state_dim,
        name="vision_encoder",
        use_conv=True,  # 使用卷积编码器
        image_size=28,  # MNIST 图像尺寸
    )
    brain.aspects.append(encoder)
    if isinstance(encoder, nn.Module):
        brain.add_module("vision_encoder", encoder)
    print(f"  ✓ 创建 EncoderAspect (卷积): vision -> internal")
    
    # 2. Observation: internal -> vision (p(obs | state))
    # 使用卷积解码器（生成 28x28 图像）
    observation_aspect = ObservationAspect(
        internal_name="internal",
        sensory_name="vision",
        state_dim=state_dim,
        obs_dim=obs_dim,
        use_conv=True,  # 使用卷积解码器
        image_size=28,  # MNIST 图像尺寸
    )
    brain.aspects.append(observation_aspect)
    if isinstance(observation_aspect, nn.Module):
        brain.add_module("observation_aspect", observation_aspect)
    print(f"  ✓ 创建 ObservationAspect: internal -> vision")
    
    # 3. Dynamics: internal + action -> internal (p(state_{t+1} | state_t, action))
    dynamics_aspect = DynamicsAspect(
        internal_name="internal",
        action_name="action",
        state_dim=state_dim,
        action_dim=act_dim,
    )
    brain.aspects.append(dynamics_aspect)
    if isinstance(dynamics_aspect, nn.Module):
        brain.add_module("dynamics_aspect", dynamics_aspect)
    print(f"  ✓ 创建 DynamicsAspect: internal + action -> internal")
    
    # 4. Classification: internal -> target
    classification_aspect = ClassificationAspect(
        internal_name="internal",
        target_name="target",
        state_dim=state_dim,
        num_classes=10,
        hidden_dim=state_dim,
        loss_weight=config.get("classification_loss_weight", 1.0),
    )
    brain.aspects.append(classification_aspect)
    if isinstance(classification_aspect, nn.Module):
        brain.add_module("classification_aspect", classification_aspect)
    print(f"  ✓ 创建 ClassificationAspect: internal -> target")
    print()
    
    # 创建优化器（所有生成模型参数）
    all_params = (
        list(encoder.parameters()) +
        list(observation_aspect.parameters()) +
        list(dynamics_aspect.parameters()) +
        list(classification_aspect.parameters())
    )
    optimizer = Adam(
        all_params,
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 1e-4),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    print(f"  ✓ 初始化 Adam 优化器: lr={config.get('learning_rate', 0.001)}")
    print()
    
    # 实验记录
    snapshots = []
    accuracy_history = []
    free_energy_history = []
    F_obs_history = []
    F_dyn_history = []
    F_class_history = []
    evolution_decisions = []  # 记录所有演化决策
    
    # 自由能停滞检测：用于触发剪枝
    # 放宽条件：增加停滞阈值，降低改善要求
    free_energy_stagnation = {
        "best_F": float('inf'),
        "stagnation_steps": 0,
        "stagnation_threshold": config.get("prune_stagnation_steps", 150),  # 放宽：150 步无改善才剪枝（原 50）
        "min_improvement": config.get("prune_min_improvement", 0.005),  # 放宽：至少改善 0.5% 才算改善（原 1%）
    }
    
    # 剪枝最小间隔：避免频繁剪枝
    last_prune_step = -1
    prune_min_interval = config.get("prune_min_interval", 20)  # 至少间隔 20 步才能再次剪枝
    
    # 初始化观察
    obs = train_interface.reset()
    prev_state = None
    prev_action = None
    
    progress = tqdm(range(num_steps), desc="MNIST Active Inference V2")
    
    try:
        for step in progress:
            # 1. 设置当前观察
            for sense, value in obs.items():
                if sense in brain.objects:
                    brain.objects[sense].set_state(value)
            
            # 获取目标标签
            target = train_interface.get_target()
            # 确保 target 是 1D 向量，且维度正确
            if target.dim() == 0:
                # 如果是标量，转换为 one-hot
                label = target.item() if hasattr(target, 'item') else int(target)
                target = torch.zeros(10, device=device)
                target[label] = 1.0
            elif target.dim() > 1:
                target = target.squeeze()
            # 确保维度匹配
            if target.shape[-1] != brain.objects["target"].dim:
                # 如果维度不匹配，截断或填充
                if target.shape[-1] > brain.objects["target"].dim:
                    target = target[..., :brain.objects["target"].dim]
                else:
                    padding = torch.zeros(brain.objects["target"].dim - target.shape[-1], device=device)
                    target = torch.cat([target, padding], dim=-1)
            brain.objects["target"].set_state(target)
            
            # 2. 网络演化：现在作为行动选择的一部分（见步骤 4）
            # 这里只记录初始结构，演化将在行动选择时触发
            num_aspects_before = len(brain.aspects)
            num_objects_before = len(brain.objects)
            
            # 3. 状态推理：给定观察，推断内部状态（最小化自由能）
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
                except Exception as e:
                    if verbose:
                        print(f"Step {step}: Inference error: {e}")
                    pass
            
            current_state = brain.objects["internal"].state.clone()
            
            # 4. 行动选择：通过优化自由能选择行动（包括演化决策）
            evolution_decision = None
            try:
                current_obs = brain.objects["vision"].state.clone()
                action, evolution_decision = optimize_action_with_evolution(
                    brain,
                    observation_aspect,
                    dynamics_aspect,
                    classification_aspect,
                    current_state,
                    current_obs,
                    target,
                    config,
                    num_action_iters=config.get("num_action_iters", 5),
                    action_lr=config.get("action_lr", 0.1),
                    device=device,
                )
                brain.objects["action"].set_state(action)
                
                # 如果演化决策建议演化，执行演化
                if evolution_decision is not None:
                    evolution_option = evolution_decision["option"]
                    if verbose:
                        print(f"\n[Step {step}] 行动选择触发演化: {evolution_option}")
                        print(f"  预期自由能降低: {evolution_decision['expected_reduction']:.4f}")
                    
                    # 记录演化决策
                    evolution_decisions.append({
                        "step": step,
                        **evolution_decision,
                    })
                    
                    # 记录演化前的结构
                    num_aspects_before_evo = len(brain.aspects)
                    num_objects_before_evo = len(brain.objects)
                    
                    # 执行演化
                    if evolution_option == "add_sensory_aspect":
                        # 触发批量创建感官 Aspects
                        brain.evolve_network(obs, target=target)
                    elif evolution_option == "add_pipeline":
                        # 触发 Pipeline 创建
                        brain.evolve_network(obs, target=target)
                    elif evolution_option == "prune":
                        # 检查剪枝最小间隔
                        if step - last_prune_step < prune_min_interval:
                            if verbose:
                                print(f"  ⏸️ 剪枝跳过: 距离上次剪枝仅 {step - last_prune_step} 步，需要间隔 {prune_min_interval} 步")
                            continue
                        
                        # 执行剪枝：移除贡献最小的 aspect
                        aspect_contributions = []
                        for asp in brain.aspects:
                            try:
                                contrib = asp.free_energy_contrib(brain.objects).item()
                                aspect_type = type(asp).__name__
                                # 只剪枝非核心 aspect
                                if aspect_type not in ["ObservationAspect", "DynamicsAspect", "ClassificationAspect", "EncoderAspect"]:
                                    aspect_contributions.append((asp, contrib))
                            except:
                                continue
                        
                        if len(aspect_contributions) > 0:
                            aspect_contributions.sort(key=lambda x: x[1])
                            weakest_aspect, weakest_contrib = aspect_contributions[0]
                            
                            # 检查剪枝的最小降低比例要求
                            prune_min_reduction_ratio = config.get("prune_min_reduction_ratio", 0.10)
                            F_before_prune = brain.compute_free_energy().item()
                            
                            # 临时移除测试
                            brain.aspects.remove(weakest_aspect)
                            F_after_prune = brain.compute_free_energy().item()
                            brain.aspects.append(weakest_aspect)  # 恢复
                            
                            reduction_ratio = (F_before_prune - F_after_prune) / F_before_prune if F_before_prune > 0 else 0
                            
                            # 只有满足最小降低比例才执行剪枝
                            if F_after_prune < F_before_prune and reduction_ratio >= prune_min_reduction_ratio:
                                # 真正移除
                                brain.aspects.remove(weakest_aspect)
                                if hasattr(brain, 'aspect_modules'):
                                    try:
                                        brain.aspect_modules.remove(weakest_aspect)
                                    except:
                                        pass
                                
                                last_prune_step = step
                                
                                if verbose:
                                    print(f"  ✂️ 剪枝: 移除了 {weakest_aspect.name} (贡献={weakest_contrib:.4f}, 降低={reduction_ratio*100:.1f}%)")
                                
                                # 重置停滞计数器（因为进行了结构改变）
                                free_energy_stagnation["stagnation_steps"] = 0
                            else:
                                if verbose:
                                    print(f"  ⏸️ 剪枝跳过: 降低比例 {reduction_ratio*100:.1f}% 不足 {prune_min_reduction_ratio*100:.1f}%")
                    
                    # 检查是否创建了新结构
                    num_aspects_after_evo = len(brain.aspects)
                    num_objects_after_evo = len(brain.objects)
                    
                    if num_aspects_after_evo > num_aspects_before_evo or num_objects_after_evo > num_objects_before_evo:
                        # 重新收集所有参数
                        all_params = []
                        for asp in brain.aspects:
                            if isinstance(asp, nn.Module):
                                all_params.extend(asp.parameters())
                        
                        # 重新创建优化器
                        optimizer = Adam(
                            all_params,
                            lr=config.get("learning_rate", 0.001),
                            weight_decay=config.get("weight_decay", 1e-4),
                            betas=(0.9, 0.999),
                            eps=1e-8,
                        )
                        if verbose:
                            print(f"  演化后结构: {num_objects_after_evo} Objects, {num_aspects_after_evo} Aspects")
                            
            except Exception as e:
                if verbose:
                    import traceback
                    print(f"Step {step}: Action optimization error: {e}")
                    print(f"  错误详情: {traceback.format_exc()}")
                # 回退：使用分类预测作为行动
                with torch.no_grad():
                    logits = classification_aspect.predict(brain.objects)
                    action = torch.softmax(logits, dim=-1)
                    brain.objects["action"].set_state(action)
            
            # 5. 执行行动，获取新观察（世界模型步进）
            if step > 0:
                obs, reward = train_interface.step(action)
            else:
                # 第一步，随机采样
                obs = train_interface.reset()
            
            # 6. 计算完整自由能（用于学习）
            with torch.no_grad():
                F_obs = observation_aspect.free_energy_contrib(brain.objects)
                # DynamicsAspect 需要 internal_next，但当前步骤还没有，所以用当前状态作为目标
                if prev_state is not None and prev_action is not None:
                    # 创建临时 internal_next 用于计算
                    temp_internal_next = brain.create_object("internal_next", dim=state_dim)
                    temp_internal_next.set_state(current_state)
                    temp_objects = brain.objects.copy()
                    temp_objects["internal_next"] = temp_internal_next
                    F_dyn = dynamics_aspect.free_energy_contrib(temp_objects)
                    # 删除临时对象
                    del brain.objects["internal_next"]
                else:
                    F_dyn = torch.tensor(0.0, device=device)
                F_class = classification_aspect.free_energy_contrib(brain.objects)
                F_total = F_obs + F_dyn + F_class
                
                F_obs_history.append(F_obs.item())
                F_dyn_history.append(F_dyn.item())
                F_class_history.append(F_class.item())
                free_energy_history.append(F_total.item())
            
            # 7. 参数学习（更新生成模型）
            if step > 0 and prev_state is not None and prev_action is not None:
                try:
                    optimizer.zero_grad()
                    
                    # 设置下一状态（用于 DynamicsAspect）
                    if "internal_next" not in brain.objects:
                        brain.objects["internal_next"] = brain.create_object("internal_next", dim=state_dim)
                    brain.objects["internal_next"].set_state(current_state)
                    
                    # 计算完整自由能
                    F_obs = observation_aspect.free_energy_contrib(brain.objects)
                    F_dyn = dynamics_aspect.free_energy_contrib(brain.objects)
                    F_class = classification_aspect.free_energy_contrib(brain.objects)
                    F_total = F_obs + F_dyn + F_class
                    
                    if torch.isfinite(F_total) and F_total.requires_grad:
                        F_total.backward()
                    
                    # 梯度裁剪
                    max_grad_norm = config.get("max_grad_norm", None)
                    if max_grad_norm is not None:
                        # 重新获取参数（可能已更新）
                        current_params = []
                        for asp in brain.aspects:
                            if isinstance(asp, nn.Module):
                                current_params.extend(asp.parameters())
                        torch.nn.utils.clip_grad_norm_(current_params, max_grad_norm)
                        
                        optimizer.step()
                        brain.sanitize_states()
                except Exception as e:
                    if verbose:
                        print(f"Step {step}: Learning error: {e}")
                    pass
            
            # 8. 自由能停滞检测：如果长时间无法降低自由能，触发剪枝
            F_current_step = F_total.item() if 'F_total' in locals() else brain.compute_free_energy().item()
            
            # 检查是否有改善
            improvement = (free_energy_stagnation["best_F"] - F_current_step) / free_energy_stagnation["best_F"] if free_energy_stagnation["best_F"] > 0 else 0
            if improvement > free_energy_stagnation["min_improvement"]:
                # 有改善，更新最佳自由能并重置停滞计数器
                free_energy_stagnation["best_F"] = F_current_step
                free_energy_stagnation["stagnation_steps"] = 0
            else:
                # 无改善，增加停滞步数
                free_energy_stagnation["stagnation_steps"] += 1
            
            # 如果停滞时间过长，强制触发剪枝（放宽条件：检查最小间隔）
            if (free_energy_stagnation["stagnation_steps"] >= free_energy_stagnation["stagnation_threshold"] and
                len(brain.aspects) > 4 and  # 至少保留 4 个核心 aspect
                step - last_prune_step >= prune_min_interval):  # 满足最小间隔
                
                if verbose:
                    print(f"\n[Step {step}] ⚠️ 自由能停滞 {free_energy_stagnation['stagnation_steps']} 步，触发强制剪枝")
                
                # 找出贡献最小的 aspect 并移除
                aspect_contributions = []
                for asp in brain.aspects:
                    try:
                        contrib = asp.free_energy_contrib(brain.objects).item()
                        aspect_type = type(asp).__name__
                        # 只剪枝非核心 aspect
                        if aspect_type not in ["ObservationAspect", "DynamicsAspect", "ClassificationAspect", "EncoderAspect"]:
                            aspect_contributions.append((asp, contrib))
                    except:
                        continue
                
                if len(aspect_contributions) > 0:
                    aspect_contributions.sort(key=lambda x: x[1])
                    weakest_aspect, weakest_contrib = aspect_contributions[0]
                    
                    # 检查剪枝的最小降低比例要求（强制剪枝时放宽要求，只要求降低 5%）
                    prune_min_reduction_ratio_forced = config.get("prune_min_reduction_ratio_forced", 0.05)  # 强制剪枝时降低到 5%
                    F_before_prune = brain.compute_free_energy().item()
                    
                    # 临时移除测试
                    brain.aspects.remove(weakest_aspect)
                    F_after_prune = brain.compute_free_energy().item()
                    brain.aspects.append(weakest_aspect)  # 恢复
                    
                    reduction_ratio = (F_before_prune - F_after_prune) / F_before_prune if F_before_prune > 0 else 0
                    
                    # 强制剪枝时，只要降低就执行（但仍有最小降低要求）
                    if F_after_prune < F_before_prune and reduction_ratio >= prune_min_reduction_ratio_forced:
                        # 真正移除
                        brain.aspects.remove(weakest_aspect)
                        if hasattr(brain, 'aspect_modules'):
                            try:
                                brain.aspect_modules.remove(weakest_aspect)
                            except:
                                pass
                        
                        last_prune_step = step
                        
                        if verbose:
                            print(f"  ✂️ 强制剪枝: 移除了 {weakest_aspect.name} (贡献={weakest_contrib:.4f}, 降低={reduction_ratio*100:.1f}%)")
                        
                        # 重置停滞计数器
                        free_energy_stagnation["stagnation_steps"] = 0
                        free_energy_stagnation["best_F"] = float('inf')  # 重置最佳自由能
                        
                        # 记录演化决策
                        evolution_decisions.append({
                            "step": step,
                            "option": "prune_forced",
                            "reason": "free_energy_stagnation",
                            "stagnation_steps": free_energy_stagnation["stagnation_steps"],
                            "pruned_aspect": weakest_aspect.name,
                            "pruned_contrib": weakest_contrib,
                        })
                    else:
                        if verbose:
                            print(f"  ⏸️ 强制剪枝跳过: 降低比例 {reduction_ratio*100:.1f}% 不足 {prune_min_reduction_ratio_forced*100:.1f}%")
                        # 即使不剪枝，也重置停滞计数器，避免持续触发
                        free_energy_stagnation["stagnation_steps"] = 0
            
            # 9. 评估准确率
            with torch.no_grad():
                logits = classification_aspect.predict(brain.objects)
                pred_label = logits.argmax().item()
                true_label = train_interface.world_model.get_label()
                correct = (pred_label == true_label)
                accuracy_history.append(1.0 if correct else 0.0)
            
            # 10. 保存快照
            if step % save_interval == 0 or step == num_steps - 1:
                structure = brain.get_network_structure()
                snapshot = {
                    "step": step,
                    "free_energy": F_total.item() if 'F_total' in locals() else 0.0,
                    "free_energy_obs": F_obs.item() if 'F_obs' in locals() else 0.0,
                    "free_energy_dyn": F_dyn.item() if 'F_dyn' in locals() else 0.0,
                    "free_energy_class": F_class.item() if 'F_class' in locals() else 0.0,
                    "accuracy": sum(accuracy_history[-100:]) / min(100, len(accuracy_history)),
                    "structure": structure,
                    "evolution_decision": evolution_decision,  # 记录演化决策
                }
                snapshots.append(snapshot)
                
                if verbose:
                    print(f"\n[Step {step}] F={F_total.item():.4f} "
                          f"(obs={F_obs.item():.4f}, dyn={F_dyn.item():.4f}, class={F_class.item():.4f}), "
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
        brain,
        val_interface,
        classification_aspect,
        num_samples=min(1000, len(val_world.dataset)),
        config=config,
        device=device,
    )
    
    # 获取演化事件
    evolution_events = []
    if brain.evolution and hasattr(brain.evolution, 'evolution_history'):
        for event in brain.evolution.evolution_history:
            evolution_events.append({
                "step": event.step,
                "event_type": event.event_type,
                "details": event.details,
                "trigger_condition": event.trigger_condition,
                "free_energy_before": event.free_energy_before,
                "free_energy_after": event.free_energy_after,
            })
    
    # 获取演化摘要
    evolution_summary = {}
    if brain.evolution:
        evolution_summary = brain.evolution.get_evolution_summary()
    
    # 保存结果
    result = {
        "num_steps": num_steps,
        "final_free_energy": free_energy_history[-1] if free_energy_history else 0.0,
        "final_accuracy": sum(accuracy_history[-100:]) / min(100, len(accuracy_history)),
        "val_accuracy": val_accuracy,
        "final_structure": brain.get_network_structure(),
        "snapshots": snapshots,
        "free_energy_history": free_energy_history,
        "F_obs_history": F_obs_history,
        "F_dyn_history": F_dyn_history,
        "F_class_history": F_class_history,
        "accuracy_history": accuracy_history,
        "evolution_events": evolution_events,
        "evolution_summary": evolution_summary,
        "evolution_decisions": evolution_decisions,  # 行动选择触发的演化决策
    }
    
    with open(output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("=" * 80)
    print("实验完成！")
    print("=" * 80)
    print(f"结果保存到: {output}")
    print(f"最终自由能: {result['final_free_energy']:.4f}")
    print(f"最终训练准确率: {result['final_accuracy']*100:.2f}%")
    print(f"验证准确率: {result['val_accuracy']*100:.2f}%")
    print(f"最终结构: {result['final_structure']['num_objects']} Objects, "
          f"{result['final_structure']['num_aspects']} Aspects")
    print("=" * 80)
    
    return result


def evaluate_accuracy(
    brain,
    world_interface,
    classification_aspect,
    num_samples: int,
    config: Dict,
    device: torch.device,
):
    """评估准确率"""
    correct = 0
    
    with torch.no_grad():
        obs = world_interface.reset()
        
        for i in range(num_samples):
            # 设置观察
            for sense, value in obs.items():
                if sense in brain.objects:
                    brain.objects[sense].set_state(value)
            
            # 获取目标
            target = world_interface.get_target()
            # 确保 target 维度正确
            if target.dim() > 1:
                target = target.squeeze()
            if target.shape[-1] != brain.objects["target"].dim:
                if target.shape[-1] > brain.objects["target"].dim:
                    target = target[..., :brain.objects["target"].dim]
                else:
                    padding = torch.zeros(brain.objects["target"].dim - target.shape[-1], device=device)
                    target = torch.cat([target, padding], dim=-1)
            brain.objects["target"].set_state(target)
            
            # 状态推理
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
                        num_iters=config.get("eval_infer_iters", 1),
                        sanitize_callback=brain.sanitize_states
                    )
                except Exception:
                    pass
            
            # 预测
            logits = classification_aspect.predict(brain.objects)
            pred_label = logits.argmax().item()
            true_label = world_interface.world_model.get_label()
            
            if pred_label == true_label:
                correct += 1
            
            # 移动到下一个样本
            action = torch.softmax(logits, dim=-1)
            obs, _ = world_interface.step(action)
    
    accuracy = correct / num_samples
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="MNIST 主动推理实验 V2")
    parser.add_argument("--steps", type=int, default=500, help="训练步数")
    parser.add_argument("--state-dim", type=int, default=128, help="状态维度")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--output", type=str, default="data/mnist_active_inference_v2.json", help="输出文件路径")
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
            "free_energy_threshold": 1.5,  # 降低阈值，更容易触发演化
            "prune_threshold": 0.01,
            "prune_min_reduction_ratio": 0.10,  # 放宽：要求至少降低 10% 才剪枝
            "prune_min_reduction_ratio_forced": 0.05,  # 强制剪枝时降低到 5%
            "prune_min_interval": 20,  # 剪枝最小间隔：至少间隔 20 步
            "prune_stagnation_steps": 150,  # 放宽：150 步无改善才强制剪枝（原 50）
            "prune_min_improvement": 0.005,  # 放宽：至少改善 0.5% 才算改善（原 1%）
            "max_objects": 20,
            "max_aspects": 500,
            "error_ema_alpha": 0.5,
            "batch_growth": {
                "base": 4,  # 启用批量创建
                "max_per_step": 8,
                "max_total": 100,
                "min_per_sense": 1,
                "error_threshold": 1.0,  # 误差阈值
                "error_multiplier": 0.7,
            },
            "pipeline_growth": {
                "enable": True,  # 启用 Pipeline 演化
                "free_energy_trigger": 1.0,  # 自由能触发阈值
                "min_interval": 10,  # 最小间隔步数
                "max_stages": 5,  # 最大 Pipeline 阶段数
                "max_depth": 6,  # 最大深度
                "initial_width": 16,  # 初始宽度
                "width_increment": 4,  # 宽度增量
                "initial_depth": 2,  # 初始深度
                "depth_increment": 1,  # 深度增量
                "use_pipeline_for_encoder": False,  # 不使用 Pipeline 作为编码器（已有 EncoderAspect）
            },
        },
        "evolution_action_threshold": 0.0,  # 演化行动阈值：已取消，只要预期自由能更低就执行
        "evolution_noise_scale": 0.01,  # 从参考 aspect 复制权重时的噪声缩放因子
        "verbose_evolution": True,  # 是否输出演化评估的详细信息
        "infer_lr": 0.01,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "classification_loss_weight": 1.0,
        "num_infer_iters": 3,
        "num_action_iters": 5,
        "action_lr": 0.1,
        "max_grad_norm": 100.0,
        "state_clip_value": 5.0,
        "eval_infer_iters": 1,
    }
    
    print("=" * 80)
    print("MNIST 主动推理实验 V2")
    print("=" * 80)
    print(f"训练步数: {args.steps}")
    print(f"状态维度: {config['state_dim']}")
    print(f"观察维度: {config['obs_dim']}")
    print(f"动作维度: {config['act_dim']}")
    print("=" * 80)
    
    result = run_experiment(
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

