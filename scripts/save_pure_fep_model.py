#!/usr/bin/env python3
"""
从改进版纯 FEP MNIST 实验结果中保存模型权重
注意：由于实验脚本没有保存权重，这个脚本需要重新训练或从检查点加载
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import argparse
import torch
import torch.nn as nn

from aonn.aspects.encoder_aspect import EncoderAspect
from aonn.aspects.world_model_aspects import ObservationAspect, DynamicsAspect, PreferenceAspect
from aonn.core.object import ObjectNode
from aonn.core.active_inference_loop import ActiveInferenceLoop


def create_fep_system(config: dict, device: torch.device):
    """创建 FEP 系统（与训练时一致）"""
    state_dim = config.get("state_dim", 128)
    obs_dim = config.get("obs_dim", 784)
    action_dim = config.get("action_dim", 10)
    use_conv = config.get("use_conv", True)
    
    # 创建 Objects
    objects = {
        "vision": ObjectNode("vision", obs_dim, device=device),
        "internal": ObjectNode("internal", state_dim, device=device, init="normal"),
        "action": ObjectNode("action", action_dim, device=device),
        "target": ObjectNode("target", action_dim, device=device),
    }
    
    # 创建生成模型 Aspects
    encoder = EncoderAspect(
        sensory_name="vision",
        internal_name="internal",
        input_dim=obs_dim,
        output_dim=state_dim,
        use_conv=use_conv,
        image_size=28 if use_conv else None,
    ).to(device)
    
    observation = ObservationAspect(
        internal_name="internal",
        sensory_name="vision",
        state_dim=state_dim,
        obs_dim=obs_dim,
        use_conv=use_conv,
        image_size=28 if use_conv else None,
    ).to(device)
    
    dynamics = DynamicsAspect(
        internal_name="internal",
        action_name="action",
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)
    
    preference = PreferenceAspect(
        internal_name="internal",
        target_name="target",
        state_dim=state_dim,
        weight=1.0,
    ).to(device)
    
    # 独立分类器
    classifier = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    
    return {
        "encoder": encoder,
        "observation": observation,
        "dynamics": dynamics,
        "preference": preference,
        "classifier": classifier,
    }


def main():
    parser = argparse.ArgumentParser(description="保存纯 FEP MNIST 模型权重")
    parser.add_argument("--experiment", type=str, 
                        default="data/pure_fep_mnist_improved_60000steps.json",
                        help="实验结果JSON文件路径")
    parser.add_argument("--output", type=str,
                        default="data/pure_fep_mnist_model.pth",
                        help="模型权重输出路径")
    
    args = parser.parse_args()
    
    device = torch.device("cpu")
    
    print("=" * 80)
    print("保存纯 FEP MNIST 模型权重")
    print("=" * 80)
    print(f"加载实验结果: {args.experiment}")
    
    # 加载配置
    with open(args.experiment, 'r') as f:
        data = json.load(f)
    
    config = data.get('config', {})
    
    # 创建模型
    models = create_fep_system(config, device)
    
    print("⚠️  注意：实验脚本没有保存模型权重")
    print("   当前保存的是随机初始化的模型架构")
    print("   要使用训练好的模型，需要修改实验脚本以保存权重")
    print()
    
    # 保存模型权重
    checkpoint = {
        "config": config,
        "encoder": models["encoder"].state_dict(),
        "observation": models["observation"].state_dict(),
        "dynamics": models["dynamics"].state_dict(),
        "preference": models["preference"].state_dict(),
        "classifier": models["classifier"].state_dict(),
    }
    
    torch.save(checkpoint, args.output)
    print(f"✅ 模型权重已保存到: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()

