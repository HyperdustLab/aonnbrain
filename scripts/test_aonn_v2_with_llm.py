#!/usr/bin/env python3
"""
测试 AONN Brain V2 与 LLMAspect 的集成
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aonn.models.aonn_brain_v2 import AONNBrainV2
from aonn.aspects.mock_llm_client import create_default_mock_llm_client


def test_aonn_v2_with_llm():
    print("=" * 70)
    print("测试 AONN Brain V2 与 LLMAspect 集成")
    print("=" * 70)
    
    # 配置
    config = {
        "input_dim": 128,
        "hidden_dims": [256, 256],
        "output_dim": 10,
        "num_aspects": 32,
        "aspect_depth": 4,
        "use_gate": False,
        "sem_dim": 128,  # 语义维度
    }
    
    device = torch.device("cpu")
    
    # 创建 LLM 客户端
    print("\n1. 创建 MockLLMClient...")
    llm_client = create_default_mock_llm_client(
        input_dim=config["sem_dim"],
        output_dim=config["sem_dim"],
        device=device,
    )
    print("   ✓ LLM 客户端创建成功")
    
    # 创建模型（带 LLMAspect）
    print("\n2. 创建 AONN Brain V2（带 LLMAspect）...")
    brain = AONNBrainV2(config=config, llm_client=llm_client, device=device)
    print("   ✓ 创建成功")
    print(f"   - 是否有 LLMAspect: {brain.has_llm_aspect()}")
    print(f"   - 语义 Object 数量: {len(brain.semantic_objects)}")
    
    # 显示网络结构
    print("\n3. 网络结构：")
    print(brain.visualize_network())
    
    # 测试前向传播
    print("\n4. 测试前向传播...")
    batch_size = 4
    x = torch.randn(batch_size, config["input_dim"])
    print(f"   输入形状: {x.shape}")
    
    brain.eval()
    with torch.no_grad():
        y = brain(x)
    
    print(f"   输出形状: {y.shape}")
    print(f"   ✓ 前向传播成功")
    
    # 测试自由能计算
    if brain.has_llm_aspect():
        print("\n5. 测试自由能计算...")
        # 设置语义 Object 的状态
        brain.semantic_objects["semantic_context"].set_state(
            torch.randn(config["sem_dim"])
        )
        brain.semantic_objects["semantic_prediction"].set_state(
            torch.randn(config["sem_dim"])
        )
        
        F = brain.compute_free_energy()
        print(f"   自由能: {F.item():.4f}")
        print(f"   ✓ 自由能计算成功")
    
    # 获取所有 Object
    print("\n6. 所有 Object：")
    all_objects = brain.get_all_objects()
    print(f"   总 Object 数: {len(all_objects)}")
    for name, obj in all_objects.items():
        print(f"     - {name}: dim={obj.dim}")
    
    print("\n" + "=" * 70)
    print("✓ 所有测试通过！AONN Brain V2 与 LLMAspect 集成成功")
    print("=" * 70)


if __name__ == "__main__":
    test_aonn_v2_with_llm()

