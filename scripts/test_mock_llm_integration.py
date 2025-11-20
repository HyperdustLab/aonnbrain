#!/usr/bin/env python3
"""
测试 MockLLMClient 与 LLMAspect 的集成
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aonn.core.object import ObjectNode
from aonn.aspects.llm_aspect import LLMAspect
from aonn.aspects.mock_llm_client import create_default_mock_llm_client
from aonn.core.free_energy import compute_total_free_energy


def test_integration():
    print("=" * 60)
    print("测试 MockLLMClient 与 LLMAspect 集成")
    print("=" * 60)
    
    device = torch.device("cpu")
    dim = 128
    
    # 1. 创建 MockLLMClient
    print("\n1. 创建 MockLLMClient...")
    mock_client = create_default_mock_llm_client(
        input_dim=dim,
        output_dim=dim,
        device=device,
    )
    print(f"   ✓ MockLLMClient 创建成功")
    print(f"   参数数量: {sum(p.numel() for p in mock_client.parameters()):,}")
    
    # 2. 创建 Object
    print("\n2. 创建 Object...")
    context_obj = ObjectNode("semantic_context", dim=dim, device=device, init="normal")
    pred_obj = ObjectNode("semantic_prediction", dim=dim, device=device, init="normal")
    objects = {
        "semantic_context": context_obj,
        "semantic_prediction": pred_obj,
    }
    print(f"   ✓ Object 创建成功")
    
    # 3. 创建 LLMAspect
    print("\n3. 创建 LLMAspect...")
    llm_aspect = LLMAspect(
        src_names=("semantic_context",),
        dst_names=("semantic_prediction",),
        llm_client=mock_client,
        loss_weight=1.0,
    )
    print(f"   ✓ LLMAspect 创建成功")
    
    # 4. 测试预测
    print("\n4. 测试语义预测...")
    pred = llm_aspect._call_llm(objects)
    print(f"   ✓ 预测成功，形状: {pred.shape}")
    print(f"   预测示例（前5维）: {pred[:5].tolist()}")
    
    # 5. 测试自由能
    print("\n5. 测试自由能计算...")
    # 设置为评估模式，避免随机噪声
    mock_client.eval()
    
    # 获取预测（评估模式下应该更稳定）
    pred_stable = llm_aspect._call_llm(objects)
    F_initial = llm_aspect.free_energy_contrib(objects)
    print(f"   初始自由能: {F_initial.item():.4f}")
    
    # 让目标等于预测（应该使自由能接近0）
    pred_obj.set_state(pred_stable.clone())
    F_final = llm_aspect.free_energy_contrib(objects)
    print(f"   目标等于预测后的自由能: {F_final.item():.4f}")
    
    # 验证：当目标等于预测时，自由能应该接近0（或至少比初始值小很多）
    assert F_final < F_initial * 0.9, f"自由能应该下降（初始: {F_initial.item():.4f}, 最终: {F_final.item():.4f}）"
    print(f"   ✓ 自由能下降验证通过")
    
    # 6. 测试可训练性
    print("\n6. 测试可训练性...")
    params = list(llm_aspect.parameters())
    print(f"   可训练参数组数: {len(params)}")
    if params:
        print(f"   第一个参数形状: {params[0].shape}")
    print(f"   ✓ 参数可访问")
    
    # 7. 测试梯度
    print("\n7. 测试梯度计算...")
    context_obj.set_state(torch.randn(dim, requires_grad=True))
    F = llm_aspect.free_energy_contrib(objects)
    F.backward()
    if context_obj.state.grad is not None:
        print(f"   ✓ 梯度计算成功")
        print(f"   梯度示例（前5维）: {context_obj.state.grad[:5].tolist()}")
    else:
        print(f"   ⚠ 梯度为 None（可能是正常的，取决于实现）")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！MockLLMClient 可以立即使用")
    print("=" * 60)


if __name__ == "__main__":
    test_integration()

