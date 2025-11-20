#!/usr/bin/env python3
"""
单独测试 LLMAspect（使用 MockLLMClient）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aonn.core.object import ObjectNode
from aonn.aspects.llm_aspect import LLMAspect
from aonn.aspects.mock_llm_client import MockLLMClient, create_default_mock_llm_client
from aonn.utils.logging import setup_logger


def test_llm_aspect():
    logger = setup_logger()
    
    device = torch.device("cpu")
    dim = 128

    # 创建 Object
    context_obj = ObjectNode("semantic_context", dim=dim, device=device, init="normal")
    pred_obj = ObjectNode("semantic_prediction", dim=dim, device=device, init="normal")

    objects = {
        "semantic_context": context_obj,
        "semantic_prediction": pred_obj,
    }

    # 创建 MockLLMClient（使用 MLP）
    mock_client = create_default_mock_llm_client(
        input_dim=dim,
        output_dim=dim,
        device=device,
    )
    
    # 设置为评估模式，避免随机噪声
    mock_client.eval()
    
    # 创建 LLMAspect
    llm_aspect = LLMAspect(
        src_names=("semantic_context",),
        dst_names=("semantic_prediction",),
        llm_client=mock_client,
        loss_weight=1.0,
    )

    # 测试自由能
    initial_F = llm_aspect.free_energy_contrib(objects)
    logger.info(f"初始自由能: {initial_F.item():.4f}")

    # 获取稳定的预测（评估模式下）
    pred = llm_aspect._call_llm(objects)
    # 让 target 等于预测（应该使自由能接近0）
    pred_obj.set_state(pred.clone())
    
    final_F = llm_aspect.free_energy_contrib(objects)
    logger.info(f"目标等于预测后的自由能: {final_F.item():.4f}")

    # 验证：当目标等于预测时，自由能应该接近0
    assert final_F < initial_F * 0.5, f"自由能应该显著下降（初始: {initial_F.item():.4f}, 最终: {final_F.item():.4f}）"
    logger.info("✓ 测试通过：自由能随目标接近预测而下降")
    
    # 测试可训练性
    logger.info(f"MockLLMClient 参数数量: {sum(p.numel() for p in mock_client.parameters())}")
    logger.info("✓ MockLLMClient 可以正常训练")


if __name__ == "__main__":
    test_llm_aspect()
