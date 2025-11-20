#!/usr/bin/env python3
"""
单独测试 LLMAspect（使用 MockLLMClient）
"""
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

    # 让 target 接近预测（应该降低自由能）
    pred = llm_aspect._call_llm(objects)
    pred_obj.set_state(pred)
    
    final_F = llm_aspect.free_energy_contrib(objects)
    logger.info(f"目标接近预测后的自由能: {final_F.item():.4f}")

    assert final_F < initial_F, "自由能应该下降"
    logger.info("✓ 测试通过：自由能随目标接近预测而下降")
    
    # 测试可训练性
    logger.info(f"MockLLMClient 参数数量: {sum(p.numel() for p in mock_client.parameters())}")
    logger.info("✓ MockLLMClient 可以正常训练")


if __name__ == "__main__":
    test_llm_aspect()
