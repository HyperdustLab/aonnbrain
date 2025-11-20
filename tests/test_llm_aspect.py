"""
测试 LLMAspect
"""
import torch
from aonn.core.object import ObjectNode
from aonn.aspects.llm_aspect import LLMAspect
from aonn.aspects.mock_llm_client import MockLLMClient, create_default_mock_llm_client


def test_llm_aspect_basic():
    dim = 64
    context = ObjectNode("semantic_context", dim=dim, init="normal")
    prediction = ObjectNode("semantic_prediction", dim=dim, init="normal")
    
    objects = {
        "semantic_context": context,
        "semantic_prediction": prediction,
    }
    
    # 使用 MockLLMClient
    mock_client = create_default_mock_llm_client(
        input_dim=dim,
        output_dim=dim,
    )
    
    llm_aspect = LLMAspect(
        src_names=("semantic_context",),
        dst_names=("semantic_prediction",),
        llm_client=mock_client,
    )
    
    # 测试自由能
    F = llm_aspect.free_energy_contrib(objects)
    assert F.item() >= 0
    
    # 让目标接近预测
    pred = llm_aspect._call_llm(objects)
    prediction.set_state(pred)
    F_low = llm_aspect.free_energy_contrib(objects)
    assert F_low < F


def test_llm_aspect_with_mock_client():
    """测试 LLMAspect 与 MockLLMClient 的集成"""
    dim = 128
    context = ObjectNode("semantic_context", dim=dim, init="normal")
    prediction = ObjectNode("semantic_prediction", dim=dim, init="normal")
    
    objects = {
        "semantic_context": context,
        "semantic_prediction": prediction,
    }
    
    # 创建可训练的 MockLLMClient
    mock_client = MockLLMClient(
        input_dim=dim,
        output_dim=dim,
        hidden_dims=[128, 256],
        trainable=True,
    )
    
    llm_aspect = LLMAspect(
        src_names=("semantic_context",),
        dst_names=("semantic_prediction",),
        llm_client=mock_client,
    )
    
    # 测试参数
    params = list(llm_aspect.parameters())
    assert len(params) > 0, "LLMAspect 应该包含可训练参数"
    
    # 测试前向传播
    pred = llm_aspect._call_llm(objects)
    assert pred.shape == (dim,), f"预测形状应该是 ({dim},)，实际是 {pred.shape}"
    
    # 测试自由能
    F = llm_aspect.free_energy_contrib(objects)
    assert F.item() >= 0

