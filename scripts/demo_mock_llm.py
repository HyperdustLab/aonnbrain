#!/usr/bin/env python3
"""
演示 MockLLMClient 的使用
"""
import torch
from aonn.aspects.mock_llm_client import MockLLMClient, create_default_mock_llm_client
from aonn.utils.logging import setup_logger


def main():
    logger = setup_logger()
    
    device = torch.device("cpu")
    dim = 128
    
    logger.info("=" * 60)
    logger.info("MockLLMClient 演示")
    logger.info("=" * 60)
    
    # 方法1：使用便捷函数创建默认客户端
    logger.info("\n1. 使用 create_default_mock_llm_client() 创建客户端")
    client1 = create_default_mock_llm_client(
        input_dim=dim,
        output_dim=dim,
        device=device,
    )
    logger.info(f"   参数数量: {sum(p.numel() for p in client1.parameters()):,}")
    
    # 测试预测
    context = torch.randn(dim)
    pred1 = client1.semantic_predict(context)
    logger.info(f"   输入形状: {context.shape}, 输出形状: {pred1.shape}")
    logger.info(f"   预测示例（前5维）: {pred1[:5].tolist()}")
    
    # 方法2：自定义创建客户端
    logger.info("\n2. 自定义创建 MockLLMClient")
    client2 = MockLLMClient(
        input_dim=dim,
        output_dim=dim,
        hidden_dims=[128, 256, 128],
        activation="tanh",
        trainable=True,
        noise_scale=0.1,
        device=device,
    )
    logger.info(f"   参数数量: {sum(p.numel() for p in client2.parameters()):,}")
    
    pred2 = client2.semantic_predict(context, temperature=0.8)
    logger.info(f"   预测示例（前5维）: {pred2[:5].tolist()}")
    
    # 方法3：创建不可训练的客户端（固定参数）
    logger.info("\n3. 创建不可训练的客户端（固定参数）")
    client3 = MockLLMClient(
        input_dim=dim,
        output_dim=dim,
        trainable=False,
        device=device,
    )
    logger.info(f"   可训练参数数量: {sum(p.numel() for p in client3.parameters())}")
    
    # 测试批量预测
    logger.info("\n4. 批量预测测试")
    batch_context = torch.randn(5, dim)
    batch_pred = client1.semantic_predict(batch_context)
    logger.info(f"   批量输入形状: {batch_context.shape}")
    logger.info(f"   批量输出形状: {batch_pred.shape}")
    
    # 测试训练模式
    logger.info("\n5. 训练模式测试")
    client1.train()
    pred_train = client1.semantic_predict(context)
    logger.info(f"   训练模式预测（前5维）: {pred_train[:5].tolist()}")
    
    client1.eval()
    pred_eval = client1.semantic_predict(context)
    logger.info(f"   评估模式预测（前5维）: {pred_eval[:5].tolist()}")
    
    logger.info("\n" + "=" * 60)
    logger.info("演示完成！MockLLMClient 可以立即用于 LLMAspect")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

