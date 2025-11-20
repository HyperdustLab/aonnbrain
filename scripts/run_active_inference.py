#!/usr/bin/env python3
"""
主动推理循环 demo
"""
import torch
import yaml
from pathlib import Path

from aonn.models.aonn_brain import AONNBrain
from aonn.pipeline.inference_agent import InferenceAgent
from aonn.utils.config import load_config
from aonn.utils.logging import setup_logger


def main():
    logger = setup_logger()
    
    # 加载配置
    config = load_config("configs/brain_default.yaml")
    train_config = load_config("configs/training_default.yaml")
    config.update(train_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建模拟 LLM 客户端
    from aonn.aspects.mock_llm_client import create_default_mock_llm_client
    llm_client = create_default_mock_llm_client(
        input_dim=config["sem_dim"],
        output_dim=config["sem_dim"],
        device=device,
    )
    logger.info("使用 MockLLMClient (MLP 模拟 LLM)")

    # 创建大脑（可以加载检查点）
    brain = AONNBrain(config=config, llm_client=llm_client, device=device).to(device)

    # 创建推理代理
    agent = InferenceAgent(brain, infer_lr=config["inference"]["infer_lr"])

    # 模拟观察
    obs = torch.randn(config["obs_dim"]).to(device)
    agent.observe(obs)
    logger.info(f"初始自由能: {agent.get_free_energy():.4f}")

    # 执行主动推理
    num_iters = config["inference"]["num_infer_iters"]
    target_objects = config["inference"]["target_objects"]
    
    logger.info(f"开始主动推理（{num_iters} 步）...")
    for i in range(num_iters):
        agent.infer(num_iters=1, target_objects=target_objects)
        F = agent.get_free_energy()
        logger.info(f"  步骤 {i+1}: 自由能 = {F:.4f}")

    logger.info(f"最终自由能: {agent.get_free_energy():.4f}")
    logger.info(f"内部状态: {agent.get_internal_state()[:5]}")  # 显示前5维
    logger.info(f"动作: {agent.get_action()[:5]}")


if __name__ == "__main__":
    main()

