#!/usr/bin/env python3
"""
训练 AONN 大脑
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import yaml
from pathlib import Path

from aonn.models.aonn_brain import AONNBrain
from aonn.pipeline.dataset import AONNDataset
from aonn.pipeline.training_loop import train_epoch
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

    # 创建模拟 LLM 客户端（使用 MLP）
    from aonn.aspects.mock_llm_client import create_default_mock_llm_client
    llm_client = create_default_mock_llm_client(
        input_dim=config["sem_dim"],
        output_dim=config["sem_dim"],
        device=device,
    )
    logger.info("使用 MockLLMClient (MLP 模拟 LLM)")

    brain = AONNBrain(config=config, llm_client=llm_client, device=device).to(device)

    # 加载数据集
    dataset_path = "data/processed/aonn_dataset.pt"
    if not Path(dataset_path).exists():
        logger.warning(f"数据集不存在: {dataset_path}，创建占位数据集...")
        # 创建占位数据
        placeholder_data = {
            "obs": [torch.randn(config["obs_dim"]) for _ in range(100)],
            "semantic_target": [torch.randn(config["sem_dim"]) for _ in range(100)],
        }
        Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(placeholder_data, dataset_path)

    dataset = AONNDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # 优化器
    optimizer = Adam(brain.parameters(), lr=config["learning_rate"])

    # 训练循环
    for epoch in range(config["num_epochs"]):
        avg_F = train_epoch(brain, dataloader, optimizer, device=device)
        logger.info(f"[epoch {epoch}] avg free energy = {avg_F:.4f}")

        # 保存检查点
        if (epoch + 1) % config.get("save_interval", 10) == 0:
            checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"brain_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": brain.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_free_energy": avg_F,
            }, checkpoint_path)
            logger.info(f"检查点已保存: {checkpoint_path}")


if __name__ == "__main__":
    main()

