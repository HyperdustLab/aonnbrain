# src/aonn/pipeline/training_loop.py
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from aonn.core.free_energy import compute_total_free_energy
from aonn.core.object import ObjectNode
from aonn.core.aspect_base import AspectBase


def train_epoch(
    brain,
    dataloader: DataLoader,
    optimizer: Adam,
    device=None,
):
    brain.train()
    total_F = 0.0
    for batch in dataloader:
        # 假设 batch 包含 "obs", "semantic_target" 等
        obs = batch["obs"].to(device)
        sem_target = batch["semantic_target"].to(device)

        # 绑定到 Object 上
        brain.objects["sensory"].set_state(obs)
        brain.objects["semantic_prediction"].set_state(sem_target)

        # 先做状态推理（可选）
        # 这里只给出简单形式，你可以引入 ActiveInferenceLoop
        # ...

        optimizer.zero_grad()
        # 使用 aspects 列表（包含所有 aspects）
        aspects = brain.aspects if hasattr(brain, 'aspects') and isinstance(brain.aspects, list) else list(brain.aspects)
        F = compute_total_free_energy(brain.objects, aspects)
        F.backward()
        optimizer.step()

        total_F += F.item()

    return total_F / len(dataloader)

