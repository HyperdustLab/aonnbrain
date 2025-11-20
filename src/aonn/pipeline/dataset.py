# src/aonn/pipeline/dataset.py
"""
数据集加载器
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, Any
import json


class AONNDataset(Dataset):
    """
    AONN 数据集：从预处理后的 .pt 文件加载
    """
    def __init__(self, data_path: str):
        """
        data_path: 指向 data/processed/aonn_dataset.pt
        期望格式：dict with keys: "obs", "semantic_target", etc.
        """
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data.get("obs", []))

    def __getitem__(self, idx):
        return {
            "obs": self.data["obs"][idx],
            "semantic_target": self.data.get("semantic_target", [None])[idx],
        }


class JSONLDataset(Dataset):
    """
    从 JSONL 文件加载原始对话数据
    """
    def __init__(self, jsonl_path: str, max_samples: int = None):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

