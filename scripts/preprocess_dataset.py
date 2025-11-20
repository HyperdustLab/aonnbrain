#!/usr/bin/env python3
"""
数据预处理脚本：将原始数据转换为训练格式
"""
import torch
import json
from pathlib import Path
from typing import List, Dict, Any


def preprocess_dialogs(jsonl_path: str, output_path: str, obs_dim: int = 128, sem_dim: int = 128):
    """
    预处理对话数据
    """
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            samples.append(data)

    # 转换为张量格式（这里简化处理，实际需要文本编码）
    processed = {
        "obs": [],
        "semantic_target": [],
    }

    for sample in samples:
        # 占位：实际需要文本编码器
        obs = torch.randn(obs_dim)
        sem_target = torch.randn(sem_dim)
        processed["obs"].append(obs)
        processed["semantic_target"].append(sem_target)

    # 保存
    torch.save(processed, output_path)
    print(f"预处理完成：{len(samples)} 个样本已保存到 {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/dialogs.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/aonn_dataset.pt")
    parser.add_argument("--obs_dim", type=int, default=128)
    parser.add_argument("--sem_dim", type=int, default=128)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preprocess_dialogs(args.input, args.output, args.obs_dim, args.sem_dim)

