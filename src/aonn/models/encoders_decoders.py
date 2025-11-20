# src/aonn/models/encoders_decoders.py
"""
文本/多模态编码解码器
用于 LLMAspect 的文本 <-> 向量转换
"""
import torch
import torch.nn as nn
from typing import Optional


class TextEncoder(nn.Module):
    """
    文本编码器：文本 -> 向量
    """
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        text_ids: [batch, seq_len]
        返回: [batch, embed_dim]
        """
        embeds = self.embedding(text_ids)
        output, (hidden, _) = self.encoder(embeds)
        # 使用最后一个时间步的隐藏状态
        return self.proj(hidden[-1])


class TextDecoder(nn.Module):
    """
    文本解码器：向量 -> 文本
    """
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.embed_dim = embed_dim

    def forward(self, vec: torch.Tensor, max_len: int = 50) -> torch.Tensor:
        """
        vec: [batch, embed_dim]
        返回: [batch, max_len, vocab_size]
        """
        batch_size = vec.shape[0]
        # 将向量扩展为序列
        input_vec = vec.unsqueeze(1).expand(-1, max_len, -1)
        output, _ = self.decoder(input_vec)
        logits = self.proj(output)
        return logits


class SimpleEmbeddingEncoder(nn.Module):
    """
    简单的嵌入编码器（用于快速原型）
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

