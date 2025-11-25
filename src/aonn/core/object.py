# src/aonn/core/object.py
from typing import Any, Dict, Optional

import torch


class ObjectNode:
    """
    AONN 中的 Object：只负责存状态 μ，不做计算。但允许附加 metadata（文本、标签等）。
    """

    def __init__(self, name: str, dim: int, device=None, init="zero"):
        self.name = name
        self.dim = dim
        self.device = device or torch.device("cpu")

        if init == "zero":
            self.state = torch.zeros(dim, device=self.device)
        elif init == "normal":
            self.state = torch.randn(dim, device=self.device) * 0.01
        else:
            raise ValueError(f"Unknown init: {init}")

        self.metadata: Dict[str, Any] = {}

    def set_state(self, tensor: torch.Tensor):
        assert tensor.shape[-1] == self.dim
        self.state = tensor.to(self.device)

    def set_metadata(self, metadata: Optional[Dict[str, Any]]):
        self.metadata = metadata or {}

    def clone_detached(self, requires_grad: bool = False):
        x = self.state.detach().clone()
        x.requires_grad_(requires_grad)
        return x

