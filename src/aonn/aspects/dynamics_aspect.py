# src/aonn/aspects/dynamics_aspect.py
"""
Dynamics Aspect: 预测状态转移
"""
import torch
import torch.nn as nn
from typing import Dict

from aonn.core.object import ObjectNode
from aonn.core.aspect_base import AspectBase


class DynamicsAspect(AspectBase, nn.Module):
    """
    状态转移模型：internal_t -> internal_{t+1}
    """
    def __init__(self, internal_name="internal", state_dim=128):
        AspectBase.__init__(self,
                            name="dynamics",
                            src_names=[internal_name],
                            dst_names=[internal_name])
        nn.Module.__init__(self)
        self.state_dim = state_dim
        self.transition = nn.Linear(state_dim, state_dim)

    def forward(self, objects: Dict[str, ObjectNode]):
        mu = objects[self.src_names[0]].state
        pred_next = self.transition(mu)
        error = pred_next - mu  # 简化：预测下一状态与当前状态的差异
        return {self.dst_names[0]: error}

    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        mu = objects[self.src_names[0]].state
        pred_next = self.transition(mu)
        error = pred_next - mu
        return 0.5 * (error ** 2).sum()

    def parameters(self):
        return list(self.transition.parameters())

