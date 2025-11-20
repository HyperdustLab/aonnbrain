# src/aonn/aspects/sensory_aspect.py
import torch
import torch.nn as nn
from typing import Dict

from aonn.core.object import ObjectNode
from aonn.core.aspect_base import AspectBase


class LinearGenerativeAspect(AspectBase, nn.Module):
    """
    internal -> sensory 的线性生成模型：
    pred = W * mu_internal
    F = 0.5 * ||sensory - pred||^2
    """

    def __init__(self, internal_name="internal", sensory_name="sensory",
                 state_dim=128, obs_dim=128):
        AspectBase.__init__(self,
                            name="linear_int2sens",
                            src_names=[internal_name],
                            dst_names=[sensory_name])
        nn.Module.__init__(self)

        self.W = nn.Parameter(0.1 * torch.randn(obs_dim, state_dim))

    def forward(self, objects: Dict[str, ObjectNode]):
        mu_int = objects[self.src_names[0]].state     # internal
        mu_obs = objects[self.dst_names[0]].state     # sensory
        pred = self.W @ mu_int
        error = mu_obs - pred
        return {self.dst_names[0]: error}

    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        mu_int = objects[self.src_names[0]].state
        mu_obs = objects[self.dst_names[0]].state
        pred = self.W @ mu_int
        error = mu_obs - pred
        return 0.5 * (error ** 2).sum()

    def parameters(self):
        return [self.W]

