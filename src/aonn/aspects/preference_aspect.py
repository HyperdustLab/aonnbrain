# src/aonn/aspects/preference_aspect.py
"""
Preference Aspect: 偏好预测
"""
import torch
import torch.nn as nn
from typing import Dict

from aonn.core.object import ObjectNode
from aonn.core.aspect_base import AspectBase


class PreferenceAspect(AspectBase, nn.Module):
    """
    偏好预测：internal -> preference
    """
    def __init__(self, internal_name="internal", preference_name="preference",
                 state_dim=128, pref_dim=32):
        AspectBase.__init__(self,
                            name="preference",
                            src_names=[internal_name],
                            dst_names=[preference_name])
        nn.Module.__init__(self)
        self.pref_predictor = nn.Linear(state_dim, pref_dim)

    def forward(self, objects: Dict[str, ObjectNode]):
        mu_int = objects[self.src_names[0]].state
        mu_pref = objects[self.dst_names[0]].state
        pred_pref = self.pref_predictor(mu_int)
        error = mu_pref - pred_pref
        return {self.dst_names[0]: error}

    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        mu_int = objects[self.src_names[0]].state
        mu_pref = objects[self.dst_names[0]].state
        pred_pref = self.pref_predictor(mu_int)
        error = mu_pref - pred_pref
        return 0.5 * (error ** 2).sum()

    def parameters(self):
        return list(self.pref_predictor.parameters())

