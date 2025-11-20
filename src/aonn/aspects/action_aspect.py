# src/aonn/aspects/action_aspect.py
"""
Action Aspect: 动作预测/生成
"""
import torch
import torch.nn as nn
from typing import Dict

from aonn.core.object import ObjectNode
from aonn.core.aspect_base import AspectBase


class ActionAspect(AspectBase, nn.Module):
    """
    动作生成：internal + intent -> action
    """
    def __init__(self, internal_name="internal", intent_name="intent",
                 action_name="action", state_dim=128, intent_dim=64, act_dim=32):
        AspectBase.__init__(self,
                            name="action",
                            src_names=[internal_name, intent_name],
                            dst_names=[action_name])
        nn.Module.__init__(self)
        self.action_predictor = nn.Linear(state_dim + intent_dim, act_dim)

    def forward(self, objects: Dict[str, ObjectNode]):
        mu_int = objects[self.src_names[0]].state
        mu_intent = objects[self.src_names[1]].state
        mu_action = objects[self.dst_names[0]].state
        combined = torch.cat([mu_int, mu_intent], dim=-1)
        pred_action = self.action_predictor(combined)
        error = mu_action - pred_action
        return {self.dst_names[0]: error}

    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        mu_int = objects[self.src_names[0]].state
        mu_intent = objects[self.src_names[1]].state
        mu_action = objects[self.dst_names[0]].state
        combined = torch.cat([mu_int, mu_intent], dim=-1)
        pred_action = self.action_predictor(combined)
        error = mu_action - pred_action
        return 0.5 * (error ** 2).sum()

    def parameters(self):
        return list(self.action_predictor.parameters())

