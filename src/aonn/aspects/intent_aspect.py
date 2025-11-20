# src/aonn/aspects/intent_aspect.py
"""
Intent Aspect: 意图预测
"""
import torch
import torch.nn as nn
from typing import Dict

from aonn.core.object import ObjectNode
from aonn.core.aspect_base import AspectBase


class IntentAspect(AspectBase, nn.Module):
    """
    意图预测：internal -> intent
    """
    def __init__(self, internal_name="internal", intent_name="intent",
                 state_dim=128, intent_dim=64):
        AspectBase.__init__(self,
                            name="intent",
                            src_names=[internal_name],
                            dst_names=[intent_name])
        nn.Module.__init__(self)
        self.intent_predictor = nn.Linear(state_dim, intent_dim)

    def forward(self, objects: Dict[str, ObjectNode]):
        mu_int = objects[self.src_names[0]].state
        mu_intent = objects[self.dst_names[0]].state
        pred_intent = self.intent_predictor(mu_int)
        error = mu_intent - pred_intent
        return {self.dst_names[0]: error}

    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        mu_int = objects[self.src_names[0]].state
        mu_intent = objects[self.dst_names[0]].state
        pred_intent = self.intent_predictor(mu_int)
        error = mu_intent - pred_intent
        return 0.5 * (error ** 2).sum()

    def parameters(self):
        return list(self.intent_predictor.parameters())

