# src/aonn/models/aonn_brain.py
from typing import Dict
import torch.nn as nn

from aonn.core.object import ObjectNode
from aonn.aspects.sensory_aspect import LinearGenerativeAspect
from aonn.aspects.llm_aspect import LLMAspect


class AONNBrain(nn.Module):
    """
    把 Object + 多种 Aspect 组装成一个"自由能大脑"
    """
    def __init__(self, config, llm_client=None, device=None):
        super().__init__()
        self.device = device

        # Object 层
        self.objects: Dict[str, ObjectNode] = {
            "sensory": ObjectNode("sensory", dim=config["obs_dim"], device=device),
            "internal": ObjectNode("internal", dim=config["state_dim"], device=device),
            "action": ObjectNode("action", dim=config["act_dim"], device=device),
            "semantic_context": ObjectNode("semantic_context", dim=config["sem_dim"], device=device),
            "semantic_prediction": ObjectNode("semantic_prediction", dim=config["sem_dim"], device=device),
        }

        # Aspect 层
        self.sensory_aspect = LinearGenerativeAspect(
            internal_name="internal",
            sensory_name="sensory",
            state_dim=config["state_dim"],
            obs_dim=config["obs_dim"],
        )
        self.llm_aspect = LLMAspect(
            src_names=("semantic_context",),
            dst_names=("semantic_prediction",),
            llm_client=llm_client,
            llm_config=config.get("llm", {}),
        )

        self.aspects = nn.ModuleList([
            self.sensory_aspect,
            self.llm_aspect,
            # TODO: dynamics_aspect, intent_aspect, action_aspect ...
        ])

