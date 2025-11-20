# src/aonn/pipeline/inference_agent.py
"""
推理代理：外部调用接口（"个人 AI 助手"）
"""
import torch
from typing import Dict, Any, Optional

from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.core.free_energy import compute_total_free_energy
from aonn.models.aonn_brain import AONNBrain


class InferenceAgent:
    """
    推理代理：封装 AONN 大脑的推理接口
    """
    def __init__(self, brain: AONNBrain, infer_lr: float = 0.1):
        self.brain = brain
        # 获取 aspects 列表（兼容 ModuleList 和普通列表）
        aspects = brain.aspects if isinstance(brain.aspects, list) else list(brain.aspects)
        self.infer_loop = ActiveInferenceLoop(
            objects=brain.objects,
            aspects=aspects,
            infer_lr=infer_lr,
            device=brain.device,
        )

    def observe(self, observation: torch.Tensor):
        """
        设置感官输入
        """
        self.brain.objects["sensory"].set_state(observation)

    def infer(self, num_iters: int = 5, target_objects=("internal", "action")):
        """
        执行主动推理
        """
        self.infer_loop.infer_states(
            target_objects=target_objects,
            num_iters=num_iters
        )

    def get_free_energy(self) -> float:
        """
        获取当前自由能
        """
        # 获取 aspects 列表（兼容 ModuleList 和普通列表）
        aspects = self.brain.aspects if isinstance(self.brain.aspects, list) else list(self.brain.aspects)
        F = compute_total_free_energy(
            self.brain.objects,
            aspects
        )
        return F.item()

    def get_action(self) -> torch.Tensor:
        """
        获取当前动作
        """
        return self.brain.objects["action"].state.clone()

    def get_internal_state(self) -> torch.Tensor:
        """
        获取内部状态
        """
        return self.brain.objects["internal"].state.clone()

