# src/aonn/core/active_inference_loop.py
from typing import Dict, List
import torch

from .object import ObjectNode
from .aspect_base import AspectBase
from .free_energy import compute_total_free_energy


class ActiveInferenceLoop:
    """
    主动推理循环：
    - 固定感官 Object
    - 对 internal / action 等 Object 做状态推理（gradient descent）
    - 可选执行一次参数学习
    """

    def __init__(
        self,
        objects: Dict[str, ObjectNode],
        aspects: List[AspectBase],
        infer_lr: float = 0.1,
        device=None,
    ):
        self.objects = objects
        self.aspects = aspects
        self.infer_lr = infer_lr
        self.device = device or torch.device("cpu")

    def infer_states(self, target_objects=("internal", "action"), num_iters: int = 5):
        """
        对指定 Object 节点（如 internal, action）执行有限步自由能下降更新。
        """
        for _ in range(num_iters):
            # 为每个待推理 Object 重置为可微叶子节点
            for name in target_objects:
                mu = self.objects[name].clone_detached(requires_grad=True)
                self.objects[name].state = mu

            # 梯度清零（这里只对 state 做手动更新）
            for name in target_objects:
                if self.objects[name].state.grad is not None:
                    self.objects[name].state.grad.zero_()

            F = compute_total_free_energy(self.objects, self.aspects)
            F.backward()

            with torch.no_grad():
                for name in target_objects:
                    mu = self.objects[name].state
                    if mu.grad is not None:
                        mu = mu - self.infer_lr * mu.grad
                    self.objects[name].state = mu.detach()

        return self.objects

