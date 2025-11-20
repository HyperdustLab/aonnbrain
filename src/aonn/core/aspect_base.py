# src/aonn/core/aspect_base.py
from abc import ABC, abstractmethod
from typing import Dict
import torch

from .object import ObjectNode


class AspectBase(ABC):
    """
    AONN 中的 Aspect 抽象：
    - 读取若干 ObjectNode 的 state
    - 输出 prediction error / 消息
    - 提供自由能贡献 F
    """

    def __init__(self, name: str, src_names, dst_names):
        self.name = name
        self.src_names = src_names if isinstance(src_names, (list, tuple)) else [src_names]
        self.dst_names = dst_names if isinstance(dst_names, (list, tuple)) else [dst_names]

    @abstractmethod
    def forward(self, objects: Dict[str, ObjectNode]) -> Dict[str, torch.Tensor]:
        """
        返回：每个 dst_object 的误差信号或局部梯度贡献
        例如：
            {
                "sensory": error_vec,
                "internal": grad_mu_internal_like_term
            }
        """
        raise NotImplementedError

    @abstractmethod
    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """
        返回该 Aspect 对总自由能 F 的标量贡献。
        """
        raise NotImplementedError

    def parameters(self):
        """
        有些 Aspect（比如 LLMAspect）可能没有传统意义上的 torch.Parameter，
        默认返回空 list，由子类根据需要覆写。
        """
        return []

