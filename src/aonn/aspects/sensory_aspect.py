# src/aonn/aspects/sensory_aspect.py
import torch
import torch.nn as nn
from typing import Dict, Optional

from aonn.core.object import ObjectNode
from aonn.core.aspect_base import AspectBase


class LinearGenerativeAspect(AspectBase, nn.Module):
    """
    internal -> sensory 的线性生成模型：
    pred = W * mu_internal
    F = 0.5 * ||sensory - pred||^2
    """

    def __init__(
        self,
        internal_name="internal",
        sensory_name="sensory",
        state_dim=128,
        obs_dim=128,
        name: str = None,
        init_weight: Optional[torch.Tensor] = None,
        reference_aspect: Optional["LinearGenerativeAspect"] = None,
        init_scale: float = 0.1,
        noise_scale: float = 0.01,
    ):
        """
        初始化 LinearGenerativeAspect
        
        Args:
            internal_name: 源 Object 名称
            sensory_name: 目标 Object 名称
            state_dim: 内部状态维度
            obs_dim: 观察维度
            name: Aspect 名称
            init_weight: 直接提供的初始权重（如果提供，优先使用）
            reference_aspect: 参考 Aspect，从其复制权重并添加小噪声
            init_scale: 随机初始化的缩放因子（当没有参考时）
            noise_scale: 从参考 Aspect 复制时的噪声缩放因子
        """
        AspectBase.__init__(self,
                            name=name or f"linear_int2sens_{sensory_name}",
                            src_names=[internal_name],
                            dst_names=[sensory_name])
        nn.Module.__init__(self)

        if init_weight is not None:
            # 直接使用提供的权重
            assert init_weight.shape == (obs_dim, state_dim), \
                f"init_weight shape {init_weight.shape} != ({obs_dim}, {state_dim})"
            self.W = nn.Parameter(init_weight.clone())
        elif reference_aspect is not None:
            # 从参考 Aspect 复制权重并添加小噪声
            ref_W = reference_aspect.W.detach().clone()
            # 确保维度匹配
            if ref_W.shape == (obs_dim, state_dim):
                # 添加小噪声，保持相似性但允许探索
                noise = noise_scale * torch.randn_like(ref_W)
                self.W = nn.Parameter(ref_W + noise)
            else:
                # 维度不匹配，使用随机初始化
                self.W = nn.Parameter(init_scale * torch.randn(obs_dim, state_dim))
        else:
            # 使用 Xavier 初始化（更好的初始化方法）
            self.W = nn.Parameter(torch.empty(obs_dim, state_dim))
            nn.init.xavier_uniform_(self.W, gain=init_scale)

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

