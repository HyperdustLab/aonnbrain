# src/aonn/core/free_energy.py
from typing import Dict, List
import torch

from .object import ObjectNode
from .aspect_base import AspectBase


def compute_total_free_energy(
    objects: Dict[str, ObjectNode],
    aspects: List[AspectBase],
    prior_terms: torch.Tensor = None,
    iteration_idx: int = None,
    is_last_iter: bool = False,
) -> torch.Tensor:
    """
    汇总所有 Aspect 的自由能贡献 + 可选先验项。
    
    Args:
        objects: Object 字典
        aspects: Aspect 列表
        prior_terms: 可选先验项
        iteration_idx: 当前迭代索引（用于 LLMAspect 等需要控制调用频率的 Aspect）
        is_last_iter: 是否是最后一次迭代
    """
    F = torch.tensor(0.0, device=next(iter(objects.values())).state.device)
    for asp in aspects:
        # 如果 Aspect 支持迭代信息，传递给它
        if hasattr(asp, 'free_energy_contrib'):
            try:
                # 尝试传递迭代信息（如果 Aspect 支持）
                contrib = asp.free_energy_contrib(objects, iteration_idx=iteration_idx, is_last_iter=is_last_iter)
            except TypeError:
                # 如果 Aspect 不支持迭代信息，使用默认调用
                contrib = asp.free_energy_contrib(objects)
            F = F + contrib
    if prior_terms is not None:
        F = F + prior_terms
    return F

