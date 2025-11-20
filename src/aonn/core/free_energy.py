# src/aonn/core/free_energy.py
from typing import Dict, List
import torch

from .object import ObjectNode
from .aspect_base import AspectBase


def compute_total_free_energy(
    objects: Dict[str, ObjectNode],
    aspects: List[AspectBase],
    prior_terms: torch.Tensor = None,
) -> torch.Tensor:
    """
    汇总所有 Aspect 的自由能贡献 + 可选先验项。
    """
    F = torch.tensor(0.0, device=next(iter(objects.values())).state.device)
    for asp in aspects:
        F = F + asp.free_energy_contrib(objects)
    if prior_terms is not None:
        F = F + prior_terms
    return F

