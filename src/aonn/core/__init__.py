"""
AONN 核心抽象模块
"""

from .object import ObjectNode
from .aspect_base import AspectBase
from .free_energy import compute_total_free_energy
from .active_inference_loop import ActiveInferenceLoop

__all__ = [
    "ObjectNode",
    "AspectBase",
    "compute_total_free_energy",
    "ActiveInferenceLoop",
]

