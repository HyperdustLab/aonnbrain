"""
AONN 核心抽象模块
"""

from .object import ObjectNode
from .aspect_base import AspectBase
from .free_energy import compute_total_free_energy
from .active_inference_loop import ActiveInferenceLoop
from .network_topology import NetworkTopology, NetworkEdge, build_network_topology
from .network_forward import forward_pass, compute_network_free_energy
from .object_layer import ObjectLayer
from .aspect_layer import AspectLayer, AspectPipeline

__all__ = [
    "ObjectNode",
    "AspectBase",
    "compute_total_free_energy",
    "ActiveInferenceLoop",
    "NetworkTopology",
    "NetworkEdge",
    "build_network_topology",
    "forward_pass",
    "compute_network_free_energy",
    "ObjectLayer",
    "AspectLayer",
    "AspectPipeline",
]

