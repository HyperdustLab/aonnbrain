"""
测试自由能计算
"""
import torch
from aonn.core.object import ObjectNode
from aonn.core.free_energy import compute_total_free_energy
from aonn.aspects.sensory_aspect import LinearGenerativeAspect


def test_compute_total_free_energy():
    dim = 16
    internal = ObjectNode("internal", dim=dim, init="normal")
    sensory = ObjectNode("sensory", dim=dim, init="normal")
    
    objects = {"internal": internal, "sensory": sensory}
    
    aspect = LinearGenerativeAspect(
        internal_name="internal",
        sensory_name="sensory",
        state_dim=dim,
        obs_dim=dim,
    )
    
    F = compute_total_free_energy(objects, [aspect])
    assert F.item() >= 0
    
    # 多个 aspect 应该累加
    aspect2 = LinearGenerativeAspect(
        internal_name="internal",
        sensory_name="sensory",
        state_dim=dim,
        obs_dim=dim,
    )
    F2 = compute_total_free_energy(objects, [aspect, aspect2])
    assert F2.item() >= F.item()

