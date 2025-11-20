"""
测试基本 Aspect
"""
import torch
from aonn.core.object import ObjectNode
from aonn.aspects.sensory_aspect import LinearGenerativeAspect


def test_linear_generative_aspect():
    dim = 32
    internal = ObjectNode("internal", dim=dim, init="normal")
    sensory = ObjectNode("sensory", dim=dim, init="normal")
    
    objects = {"internal": internal, "sensory": sensory}
    
    aspect = LinearGenerativeAspect(
        internal_name="internal",
        sensory_name="sensory",
        state_dim=dim,
        obs_dim=dim,
    )
    
    # 测试 forward
    errors = aspect.forward(objects)
    assert "sensory" in errors
    assert errors["sensory"].shape == (dim,)
    
    # 测试自由能
    F = aspect.free_energy_contrib(objects)
    assert F.item() >= 0  # 自由能应该非负
    
    # 当预测接近目标时，自由能应该降低
    pred = aspect.W @ internal.state
    sensory.set_state(pred)
    F_low = aspect.free_energy_contrib(objects)
    assert F_low < F

