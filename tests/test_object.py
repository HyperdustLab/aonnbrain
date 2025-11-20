"""
测试 ObjectNode
"""
import torch
import pytest
from aonn.core.object import ObjectNode


def test_object_creation():
    obj = ObjectNode("test", dim=10, init="zero")
    assert obj.name == "test"
    assert obj.dim == 10
    assert obj.state.shape == (10,)
    assert torch.allclose(obj.state, torch.zeros(10))


def test_object_set_state():
    obj = ObjectNode("test", dim=5)
    new_state = torch.randn(5)
    obj.set_state(new_state)
    assert torch.allclose(obj.state, new_state)


def test_object_clone_detached():
    obj = ObjectNode("test", dim=3, init="normal")
    cloned = obj.clone_detached(requires_grad=True)
    assert cloned.requires_grad
    assert not obj.state.requires_grad
    assert torch.allclose(cloned, obj.state)

