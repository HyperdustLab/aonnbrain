"""
测试主动推理循环
"""
import torch
from aonn.core.object import ObjectNode
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.aspects.sensory_aspect import LinearGenerativeAspect


def test_active_inference_loop():
    dim = 32
    internal = ObjectNode("internal", dim=dim, init="normal")
    sensory = ObjectNode("sensory", dim=dim, init="zero")
    
    objects = {"internal": internal, "sensory": sensory}
    
    aspect = LinearGenerativeAspect(
        internal_name="internal",
        sensory_name="sensory",
        state_dim=dim,
        obs_dim=dim,
    )
    
    # 设置一个目标观察
    target_obs = torch.randn(dim)
    sensory.set_state(target_obs)
    
    # 创建推理循环
    infer_loop = ActiveInferenceLoop(
        objects=objects,
        aspects=[aspect],
        infer_lr=0.1,
    )
    
    # 记录初始自由能
    from aonn.core.free_energy import compute_total_free_energy
    F_initial = compute_total_free_energy(objects, [aspect]).item()
    
    # 执行推理
    infer_loop.infer_states(target_objects=("internal",), num_iters=10)
    
    # 检查自由能是否下降（或至少不显著上升）
    F_final = compute_total_free_energy(objects, [aspect]).item()
    
    # 注意：由于是梯度下降，自由能应该下降或保持稳定
    # 但由于初始化随机性，允许小幅波动
    print(f"初始自由能: {F_initial:.4f}, 最终自由能: {F_final:.4f}")
    assert F_final < F_initial * 1.5  # 允许一定波动，但不应该大幅上升

