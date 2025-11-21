# src/aonn/core/active_inference_loop.py
from typing import Dict, List
import torch

from .object import ObjectNode
from .aspect_base import AspectBase
from .free_energy import compute_total_free_energy


class ActiveInferenceLoop:
    """
    主动推理循环：
    - 固定感官 Object
    - 对 internal / action 等 Object 做状态推理（gradient descent）
    - 可选执行一次参数学习
    """

    def __init__(
        self,
        objects: Dict[str, ObjectNode],
        aspects: List[AspectBase],
        infer_lr: float = 0.1,
        max_grad_norm: float = None,
        device=None,
    ):
        self.objects = objects
        self.aspects = aspects
        self.infer_lr = infer_lr
        self.max_grad_norm = max_grad_norm  # 梯度裁剪阈值
        self.device = device or torch.device("cpu")

    def infer_states(self, target_objects=("internal", "action"), num_iters: int = 5, sanitize_callback=None):
        """
        对指定 Object 节点（如 internal, action）执行有限步自由能下降更新。
        
        Args:
            target_objects: 要推理的 Object 名称列表
            num_iters: 推理迭代次数
            sanitize_callback: 可选的清理回调函数，在每次迭代后调用（用于状态裁剪）
        """
        for iter_idx in range(num_iters):
            # 为每个待推理 Object 重置为可微叶子节点
            for name in target_objects:
                mu = self.objects[name].clone_detached(requires_grad=True)
                self.objects[name].state = mu

            # 梯度清零（这里只对 state 做手动更新）
            # 只处理叶子张量，避免警告
            for name in target_objects:
                state = self.objects[name].state
                if state.requires_grad and state.is_leaf and state.grad is not None:
                    state.grad.zero_()

            # 确保所有 Object 状态都是 detached 的，避免图冲突
            # 只处理非目标对象，避免影响当前迭代的目标对象
            for name in self.objects:
                if name not in target_objects:
                    state = self.objects[name].state
                    # 如果不是叶子张量，直接 detach；如果是叶子张量且有梯度，也 detach
                    if not state.is_leaf or (state.requires_grad and state.is_leaf and state.grad is not None):
                        self.objects[name].state = state.detach()
            
            F = compute_total_free_energy(self.objects, self.aspects)
            # 最后一次迭代不需要 retain_graph
            retain_graph = (iter_idx < num_iters - 1)
            try:
                F.backward(retain_graph=retain_graph)
            except RuntimeError as e:
                if "backward through the graph a second time" in str(e) or "freed" in str(e):
                    # 如果图已经被释放，跳过这次迭代
                    # 重新创建所有目标对象状态为 detached，避免下次迭代出错
                    for name in target_objects:
                        mu = self.objects[name].clone_detached(requires_grad=False)
                        self.objects[name].state = mu
                    # 跳过这次迭代的更新
                    continue
                else:
                    raise

            with torch.no_grad():
                for name in target_objects:
                    mu = self.objects[name].state
                    # 只处理叶子张量的梯度
                    if mu.requires_grad and mu.is_leaf and mu.grad is not None:
                        grad = mu.grad
                        # 梯度裁剪（如果设置了 max_grad_norm）
                        if self.max_grad_norm is not None and self.max_grad_norm > 0:
                            grad_norm = torch.norm(grad)
                            if grad_norm > self.max_grad_norm:
                                grad = grad * (self.max_grad_norm / grad_norm)
                        mu = mu - self.infer_lr * grad
                    self.objects[name].state = mu.detach()
            
            # 在每次迭代后调用清理回调（用于状态裁剪）
            if sanitize_callback is not None:
                sanitize_callback()

        return self.objects

