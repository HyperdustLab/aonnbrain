# src/aonn/core/object_layer.py
"""
Object Layer：状态层（Vertical MB）
所有"有状态的东西"都落在 Object Layer 上
"""
from typing import Dict, List
import torch

from .object import ObjectNode


class ObjectLayer:
    """
    Object Layer：一层状态组织
    
    结构：
    - 包含多个 Object（Vertical MB）
    - 每个 Object 有自己的状态（S/A/internal）
    - Object Layer 只存储状态，不执行计算
    """
    
    def __init__(self, name: str, objects: Dict[str, ObjectNode]):
        """
        Args:
            name: Layer 名称（如 "input", "hidden_1", "output"）
            objects: Object 字典
        """
        self.name = name
        self.objects = objects
    
    def get_state_vector(self) -> torch.Tensor:
        """
        将所有 Object 的状态拼接成向量
        
        Returns:
            [dim_total] 或 [batch, dim_total]
        """
        states = [obj.state for obj in self.objects.values()]
        return torch.cat(states, dim=-1)
    
    def set_state_vector(self, state_vector: torch.Tensor):
        """
        从向量设置所有 Object 的状态
        
        Args:
            state_vector: [dim_total] 或 [batch, dim_total]
        """
        dims = [obj.dim for obj in self.objects.values()]
        dim_total = sum(dims)
        
        # 检查维度
        if len(state_vector.shape) == 1:
            # 单样本：[dim_total]
            assert state_vector.shape[0] == dim_total, \
                f"维度不匹配: 期望 {dim_total}, 实际 {state_vector.shape[0]}"
            # 分割并设置
            start_idx = 0
            for obj, dim in zip(self.objects.values(), dims):
                end_idx = start_idx + dim
                obj.set_state(state_vector[start_idx:end_idx])
                start_idx = end_idx
        else:
            # 批量：[batch, dim_total]
            assert state_vector.shape[-1] == dim_total, \
                f"维度不匹配: 期望 {dim_total}, 实际 {state_vector.shape[-1]}"
            # 分割并设置
            start_idx = 0
            for obj, dim in zip(self.objects.values(), dims):
                end_idx = start_idx + dim
                # 提取对应的切片 [batch, dim]
                slice_tensor = state_vector[..., start_idx:end_idx]
                obj.set_state(slice_tensor)
                start_idx = end_idx
    
    def get_total_dim(self) -> int:
        """获取总维度"""
        return sum(obj.dim for obj in self.objects.values())
    
    def __len__(self):
        """Object 数量"""
        return len(self.objects)
    
    def __iter__(self):
        """迭代 Object"""
        return iter(self.objects.items())
    
    def __getitem__(self, name: str) -> ObjectNode:
        """获取 Object"""
        return self.objects[name]

