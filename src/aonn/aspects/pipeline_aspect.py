# src/aonn/aspects/pipeline_aspect.py
"""
PipelineAspect：将 AspectPipeline 包装为 AspectBase
实现向量化 Pipeline 和对象化 Aspect 的统一接口
"""
from typing import Dict, List
import torch
import torch.nn as nn

from aonn.core.aspect_base import AspectBase
from aonn.core.aspect_layer import AspectPipeline
from aonn.core.object import ObjectNode


class PipelineAspect(AspectBase, nn.Module):
    """
    将 AspectPipeline 包装为 Aspect
    
    实现向量化 Pipeline 和对象化 Aspect 的统一接口：
    - 实现 AspectBase 接口（forward, free_energy_contrib）
    - 内部使用 AspectPipeline 进行高效批量处理
    - 支持多源 Object 的状态拼接
    """
    
    def __init__(
        self,
        src_names: List[str],
        dst_names: List[str],
        input_dim: int,
        output_dim: int,
        num_aspects: int,
        depth: int,
        name: str = "pipeline_aspect",
        use_gate: bool = False,
    ):
        """
        Args:
            src_names: 源 Object 名称列表
            dst_names: 目标 Object 名称列表
            input_dim: 输入维度（所有源 Object 的总维度）
            output_dim: 输出维度（目标 Object 的维度）
            num_aspects: Pipeline 中每层的 Aspect 数量
            depth: Pipeline 深度（Aspect Layer 层数）
            name: Aspect 名称
            use_gate: 是否使用 gate 机制
        """
        AspectBase.__init__(self, name=name, src_names=src_names, dst_names=dst_names)
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_aspects = num_aspects
        self.depth = depth
        
        # 创建内部的 AspectPipeline
        self.pipeline = AspectPipeline(
            input_dim=input_dim,
            output_dim=output_dim,
            num_aspects=num_aspects,
            depth=depth,
            use_gate=use_gate,
        )
    
    def forward(self, objects: Dict[str, ObjectNode]) -> Dict[str, torch.Tensor]:
        """
        计算预测误差
        
        Args:
            objects: Object 字典
        
        Returns:
            每个目标 Object 的误差字典
        """
        # 从源 Object 获取状态
        src_state = self._get_src_state(objects)
        
        # 通过 Pipeline 处理
        pred = self.pipeline(src_state)
        
        # 计算误差（对每个目标 Object）
        errors = {}
        for dst_name in self.dst_names:
            if dst_name in objects:
                dst_state = objects[dst_name].state
                # 如果维度不匹配，进行适配
                if dst_state.shape[-1] != pred.shape[-1]:
                    # 如果 pred 是 [output_dim]，dst 是 [other_dim]
                    # 只取前 output_dim 维进行比较
                    min_dim = min(pred.shape[-1], dst_state.shape[-1])
                    error = dst_state[..., :min_dim] - pred[..., :min_dim]
                else:
                    error = dst_state - pred
                errors[dst_name] = error
            else:
                # 如果目标 Object 不存在，使用零误差
                errors[dst_name] = torch.zeros_like(pred)
        
        return errors
    
    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """
        计算自由能贡献
        
        Args:
            objects: Object 字典
        
        Returns:
            自由能贡献（标量）
        """
        error_dict = self.forward(objects)
        
        # 对所有目标 Object 的误差求和
        total_error = torch.tensor(0.0, device=next(iter(objects.values())).state.device)
        for dst_name, error in error_dict.items():
            total_error = total_error + 0.5 * (error ** 2).sum()
        
        return total_error
    
    def _get_src_state(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """
        从源 Object 获取状态（支持多源拼接）
        
        Args:
            objects: Object 字典
        
        Returns:
            拼接后的状态向量
        """
        states = []
        for src_name in self.src_names:
            if src_name in objects:
                state = objects[src_name].state
                states.append(state)
            else:
                # 如果源 Object 不存在，使用零向量
                # 需要知道维度，这里简化处理
                raise ValueError(f"源 Object '{src_name}' 不存在")
        
        # 如果只有一个源，直接返回
        if len(states) == 1:
            return states[0]
        
        # 多个源，拼接
        return torch.cat(states, dim=-1)
    
    def parameters(self):
        """返回 Pipeline 的参数"""
        return self.pipeline.parameters()

