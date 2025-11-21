# src/aonn/aspects/encoder_aspect.py
"""
EncoderAspect：卷积编码器
用于将 sensory 编码为 internal 状态（如 vision -> internal）
支持 MNIST 等图像数据的卷积编码
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from aonn.core.object import ObjectNode
from aonn.core.aspect_base import AspectBase


class EncoderAspect(AspectBase, nn.Module):
    """
    卷积编码器：sensory -> internal
    
    对于图像数据（如 MNIST 28x28），使用卷积编码器
    对于向量数据，使用线性编码器
    """
    
    def __init__(
        self,
        sensory_name: str = "vision",
        internal_name: str = "internal",
        input_dim: int = 784,
        output_dim: int = 128,
        name: str = "encoder",
        use_conv: bool = True,
        image_size: Optional[int] = None,
    ):
        """
        Args:
            sensory_name: 源 Object 名称（如 "vision"）
            internal_name: 目标 Object 名称（如 "internal"）
            input_dim: 输入维度（如 784 for MNIST）
            output_dim: 输出维度（如 128）
            name: Aspect 名称
            use_conv: 是否使用卷积编码器（对于图像数据）
            image_size: 图像尺寸（如 28 for MNIST），如果为 None 则从 input_dim 推断
        """
        AspectBase.__init__(self, name=name, src_names=[sensory_name], dst_names=[internal_name])
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_conv = use_conv
        
        # 推断图像尺寸
        if image_size is None and use_conv:
            # 假设是正方形图像
            import math
            image_size = int(math.sqrt(input_dim))
            if image_size * image_size != input_dim:
                # 如果不是完全平方数，使用线性编码器
                use_conv = False
        
        self.image_size = image_size
        
        if use_conv and image_size is not None:
            # 卷积编码器（用于图像数据）
            # 输入：1 x 28 x 28
            # 输出：128 维向量
            # 标准 VAE 编码器架构：28 -> 14 -> 7 -> 3
            self.encoder = nn.Sequential(
                # 第一层卷积：28x28 -> 14x14
                nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28 -> 14
                nn.ReLU(),
                # 第二层卷积：14x14 -> 7x7
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14 -> 7
                nn.ReLU(),
                # 第三层卷积：7x7 -> 3x3
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 7 -> 3
                nn.ReLU(),
                # Flatten: 128 x 3 x 3 = 1152
                nn.Flatten(),
                # 线性层：1152 -> 128
                nn.Linear(128 * 3 * 3, output_dim),
            )
        else:
            # 线性编码器（用于向量数据）
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, output_dim * 2),
                nn.ReLU(),
                nn.Linear(output_dim * 2, output_dim),
            )
    
    def forward(self, objects: Dict[str, ObjectNode]) -> Dict[str, torch.Tensor]:
        """计算预测误差"""
        sensory_state = objects[self.src_names[0]].state
        internal_state = objects[self.dst_names[0]].state
        
        # 处理输入：如果是向量，可能需要 reshape
        if self.use_conv and self.image_size is not None:
            # Reshape: [784] -> [1, 1, 28, 28]
            if sensory_state.dim() == 1:
                sensory_state = sensory_state.view(1, 1, self.image_size, self.image_size)
            elif sensory_state.dim() == 2:
                batch_size = sensory_state.shape[0]
                sensory_state = sensory_state.view(batch_size, 1, self.image_size, self.image_size)
        
        # 编码预测
        pred = self.encoder(sensory_state)
        
        # 确保 pred 和 internal_state 维度匹配
        if pred.dim() > 1:
            pred = pred.squeeze(0)
        if internal_state.dim() > 1:
            internal_state = internal_state.squeeze(0)
        
        # 误差
        error = internal_state - pred
        return {self.dst_names[0]: error}
    
    def free_energy_contrib(self, objects: Dict[str, ObjectNode], **kwargs) -> torch.Tensor:
        """计算自由能贡献"""
        sensory_state = objects[self.src_names[0]].state
        internal_state = objects[self.dst_names[0]].state
        
        # 处理输入
        if self.use_conv and self.image_size is not None:
            if sensory_state.dim() == 1:
                sensory_state = sensory_state.view(1, 1, self.image_size, self.image_size)
            elif sensory_state.dim() == 2:
                batch_size = sensory_state.shape[0]
                sensory_state = sensory_state.view(batch_size, 1, self.image_size, self.image_size)
        
        pred = self.encoder(sensory_state)
        
        # 确保维度匹配
        if pred.dim() > 1:
            pred = pred.squeeze(0)
        if internal_state.dim() > 1:
            internal_state = internal_state.squeeze(0)
        
        error = internal_state - pred
        
        # 自由能贡献 = 0.5 * ||error||^2
        return 0.5 * (error ** 2).sum()
    
    def parameters(self):
        return list(self.encoder.parameters())

