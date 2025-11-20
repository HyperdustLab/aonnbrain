# src/aonn/core/aspect_layer.py
"""
Aspect Layer：纯神经元层（横切 MB）
隐层只有 Aspect，完全没有普通 object 点
"""
from typing import List, Dict
import torch
import torch.nn as nn

from .object import ObjectNode
from .aspect_base import AspectBase


class AspectLayer(nn.Module):
    """
    Aspect Layer：一层横切的神经元组织
    
    结构：
    - 输入：上一层 Object Layer 的状态
    - 处理：M 个 Aspect（神经元）并行计算
    - 输出：对下一层 Object Layer 的影响
    
    每个 Aspect 是一个横切的神经元：
    - 读取上一层所有（或部分）Object 的状态
    - 计算一个功能特征
    - 对下一层所有（或部分）Object 施加影响
    """
    
    def __init__(
        self,
        input_dim: int,
        num_aspects: int,
        output_dim: int,
        use_gate: bool = False,
    ):
        """
        Args:
            input_dim: 输入 Object Layer 的总维度
            num_aspects: Aspect（神经元）数量 M
            output_dim: 输出 Object Layer 的总维度
            use_gate: 是否使用 gate 机制
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_aspects = num_aspects
        self.output_dim = output_dim
        self.use_gate = use_gate
        
        # W: 所有 Aspect 的读取方向 [input_dim, num_aspects]
        self.W = nn.Linear(input_dim, num_aspects, bias=True)
        
        # V: 所有 Aspect 的写回方向 [num_aspects, output_dim]
        self.V = nn.Linear(num_aspects, output_dim, bias=False)
        
        # 如果输入输出维度不同，需要投影层
        if input_dim != output_dim:
            self.proj = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.proj = None
        
        # G: gate 机制（可选）
        if use_gate:
            self.G = nn.Linear(input_dim, num_aspects, bias=True)
        else:
            self.G = None
        
        self.act = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
        - 如果 input_dim == output_dim: x' = ReLU(x + V(ReLU(Wx)))
        - 如果 input_dim != output_dim: x' = ReLU(proj(x) + V(ReLU(Wx)))
        
        Args:
            x: [batch, input_dim] 或 [input_dim]
        
        Returns:
            [batch, output_dim] 或 [output_dim]
        """
        # 处理单样本
        was_1d = len(x.shape) == 1
        if was_1d:
            x = x.unsqueeze(0)
        
        # Aspect 激活：a = W(x) -> [B, M]
        a = self.W(x)
        
        # 非线性激活：z = ReLU(a)
        z = self.act(a)
        
        # Gate 机制（可选）
        if self.G is not None:
            g = torch.sigmoid(self.G(x))
            z = z * g
        
        # Aspect 写回：delta = V(z) -> [B, output_dim]
        delta = self.V(z)
        
        # 残差更新
        if self.input_dim == self.output_dim:
            # 维度相同：直接残差
            out = x + delta
        else:
            # 维度不同：先投影 x，再残差
            x_proj = self.proj(x)
            out = x_proj + delta
        
        out = self.act(out)
        
        # 恢复原始形状
        if was_1d:
            out = out.squeeze(0)
        
        return out


class AspectPipeline(nn.Module):
    """
    Aspect Pipeline：连续的 Aspect Layer
    
    结构：Aspect → Aspect → Aspect → ...
    中间完全没有 Object Layer，只有 Aspect（神经元）
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_aspects: int,
        depth: int,
        use_gate: bool = False,
    ):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            num_aspects: 每层 Aspect 数量
            depth: Aspect Layer 的深度
            use_gate: 是否使用 gate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_aspects = num_aspects
        self.depth = depth
        
        # 堆叠多个 Aspect Layer
        layers = []
        for i in range(depth):
            # 第一层：input_dim -> hidden_dim
            # 中间层：hidden_dim -> hidden_dim
            # 最后一层：hidden_dim -> output_dim
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = output_dim  # 中间层保持维度
            
            if i == depth - 1:
                layer_output_dim = output_dim
            else:
                layer_output_dim = output_dim  # 中间层保持维度
            
            layers.append(
                AspectLayer(
                    input_dim=layer_input_dim,
                    num_aspects=num_aspects,
                    output_dim=layer_output_dim,
                    use_gate=use_gate,
                )
            )
        self.pipeline = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        通过 Aspect Pipeline：x → Aspect → Aspect → ... → x'
        """
        return self.pipeline(x)

