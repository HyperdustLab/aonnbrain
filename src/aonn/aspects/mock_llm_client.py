# src/aonn/aspects/mock_llm_client.py
"""
模拟 LLM 客户端：使用本地 MLP 模拟 LLM 输出
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class MockLLMClient(nn.Module):
    """
    使用 MLP 模拟 LLM 的语义预测功能
    
    这个客户端可以：
    1. 作为可训练模块（trainable=True）
    2. 作为固定参数模块（trainable=False）
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        hidden_dims: list = None,
        activation: str = "relu",
        trainable: bool = True,
        noise_scale: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            input_dim: 输入向量维度（语义上下文维度）
            output_dim: 输出向量维度（预测语义维度）
            hidden_dims: MLP 隐藏层维度列表，例如 [256, 512, 256]
            activation: 激活函数类型 ("relu", "tanh", "gelu")
            trainable: 是否可训练
            noise_scale: 输出噪声尺度（用于模拟 LLM 的不确定性）
            device: 设备
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.trainable = trainable
        self.noise_scale = noise_scale
        self.device = device or torch.device("cpu")
        
        # 构建 MLP
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 最后一层不加激活函数
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                # 添加 dropout 增加随机性
                layers.append(nn.Dropout(0.1))
        
        self.mlp = nn.Sequential(*layers).to(self.device)
        
        # 初始化权重
        self._init_weights()
        
        # 如果不可训练，冻结参数
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
        
        # 用于存储最后一次生成的文本描述（模拟真实 LLM）
        self._last_generated_text = ""
    
    def _init_weights(self):
        """初始化 MLP 权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def semantic_predict(
        self,
        context_vec: torch.Tensor,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        模拟 LLM 的语义预测
        
        Args:
            context_vec: 输入语义上下文向量 [batch, input_dim] 或 [input_dim]
            temperature: 温度参数（控制输出的随机性）
            **kwargs: 其他配置参数（兼容真实 LLM API）
        
        Returns:
            预测的语义向量 [batch, output_dim] 或 [output_dim]
        """
        # 确保输入在正确的设备上
        context_vec = context_vec.to(self.device)
        
        # 处理单样本输入
        was_1d = len(context_vec.shape) == 1
        if was_1d:
            context_vec = context_vec.unsqueeze(0)
        
        # 通过 MLP
        pred = self.mlp(context_vec)
        
        # 应用温度缩放（模拟 LLM 的采样行为）
        if temperature != 1.0:
            pred = pred / temperature
        
        # 添加噪声（模拟 LLM 的不确定性）
        if self.noise_scale > 0 and self.training:
            noise = torch.randn_like(pred) * self.noise_scale
            pred = pred + noise
        
        # 生成模拟的语义描述文本（基于输入向量的特征）
        # 提取前几个维度作为特征，生成描述性文本
        context_summary = context_vec.detach().cpu()
        if len(context_summary.shape) > 1:
            context_summary = context_summary[0]
        
        # 转换为 numpy 用于计算
        context_np = context_summary.numpy() if hasattr(context_summary, 'numpy') else context_summary.detach().cpu().numpy()
        
        # 生成一个基于数值特征的描述
        max_val = float(context_np.max()) if len(context_np) > 0 else 0.0
        min_val = float(context_np.min()) if len(context_np) > 0 else 0.0
        mean_val = float(context_np.mean()) if len(context_np) > 0 else 0.0
        
        # 根据特征值生成描述性文本
        if abs(mean_val) < 0.1:
            desc = "语义状态接近中性，无明显倾向"
        elif mean_val > 0.5:
            desc = f"语义状态呈现正向激活（峰值{max_val:.2f}），可能表示积极意图或目标导向"
        elif mean_val < -0.5:
            desc = f"语义状态呈现负向激活（谷值{min_val:.2f}），可能表示回避或抑制"
        else:
            desc = f"语义状态处于中等水平（均值{mean_val:.2f}），包含混合特征"
        
        self._last_generated_text = desc
        
        # 恢复原始形状
        if was_1d:
            pred = pred.squeeze(0)
        
        return pred
    
    def forward(self, context_vec: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        PyTorch forward 方法（兼容 nn.Module）
        """
        return self.semantic_predict(context_vec, **kwargs)
    
    def parameters(self):
        """
        返回可训练参数（如果 trainable=False，返回空）
        """
        if self.trainable:
            return super().parameters()
        else:
            return []


def create_default_mock_llm_client(
    input_dim: int = 128,
    output_dim: int = 128,
    device: Optional[torch.device] = None,
) -> MockLLMClient:
    """
    创建默认的模拟 LLM 客户端
    
    这是一个便捷函数，用于快速创建可用的客户端
    """
    return MockLLMClient(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[256, 512, 256],
        activation="relu",
        trainable=True,
        noise_scale=0.05,
        device=device,
    )

