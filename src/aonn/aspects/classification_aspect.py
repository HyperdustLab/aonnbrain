# src/aonn/aspects/classification_aspect.py
"""
ClassificationAspect：分类专用 Aspect
用于从内部状态预测类别（如 MNIST 数字分类）
"""
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from aonn.core.aspect_base import AspectBase
from aonn.core.object import ObjectNode


class ClassificationAspect(AspectBase, nn.Module):
    """
    分类 Aspect：从 internal 状态预测类别
    
    自由能 = 交叉熵损失
    F = -log(p(target|internal))
    
    用于分类任务（如 MNIST 手写数字识别）
    """
    
    def __init__(
        self,
        internal_name: str = "internal",
        target_name: str = "target",
        state_dim: int = 256,
        num_classes: int = 10,
        hidden_dim: int = 128,
        name: str = "classification",
        loss_weight: float = 1.0,
    ):
        """
        Args:
            internal_name: 内部状态 Object 名称
            target_name: 目标类别 Object 名称（one-hot 编码）
            state_dim: 内部状态维度
            num_classes: 类别数量
            hidden_dim: 分类器隐藏层维度
            name: Aspect 名称
            loss_weight: 损失权重
        """
        AspectBase.__init__(self, name=name, src_names=[internal_name], dst_names=[target_name])
        nn.Module.__init__(self)
        
        self.state_dim = state_dim
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        
        # 分类器网络
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, objects: Dict[str, ObjectNode]) -> Dict[str, torch.Tensor]:
        """
        计算预测误差
        
        Args:
            objects: Object 字典
        
        Returns:
            目标 Object 的误差字典
        """
        internal = objects[self.src_names[0]].state
        logits = self.classifier(internal)
        
        target = objects[self.dst_names[0]].state  # one-hot 编码
        pred_probs = torch.softmax(logits, dim=-1)
        
        # 误差 = 目标概率 - 预测概率
        error = target - pred_probs
        return {self.dst_names[0]: error}
    
    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """
        计算自由能贡献（交叉熵损失）
        
        Args:
            objects: Object 字典
        
        Returns:
            自由能贡献（标量）
        """
        internal = objects[self.src_names[0]].state
        logits = self.classifier(internal)
        
        target = objects[self.dst_names[0]].state  # one-hot 编码
        
        # 获取目标类别索引
        target_class = target.argmax(dim=-1)
        
        # 交叉熵损失
        # logits 需要是 [batch, num_classes]，target_class 需要是 [batch]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if target_class.dim() == 0:
            target_class = target_class.unsqueeze(0)
        
        F = F.cross_entropy(logits, target_class, reduction='none')
        
        # 如果原来是标量，返回标量
        if F.dim() > 0:
            F = F.squeeze(0)
        
        return self.loss_weight * F
    
    def predict(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """
        预测类别（用于推理）
        
        Args:
            objects: Object 字典
        
        Returns:
            类别 logits [num_classes]
        """
        internal = objects[self.src_names[0]].state
        logits = self.classifier(internal)
        return logits
    
    def predict_class(self, objects: Dict[str, ObjectNode]) -> int:
        """
        预测类别索引
        
        Args:
            objects: Object 字典
        
        Returns:
            预测的类别索引
        """
        logits = self.predict(objects)
        return logits.argmax(dim=-1).item()
    
    def parameters(self):
        """返回分类器的参数"""
        return list(self.classifier.parameters())

