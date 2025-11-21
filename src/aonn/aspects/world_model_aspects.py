# src/aonn/aspects/world_model_aspects.py
"""
世界模型的 Aspect：用于学习真实世界的生成模型

根据自由能原理，AONN 需要学习：
1. Dynamics Aspect: 状态转移模型 p(s_{t+1} | s_t, a_t)
2. Observation Aspect: 观察生成模型 p(o_t | s_t)
3. Preference Aspect: 偏好/目标模型（先验）
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
from aonn.core.object import ObjectNode
from aonn.core.aspect_base import AspectBase


class DynamicsAspect(AspectBase, nn.Module):
    """
    状态转移模型：internal_t + action → internal_{t+1}
    
    学习真实世界的动力学：p(s_{t+1} | s_t, a_t)
    """
    
    def __init__(
        self,
        internal_name: str = "internal",
        action_name: str = "action",
        state_dim: int = 32,
        action_dim: int = 8,
        hidden_dim: Optional[int] = None,
    ):
        AspectBase.__init__(
            self,
            name="dynamics",
            src_names=[internal_name, action_name],
            dst_names=[internal_name]
        )
        nn.Module.__init__(self)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        hidden_dim = hidden_dim or (state_dim * 2)
        
        # 状态转移模型（可学习）
        self.transition = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def forward(self, objects: Dict[str, ObjectNode]) -> Dict[str, torch.Tensor]:
        """
        预测下一状态：pred_next = f(internal, action)
        """
        internal = objects[self.src_names[0]].state
        action = objects[self.src_names[1]].state
        
        combined = torch.cat([internal, action], dim=-1)
        pred_next = self.transition(combined)
        
        # 返回预测误差
        target_next = objects[self.dst_names[0]].state
        error = target_next - pred_next
        
        return {self.dst_names[0]: error}
    
    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """
        自由能贡献：F = 0.5 * ||internal_{t+1} - pred(internal_t, action)||²
        """
        # 检查必要的 Object 是否存在
        if self.src_names[0] not in objects or self.src_names[1] not in objects:
            return torch.tensor(0.0, device=self.transition[0].weight.device)
        
        internal = objects[self.src_names[0]].state
        action = objects[self.src_names[1]].state
        
        # 如果 dst 不存在，使用 internal 作为目标（自预测）
        if self.dst_names[0] not in objects:
            target_next = internal
        else:
            target_next = objects[self.dst_names[0]].state
        
        combined = torch.cat([internal, action], dim=-1)
        pred_next = self.transition(combined)
        
        error = target_next - pred_next
        return 0.5 * (error ** 2).sum()
    
    def predict_next_state(self, internal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测下一状态（用于推理）"""
        combined = torch.cat([internal, action], dim=-1)
        return self.transition(combined)
    
    def parameters(self):
        return list(self.transition.parameters())


class ObservationAspect(AspectBase, nn.Module):
    """
    观察生成模型：internal → sensory
    
    学习观察生成：p(o_t | s_t)
    支持卷积解码器（用于图像数据）和线性解码器（用于向量数据）
    """
    
    def __init__(
        self,
        internal_name: str = "internal",
        sensory_name: str = "sensory",
        state_dim: int = 32,
        obs_dim: int = 16,
        hidden_dim: Optional[int] = None,
        use_conv: bool = True,
        image_size: Optional[int] = None,
    ):
        AspectBase.__init__(
            self,
            name="observation",
            src_names=[internal_name],
            dst_names=[sensory_name]
        )
        nn.Module.__init__(self)
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.use_conv = use_conv
        
        # 推断图像尺寸
        if image_size is None and use_conv:
            # 假设是正方形图像
            import math
            image_size = int(math.sqrt(obs_dim))
            if image_size * image_size != obs_dim:
                # 如果不是完全平方数，使用线性解码器
                use_conv = False
        
        self.image_size = image_size
        
        if use_conv and image_size is not None:
            # 卷积解码器（用于图像数据）
            # 输入：128 维向量
            # 输出：1 x 28 x 28
            # 标准 VAE 解码器架构：128 -> 1152 -> 3 -> 7 -> 14 -> 28
            # 使用 View 层来替代 Unflatten（更兼容）
            class View(nn.Module):
                def __init__(self, shape):
                    super().__init__()
                    self.shape = shape
                def forward(self, x):
                    return x.view(-1, *self.shape)
            
            self.observation_model = nn.Sequential(
                # 线性层：128 -> 1152
                nn.Linear(state_dim, 128 * 3 * 3),
                nn.ReLU(),
                # Reshape: 1152 -> 128 x 3 x 3
                View((128, 3, 3)),
                # 转置卷积：3x3 -> 7x7 (使用 output_padding=1)
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),  # 3 -> 7
                nn.ReLU(),
                # 转置卷积：7x7 -> 14x14
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7 -> 14
                nn.ReLU(),
                # 转置卷积：14x14 -> 28x28
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 14 -> 28
                nn.Sigmoid(),  # 输出像素值在 [0, 1]
            )
        else:
            # 线性解码器（用于向量数据）
            hidden_dim = hidden_dim or (obs_dim * 2)
            self.observation_model = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim),
            )
    
    def forward(self, objects: Dict[str, ObjectNode]) -> Dict[str, torch.Tensor]:
        """
        预测观察：pred_obs = f(internal)
        """
        internal = objects[self.src_names[0]].state
        target_obs = objects[self.dst_names[0]].state
        
        # 确保 internal 是 2D [batch, dim] 或 1D [dim]
        if internal.dim() == 1:
            internal = internal.unsqueeze(0)
        
        pred_obs = self.observation_model(internal)
        
        # 如果是卷积解码器，需要 flatten
        if self.use_conv and self.image_size is not None:
            if pred_obs.dim() == 4:
                # [batch, 1, 28, 28] -> [batch, 784]
                batch_size = pred_obs.shape[0]
                pred_obs = pred_obs.view(batch_size, -1)
            elif pred_obs.dim() == 3:
                # [1, 28, 28] -> [784]
                pred_obs = pred_obs.view(-1)
        
        # 确保 pred_obs 和 target_obs 维度匹配
        if pred_obs.dim() > 1:
            pred_obs = pred_obs.squeeze(0)
        if target_obs.dim() > 1:
            target_obs = target_obs.squeeze(0)
        
        # 返回预测误差
        error = target_obs - pred_obs
        
        return {self.dst_names[0]: error}
    
    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """
        自由能贡献：F = 0.5 * ||sensory - pred(internal)||²
        """
        internal = objects[self.src_names[0]].state
        target_obs = objects[self.dst_names[0]].state
        
        # 确保 internal 是 2D
        if internal.dim() == 1:
            internal = internal.unsqueeze(0)
        
        pred_obs = self.observation_model(internal)
        
        # 如果是卷积解码器，需要 flatten
        if self.use_conv and self.image_size is not None:
            if pred_obs.dim() == 4:
                batch_size = pred_obs.shape[0]
                pred_obs = pred_obs.view(batch_size, -1)
            elif pred_obs.dim() == 3:
                pred_obs = pred_obs.view(-1)
        
        # 确保维度匹配
        if pred_obs.dim() > 1:
            pred_obs = pred_obs.squeeze(0)
        if target_obs.dim() > 1:
            target_obs = target_obs.squeeze(0)
        
        error = target_obs - pred_obs
        
        return 0.5 * (error ** 2).sum()
    
    def predict_observation(self, internal: torch.Tensor) -> torch.Tensor:
        """预测观察（用于推理）"""
        if internal.dim() == 1:
            internal = internal.unsqueeze(0)
        
        pred_obs = self.observation_model(internal)
        
        # 如果是卷积解码器，需要 flatten
        if self.use_conv and self.image_size is not None:
            if pred_obs.dim() == 4:
                batch_size = pred_obs.shape[0]
                pred_obs = pred_obs.view(batch_size, -1)
            elif pred_obs.dim() == 3:
                pred_obs = pred_obs.view(-1)
        
        if pred_obs.dim() > 1:
            pred_obs = pred_obs.squeeze(0)
        
        return pred_obs
    
    def parameters(self):
        return list(self.observation_model.parameters())


class PreferenceAspect(AspectBase, nn.Module):
    """
    偏好/目标模型：internal → preference
    
    将目标状态转化为先验项，成为自由能的一部分
    F_preference = -log p(internal | target) ≈ 0.5 * ||internal - target||²
    """
    
    def __init__(
        self,
        internal_name: str = "internal",
        target_name: str = "target",
        state_dim: int = 32,
        weight: float = 1.0,
    ):
        AspectBase.__init__(
            self,
            name="preference",
            src_names=[internal_name],
            dst_names=[target_name]
        )
        nn.Module.__init__(self)
        
        self.state_dim = state_dim
        self.weight = weight
        
        # 目标状态（可学习或固定）
        self.target_state = nn.Parameter(torch.randn(state_dim) * 0.5)
    
    def forward(self, objects: Dict[str, ObjectNode]) -> Dict[str, torch.Tensor]:
        """
        计算偏好误差：error = internal - target
        """
        internal = objects[self.src_names[0]].state
        error = internal - self.target_state
        
        return {self.dst_names[0]: error}
    
    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """
        自由能贡献（先验项）：F = 0.5 * weight * ||internal - target||²
        这相当于 -log p(internal | target)
        """
        if self.src_names[0] not in objects:
            return torch.tensor(0.0, device=self.target_state.device)
        
        internal = objects[self.src_names[0]].state
        error = internal - self.target_state
        
        return 0.5 * self.weight * (error ** 2).sum()
    
    def set_target(self, target: torch.Tensor):
        """设置目标状态"""
        with torch.no_grad():
            self.target_state.data = target.to(self.target_state.device)
    
    def get_target(self) -> torch.Tensor:
        """获取目标状态"""
        return self.target_state.data.clone()
    
    def parameters(self):
        return [self.target_state]


class WorldModelAspectSet:
    """
    世界模型 Aspect 集合：统一管理 dynamics, observation(多感官), preference
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 8,
        observation_dims: Optional[Dict[str, int]] = None,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_dims = observation_dims or {"vision": 16, "olfactory": 8, "proprio": 4}
        self.device = device or torch.device("cpu")
        
        # Dynamics
        self.dynamics = DynamicsAspect(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)
        
        # Observation aspects（多感官）
        self.observation_aspects = []
        for sense_name, dim in self.observation_dims.items():
            aspect = ObservationAspect(
                internal_name="internal",
                sensory_name=sense_name,
                state_dim=state_dim,
                obs_dim=dim,
            ).to(self.device)
            self.observation_aspects.append(aspect)
        
        # Preference
        self.preference = PreferenceAspect(
            state_dim=state_dim,
        ).to(self.device)
        
        # 汇总
        self.aspects = [self.dynamics, *self.observation_aspects, self.preference]
    
    def get_all_parameters(self):
        """获取所有可训练参数"""
        params = []
        for aspect in self.aspects:
            params.extend(aspect.parameters())
        return params
    
    def set_target_state(self, target: torch.Tensor):
        """设置目标状态（用于 preference）"""
        self.preference.set_target(target)

