# src/aonn/models/world_model.py
"""
世界模型（World Model）：模拟环境，提供观察和奖励
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np


class SimpleWorldModel:
    """
    简单的世界模型：模拟一个可交互的环境
    
    环境特性：
    - 状态空间：连续状态
    - 动作空间：连续动作
    - 观察空间：部分可观察
    - 奖励：基于状态和动作
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 8,
        obs_dim: int = 16,
        reward_dim: int = 1,
        device=None,
        state_noise_std: float = 0.01,
        observation_noise_std: float = 0.01,
        target_drift_std: float = 0.0,
        vision_dim: Optional[int] = None,
        olfactory_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.reward_dim = reward_dim
        self.device = device or torch.device("cpu")
        self.state_noise_std = state_noise_std
        self.observation_noise_std = observation_noise_std
        self.target_drift_std = target_drift_std

        # 感官维度配置（若未提供则按照 obs_dim 拆分）
        if vision_dim is None and olfactory_dim is None and proprio_dim is None:
            vision_dim = obs_dim * 3 // 4
            olfactory_dim = max(obs_dim // 5, 4)
            proprio_dim = obs_dim - vision_dim - olfactory_dim
        elif vision_dim is None or olfactory_dim is None or proprio_dim is None:
            raise ValueError("必须同时提供 vision_dim/olfactory_dim/proprio_dim 或全部省略。")

        self.vision_dim = vision_dim
        self.olfactory_dim = olfactory_dim
        self.proprio_dim = proprio_dim
        self.obs_dim = self.vision_dim + self.olfactory_dim + self.proprio_dim
        
        # 环境状态（隐藏状态）
        self.hidden_state = torch.randn(state_dim, device=self.device) * 0.1
        
        # 状态转移模型（真实世界动力学）
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, state_dim),
        ).to(self.device)
        
        # 观察模型（多模态）
        self.vision_model = nn.Sequential(
            nn.Linear(state_dim, max(self.vision_dim * 2, state_dim)),
            nn.ReLU(),
            nn.Linear(max(self.vision_dim * 2, state_dim), self.vision_dim),
        ).to(self.device)
        self.olfactory_model = nn.Sequential(
            nn.Linear(state_dim, max(self.olfactory_dim * 2, state_dim)),
            nn.ReLU(),
            nn.Linear(max(self.olfactory_dim * 2, state_dim), self.olfactory_dim),
        ).to(self.device)
        self.proprio_model = nn.Sequential(
            nn.Linear(state_dim, max(self.proprio_dim * 2, state_dim)),
            nn.ReLU(),
            nn.Linear(max(self.proprio_dim * 2, state_dim), self.proprio_dim),
        ).to(self.device)
        
        # 奖励模型（从状态和动作到奖励）
        self.reward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, reward_dim * 2),
            nn.ReLU(),
            nn.Linear(reward_dim * 2, reward_dim),
        ).to(self.device)
        
        # 目标状态（用于计算奖励）
        self.target_state = torch.randn(state_dim, device=self.device) * 0.5
    
    def reset(self) -> torch.Tensor:
        """重置环境，返回初始观察"""
        self.hidden_state = torch.randn(self.state_dim, device=self.device) * 0.1
        return self.get_multimodal_observation()
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        执行一步动作
        
        Args:
            action: [action_dim] 动作向量
        
        Returns:
            (observation, reward, done)
        """
        # 状态转移
        combined = torch.cat([self.hidden_state, action], dim=-1)
        noise = self.state_noise_std * torch.randn_like(self.hidden_state)
        self.hidden_state = self.dynamics(combined) + noise
        
        # 获取观察
        observation = self.get_multimodal_observation()
        
        # 计算奖励
        reward = self.get_reward(action)
        
        # 判断是否结束（简化：固定步数）
        done = False
        
        return observation, reward, done
    
    def get_multimodal_observation(self) -> Dict[str, torch.Tensor]:
        """获取多模态观察"""
        vision = self.vision_model(self.hidden_state)
        olfactory = self.olfactory_model(self.hidden_state)
        proprio = self.proprio_model(self.hidden_state)
        if self.observation_noise_std > 0:
            vision = vision + self.observation_noise_std * torch.randn_like(vision)
            olfactory = olfactory + self.observation_noise_std * torch.randn_like(olfactory)
            proprio = proprio + self.observation_noise_std * torch.randn_like(proprio)
        return {
            "vision": vision,
            "olfactory": olfactory,
            "proprio": proprio,
        }

    def get_observation(self) -> torch.Tensor:
        """兼容旧接口：返回拼接后的观察向量"""
        obs_dict = self.get_multimodal_observation()
        return torch.cat([obs_dict["vision"], obs_dict["olfactory"], obs_dict["proprio"]], dim=-1)
    
    def get_reward(self, action: torch.Tensor) -> torch.Tensor:
        """计算奖励"""
        # 奖励 = 接近目标状态 + 动作平滑度
        state_error = torch.norm(self.hidden_state - self.target_state)
        action_penalty = 0.1 * torch.norm(action)
        reward = -state_error - action_penalty

        # 目标状态轻微漂移，增加任务难度
        if self.target_drift_std > 0:
            drift = self.target_drift_std * torch.randn_like(self.target_state)
            self.target_state = (self.target_state + drift).clamp(-2.0, 2.0)
        
        return reward.unsqueeze(0)  # [1]
    
    def get_true_state(self) -> torch.Tensor:
        """获取真实状态（用于评估，实际不可见）"""
        return self.hidden_state.clone()
    
    def set_target_state(self, target: torch.Tensor):
        """设置目标状态"""
        self.target_state = target.to(self.device)


class WorldModelInterface:
    """
    世界模型接口：为 AONN 提供标准化的环境交互
    """
    
    def __init__(self, world_model: SimpleWorldModel):
        self.world_model = world_model
    
    def get_observation(self) -> Dict[str, torch.Tensor]:
        """获取观察（返回多模态字典）"""
        return self.world_model.get_multimodal_observation()
    
    def get_reward(self, action: torch.Tensor) -> torch.Tensor:
        """获取奖励（用于学习）"""
        return self.world_model.get_reward(action)
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """执行动作，返回多模态观察和奖励"""
        obs, reward, done = self.world_model.step(action)
        return obs, reward
    
    def reset(self) -> torch.Tensor:
        """重置环境"""
        return self.world_model.reset()
    
    def get_true_state(self) -> torch.Tensor:
        """获取真实状态（用于评估）"""
        return self.world_model.get_true_state()

