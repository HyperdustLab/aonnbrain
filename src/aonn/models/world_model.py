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
        device=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.reward_dim = reward_dim
        self.device = device or torch.device("cpu")
        
        # 环境状态（隐藏状态）
        self.hidden_state = torch.randn(state_dim, device=self.device) * 0.1
        
        # 状态转移模型（真实世界动力学）
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, state_dim),
        ).to(self.device)
        
        # 观察模型（从状态到观察）
        self.observation_model = nn.Sequential(
            nn.Linear(state_dim, obs_dim * 2),
            nn.ReLU(),
            nn.Linear(obs_dim * 2, obs_dim),
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
        return self.get_observation()
    
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
        self.hidden_state = self.dynamics(combined) + 0.01 * torch.randn_like(self.hidden_state)
        
        # 获取观察
        observation = self.get_observation()
        
        # 计算奖励
        reward = self.get_reward(action)
        
        # 判断是否结束（简化：固定步数）
        done = False
        
        return observation, reward, done
    
    def get_observation(self) -> torch.Tensor:
        """获取当前观察（部分可观察）"""
        obs = self.observation_model(self.hidden_state)
        # 添加噪声
        obs = obs + 0.01 * torch.randn_like(obs)
        return obs
    
    def get_reward(self, action: torch.Tensor) -> torch.Tensor:
        """计算奖励"""
        # 奖励 = 接近目标状态 + 动作平滑度
        state_error = torch.norm(self.hidden_state - self.target_state)
        action_penalty = 0.1 * torch.norm(action)
        reward = -state_error - action_penalty
        
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
    
    def get_observation(self) -> torch.Tensor:
        """获取观察（对应 sensory Object）"""
        return self.world_model.get_observation()
    
    def get_reward(self, action: torch.Tensor) -> torch.Tensor:
        """获取奖励（用于学习）"""
        return self.world_model.get_reward(action)
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行动作，返回观察和奖励"""
        obs, reward, done = self.world_model.step(action)
        return obs, reward
    
    def reset(self) -> torch.Tensor:
        """重置环境"""
        return self.world_model.reset()
    
    def get_true_state(self) -> torch.Tensor:
        """获取真实状态（用于评估）"""
        return self.world_model.get_true_state()

