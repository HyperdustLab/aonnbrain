# src/aonn/models/office_ai_world_model.py
"""
Office AI 世界模型：模拟办公场景的智能体环境

状态空间：
- 文档状态：256 维（文档内容、格式、结构）
- 任务状态：128 维（待办事项、优先级、进度）
- 日程状态：64 维（会议、事件、时间安排）
- 上下文状态：128 维（当前工作上下文、历史记录）

感官空间：
- 文档内容：256 维（文档文本、格式特征）
- 表格数据：128 维（表格结构、数据特征）
- 日历事件：64 维（日程安排、会议信息）

动作空间：
- 编辑操作：64 维（文档编辑、表格修改）
- 发送操作：32 维（邮件发送、消息发送）
- 安排操作：32 维（日程安排、任务分配）

复杂度：⭐⭐⭐ (中等偏上)
- 比 GeneralAI 简单（状态 768 vs 2112，观察 448 vs 1408）
- 比 LineWorm 复杂（多模态、结构化数据）
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficeAIWorldModel(nn.Module):
    """
    Office AI 世界模型
    
    特性：
    - 专注于办公场景（文档、表格、邮件、日程）
    - 结构化状态空间
    - 多模态感官输入
    - 办公动作空间
    """
    
    def __init__(
        self,
        # 状态维度配置
        document_dim: int = 256,  # 文档状态
        task_dim: int = 128,      # 任务状态
        schedule_dim: int = 64,   # 日程状态
        context_dim: int = 128,   # 上下文状态
        # 感官维度配置
        document_obs_dim: int = 256,  # 文档观察
        table_obs_dim: int = 128,     # 表格观察
        calendar_obs_dim: int = 64,   # 日历观察
        # 动作维度配置
        action_dim: int = 128,  # 办公动作（编辑、发送、安排）
        # 其他配置
        device: Optional[torch.device] = None,
        state_noise_std: float = 0.01,
        observation_noise_std: float = 0.01,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        
        # ========== 状态空间 ==========
        self.document_dim = document_dim
        self.task_dim = task_dim
        self.schedule_dim = schedule_dim
        self.context_dim = context_dim
        self.total_state_dim = document_dim + task_dim + schedule_dim + context_dim
        
        # ========== 感官空间 ==========
        self.document_obs_dim = document_obs_dim
        self.table_obs_dim = table_obs_dim
        self.calendar_obs_dim = calendar_obs_dim
        self.total_obs_dim = document_obs_dim + table_obs_dim + calendar_obs_dim
        
        # ========== 动作空间 ==========
        self.action_dim = action_dim
        
        # ========== 噪声配置 ==========
        self.state_noise_std = state_noise_std
        self.observation_noise_std = observation_noise_std
        
        # ========== 状态初始化 ==========
        self.document_state = torch.randn(document_dim, device=self.device) * 0.1
        self.task_state = torch.zeros(task_dim, device=self.device)
        self.schedule_state = torch.randn(schedule_dim, device=self.device) * 0.1
        self.context_state = torch.randn(context_dim, device=self.device) * 0.1
        
        # ========== 状态转移模型 ==========
        hidden_dim = max(256, self.total_state_dim // 2)
        self.dynamics = nn.Sequential(
            nn.Linear(self.total_state_dim + action_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.total_state_dim),
        ).to(self.device)
        
        # ========== 观察生成模型 ==========
        # 文档观察模型
        self.document_obs_model = nn.Sequential(
            nn.Linear(self.total_state_dim, document_obs_dim * 2),
            nn.ReLU(),
            nn.Linear(document_obs_dim * 2, document_obs_dim),
        ).to(self.device)
        
        # 表格观察模型
        self.table_obs_model = nn.Sequential(
            nn.Linear(self.total_state_dim, table_obs_dim * 2),
            nn.ReLU(),
            nn.Linear(table_obs_dim * 2, table_obs_dim),
        ).to(self.device)
        
        # 日历观察模型
        self.calendar_obs_model = nn.Sequential(
            nn.Linear(self.total_state_dim, calendar_obs_dim * 2),
            nn.ReLU(),
            nn.Linear(calendar_obs_dim * 2, calendar_obs_dim),
        ).to(self.device)
        
        # ========== 奖励模型 ==========
        self.reward_model = nn.Sequential(
            nn.Linear(self.total_state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)
    
    def _get_full_state(self) -> torch.Tensor:
        """获取完整状态向量"""
        return torch.cat([
            self.document_state,
            self.task_state,
            self.schedule_state,
            self.context_state,
        ], dim=-1)
    
    def _set_full_state(self, state: torch.Tensor):
        """设置完整状态向量"""
        idx = 0
        self.document_state = state[idx:idx+self.document_dim].clone()
        idx += self.document_dim
        self.task_state = state[idx:idx+self.task_dim].clone()
        idx += self.task_dim
        self.schedule_state = state[idx:idx+self.schedule_dim].clone()
        idx += self.schedule_dim
        self.context_state = state[idx:idx+self.context_dim].clone()
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境"""
        self.document_state = torch.randn(self.document_dim, device=self.device) * 0.1
        self.task_state = torch.zeros(self.task_dim, device=self.device)
        self.schedule_state = torch.randn(self.schedule_dim, device=self.device) * 0.1
        self.context_state = torch.randn(self.context_dim, device=self.device) * 0.1
        return self.get_observation()
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        """执行一步动作"""
        current_full_state = self._get_full_state()
        combined = torch.cat([current_full_state, action], dim=-1)
        
        # 状态转移（添加噪声）
        noise = self.state_noise_std * torch.randn_like(current_full_state)
        next_full_state = self.dynamics(combined) + noise
        self._set_full_state(next_full_state)
        
        # 生成观察
        observation = self.get_observation()
        
        # 计算奖励
        reward = self.get_reward(action)
        
        # 简化：永远不结束
        done = False
        
        return observation, reward, done
    
    def get_observation(self) -> Dict[str, torch.Tensor]:
        """获取多模态观察"""
        current_full_state = self._get_full_state()
        
        # 生成各模态观察
        document_obs = self.document_obs_model(current_full_state)
        table_obs = self.table_obs_model(current_full_state)
        calendar_obs = self.calendar_obs_model(current_full_state)
        
        # 添加观察噪声
        if self.observation_noise_std > 0:
            document_obs += self.observation_noise_std * torch.randn_like(document_obs)
            table_obs += self.observation_noise_std * torch.randn_like(table_obs)
            calendar_obs += self.observation_noise_std * torch.randn_like(calendar_obs)
        
        return {
            "document": document_obs,
            "table": table_obs,
            "calendar": calendar_obs,
        }
    
    def get_reward(self, action: torch.Tensor) -> torch.Tensor:
        """计算奖励"""
        current_full_state = self._get_full_state()
        combined = torch.cat([current_full_state, action], dim=-1)
        reward = self.reward_model(combined)
        return reward.squeeze(0)
    
    def get_true_state(self) -> torch.Tensor:
        """获取真实状态（用于学习）"""
        return self._get_full_state().detach().clone()


class OfficeAIWorldInterface:
    """Office AI 世界模型接口"""
    
    def __init__(self, world_model: OfficeAIWorldModel):
        self.world_model = world_model
    
    def reset(self) -> Dict[str, torch.Tensor]:
        return self.world_model.reset()
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        obs, reward, done = self.world_model.step(action)
        return obs, reward
    
    def get_observation(self) -> Dict[str, torch.Tensor]:
        return self.world_model.get_observation()
    
    def get_true_state(self) -> torch.Tensor:
        return self.world_model.get_true_state()

