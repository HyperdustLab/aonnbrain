# src/aonn/models/general_ai_world_model.py
"""
通用AI智能体世界模型：模拟通用AI智能体的复杂环境

状态空间：
- 语义状态：1024-4096 维（语言/概念表示）
- 记忆状态：512-2048 维（工作记忆 + 长期记忆索引）
- 上下文状态：256-1024 维（对话历史、任务上下文）
- 物理状态：64-256 维（位置、姿态、工具状态）
- 目标状态：256-512 维（当前任务目标、子目标栈）

感官空间：
- 视觉：512-2048 维
- 语言：512-2048 维
- 音频：128-512 维
- 多模态融合：256-1024 维

动作空间：
- 语言生成：离散 token 序列
- 工具调用：结构化动作
- 多模态输出：文本 + 图像 + 代码
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralAIWorldModel:
    """
    通用AI智能体世界模型
    
    特性：
    - 多层级状态空间（物理 → 语义 → 抽象）
    - 多模态感官输入
    - 复杂动作空间（语言、工具调用）
    - 长期记忆机制
    - 多目标奖励函数
    """
    
    def __init__(
        self,
        # 状态维度配置
        semantic_dim: int = 1024,
        memory_dim: int = 512,
        context_dim: int = 256,
        physical_dim: int = 64,
        goal_dim: int = 256,
        # 感官维度配置
        vision_dim: int = 512,
        language_dim: int = 512,
        audio_dim: int = 128,
        multimodal_dim: int = 256,
        # 动作维度配置
        action_dim: int = 256,  # 混合动作空间（语言 + 工具）
        vocab_size: int = 50000,  # 词汇表大小（用于语言生成）
        # 其他配置
        device: Optional[torch.device] = None,
        state_noise_std: float = 0.01,
        observation_noise_std: float = 0.01,
        enable_tools: bool = True,
        max_tools: int = 10,
    ):
        self.device = device or torch.device("cpu")
        
        # ========== 状态空间 ==========
        self.semantic_dim = semantic_dim
        self.memory_dim = memory_dim
        self.context_dim = context_dim
        self.physical_dim = physical_dim
        self.goal_dim = goal_dim
        self.total_state_dim = semantic_dim + memory_dim + context_dim + physical_dim + goal_dim
        
        # ========== 感官空间 ==========
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.audio_dim = audio_dim
        self.multimodal_dim = multimodal_dim
        self.total_obs_dim = vision_dim + language_dim + audio_dim + multimodal_dim
        
        # ========== 动作空间 ==========
        self.action_dim = action_dim
        self.vocab_size = vocab_size
        self.enable_tools = enable_tools
        self.max_tools = max_tools
        
        # ========== 噪声配置 ==========
        self.state_noise_std = state_noise_std
        self.observation_noise_std = observation_noise_std
        
        # ========== 状态初始化 ==========
        self.semantic_state = torch.randn(semantic_dim, device=self.device) * 0.1
        self.memory_state = torch.zeros(memory_dim, device=self.device)
        self.context_state = torch.randn(context_dim, device=self.device) * 0.1
        self.physical_state = torch.randn(physical_dim, device=self.device) * 0.1
        self.goal_state = torch.randn(goal_dim, device=self.device) * 0.5
        
        # ========== 长期记忆库（简化：固定大小）==========
        self.long_term_memory = torch.randn(1000, memory_dim, device=self.device) * 0.1
        self.memory_index = 0
        
        # ========== 工具状态 ==========
        self.available_tools: List[str] = []
        self.tool_results: Dict[str, torch.Tensor] = {}
        if enable_tools:
            self.available_tools = [f"tool_{i}" for i in range(max_tools)]
        
        # ========== 状态转移模型 ==========
        # 使用 Transformer 风格的模型处理序列依赖
        hidden_dim = max(512, self.total_state_dim // 2)
        self.dynamics = nn.Sequential(
            nn.Linear(self.total_state_dim + action_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.total_state_dim),
        ).to(self.device)
        
        # ========== 感官编码器 ==========
        # 视觉编码器
        self.vision_encoder = nn.Sequential(
            nn.Linear(self.total_state_dim, vision_dim * 2),
            nn.LayerNorm(vision_dim * 2),
            nn.ReLU(),
            nn.Linear(vision_dim * 2, vision_dim),
        ).to(self.device)
        
        # 语言编码器
        self.language_encoder = nn.Sequential(
            nn.Linear(self.total_state_dim, language_dim * 2),
            nn.LayerNorm(language_dim * 2),
            nn.ReLU(),
            nn.Linear(language_dim * 2, language_dim),
        ).to(self.device)
        
        # 音频编码器
        self.audio_encoder = nn.Sequential(
            nn.Linear(self.total_state_dim, audio_dim * 2),
            nn.LayerNorm(audio_dim * 2),
            nn.ReLU(),
            nn.Linear(audio_dim * 2, audio_dim),
        ).to(self.device)
        
        # 多模态融合编码器
        self.multimodal_encoder = nn.Sequential(
            nn.Linear(vision_dim + language_dim + audio_dim, multimodal_dim * 2),
            nn.LayerNorm(multimodal_dim * 2),
            nn.ReLU(),
            nn.Linear(multimodal_dim * 2, multimodal_dim),
        ).to(self.device)
        
        # ========== 记忆更新模型 ==========
        self.memory_updater = nn.Sequential(
            nn.Linear(self.total_state_dim + context_dim, memory_dim * 2),
            nn.LayerNorm(memory_dim * 2),
            nn.ReLU(),
            nn.Linear(memory_dim * 2, memory_dim),
        ).to(self.device)
        
        # ========== 奖励模型（多目标）==========
        reward_dim = 10  # 任务完成度、知识获取、社交反馈、安全性等
        self.reward_model = nn.Sequential(
            nn.Linear(self.total_state_dim + action_dim, reward_dim * 2),
            nn.LayerNorm(reward_dim * 2),
            nn.ReLU(),
            nn.Linear(reward_dim * 2, reward_dim),
        ).to(self.device)
        
        # ========== 工具调用模型 ==========
        if enable_tools:
            self.tool_selector = nn.Sequential(
                nn.Linear(self.total_state_dim, max_tools * 2),
                nn.LayerNorm(max_tools * 2),
                nn.ReLU(),
                nn.Linear(max_tools * 2, max_tools),
            ).to(self.device)
    
    def get_full_state(self) -> torch.Tensor:
        """获取完整状态向量"""
        return torch.cat([
            self.semantic_state,
            self.memory_state,
            self.context_state,
            self.physical_state,
            self.goal_state,
        ], dim=-1)
    
    def set_full_state(self, state: torch.Tensor):
        """设置完整状态向量"""
        assert state.shape[-1] == self.total_state_dim
        idx = 0
        self.semantic_state = state[idx:idx+self.semantic_dim].clone()
        idx += self.semantic_dim
        self.memory_state = state[idx:idx+self.memory_dim].clone()
        idx += self.memory_dim
        self.context_state = state[idx:idx+self.context_dim].clone()
        idx += self.context_dim
        self.physical_state = state[idx:idx+self.physical_dim].clone()
        idx += self.physical_dim
        self.goal_state = state[idx:idx+self.goal_dim].clone()
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境"""
        self.semantic_state = torch.randn(self.semantic_dim, device=self.device) * 0.1
        self.memory_state = torch.zeros(self.memory_dim, device=self.device)
        self.context_state = torch.randn(self.context_dim, device=self.device) * 0.1
        self.physical_state = torch.randn(self.physical_dim, device=self.device) * 0.1
        self.goal_state = torch.randn(self.goal_dim, device=self.device) * 0.5
        self.memory_index = 0
        self.tool_results.clear()
        return self.get_multimodal_observation()
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        """
        执行一步动作
        
        Args:
            action: [action_dim] 动作向量（可能包含语言token、工具调用等）
        
        Returns:
            (observation, reward, done)
        """
        # 状态转移
        full_state = self.get_full_state()
        combined = torch.cat([full_state, action], dim=-1)
        noise = self.state_noise_std * torch.randn_like(full_state)
        next_state = self.dynamics(combined) + noise
        self.set_full_state(next_state)
        
        # 更新记忆（定期写入长期记忆）
        if torch.rand(1).item() < 0.1:  # 10% 概率更新记忆
            memory_update = self.memory_updater(
                torch.cat([full_state, self.context_state], dim=-1)
            )
            self.long_term_memory[self.memory_index % 1000] = memory_update
            self.memory_index += 1
        
        # 工具调用（如果启用）
        if self.enable_tools and action.shape[0] >= self.max_tools:
            tool_logits = self.tool_selector(full_state)
            tool_probs = F.softmax(tool_logits, dim=-1)
            selected_tool_idx = torch.argmax(tool_probs).item()
            if selected_tool_idx < len(self.available_tools):
                tool_name = self.available_tools[selected_tool_idx]
                # 模拟工具执行结果
                tool_result = torch.randn(self.memory_dim, device=self.device) * 0.1
                self.tool_results[tool_name] = tool_result
                # 将工具结果融合到状态中
                self.memory_state = self.memory_state + 0.1 * tool_result
        
        # 获取观察
        observation = self.get_multimodal_observation()
        
        # 计算奖励
        reward = self.get_reward(action)
        
        # 判断是否结束（简化：基于任务完成度）
        done = reward[0].item() > 0.9  # 任务完成度 > 0.9
        
        return observation, reward, done
    
    def get_multimodal_observation(self) -> Dict[str, torch.Tensor]:
        """获取多模态观察"""
        full_state = self.get_full_state()
        
        # 各模态编码
        vision = self.vision_encoder(full_state)
        language = self.language_encoder(full_state)
        audio = self.audio_encoder(full_state)
        
        # 添加观察噪声
        if self.observation_noise_std > 0:
            vision = vision + self.observation_noise_std * torch.randn_like(vision)
            language = language + self.observation_noise_std * torch.randn_like(language)
            audio = audio + self.observation_noise_std * torch.randn_like(audio)
        
        # 多模态融合
        multimodal_input = torch.cat([vision, language, audio], dim=-1)
        multimodal = self.multimodal_encoder(multimodal_input)
        
        return {
            "vision": vision,
            "language": language,
            "audio": audio,
            "multimodal": multimodal,
        }
    
    def get_reward(self, action: torch.Tensor) -> torch.Tensor:
        """计算多目标奖励"""
        full_state = self.get_full_state()
        combined = torch.cat([full_state, action], dim=-1)
        reward_vector = self.reward_model(combined)
        
        # 奖励向量包含：
        # [0]: 任务完成度
        # [1]: 知识获取
        # [2]: 社交反馈
        # [3]: 安全性
        # [4-9]: 其他目标
        
        # 归一化到合理范围
        reward_vector = torch.tanh(reward_vector)
        
        # 返回总奖励（加权和）
        weights = torch.tensor([0.4, 0.2, 0.2, 0.1, 0.1], device=self.device)
        total_reward = torch.sum(reward_vector[:5] * weights)
        
        return total_reward.unsqueeze(0)
    
    def get_true_state(self) -> torch.Tensor:
        """获取真实状态（用于评估）"""
        return self.get_full_state().clone()
    
    def set_target_state(self, target: torch.Tensor):
        """设置目标状态（用于 preference）"""
        if target.shape[-1] == self.goal_dim:
            self.goal_state = target.to(self.device)
        elif target.shape[-1] == self.total_state_dim:
            self.set_full_state(target)
        else:
            raise ValueError(f"目标状态维度不匹配: {target.shape[-1]} vs {self.goal_dim} or {self.total_state_dim}")


class GeneralAIWorldInterface:
    """
    通用AI智能体世界模型接口
    """
    
    def __init__(self, world_model: GeneralAIWorldModel):
        self.world_model = world_model
    
    def get_observation(self) -> Dict[str, torch.Tensor]:
        """获取多模态观察"""
        return self.world_model.get_multimodal_observation()
    
    def get_reward(self, action: torch.Tensor) -> torch.Tensor:
        """获取奖励"""
        return self.world_model.get_reward(action)
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """执行动作"""
        obs, reward, done = self.world_model.step(action)
        return obs, reward
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境"""
        return self.world_model.reset()
    
    def get_true_state(self) -> torch.Tensor:
        """获取真实状态"""
        return self.world_model.get_true_state()

