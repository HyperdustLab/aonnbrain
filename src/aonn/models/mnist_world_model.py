# src/aonn/models/mnist_world_model.py
"""
MNIST 世界模型：将 MNIST 数据集包装为 AONN 可交互的环境
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

try:
    from torchvision import datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class MNISTWorldModel:
    """
    MNIST 世界模型：将 MNIST 数据集包装为 AONN 可交互的环境
    
    状态空间：
    - 图像特征表示：256 维（内部状态）
    
    观察空间：
    - 图像像素：784 维（28x28）
    
    动作空间：
    - 分类输出：10 维（10个数字类别）
    
    目标：
    - 正确分类标签：10 维 one-hot 编码
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 10,  # 10个类别
        obs_dim: int = 784,    # 28x28
        device=None,
        dataset=None,  # MNIST 数据集
        data_root: str = './data',
        train: bool = True,
    ):
        """
        Args:
            state_dim: 状态维度（内部表示）
            action_dim: 动作维度（类别数）
            obs_dim: 观察维度（图像像素数）
            device: 设备
            dataset: 预加载的数据集（可选）
            data_root: 数据根目录
            train: 是否使用训练集
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.device = device or torch.device("cpu")
        self.data_root = data_root
        self.train = train
        
        # 当前样本索引
        self.current_idx = 0
        self.current_image = None
        self.current_label = None
        
        # 加载数据集
        if dataset is not None:
            self.dataset = dataset
        elif TORCHVISION_AVAILABLE:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.dataset = datasets.MNIST(
                root=data_root, 
                train=train, 
                download=True, 
                transform=transform
            )
        else:
            raise ImportError(
                "需要 torchvision 库来加载 MNIST 数据集。"
                "请运行: pip install torchvision"
            )
        
        # 图像编码器（可选，用于生成状态）
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim),
        ).to(self.device)
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """
        重置到随机样本，返回观察
        
        Returns:
            观察字典 {"vision": image_tensor}
        """
        self.current_idx = torch.randint(0, len(self.dataset), (1,)).item()
        image, label = self.dataset[self.current_idx]
        
        # 展平图像 [1, 28, 28] -> [784]
        self.current_image = image.flatten().to(self.device)
        self.current_label = label
        
        return {"vision": self.current_image}
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool]:
        """
        执行一步（分类预测）
        
        Args:
            action: [10] 分类预测（logits 或概率）
        
        Returns:
            (observation, reward, done)
        """
        # 计算奖励（分类正确性）
        pred_class = action.argmax().item()
        correct = (pred_class == self.current_label)
        reward = 1.0 if correct else -0.1
        
        # 移动到下一个样本
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        image, label = self.dataset[self.current_idx]
        self.current_image = image.flatten().to(self.device)
        self.current_label = label
        
        # 返回新观察
        obs = {"vision": self.current_image}
        done = False
        
        return obs, reward, done
    
    def get_observation(self) -> Dict[str, torch.Tensor]:
        """获取当前观察"""
        if self.current_image is None:
            return self.reset()
        return {"vision": self.current_image}
    
    def get_true_state(self) -> torch.Tensor:
        """
        获取真实状态（用于学习）
        
        返回图像的特征表示（通过编码器）
        """
        if self.current_image is None:
            return torch.zeros(self.state_dim, device=self.device)
        
        with torch.no_grad():
            state = self.encoder(self.current_image)
        return state
    
    def get_target(self) -> torch.Tensor:
        """
        获取目标（one-hot 编码的标签）
        
        Returns:
            [10] one-hot 向量
        """
        target = torch.zeros(10, device=self.device)
        if self.current_label is not None:
            target[self.current_label] = 1.0
        return target
    
    def get_label(self) -> Optional[int]:
        """获取当前标签（类别索引）"""
        return self.current_label


class MNISTWorldInterface:
    """MNIST 世界模型接口（兼容其他世界模型接口）"""
    
    def __init__(self, world_model: MNISTWorldModel):
        self.world_model = world_model
    
    def reset(self) -> Dict[str, torch.Tensor]:
        return self.world_model.reset()
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        执行一步
        
        Returns:
            (observation, reward)
        """
        obs, reward, done = self.world_model.step(action)
        return obs, reward
    
    def get_observation(self) -> Dict[str, torch.Tensor]:
        return self.world_model.get_observation()
    
    def get_true_state(self) -> torch.Tensor:
        return self.world_model.get_true_state()
    
    def get_target(self) -> torch.Tensor:
        return self.world_model.get_target()

