# src/aonn/models/lineworm_world_model.py
"""
线虫世界模型：模拟 2D 环境中的化学/温度/触觉场景
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SourceField:
    position: torch.Tensor
    strength: float
    sigma: float


class LineWormWorldModel:
    """
    线虫级世界模型：
    - 状态：隐藏向量 + 位置 + 朝向 + 内部能量
    - 感官：chemo / thermo / touch
    - 动作：forward (推进) + turn (转向)
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 32,
        chemo_dim: int = 128,
        thermo_dim: int = 32,
        touch_dim: int = 64,
        plane_size: float = 10.0,
        preferred_temp: float = 0.2,
        device: Optional[torch.device] = None,
        noise_config: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chemo_dim = chemo_dim
        self.thermo_dim = thermo_dim
        self.touch_dim = touch_dim
        self.device = device or torch.device("cpu")
        self.preferred_temp = preferred_temp
        self.plane_size = plane_size

        self.hidden_state = torch.zeros(state_dim, device=self.device)
        self.position = torch.zeros(2, device=self.device)
        self.heading = torch.zeros(1, device=self.device)
        self.energy = torch.tensor(1.0, device=self.device)

        hidden = max(64, chemo_dim // 2)
        self.chemo_encoder = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, chemo_dim),
        ).to(self.device)

        hidden_t = max(32, thermo_dim)
        self.thermo_encoder = nn.Sequential(
            nn.Linear(4, hidden_t),
            nn.ReLU(),
            nn.Linear(hidden_t, thermo_dim),
        ).to(self.device)

        hidden_touch = max(64, touch_dim)
        self.touch_encoder = nn.Sequential(
            nn.Linear(8, hidden_touch),
            nn.ReLU(),
            nn.Linear(hidden_touch, touch_dim),
        ).to(self.device)

        hidden_state_updater = max(128, state_dim)
        self.state_updater = nn.Sequential(
            nn.Linear(state_dim + 6, hidden_state_updater),
            nn.Tanh(),
            nn.Linear(hidden_state_updater, state_dim),
        ).to(self.device)

        self.chemo_sources: List[SourceField] = [
            SourceField(torch.tensor([3.0, 2.5], device=self.device), 1.2, 2.0),
            SourceField(torch.tensor([-2.0, -3.5], device=self.device), 0.8, 1.5),
        ]
        self.heat_sources: List[SourceField] = [
            SourceField(torch.tensor([4.0, -4.0], device=self.device), 1.5, 3.0),
        ]
        self.obstacles = [
            (torch.tensor([0.0, 0.0], device=self.device), 1.0),
            (torch.tensor([-3.0, 3.0], device=self.device), 0.8),
        ]
        default_noise = {
            "chemo": {
                "std": 0.04,
                "temporal_corr": 0.97,
                "spatial_scale": 3.0,
                "basis_size": 6,
                "amplitude": 0.4,
            },
            "thermo": {
                "std": 0.03,
                "temporal_corr": 0.95,
                "spatial_scale": 4.5,
                "basis_size": 5,
                "amplitude": 0.25,
            },
        }
        cfg = default_noise
        if noise_config is not None:
            for k, v in noise_config.items():
                if k in cfg:
                    cfg[k].update(v)
        self.chemo_noise_params = cfg["chemo"]
        self.thermo_noise_params = cfg["thermo"]
        self.chemo_noise_basis = torch.empty(
            self.chemo_noise_params["basis_size"], 2, device=self.device
        ).uniform_(-self.plane_size, self.plane_size)
        self.thermo_noise_basis = torch.empty(
            self.thermo_noise_params["basis_size"], 2, device=self.device
        ).uniform_(-self.plane_size, self.plane_size)
        self.chemo_noise_state = torch.zeros(
            self.chemo_noise_params["basis_size"], device=self.device
        )
        self.thermo_noise_state = torch.zeros(
            self.thermo_noise_params["basis_size"], device=self.device
        )
        self._noise_dirty = True

    def reset(self) -> Dict[str, torch.Tensor]:
        self.hidden_state = torch.randn(self.state_dim, device=self.device) * 0.1
        self.position = torch.empty(2, device=self.device).uniform_(
            -self.plane_size / 2, self.plane_size / 2
        )
        self.heading = torch.rand(1, device=self.device) * 2 * torch.pi
        self.energy = torch.tensor(1.0, device=self.device)
        self.chemo_noise_state.zero_()
        self.thermo_noise_state.zero_()
        self._noise_dirty = True
        return self.get_observation()

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        # 使用前两个动作分量：推进 + 转向
        forward = torch.tanh(action[0]) * 0.4
        turn = torch.tanh(action[1]) * 0.3
        self.heading = (self.heading + turn) % (2 * torch.pi)

        dx = forward * torch.cos(self.heading)
        dy = forward * torch.sin(self.heading)
        self.position = self.position + torch.cat([dx, dy])

        # 边界处理
        self.position = torch.clamp(self.position, -self.plane_size, self.plane_size)

        # 更新隐藏状态
        control = torch.stack([forward, turn]).detach()
        features = torch.cat(
            [self.position.detach(), self.heading.detach(), self.energy.detach().unsqueeze(0), control]
        )
        self.hidden_state = self.state_updater(
            torch.cat([self.hidden_state, features])
        )

        # 能量消耗
        self.energy = torch.clamp(self.energy - 0.01 + self._chemo_scalar(self.position) * 0.002, 0.0, 2.0)
        self._noise_dirty = True
        obs = self.get_observation()
        reward = self._compute_reward(obs)
        done = bool(self.energy.item() <= 0.05)
        return obs, reward, done

    def get_observation(self) -> Dict[str, torch.Tensor]:
        if self._noise_dirty:
            self._advance_field_noise()
            self._noise_dirty = False
        chemo_value = self._chemo_scalar(self.position)
        thermo_value = self._thermo_scalar(self.position)
        touch_features = self._touch_features(self.position)

        chemo_vec = self.chemo_encoder(self._feature_vector(chemo_value))
        thermo_vec = self.thermo_encoder(self._feature_vector(thermo_value))
        touch_vec = self.touch_encoder(touch_features)

        return {
            "chemo": chemo_vec,
            "thermo": thermo_vec,
            "touch": touch_vec,
        }

    def get_reward(self, action: torch.Tensor) -> torch.Tensor:
        obs = self.get_observation()
        return self._compute_reward(obs)

    def get_true_state(self) -> torch.Tensor:
        return self.hidden_state.detach().clone()

    def _compute_reward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        chemo = obs["chemo"].mean()
        thermo = obs["thermo"].mean()
        touch_penalty = torch.relu(obs["touch"].abs().mean() - 0.3)
        temp_penalty = torch.abs(thermo - self.preferred_temp)
        reward = chemo - 0.4 * temp_penalty - 0.5 * touch_penalty
        return reward.unsqueeze(0)

    def _feature_vector(self, value: torch.Tensor) -> torch.Tensor:
        v = value.detach()
        return torch.tensor(
            [
                v.item(),
                (v ** 2).item(),
                torch.sin(v / 2).item(),
                torch.cos(v / 2).item(),
            ],
            device=self.device,
        )

    def _chemo_scalar(self, pos: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.device)
        for src in self.chemo_sources:
            dist_sq = torch.sum((pos - src.position) ** 2)
            total += src.strength * torch.exp(-dist_sq / (2 * src.sigma ** 2))
        noise = self._evaluate_noise_field(
            pos,
            self.chemo_noise_basis,
            self.chemo_noise_state,
            self.chemo_noise_params,
        )
        total = total + noise
        return torch.clamp(total, 0.0, 2.5)

    def _thermo_scalar(self, pos: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.device)
        for src in self.heat_sources:
            dist_sq = torch.sum((pos - src.position) ** 2)
            total += src.strength * torch.exp(-dist_sq / (2 * src.sigma ** 2))
        noise = self._evaluate_noise_field(
            pos,
            self.thermo_noise_basis,
            self.thermo_noise_state,
            self.thermo_noise_params,
        )
        total = total + noise
        return torch.clamp(total, -1.0, 2.5)

    def _touch_features(self, pos: torch.Tensor) -> torch.Tensor:
        directions = torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            device=self.device,
        )
        distances = []
        for direction in directions:
            ray = pos.clone()
            dist = 0.0
            for _ in range(20):
                ray = ray + direction * 0.3
                dist += 0.3
                if torch.any(ray.abs() > self.plane_size):
                    break
                for center, radius in self.obstacles:
                    if torch.norm(ray - center) <= radius:
                        break
            distances.append(dist)
        heading = self.heading.detach()
        features = torch.tensor(distances + [self.energy.item(), torch.sin(heading).item(),
                                             torch.cos(heading).item(), 1.0], device=self.device)
        return features

    def _advance_field_noise(self):
        self.chemo_noise_state = (
            self.chemo_noise_state * self.chemo_noise_params["temporal_corr"]
            + torch.randn_like(self.chemo_noise_state) * self.chemo_noise_params["std"]
        )
        self.thermo_noise_state = (
            self.thermo_noise_state * self.thermo_noise_params["temporal_corr"]
            + torch.randn_like(self.thermo_noise_state) * self.thermo_noise_params["std"]
        )

    def _evaluate_noise_field(
        self,
        pos: torch.Tensor,
        basis: torch.Tensor,
        coeffs: torch.Tensor,
        params: Dict[str, float],
    ) -> torch.Tensor:
        if basis.nelement() == 0:
            return torch.tensor(0.0, device=self.device)
        spatial_scale = params["spatial_scale"]
        weights = torch.exp(
            -torch.sum((basis - pos) ** 2, dim=1) / (2 * spatial_scale ** 2)
        )
        if torch.sum(weights) < 1e-6:
            return torch.tensor(0.0, device=self.device)
        value = torch.dot(weights, coeffs) / (weights.sum() + 1e-6)
        return value * params.get("amplitude", 0.3)


class LineWormWorldInterface:
    def __init__(self, world_model: LineWormWorldModel):
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

