# src/aonn/core/evolution.py
"""
AONN 网络动态演化机制
支持 Object、Aspect、Pipeline 的动态创建和删除
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
from collections import defaultdict
import math

from .object import ObjectNode
from .aspect_base import AspectBase
from .free_energy import compute_total_free_energy


@dataclass
class EvolutionEvent:
    """演化事件记录"""
    step: int
    event_type: str  # "create_object", "create_aspect", "create_pipeline", "prune"
    details: Dict
    trigger_condition: str
    free_energy_before: float
    free_energy_after: float


class NetworkEvolution:
    """
    网络演化管理器
    
    功能：
    1. 动态创建 Object、Aspect、Pipeline
    2. 根据自由能变化决定演化
    3. 剪枝不重要的连接
    4. 记录演化历史
    """
    
    def __init__(
        self,
        free_energy_threshold: float = 1.0,
        prune_threshold: float = 0.01,
        max_objects: int = 100,
        max_aspects: int = 1000,
        evolution_rate: float = 0.1,
        batch_growth: Optional[Dict] = None,
    ):
        self.free_energy_threshold = free_energy_threshold
        self.prune_threshold = prune_threshold
        self.max_objects = max_objects
        self.max_aspects = max_aspects
        self.evolution_rate = evolution_rate
        self.batch_growth = batch_growth or {}
        
        # 演化历史
        self.evolution_history: List[EvolutionEvent] = []
        self.step_count = 0
        
        # 统计信息
        self.stats = {
            "objects_created": 0,
            "aspects_created": 0,
            "pipelines_created": 0,
            "pruned_count": 0,
        }
    
    def should_create_object(
        self,
        free_energy: float,
        num_objects: int,
        error_distribution: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """
        判断是否应该创建新 Object
        
        Returns:
            (should_create, reason)
        """
        if num_objects >= self.max_objects:
            return False, "max_objects_reached"
        
        # 如果自由能持续高，可能需要新 Object
        if free_energy > self.free_energy_threshold * 2:
            return True, "high_free_energy"
        
        # 如果某个 Object 的误差持续高，可能需要分解
        for obj_name, error in error_distribution.items():
            if error > self.free_energy_threshold:
                return True, f"high_error_in_{obj_name}"
        
        return False, None
    
    def should_create_aspect(
        self,
        src_name: str,
        dst_name: str,
        potential_free_energy_reduction: float,
        num_aspects: int
    ) -> Tuple[bool, Optional[str]]:
        """
        判断是否应该创建新 Aspect
        
        Returns:
            (should_create, reason)
        """
        if num_aspects >= self.max_aspects:
            return False, "max_aspects_reached"
        
        # 如果创建新 Aspect 能显著降低自由能
        if potential_free_energy_reduction > self.free_energy_threshold:
            return True, f"significant_reduction_{potential_free_energy_reduction:.2f}"
        
        return False, None
    
    def should_prune(
        self,
        aspect: AspectBase,
        free_energy_contrib: float,
        weight: float = 1.0
    ) -> Tuple[bool, Optional[str]]:
        """
        判断是否应该剪枝某个 Aspect
        
        Returns:
            (should_prune, reason)
        """
        # 如果自由能贡献很小且权重很低
        if free_energy_contrib < self.prune_threshold and weight < self.prune_threshold:
            return True, "low_contribution_and_weight"
        
        # 如果自由能贡献为负（说明是噪声）
        if free_energy_contrib < 0:
            return True, "negative_contribution"
        
        return False, None
    
    def plan_aspect_batch(
        self,
        error_value: float,
        current_target_count: int,
        remaining_capacity: int,
        growth_policy: Optional[Dict] = None,
    ) -> int:
        """
        根据误差和策略决定一次性创建多少个 Aspect
        """
        if remaining_capacity <= 0 or error_value <= 0:
            return 0
        if not math.isfinite(error_value):
            return 0
        
        policy = growth_policy or self.batch_growth
        threshold = policy.get("error_threshold", self.free_energy_threshold)
        if threshold <= 0:
            threshold = self.free_energy_threshold
        if threshold <= 0:
            threshold = 1e-6
        
        if error_value < threshold:
            return 0
        
        base = max(1, int(policy.get("base", 4)))
        max_per_step = max(1, int(policy.get("max_per_step", 32)))
        max_total = policy.get("max_total", None)
        
        error_ratio = max(1, int(error_value / threshold))
        plan = base * error_ratio
        plan = min(plan, max_per_step, remaining_capacity)
        
        if max_total is not None:
            remaining_for_target = max(0, int(max_total) - current_target_count)
            plan = min(plan, remaining_for_target)
        
        return max(plan, 0)
    
    def record_event(
        self,
        event_type: str,
        details: Dict,
        trigger_condition: str,
        free_energy_before: float,
        free_energy_after: float
    ):
        """记录演化事件"""
        event = EvolutionEvent(
            step=self.step_count,
            event_type=event_type,
            details=details,
            trigger_condition=trigger_condition,
            free_energy_before=free_energy_before,
            free_energy_after=free_energy_after
        )
        self.evolution_history.append(event)
        
        # 更新统计
        if event_type == "create_object":
            self.stats["objects_created"] += 1
        elif event_type == "create_aspect":
            self.stats["aspects_created"] += 1
        elif event_type == "create_pipeline":
            self.stats["pipelines_created"] += 1
        elif event_type == "prune":
            self.stats["pruned_count"] += 1
    
    def get_evolution_summary(self) -> Dict:
        """获取演化摘要"""
        return {
            "total_steps": self.step_count,
            "total_events": len(self.evolution_history),
            "stats": self.stats.copy(),
            "recent_events": [
                {
                    "step": e.step,
                    "type": e.event_type,
                    "condition": e.trigger_condition,
                    "F_before": e.free_energy_before,
                    "F_after": e.free_energy_after,
                }
                for e in self.evolution_history[-10:]  # 最近10个事件
            ]
        }
    
    def increment_step(self):
        """增加步数"""
        self.step_count += 1

