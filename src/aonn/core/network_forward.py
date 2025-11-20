# src/aonn/core/network_forward.py
"""
AONN 网络前向传播
基于网络拓扑结构执行前向传播
"""
from typing import Dict, List, Optional
import torch

from .object import ObjectNode
from .aspect_base import AspectBase
from .network_topology import NetworkTopology


def forward_pass(
    topology: NetworkTopology,
    fixed_objects: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    执行网络前向传播
    
    Args:
        topology: 网络拓扑结构
        fixed_objects: 固定的 Object 列表（不更新状态）
    
    Returns:
        所有 Object 的状态字典
    """
    if fixed_objects is None:
        fixed_objects = []
    
    # 获取拓扑排序
    topo_order = topology.get_topological_order()
    
    # 按拓扑顺序处理每个 Object
    for obj_name in topo_order:
        if obj_name in fixed_objects:
            continue  # 跳过固定的 Object
        
        # 获取所有指向此 Object 的边
        incoming_edges = topology.get_edges_to(obj_name)
        
        if not incoming_edges:
            continue  # 没有输入边，跳过
        
        # 收集所有 Aspect 的预测
        predictions = []
        for edge in incoming_edges:
            aspect = edge.aspect
            # 计算 Aspect 的预测
            pred_dict = aspect.forward(topology.objects)
            if obj_name in pred_dict:
                predictions.append(pred_dict[obj_name] * edge.weight)
        
        # 如果有预测，更新 Object 状态（这里简化处理）
        # 实际应该根据自由能最小化来更新
        if predictions:
            # 简单平均（实际应该用自由能最小化）
            avg_pred = torch.stack(predictions).mean(dim=0)
            topology.objects[obj_name].set_state(avg_pred)
    
    # 返回所有状态
    return {name: obj.state.clone() for name, obj in topology.objects.items()}


def compute_network_free_energy(topology: NetworkTopology) -> torch.Tensor:
    """
    计算整个网络的自由能
    """
    from .free_energy import compute_total_free_energy
    return compute_total_free_energy(topology.objects, topology.aspects)

