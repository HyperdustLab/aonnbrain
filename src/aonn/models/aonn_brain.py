# src/aonn/models/aonn_brain.py
from typing import Dict
import torch.nn as nn

from aonn.core.object import ObjectNode
from aonn.core.network_topology import NetworkTopology, build_network_topology
from aonn.aspects.sensory_aspect import LinearGenerativeAspect
from aonn.aspects.llm_aspect import LLMAspect


class AONNBrain(nn.Module):
    """
    把 Object + 多种 Aspect 组装成一个"自由能大脑"
    
    网络架构：
    - Object 节点：存储状态
    - Aspect 边：连接 Object，计算预测和自由能
    - 网络拓扑：定义完整的连接图
    """
    def __init__(self, config, llm_client=None, device=None):
        super().__init__()
        self.device = device

        # Object 层
        self.objects: Dict[str, ObjectNode] = {
            "sensory": ObjectNode("sensory", dim=config["obs_dim"], device=device),
            "internal": ObjectNode("internal", dim=config["state_dim"], device=device),
            "action": ObjectNode("action", dim=config["act_dim"], device=device),
            "semantic_context": ObjectNode("semantic_context", dim=config["sem_dim"], device=device),
            "semantic_prediction": ObjectNode("semantic_prediction", dim=config["sem_dim"], device=device),
        }

        # Aspect 层
        self.sensory_aspect = LinearGenerativeAspect(
            internal_name="internal",
            sensory_name="sensory",
            state_dim=config["state_dim"],
            obs_dim=config["obs_dim"],
        )
        self.llm_aspect = LLMAspect(
            src_names=("semantic_context",),
            dst_names=("semantic_prediction",),
            llm_client=llm_client,
            llm_config=config.get("llm", {}),
        )

        # 注意：LLMAspect 不是 nn.Module，所以不能直接放入 ModuleList
        # 使用普通列表存储所有 aspects
        self.aspects = [self.sensory_aspect, self.llm_aspect]
        # 只将 nn.Module 子类的 Aspect 注册到 ModuleList（用于参数管理）
        self.aspect_modules = nn.ModuleList([self.sensory_aspect])
        # 如果 llm_client 是 nn.Module，也注册它
        if llm_client is not None and isinstance(llm_client, nn.Module):
            self.aspect_modules.append(llm_client)
        
        # 构建网络拓扑结构
        self.topology = build_network_topology(self.objects, self.aspects)
        
        # TODO: dynamics_aspect, intent_aspect, action_aspect ...
    
    def get_network_graph(self) -> Dict:
        """获取网络图结构"""
        return self.topology.get_network_graph()
    
    def visualize_network(self) -> str:
        """可视化网络结构"""
        return self.topology.visualize_network()

