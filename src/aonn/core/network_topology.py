# src/aonn/core/network_topology.py
"""
AONN 网络拓扑结构定义
定义 Object 节点和 Aspect 边之间的连接关系
"""
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .object import ObjectNode
from .aspect_base import AspectBase


@dataclass
class NetworkEdge:
    """网络边：连接源 Object 和目标 Object"""
    src: str  # 源 Object 名称
    dst: str  # 目标 Object 名称
    aspect: AspectBase  # 负责此连接的 Aspect
    weight: float = 1.0  # 边的权重


class NetworkTopology:
    """
    AONN 网络拓扑结构
    
    定义：
    - Object 节点集合
    - Aspect 边集合
    - 网络连接图
    """
    
    def __init__(self):
        self.objects: Dict[str, ObjectNode] = {}
        self.aspects: List[AspectBase] = []
        self.edges: List[NetworkEdge] = []
        
        # 图结构：邻接表
        self.incoming_edges: Dict[str, List[NetworkEdge]] = defaultdict(list)  # dst -> edges
        self.outgoing_edges: Dict[str, List[NetworkEdge]] = defaultdict(list)  # src -> edges
        
        # Aspect 到边的映射
        self.aspect_to_edges: Dict[AspectBase, List[NetworkEdge]] = defaultdict(list)
    
    def add_object(self, name: str, obj: ObjectNode):
        """添加 Object 节点"""
        self.objects[name] = obj
    
    def add_aspect(self, aspect: AspectBase):
        """添加 Aspect 并自动创建边"""
        self.aspects.append(aspect)
        
        # 为每个 src -> dst 对创建边
        for src_name in aspect.src_names:
            for dst_name in aspect.dst_names:
                edge = NetworkEdge(
                    src=src_name,
                    dst=dst_name,
                    aspect=aspect,
                    weight=1.0
                )
                self.edges.append(edge)
                self.incoming_edges[dst_name].append(edge)
                self.outgoing_edges[src_name].append(edge)
                self.aspect_to_edges[aspect].append(edge)
    
    def get_edges_to(self, obj_name: str) -> List[NetworkEdge]:
        """获取指向某个 Object 的所有边"""
        return self.incoming_edges[obj_name]
    
    def get_edges_from(self, obj_name: str) -> List[NetworkEdge]:
        """获取从某个 Object 出发的所有边"""
        return self.outgoing_edges[obj_name]
    
    def get_aspect_edges(self, aspect: AspectBase) -> List[NetworkEdge]:
        """获取某个 Aspect 的所有边"""
        return self.aspect_to_edges[aspect]
    
    def get_topological_order(self) -> List[str]:
        """
        获取拓扑排序（用于前向传播）
        返回 Object 名称的有序列表
        """
        # 简单的拓扑排序（假设无环）
        in_degree = {name: len(self.incoming_edges[name]) for name in self.objects}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # 减少相邻节点的入度
            for edge in self.outgoing_edges[node]:
                dst = edge.dst
                in_degree[dst] -= 1
                if in_degree[dst] == 0:
                    queue.append(dst)
        
        # 如果还有节点未处理，说明有环（暂时忽略）
        return result
    
    def get_network_graph(self) -> Dict:
        """
        获取网络图结构（用于可视化）
        返回：{
            "nodes": [{"name": ..., "type": "object"}],
            "edges": [{"src": ..., "dst": ..., "aspect": ...}]
        }
        """
        nodes = [
            {"name": name, "type": "object", "dim": obj.dim}
            for name, obj in self.objects.items()
        ]
        
        edges = [
            {
                "src": edge.src,
                "dst": edge.dst,
                "aspect": edge.aspect.name,
                "weight": edge.weight
            }
            for edge in self.edges
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "num_objects": len(self.objects),
            "num_aspects": len(self.aspects),
            "num_edges": len(self.edges)
        }
    
    def visualize_network(self) -> str:
        """
        生成网络结构的文本可视化
        """
        lines = []
        lines.append("=" * 60)
        lines.append("AONN 网络拓扑结构")
        lines.append("=" * 60)
        lines.append(f"\nObject 节点 ({len(self.objects)} 个):")
        for name, obj in self.objects.items():
            lines.append(f"  - {name}: dim={obj.dim}")
        
        lines.append(f"\nAspect 边 ({len(self.aspects)} 个):")
        for aspect in self.aspects:
            edges = self.get_aspect_edges(aspect)
            for edge in edges:
                lines.append(f"  - {aspect.name}: {edge.src} -> {edge.dst}")
        
        lines.append(f"\n网络连接图:")
        for obj_name in self.objects:
            incoming = self.get_edges_to(obj_name)
            outgoing = self.get_edges_from(obj_name)
            if incoming or outgoing:
                lines.append(f"  {obj_name}:")
                if incoming:
                    srcs = [e.src for e in incoming]
                    lines.append(f"    <- {', '.join(srcs)}")
                if outgoing:
                    dsts = [e.dst for e in outgoing]
                    lines.append(f"    -> {', '.join(dsts)}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def build_network_topology(
    objects: Dict[str, ObjectNode],
    aspects: List[AspectBase]
) -> NetworkTopology:
    """
    从 Object 和 Aspect 构建网络拓扑
    """
    topology = NetworkTopology()
    
    # 添加所有 Object
    for name, obj in objects.items():
        topology.add_object(name, obj)
    
    # 添加所有 Aspect（自动创建边）
    for aspect in aspects:
        topology.add_aspect(aspect)
    
    return topology

