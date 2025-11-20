# src/aonn/models/aonn_brain_v2.py
"""
AONN Brain V2：按照 AONN网络.txt 设计的正确架构

核心特点：
1. 状态与计算完全分离
2. Object Layer（状态层）：只有状态，不计算
3. Aspect Layer（计算层）：只有神经元/Aspect，没有 Object
4. 深度网络：Object → Aspect → Aspect → ... → Aspect → Object
"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from aonn.core.object import ObjectNode
from aonn.core.object_layer import ObjectLayer
from aonn.core.aspect_layer import AspectLayer, AspectPipeline
from aonn.core.aspect_base import AspectBase
from aonn.core.network_topology import NetworkTopology, build_network_topology


class AONNBrainV2(nn.Module):
    """
    AONN Brain V2：正确的深度 AONN 架构
    
    结构：
    - Object Layer (0) → Aspect Pipeline → Object Layer (1) → Aspect Pipeline → ... → Object Layer (L)
    
    特点：
    - Object Layer：只存储状态（Vertical MB）
    - Aspect Pipeline：只有神经元/Aspect（Cross-MB），没有 Object
    - 隐层完全是 Aspect，没有普通 object 点
    """
    
    def __init__(
        self,
        config: Dict,
        llm_client=None,
        device=None,
    ):
        """
        Args:
            config: 配置字典，包含：
                - input_dim: 输入维度
                - hidden_dims: 隐藏层维度列表
                - output_dim: 输出维度
                - num_aspects: 每层 Aspect 数量
                - aspect_depth: Aspect Pipeline 深度
                - use_gate: 是否使用 gate
            llm_client: LLM 客户端（可选）
            device: 设备
        """
        super().__init__()
        self.device = device or torch.device("cpu")
        self.config = config
        
        # ========== Object Layers ==========
        # Object Layer 0: 输入层
        input_dim = config.get("input_dim", 128)
        self.input_layer = ObjectLayer(
            "input",
            {
                "input": ObjectNode("input", dim=input_dim, device=self.device)
            }
        )
        
        # Object Layer 1..L-1: 隐藏层
        hidden_dims = config.get("hidden_dims", [256, 256])
        self.hidden_layers: List[ObjectLayer] = []
        for i, dim in enumerate(hidden_dims):
            layer = ObjectLayer(
                f"hidden_{i+1}",
                {
                    f"hidden_{i+1}": ObjectNode(f"hidden_{i+1}", dim=dim, device=self.device)
                }
            )
            self.hidden_layers.append(layer)
        
        # Object Layer L: 输出层
        output_dim = config.get("output_dim", 10)
        self.output_layer = ObjectLayer(
            "output",
            {
                "output": ObjectNode("output", dim=output_dim, device=self.device)
            }
        )
        
        # 所有 Object Layers
        self.object_layers: List[ObjectLayer] = [
            self.input_layer,
            *self.hidden_layers,
            self.output_layer,
        ]
        
        # ========== Aspect Pipelines ==========
        # Aspect Pipeline 连接相邻的 Object Layers
        num_aspects = config.get("num_aspects", 32)
        aspect_depth = config.get("aspect_depth", 4)
        use_gate = config.get("use_gate", False)
        
        self.aspect_pipelines: List[AspectPipeline] = []
        
        # Input → Hidden_1
        if len(self.hidden_layers) > 0:
            first_hidden_dim = hidden_dims[0]
            pipeline = AspectPipeline(
                input_dim=input_dim,
                output_dim=first_hidden_dim,
                num_aspects=num_aspects,
                depth=aspect_depth,
                use_gate=use_gate,
            )
            self.aspect_pipelines.append(pipeline)
        
        # Hidden_i → Hidden_{i+1}
        for i in range(len(self.hidden_layers) - 1):
            input_dim_i = hidden_dims[i]
            output_dim_i = hidden_dims[i + 1]
            pipeline = AspectPipeline(
                input_dim=input_dim_i,
                output_dim=output_dim_i,
                num_aspects=num_aspects,
                depth=aspect_depth,
                use_gate=use_gate,
            )
            self.aspect_pipelines.append(pipeline)
        
        # Hidden_L → Output
        if len(self.hidden_layers) > 0:
            last_hidden_dim = hidden_dims[-1]
            pipeline = AspectPipeline(
                input_dim=last_hidden_dim,
                output_dim=output_dim,
                num_aspects=num_aspects,
                depth=aspect_depth,
                use_gate=use_gate,
            )
            self.aspect_pipelines.append(pipeline)
        
        # 注册为 Module
        self.aspect_modules = nn.ModuleList(self.aspect_pipelines)
        
        # ========== 收集所有 Object（用于拓扑构建）==========
        all_objects: Dict[str, ObjectNode] = {}
        for layer in self.object_layers:
            all_objects.update(layer.objects)
        
        # ========== LLMAspect（可选，用于语义增强）==========
        # 如果提供了 llm_client，可以添加语义相关的 Object Layer 和 LLMAspect
        self.llm_aspect = None
        self.semantic_objects: Dict[str, ObjectNode] = {}
        
        if llm_client is not None:
            # 添加语义相关的 Object
            sem_dim = config.get("sem_dim", 128)
            self.semantic_objects = {
                "semantic_context": ObjectNode("semantic_context", dim=sem_dim, device=self.device),
                "semantic_prediction": ObjectNode("semantic_prediction", dim=sem_dim, device=self.device),
            }
            
            # 创建 LLMAspect
            from aonn.aspects.llm_aspect import LLMAspect
            self.llm_aspect = LLMAspect(
                src_names=("semantic_context",),
                dst_names=("semantic_prediction",),
                llm_client=llm_client,
                llm_config=config.get("llm", {}),
            )
            
            # 将语义 Object 添加到 all_objects
            all_objects.update(self.semantic_objects)
        
        # ========== 传统 Aspect（用于自由能框架）==========
        # 保留传统 Aspect 用于自由能计算和主动推理
        self.traditional_aspects: List[AspectBase] = []
        if self.llm_aspect is not None:
            self.traditional_aspects.append(self.llm_aspect)
        
        # ========== 网络拓扑（用于可视化）==========
        # 构建拓扑（简化版，主要用于可视化）
        self.topology = NetworkTopology()
        for name, obj in all_objects.items():
            self.topology.add_object(name, obj)
        
        # 添加传统 Aspect 到拓扑（如果存在）
        for aspect in self.traditional_aspects:
            self.topology.add_aspect(aspect)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：Object → Aspect → Aspect → ... → Object
        
        Args:
            x: [batch, input_dim] 输入
        
        Returns:
            [batch, output_dim] 输出
        """
        # 设置输入层状态
        self.input_layer.set_state_vector(x)
        
        # 通过所有 Object Layer 和 Aspect Pipeline
        current_state = self.input_layer.get_state_vector()
        
        for i, pipeline in enumerate(self.aspect_pipelines):
            # 通过 Aspect Pipeline
            current_state = pipeline(current_state)
            
            # 设置下一层 Object Layer 的状态
            # 注意：pipeline 的输出维度应该匹配下一层的输入维度
            if i < len(self.hidden_layers):
                # 设置隐藏层状态
                self.hidden_layers[i].set_state_vector(current_state)
                # 更新当前状态为隐藏层状态（用于下一个 pipeline）
                current_state = self.hidden_layers[i].get_state_vector()
            else:
                # 设置输出层状态
                self.output_layer.set_state_vector(current_state)
        
        # 返回输出层状态
        return self.output_layer.get_state_vector()
    
    def get_network_structure(self) -> Dict:
        """
        获取网络结构信息
        """
        return {
            "num_object_layers": len(self.object_layers),
            "num_aspect_pipelines": len(self.aspect_pipelines),
            "object_layers": [
                {
                    "name": layer.name,
                    "num_objects": len(layer),
                    "total_dim": layer.get_total_dim(),
                }
                for layer in self.object_layers
            ],
            "aspect_pipelines": [
                {
                    "depth": pipeline.depth,
                    "num_aspects": pipeline.num_aspects,
                    "input_dim": pipeline.input_dim,
                    "output_dim": pipeline.output_dim,
                }
                for pipeline in self.aspect_pipelines
            ],
            "has_llm_aspect": self.has_llm_aspect(),
            "num_semantic_objects": len(self.semantic_objects),
        }
    
    def visualize_network(self) -> str:
        """
        可视化网络结构
        """
        lines = []
        lines.append("=" * 70)
        lines.append("AONN Brain V2 网络结构")
        lines.append("=" * 70)
        lines.append("\n架构：Object Layer → Aspect Pipeline → Object Layer → ...")
        lines.append("\nObject Layers（状态层）：")
        for layer in self.object_layers:
            lines.append(f"  - {layer.name}: {len(layer)} objects, dim={layer.get_total_dim()}")
        
        lines.append("\nAspect Pipelines（计算层，纯神经元）：")
        for i, pipeline in enumerate(self.aspect_pipelines):
            lines.append(
                f"  - Pipeline {i+1}: depth={pipeline.depth}, "
                f"aspects={pipeline.num_aspects}, "
                f"{pipeline.input_dim}→{pipeline.output_dim}"
            )
        
        lines.append("\n网络流程：")
        for i, layer in enumerate(self.object_layers):
            lines.append(f"  {layer.name}")
            if i < len(self.aspect_pipelines):
                lines.append(f"    ↓ [Aspect Pipeline {i+1}]")
        
        # 显示 LLMAspect（如果存在）
        if self.llm_aspect is not None:
            lines.append("\n语义增强（LLMAspect）：")
            lines.append(f"  - semantic_context → [LLMAspect] → semantic_prediction")
            lines.append(f"  - LLM 客户端: {'已配置' if self.llm_aspect.llm_client is not None else '未配置'}")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def get_all_objects(self) -> Dict[str, ObjectNode]:
        """获取所有 Object（包括语义 Object）"""
        all_objects = {}
        for layer in self.object_layers:
            all_objects.update(layer.objects)
        # 添加语义 Object（如果存在）
        all_objects.update(self.semantic_objects)
        return all_objects
    
    def has_llm_aspect(self) -> bool:
        """检查是否有 LLMAspect"""
        return self.llm_aspect is not None
    
    def compute_free_energy(self) -> torch.Tensor:
        """
        计算自由能（如果存在传统 Aspect）
        """
        if len(self.traditional_aspects) == 0:
            return torch.tensor(0.0, device=self.device)
        
        from aonn.core.free_energy import compute_total_free_energy
        all_objects = self.get_all_objects()
        return compute_total_free_energy(all_objects, self.traditional_aspects)

