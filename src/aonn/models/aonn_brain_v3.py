# src/aonn/models/aonn_brain_v3.py
"""
AONN Brain V3：支持动态演化的自由能大脑

核心特性：
1. 从最小初始网络开始
2. 根据自由能动态演化
3. 支持世界模型交互
4. 自我模型观察和可视化
"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from aonn.core.object import ObjectNode
from aonn.core.object_layer import ObjectLayer
from aonn.core.aspect_layer import AspectLayer, AspectPipeline
from aonn.core.aspect_base import AspectBase
from aonn.core.free_energy import compute_total_free_energy
from aonn.core.evolution import NetworkEvolution, EvolutionEvent
from aonn.aspects.sensory_aspect import LinearGenerativeAspect
from aonn.aspects.llm_aspect import LLMAspect
from aonn.aspects.world_model_aspects import WorldModelAspectSet


class AONNBrainV3(nn.Module):
    """
    AONN Brain V3：动态演化的自由能大脑
    
    初始架构（最小）：
    - Object: sensory, internal
    - Aspect: 无（或一个基础 identity aspect）
    
    演化过程：
    - 根据自由能动态创建 Object、Aspect、Pipeline
    - 剪枝不重要的连接
    - 记录演化历史
    """
    
    def __init__(
        self,
        config: Dict,
        llm_client=None,
        device=None,
        enable_evolution: bool = True,
    ):
        """
        Args:
            config: 配置字典
            llm_client: LLM 客户端（可选）
            device: 设备
            enable_evolution: 是否启用动态演化
        """
        super().__init__()
        self.device = device or torch.device("cpu")
        self.config = config
        self.enable_evolution = enable_evolution
        
        # ========== 最小初始架构 ==========
        # 只有基础的 Object
        obs_dim = config.get("obs_dim", 16)
        state_dim = config.get("state_dim", 32)
        act_dim = config.get("act_dim", 8)
        
        self.objects: Dict[str, ObjectNode] = {
            "sensory": ObjectNode("sensory", dim=obs_dim, device=self.device),
            "internal": ObjectNode("internal", dim=state_dim, device=self.device),
        }
        
        # 初始 Aspect（最小）
        self.aspects: List[AspectBase] = []
        
        # 初始 Pipeline（无）
        self.aspect_pipelines: List[AspectPipeline] = []
        self.aspect_modules = nn.ModuleList()
        
        # ========== 演化管理器 ==========
        if enable_evolution:
            self.evolution = NetworkEvolution(
                free_energy_threshold=config.get("evolution", {}).get("free_energy_threshold", 1.0),
                prune_threshold=config.get("evolution", {}).get("prune_threshold", 0.01),
                max_objects=config.get("evolution", {}).get("max_objects", 100),
                max_aspects=config.get("evolution", {}).get("max_aspects", 1000),
            )
        else:
            self.evolution = None
        
        # ========== 世界模型 Aspect（可学习的生成模型）==========
        self.world_model_aspects: Optional[WorldModelAspectSet] = None
        self.enable_world_model_learning = config.get("enable_world_model_learning", True)
        
        if self.enable_world_model_learning:
            # 创建 action Object（dynamics aspect 需要）
            if "action" not in self.objects:
                self.objects["action"] = ObjectNode("action", dim=act_dim, device=self.device)
            
            # 创建 target Object（用于 preference）
            if "target" not in self.objects:
                self.objects["target"] = ObjectNode("target", dim=state_dim, device=self.device)
            
            self.world_model_aspects = WorldModelAspectSet(
                state_dim=state_dim,
                action_dim=act_dim,
                obs_dim=obs_dim,
                device=self.device
            )
            # 将世界模型 Aspect 添加到 aspects 列表
            self.aspects.extend(self.world_model_aspects.aspects)
            # 注册到 ModuleList
            for aspect in self.world_model_aspects.aspects:
                if isinstance(aspect, nn.Module):
                    self.aspect_modules.append(aspect)
        
        # ========== 可选组件 ==========
        self.llm_aspect = None
        self.semantic_objects: Dict[str, ObjectNode] = {}
        
        if llm_client is not None:
            sem_dim = config.get("sem_dim", 128)
            self.semantic_objects = {
                "semantic_context": ObjectNode("semantic_context", dim=sem_dim, device=self.device),
                "semantic_prediction": ObjectNode("semantic_prediction", dim=sem_dim, device=self.device),
            }
            self.objects.update(self.semantic_objects)
            
            self.llm_aspect = LLMAspect(
                src_names=("semantic_context",),
                dst_names=("semantic_prediction",),
                llm_client=llm_client,
                llm_config=config.get("llm", {}),
            )
            self.aspects.append(self.llm_aspect)
        
        # ========== 网络拓扑 ==========
        self.topology = None  # 将在需要时构建
        
        # ========== 自我模型观察 ==========
        self.self_model_history: List[Dict] = []
    
    def create_object(
        self,
        name: str,
        dim: int,
        init_state: Optional[torch.Tensor] = None
    ) -> ObjectNode:
        """
        动态创建新 Object
        """
        if name in self.objects:
            return self.objects[name]
        
        obj = ObjectNode(name, dim=dim, device=self.device)
        if init_state is not None:
            obj.set_state(init_state)
        
        self.objects[name] = obj
        
        if self.evolution:
            F_before = self.compute_free_energy().item()
            F_after = self.compute_free_energy().item()
            self.evolution.record_event(
                "create_object",
                {"name": name, "dim": dim},
                "dynamic_creation",
                F_before,
                F_after
            )
        
        return obj
    
    def create_aspect(
        self,
        aspect_type: str,
        src_names: List[str],
        dst_names: List[str],
        **kwargs
    ) -> AspectBase:
        """
        动态创建新 Aspect
        """
        if aspect_type == "sensory":
            aspect = LinearGenerativeAspect(
                internal_name=src_names[0],
                sensory_name=dst_names[0],
                state_dim=self.objects[src_names[0]].dim,
                obs_dim=self.objects[dst_names[0]].dim,
            )
        elif aspect_type == "llm":
            aspect = LLMAspect(
                src_names=src_names,
                dst_names=dst_names,
                llm_client=kwargs.get("llm_client"),
                llm_config=kwargs.get("llm_config", {}),
            )
        else:
            raise ValueError(f"Unknown aspect type: {aspect_type}")
        
        self.aspects.append(aspect)
        
        # 如果是 nn.Module，注册到 ModuleList
        if isinstance(aspect, nn.Module):
            if not hasattr(self, 'aspect_modules') or self.aspect_modules is None:
                self.aspect_modules = nn.ModuleList()
            self.aspect_modules.append(aspect)
        
        if self.evolution:
            F_before = self.compute_free_energy().item()
            F_after = self.compute_free_energy().item()
            self.evolution.record_event(
                "create_aspect",
                {"type": aspect_type, "src": src_names, "dst": dst_names},
                "dynamic_creation",
                F_before,
                F_after
            )
        
        return aspect
    
    def create_pipeline(
        self,
        input_layer_name: str,
        output_layer_name: str,
        num_aspects: int = 32,
        depth: int = 4,
    ) -> AspectPipeline:
        """
        动态创建新 Pipeline
        """
        input_dim = self.objects[input_layer_name].dim
        output_dim = self.objects[output_layer_name].dim
        
        pipeline = AspectPipeline(
            input_dim=input_dim,
            output_dim=output_dim,
            num_aspects=num_aspects,
            depth=depth,
            use_gate=False,
        )
        
        self.aspect_pipelines.append(pipeline)
        if not hasattr(self, 'aspect_modules') or self.aspect_modules is None:
            self.aspect_modules = nn.ModuleList()
        self.aspect_modules.append(pipeline)
        
        if self.evolution:
            F_before = self.compute_free_energy().item()
            F_after = self.compute_free_energy().item()
            self.evolution.record_event(
                "create_pipeline",
                {"input": input_layer_name, "output": output_layer_name, "depth": depth},
                "dynamic_creation",
                F_before,
                F_after
            )
        
        return pipeline
    
    def evolve_network(self, observation: torch.Tensor, target: Optional[torch.Tensor] = None):
        """
        网络演化：根据自由能决定是否创建新组件
        """
        if not self.enable_evolution or self.evolution is None:
            return
        
        self.evolution.increment_step()
        
        # 设置当前观察
        self.objects["sensory"].set_state(observation)
        if target is not None:
            if "semantic_prediction" in self.objects:
                self.objects["semantic_prediction"].set_state(target)
        
        # 计算当前自由能
        F_current = self.compute_free_energy().item()
        
        # 检查是否需要创建新 Aspect（sensory → internal）
        if len([a for a in self.aspects if isinstance(a, LinearGenerativeAspect)]) == 0:
            # 创建基础 sensory aspect
            self.create_aspect(
                "sensory",
                src_names=["internal"],
                dst_names=["sensory"]
            )
        
        # 检查是否需要创建新 Object（如 action）
        if "action" not in self.objects and F_current > self.evolution.free_energy_threshold:
            act_dim = self.config.get("act_dim", 8)
            self.create_object("action", dim=act_dim)
        
        # 检查是否需要创建 Pipeline
        if len(self.aspect_pipelines) == 0 and "action" in self.objects:
            # 创建 internal → action 的 pipeline
            self.create_pipeline(
                "internal",
                "action",
                num_aspects=16,
                depth=2
            )
    
    def compute_free_energy(self) -> torch.Tensor:
        """计算总自由能"""
        return compute_total_free_energy(self.objects, self.aspects)
    
    def learn_world_model(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        target_state: Optional[torch.Tensor] = None,
        learning_rate: float = 0.001,
    ):
        """
        学习世界模型参数（通过自由能最小化）
        
        Args:
            observation: 当前观察
            action: 执行的动作
            next_observation: 下一观察
            target_state: 目标状态（用于 preference）
            learning_rate: 学习率
        """
        if not self.enable_world_model_learning or self.world_model_aspects is None:
            return
        
        # 创建优化器（如果还没有）
        if not hasattr(self, '_world_model_optimizer'):
            self._world_model_optimizer = torch.optim.Adam(
                self.world_model_aspects.get_all_parameters(),
                lr=learning_rate
            )
        
        # 设置 Object 状态（使用 detach 避免梯度图问题）
        self.objects["sensory"].set_state(observation.detach())
        if "action" in self.objects:
            self.objects["action"].set_state(action.detach())
        
        # 获取当前 internal 状态（detach）
        current_internal = self.objects["internal"].state.detach()
        
        # 预测下一状态（使用 dynamics）
        pred_next_internal = self.world_model_aspects.dynamics.predict_next_state(
            current_internal, action.detach()
        )
        
        # 设置下一观察（用于计算 observation aspect 的自由能）
        if "sensory_next" not in self.objects:
            self.objects["sensory_next"] = ObjectNode("sensory_next", dim=observation.shape[-1], device=self.device)
        self.objects["sensory_next"].set_state(next_observation.detach())
        
        # 临时修改 observation aspect 的 dst 为 sensory_next
        original_dst = self.world_model_aspects.observation.dst_names[0]
        self.world_model_aspects.observation.dst_names[0] = "sensory_next"
        
        # 设置目标状态（用于 preference）
        if target_state is not None and "target" in self.objects:
            self.objects["target"].set_state(target_state.detach())
            self.world_model_aspects.set_target_state(target_state.detach())
        
        # 计算自由能并更新参数
        self._world_model_optimizer.zero_grad()
        
        # 重新设置 internal 为可微的（用于计算梯度）
        self.objects["internal"].set_state(pred_next_internal)
        self.objects["sensory"].set_state(observation)  # 也需要可微
        
        # 只计算世界模型 Aspect 的自由能
        F = torch.tensor(0.0, device=self.device)
        for aspect in self.world_model_aspects.aspects:
            F = F + aspect.free_energy_contrib(self.objects)
        
        F.backward()
        self._world_model_optimizer.step()
        
        # 恢复 observation aspect 的 dst
        self.world_model_aspects.observation.dst_names[0] = original_dst
        
        # 恢复 internal 为 detach 状态
        self.objects["internal"].set_state(pred_next_internal.detach())
    
    def set_world_model_target(self, target_state: torch.Tensor):
        """设置世界模型的目标状态（用于 preference）"""
        if self.world_model_aspects is not None:
            self.world_model_aspects.set_target_state(target_state)
            if "target" in self.objects:
                self.objects["target"].set_state(target_state)
    
    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        """
        if x is not None:
            self.objects["sensory"].set_state(x)
        
        # 如果有 Pipeline，使用 Pipeline
        if len(self.aspect_pipelines) > 0:
            current_state = self.objects["internal"].state
            for pipeline in self.aspect_pipelines:
                current_state = pipeline(current_state)
            return current_state
        
        # 否则使用传统 Aspect
        # 这里简化处理，实际应该通过自由能最小化
        return self.objects["internal"].state
    
    def get_network_structure(self) -> Dict:
        """获取网络结构信息"""
        return {
            "num_objects": len(self.objects),
            "num_aspects": len(self.aspects),
            "num_pipelines": len(self.aspect_pipelines),
            "objects": {
                name: {"dim": obj.dim, "state_norm": torch.norm(obj.state).item()}
                for name, obj in self.objects.items()
            },
            "aspects": [
                {
                    "name": a.name,
                    "src": a.src_names,
                    "dst": a.dst_names,
                    "type": type(a).__name__
                }
                for a in self.aspects
            ],
            "pipelines": [
                {
                    "depth": p.depth,
                    "num_aspects": p.num_aspects,
                    "input_dim": p.input_dim,
                    "output_dim": p.output_dim,
                }
                for p in self.aspect_pipelines
            ],
        }
    
    def observe_self_model(self) -> Dict:
        """
        观察自我模型：记录当前网络结构
        """
        snapshot = {
            "step": self.evolution.step_count if self.evolution else 0,
            "free_energy": self.compute_free_energy().item(),
            "structure": self.get_network_structure(),
            "evolution_stats": self.evolution.get_evolution_summary() if self.evolution else None,
        }
        self.self_model_history.append(snapshot)
        return snapshot
    
    def get_evolution_history(self) -> List[EvolutionEvent]:
        """获取演化历史"""
        if self.evolution:
            return self.evolution.evolution_history
        return []
    
    def visualize_network(self) -> str:
        """可视化网络结构"""
        lines = []
        lines.append("=" * 70)
        lines.append("AONN Brain V3 网络结构（动态演化）")
        lines.append("=" * 70)
        
        structure = self.get_network_structure()
        lines.append(f"\nObject 节点 ({structure['num_objects']} 个):")
        for name, info in structure["objects"].items():
            lines.append(f"  - {name}: dim={info['dim']}, ||state||={info['state_norm']:.4f}")
        
        lines.append(f"\nAspect ({structure['num_aspects']} 个):")
        for aspect_info in structure["aspects"]:
            lines.append(f"  - {aspect_info['name']}: {aspect_info['src']} → {aspect_info['dst']} ({aspect_info['type']})")
        
        lines.append(f"\nPipeline ({structure['num_pipelines']} 个):")
        for i, pipe_info in enumerate(structure["pipelines"]):
            lines.append(f"  - Pipeline {i+1}: {pipe_info['input_dim']}→{pipe_info['output_dim']}, "
                        f"depth={pipe_info['depth']}, aspects={pipe_info['num_aspects']}")
        
        if self.evolution:
            stats = self.evolution.get_evolution_summary()
            lines.append(f"\n演化统计:")
            lines.append(f"  - 总步数: {stats['total_steps']}")
            lines.append(f"  - 总事件: {stats['total_events']}")
            lines.append(f"  - Objects 创建: {stats['stats']['objects_created']}")
            lines.append(f"  - Aspects 创建: {stats['stats']['aspects_created']}")
            lines.append(f"  - Pipelines 创建: {stats['stats']['pipelines_created']}")
            lines.append(f"  - 剪枝次数: {stats['stats']['pruned_count']}")
        
        lines.append("=" * 70)
        return "\n".join(lines)

