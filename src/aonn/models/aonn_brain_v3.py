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
from aonn.aspects.pipeline_aspect import PipelineAspect
from aonn.aspects.classification_aspect import ClassificationAspect


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

        # 感官维度配置
        vision_dim = config.get("vision_dim", max(obs_dim * 3 // 4, 8))
        olfactory_dim = config.get("olfactory_dim", max(obs_dim // 5, 4))
        proprio_dim = config.get("proprio_dim", max(obs_dim - vision_dim - olfactory_dim, 4))
        default_sense_dims = {
            "vision": vision_dim,
            "olfactory": olfactory_dim,
            "proprio": proprio_dim,
        }
        self.sense_dims = config.get("sense_dims", default_sense_dims)
        # 确保所有维度为正
        for sense, dim in self.sense_dims.items():
            if dim <= 0:
                raise ValueError(f"感官 {sense} 的维度必须为正，当前: {dim}")
        self.senses = list(self.sense_dims.keys())
        self._sensory_aspect_counters = {sense: 0 for sense in self.senses}
        
        evo_cfg = config.get("evolution", {})
        batch_cfg = evo_cfg.get("batch_growth", {})
        self.batch_growth_cfg = {
            "base": batch_cfg.get("base", 8),
            "max_per_step": batch_cfg.get("max_per_step", 64),
            "max_total": batch_cfg.get("max_total", 256),
            "error_multiplier": batch_cfg.get("error_multiplier", 1.0),
            "min_per_sense": batch_cfg.get("min_per_sense", 1),
            "error_threshold": batch_cfg.get("error_threshold", evo_cfg.get("free_energy_threshold", 1.0)),
        }
        self.error_ema_alpha = batch_cfg.get("error_ema_alpha", evo_cfg.get("error_ema_alpha", 0.3))
        self.sensory_error_ema = {sense: 0.0 for sense in self.senses}
        self.state_clip_value = float(config.get("state_clip_value", 10.0))
        
        self.objects: Dict[str, ObjectNode] = {
            "internal": ObjectNode("internal", dim=state_dim, device=self.device),
        }
        for sense_name, dim in self.sense_dims.items():
            self.objects[sense_name] = ObjectNode(sense_name, dim=dim, device=self.device)
        
        # 初始 Aspect（最小）
        self.aspects: List[AspectBase] = []
        
        # 初始 Pipeline（无）
        self.aspect_pipelines: List[AspectPipeline] = []
        self.pipeline_specs: List[Dict] = []
        self.aspect_modules = nn.ModuleList()
        pipeline_cfg = config.get("pipeline_growth", {})
        self.pipeline_growth_cfg = {
            "enable": pipeline_cfg.get("enable", True),
            "initial_depth": pipeline_cfg.get("initial_depth", 2),
            "initial_width": pipeline_cfg.get("initial_width", 16),
            "depth_increment": pipeline_cfg.get("depth_increment", 1),
            "width_increment": pipeline_cfg.get("width_increment", 0),
            "max_stages": max(1, pipeline_cfg.get("max_stages", 3)),
            "min_interval": pipeline_cfg.get("min_interval", 80),
            "free_energy_trigger": pipeline_cfg.get("free_energy_trigger"),
            "max_depth": pipeline_cfg.get("max_depth", 8),
        }
        self.pipeline_growth_state = {"last_expand_step": 0}
        
        # ========== 演化管理器 ==========
        if enable_evolution:
            self.evolution = NetworkEvolution(
                free_energy_threshold=evo_cfg.get("free_energy_threshold", 1.0),
                prune_threshold=evo_cfg.get("prune_threshold", 0.01),
                max_objects=evo_cfg.get("max_objects", 100),
                max_aspects=evo_cfg.get("max_aspects", 1000),
                batch_growth=batch_cfg,
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
                observation_dims=self.sense_dims,
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
            
            llm_config = config.get("llm", {})
            self.llm_aspect = LLMAspect(
                src_names=("semantic_context",),
                dst_names=("semantic_prediction",),
                llm_client=llm_client,
                llm_config=llm_config,
                call_frequency=llm_config.get("call_frequency", "last_iter_only"),  # 默认只在最后一次迭代调用
                call_every_n_steps=llm_config.get("call_every_n_steps", 1),
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
        name: Optional[str] = None,
        **kwargs
    ) -> AspectBase:
        """
        动态创建新 Aspect
        
        Args:
            aspect_type: Aspect 类型
            src_names: 源 Object 名称列表
            dst_names: 目标 Object 名称列表
            name: Aspect 名称
            **kwargs: 额外参数，包括：
                - obs_dim: 观察维度
                - init_weight: 初始权重（可选）
                - reference_aspect: 参考 Aspect（可选）
                - init_scale: 初始化缩放因子（默认 0.1）
                - noise_scale: 从参考复制时的噪声缩放（默认 0.01）
        """
        if aspect_type == "sensory":
            aspect = LinearGenerativeAspect(
                internal_name=src_names[0],
                sensory_name=dst_names[0],
                state_dim=self.objects[src_names[0]].dim,
                obs_dim=kwargs.get("obs_dim", self.objects[dst_names[0]].dim),
                name=name,
                init_weight=kwargs.get("init_weight"),
                reference_aspect=kwargs.get("reference_aspect"),
                init_scale=kwargs.get("init_scale", 0.1),
                noise_scale=kwargs.get("noise_scale", 0.01),
            )
        elif aspect_type == "llm":
            llm_config = kwargs.get("llm_config", {})
            aspect = LLMAspect(
                name=name or "llm_aspect",
                src_names=src_names,
                dst_names=dst_names,
                llm_client=kwargs.get("llm_client"),
                llm_config=llm_config,
                call_frequency=llm_config.get("call_frequency", "last_iter_only"),
                call_every_n_steps=llm_config.get("call_every_n_steps", 1),
            )
        elif aspect_type == "pipeline":
            # 创建 Pipeline Aspect（向量化）
            aspect = PipelineAspect(
                src_names=src_names,
                dst_names=dst_names,
                input_dim=kwargs.get("input_dim"),
                output_dim=kwargs.get("output_dim"),
                num_aspects=kwargs.get("num_aspects", 32),
                depth=kwargs.get("depth", 2),
                name=name or "pipeline_aspect",
                use_gate=kwargs.get("use_gate", False),
            )
        elif aspect_type == "classification":
            # 创建分类 Aspect
            aspect = ClassificationAspect(
                internal_name=src_names[0],
                target_name=dst_names[0],
                state_dim=kwargs.get("state_dim", self.objects[src_names[0]].dim if src_names[0] in self.objects else 256),
                num_classes=kwargs.get("num_classes", self.objects[dst_names[0]].dim if dst_names[0] in self.objects else 10),
                hidden_dim=kwargs.get("hidden_dim", 128),
                name=name or "classification",
                loss_weight=kwargs.get("loss_weight", 1.0),
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
    
    def create_unified_aspect(
        self,
        aspect_type: str,  # "pipeline", "individual", "classification", etc.
        src_names: List[str],
        dst_names: List[str],
        name: Optional[str] = None,
        **kwargs
    ) -> AspectBase:
        """
        统一创建 Aspect，自动选择最优实现方式
        
        Args:
            aspect_type: Aspect 类型
                - "pipeline": 向量化 Pipeline（高效批量处理）
                - "individual": 独立 Aspect（灵活、可解释）
                - "classification": 分类 Aspect
                - "sensory": 感官 Aspect
                - "llm": LLM Aspect
            src_names: 源 Object 名称列表
            dst_names: 目标 Object 名称列表
            name: Aspect 名称
            **kwargs: 额外参数
                - use_pipeline: 是否强制使用 Pipeline
                - input_dim, output_dim, num_aspects, depth: Pipeline 参数
                - state_dim, num_classes, hidden_dim: 分类器参数
        
        Returns:
            AspectBase 实例
        """
        # 自动选择：如果指定 use_pipeline 或 aspect_type == "pipeline"
        use_pipeline = kwargs.get("use_pipeline", False) or aspect_type == "pipeline"
        
        if use_pipeline:
            # 使用向量化 Pipeline（高效批量处理）
            # 计算输入维度（所有源 Object 的总维度）
            input_dim = kwargs.get("input_dim")
            if input_dim is None:
                input_dim = sum(self.objects[name].dim for name in src_names if name in self.objects)
            
            # 计算输出维度（目标 Object 的维度）
            output_dim = kwargs.get("output_dim")
            if output_dim is None:
                output_dim = self.objects[dst_names[0]].dim if dst_names[0] in self.objects else input_dim
            
            return self.create_aspect(
                aspect_type="pipeline",
                src_names=src_names,
                dst_names=dst_names,
                name=name,
                input_dim=input_dim,
                output_dim=output_dim,
                num_aspects=kwargs.get("num_aspects", 32),
                depth=kwargs.get("depth", 2),
                use_gate=kwargs.get("use_gate", False),
            )
        else:
            # 使用独立 Aspect（灵活、可解释）
            individual_type = kwargs.get("individual_type", aspect_type)
            return self.create_aspect(
                aspect_type=individual_type,
                src_names=src_names,
                dst_names=dst_names,
                name=name,
                **{k: v for k, v in kwargs.items() 
                   if k not in ["use_pipeline", "input_dim", "output_dim", 
                               "num_aspects", "depth", "individual_type"]}
            )
    
    def update_semantic_context(self, source: Optional[str] = None, world_semantic_state: Optional[torch.Tensor] = None):
        """
        更新 semantic_context Object 的状态
        
        优先级：
        1. 如果提供了 world_semantic_state，直接使用（从世界模型同步）
        2. 如果 source 指定了 Object 名称，从该 Object 提取语义部分
        3. 默认从 internal Object 提取前 sem_dim 维作为语义状态
        
        Args:
            source: 源 Object 名称（如 "internal"），默认为 "internal"
            world_semantic_state: 世界模型的语义状态向量（可选）
        """
        if "semantic_context" not in self.objects:
            return
        
        sem_dim = self.objects["semantic_context"].dim
        
        if world_semantic_state is not None:
            # 方案 1: 从世界模型的语义状态同步
            # 检查 world_semantic_state 是否全为零或包含 NaN/Inf
            if torch.all(world_semantic_state == 0) or not torch.isfinite(world_semantic_state).all():
                # 如果全为零或包含 NaN/Inf，则从 internal 提取
                if "internal" in self.objects:
                    internal_state = self.objects["internal"].state
                    if internal_state.shape[-1] >= sem_dim:
                        semantic_vec = internal_state[:sem_dim].detach().clone()
                    else:
                        padding = torch.zeros(sem_dim - internal_state.shape[-1], device=internal_state.device)
                        semantic_vec = torch.cat([internal_state.detach().clone(), padding], dim=-1)
                    self.objects["semantic_context"].set_state(semantic_vec)
                return
            if world_semantic_state.shape[-1] >= sem_dim:
                semantic_vec = world_semantic_state[:sem_dim].detach().clone()
            else:
                # 如果维度不够，用零填充
                padding = torch.zeros(sem_dim - world_semantic_state.shape[-1], device=world_semantic_state.device)
                semantic_vec = torch.cat([world_semantic_state.detach().clone(), padding], dim=-1)
            # 清理 NaN/Inf
            semantic_vec = torch.nan_to_num(semantic_vec, nan=0.0, posinf=1.0, neginf=-1.0)
            self.objects["semantic_context"].set_state(semantic_vec)
        elif source is None or source == "internal":
            # 方案 2: 从 internal Object 提取语义部分
            if "internal" in self.objects:
                internal_state = self.objects["internal"].state
                if internal_state.shape[-1] >= sem_dim:
                    # 提取前 sem_dim 维
                    semantic_vec = internal_state[:sem_dim].detach().clone()
                else:
                    # 如果维度不够，用零填充
                    padding = torch.zeros(sem_dim - internal_state.shape[-1], device=internal_state.device)
                    semantic_vec = torch.cat([internal_state.detach().clone(), padding], dim=-1)
                self.objects["semantic_context"].set_state(semantic_vec)
        elif source in self.objects:
            # 方案 3: 从指定的 Object 提取
            source_state = self.objects[source].state
            if source_state.shape[-1] >= sem_dim:
                semantic_vec = source_state[:sem_dim].detach().clone()
            else:
                padding = torch.zeros(sem_dim - source_state.shape[-1], device=source_state.device)
                semantic_vec = torch.cat([source_state.detach().clone(), padding], dim=-1)
            self.objects["semantic_context"].set_state(semantic_vec)
    
    def create_sensory_aspect_batch(
        self,
        sense_name: str,
        count: int,
        use_weight_copy: bool = True,
        noise_scale: float = 0.01,
    ) -> List[AspectBase]:
        """
        批量创建感官 Aspect，用于快速扩展"神经元"数量
        
        Args:
            sense_name: 感官名称
            count: 要创建的数量
            use_weight_copy: 是否从已有 Aspects 复制权重（默认 True）
            noise_scale: 复制权重时的噪声缩放因子（默认 0.01）
        
        策略：
        1. 如果已有同类型的 Aspects，从其中随机选择一个复制权重
        2. 如果没有，使用随机初始化
        3. 复制时添加小噪声，保持多样性
        """
        created: List[AspectBase] = []
        if count <= 0:
            return created
        
        # 查找已有的同类型 Aspects（相同 sensory_name）
        existing_aspects = [
            asp for asp in self.aspects
            if isinstance(asp, LinearGenerativeAspect) and asp.dst_names[0] == sense_name
        ]
        
        # 选择参考 Aspect（如果有的话）
        reference_aspect = None
        if use_weight_copy and len(existing_aspects) > 0:
            # 选择自由能贡献最小的 Aspect 作为参考（通常学习得最好）
            ref_contribs = [
                (asp, asp.free_energy_contrib(self.objects).item())
                for asp in existing_aspects
            ]
            ref_contribs.sort(key=lambda x: x[1])  # 按自由能贡献排序
            reference_aspect = ref_contribs[0][0]  # 选择贡献最小的
        
        for _ in range(count):
            self._sensory_aspect_counters[sense_name] += 1
            asp_name = f"sensory_{sense_name}_{self._sensory_aspect_counters[sense_name]}"
            created.append(
                self.create_aspect(
                    "sensory",
                    src_names=["internal"],
                    dst_names=[sense_name],
                    name=asp_name,
                    obs_dim=self.sense_dims[sense_name],
                    reference_aspect=reference_aspect,
                    noise_scale=noise_scale,
                )
            )
        return created
    
    def create_pipeline(
        self,
        input_layer_name: str,
        output_layer_name: str,
        num_aspects: int = 32,
        depth: int = 4,
        position: Optional[int] = None,
        metadata: Optional[Dict] = None,
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
        
        if position is None or position >= len(self.aspect_pipelines):
            self.aspect_pipelines.append(pipeline)
            self.pipeline_specs.append({
                "input": input_layer_name,
                "output": output_layer_name,
                "metadata": metadata or {},
            })
        else:
            self.aspect_pipelines.insert(position, pipeline)
            self.pipeline_specs.insert(position, {
                "input": input_layer_name,
                "output": output_layer_name,
                "metadata": metadata or {},
            })
        if not hasattr(self, 'aspect_modules') or self.aspect_modules is None:
            self.aspect_modules = nn.ModuleList()
        self.aspect_modules.append(pipeline)
        
        if self.evolution:
            F_before = self.compute_free_energy().item()
            F_after = self.compute_free_energy().item()
            self.evolution.record_event(
                "create_pipeline",
                {
                    "input": input_layer_name,
                    "output": output_layer_name,
                    "depth": depth,
                    "num_aspects": num_aspects,
                    "position": position if position is not None else len(self.aspect_pipelines) - 1,
                    "metadata": metadata or {},
                },
                "dynamic_creation",
                F_before,
                F_after
            )
        
        return pipeline
    
    def ensure_action_pipeline(self):
        """
        确保至少存在一条 internal→action 的 Pipeline 作为最终动作头
        """
        if "action" not in self.objects:
            return
        if len(self.aspect_pipelines) == 0:
            self.create_pipeline(
                "internal",
                "action",
                num_aspects=self.pipeline_growth_cfg.get("initial_width", 16),
                depth=self.pipeline_growth_cfg.get("initial_depth", 2),
                metadata={"stage": "action"}
            )
    
    def maybe_expand_action_pipeline(self, free_energy_value: float):
        """
        根据自由能和演化步数，动态插入新的 internal→internal Pipeline，
        增加动作头之前的深度
        """
        if not self.pipeline_growth_cfg.get("enable", True):
            return
        if len(self.aspect_pipelines) == 0 or self.evolution is None:
            return
        if len(self.aspect_pipelines) >= self.pipeline_growth_cfg.get("max_stages", 3):
            return
        # 检测 NaN/Inf，如果是则不扩展
        import math
        if not math.isfinite(free_energy_value):
            return
        step = self.evolution.step_count
        last_expand = self.pipeline_growth_state.get("last_expand_step", 0)
        if step - last_expand < self.pipeline_growth_cfg.get("min_interval", 80):
            return
        trigger = self.pipeline_growth_cfg.get("free_energy_trigger")
        # 如果设置了自由能触发阈值，只有当自由能高于阈值时才扩展
        if trigger is not None and free_energy_value <= trigger:
            return
        stage_index = len(self.aspect_pipelines) - 1  # 预留最后一个动作头
        width = max(1, self.pipeline_growth_cfg.get("initial_width", 16) + stage_index * self.pipeline_growth_cfg.get("width_increment", 0))
        depth = min(
            self.pipeline_growth_cfg.get("initial_depth", 2) + stage_index * self.pipeline_growth_cfg.get("depth_increment", 1),
            self.pipeline_growth_cfg.get("max_depth", 8),
        )
        insert_pos = max(0, len(self.aspect_pipelines) - 1)
        self.create_pipeline(
            "internal",
            "internal",
            num_aspects=width,
            depth=depth,
            position=insert_pos,
            metadata={"stage": f"latent_{len(self.aspect_pipelines)}"}
        )
        self.pipeline_growth_state["last_expand_step"] = step
    
    def sanitize_states(self):
        """
        保证所有 Object state 有限且处于可控范围，避免 NaN/Inf 继续扩散
        """
        clip = self.state_clip_value
        for obj in self.objects.values():
            state = obj.state
            if state is None:
                continue
            # 总是清理：先处理 NaN/Inf，再裁剪范围
            cleaned = torch.nan_to_num(state, nan=0.0, posinf=clip or 10.0, neginf=-(clip or 10.0))
            if clip is not None and clip > 0:
                cleaned = torch.clamp(cleaned, -clip, clip)
            # 如果状态有变化，更新它
            if not torch.equal(state, cleaned):
                obj.set_state(cleaned.detach())
    
    
    def evolve_network(self, observation: Dict[str, torch.Tensor], target: Optional[torch.Tensor] = None):
        """
        网络演化：根据自由能决定是否创建新组件
        """
        if not self.enable_evolution or self.evolution is None:
            return
        
        self.evolution.increment_step()
        
        # 设置当前观察
        for sense, value in observation.items():
            if sense in self.objects:
                self.objects[sense].set_state(value)
        if target is not None and "target" in self.objects:
            self.objects["target"].set_state(target)
        
        # 更新 semantic_context（从 internal 状态提取语义部分）
        self.update_semantic_context(source="internal")
        
        # 计算当前自由能
        F_current = self.compute_free_energy().item()
        
        # 如果自由能是 NaN/Inf，清理状态并跳过演化
        import math
        if not math.isfinite(F_current):
            self.sanitize_states()
            return
        
        # 感官 Aspect 的自由能贡献统计
        sense_error_map = {sense: 0.0 for sense in self.senses}
        sense_aspect_count = {sense: 0 for sense in self.senses}
        for aspect in self.aspects:
            if not isinstance(aspect, LinearGenerativeAspect):
                continue
            sense_name = aspect.dst_names[0]
            sense_aspect_count[sense_name] = sense_aspect_count.get(sense_name, 0) + 1
            try:
                contrib = aspect.free_energy_contrib(self.objects)
                sense_error_map[sense_name] = sense_error_map.get(sense_name, 0.0) + float(contrib.detach().item())
            except Exception:
                continue
        
        # 更新 EMA
        alpha = max(0.0, min(1.0, self.error_ema_alpha))
        for sense in self.senses:
            prev = self.sensory_error_ema.get(sense, 0.0)
            current = sense_error_map.get(sense, 0.0)
            ema = current if alpha == 0.0 else (1 - alpha) * prev + alpha * current
            self.sensory_error_ema[sense] = ema
        
        # 确保每个感官至少拥有最小数量的 Aspect
        min_required = self.batch_growth_cfg.get("min_per_sense", 1)
        if min_required > 0:
            for sense in self.senses:
                current_count = sense_aspect_count.get(sense, 0)
                if current_count < min_required:
                    needed = min_required - current_count
                    self.create_sensory_aspect_batch(sense, needed)
                    sense_aspect_count[sense] = current_count + needed
        
        # 批量扩增高误差感官的 Aspect
        remaining_capacity = 0
        if self.evolution is not None:
            remaining_capacity = max(0, self.evolution.max_aspects - len(self.aspects))
        if remaining_capacity > 0:
            for sense in self.senses:
                if remaining_capacity <= 0:
                    break
                error_metric = self.sensory_error_ema[sense] * self.batch_growth_cfg.get("error_multiplier", 1.0)
                current_count = sense_aspect_count.get(sense, 0)
                plan = self.evolution.plan_aspect_batch(
                    error_value=error_metric,
                    current_target_count=current_count,
                    remaining_capacity=remaining_capacity,
                    growth_policy=self.batch_growth_cfg,
                )
                if plan > 0:
                    self.create_sensory_aspect_batch(sense, plan)
                    sense_aspect_count[sense] = current_count + plan
                    remaining_capacity = max(0, self.evolution.max_aspects - len(self.aspects))
        
        # 检查是否需要创建新 Object（如 action）
        if "action" not in self.objects and F_current > self.evolution.free_energy_threshold:
            act_dim = self.config.get("act_dim", 8)
            self.create_object("action", dim=act_dim)
        
        # 确保存在动作 Pipeline，并在自由能较低时增加深度
        self.ensure_action_pipeline()
        self.maybe_expand_action_pipeline(F_current)
        self.sanitize_states()
    
    def compute_free_energy(self) -> torch.Tensor:
        """计算总自由能"""
        F = compute_total_free_energy(self.objects, self.aspects)
        # 检测 NaN/Inf，如果出现则返回 0 并清理状态
        if not torch.isfinite(F):
            # 清理所有状态
            self.sanitize_states()
            # 返回一个小的非零值，避免完全停止演化
            return torch.tensor(1e-6, device=F.device, dtype=F.dtype)
        return F
    
    def learn_world_model(
        self,
        observation: Dict[str, torch.Tensor],
        action: torch.Tensor,
        next_observation: Dict[str, torch.Tensor],
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
            # 收集所有可学习参数：world_model_aspects + LinearGenerativeAspect
            all_params = list(self.world_model_aspects.get_all_parameters())
            # 添加 LinearGenerativeAspect 的参数（包括 touch Aspects）
            for aspect in self.aspects:
                if isinstance(aspect, LinearGenerativeAspect):
                    all_params.extend(aspect.parameters())
            self._world_model_optimizer = torch.optim.Adam(
                all_params,
                lr=learning_rate
            )
        else:
            # 如果优化器已存在，检查是否需要添加新的 LinearGenerativeAspect 参数
            optimizer_param_ids = set()
            for group in self._world_model_optimizer.param_groups:
                for p in group['params']:
                    optimizer_param_ids.add(id(p))
            
            # 检查是否有新的 LinearGenerativeAspect 参数未加入优化器
            new_params = []
            for aspect in self.aspects:
                if isinstance(aspect, LinearGenerativeAspect):
                    for p in aspect.parameters():
                        if id(p) not in optimizer_param_ids:
                            new_params.append(p)
                            optimizer_param_ids.add(id(p))
            
            # 如果有新参数，添加到优化器
            if new_params:
                self._world_model_optimizer.add_param_group({'params': new_params, 'lr': learning_rate})
        
        # 设置感官 Object 状态（使用 detach 避免梯度图问题）
        for sense, value in observation.items():
            if sense in self.objects:
                self.objects[sense].set_state(value.detach())
        if "action" in self.objects:
            self.objects["action"].set_state(action.detach())
        
        # 获取当前 internal 状态（detach）
        current_internal = self.objects["internal"].state.detach()
        
        # 预测下一状态（使用 dynamics）
        pred_next_internal = self.world_model_aspects.dynamics.predict_next_state(
            current_internal, action.detach()
        )
        
        # 设置下一观察（用于计算 observation aspect 的自由能）
        next_targets = {}
        for sense, value in next_observation.items():
            target_name = f"{sense}_next"
            if target_name not in self.objects:
                self.objects[target_name] = ObjectNode(target_name, dim=value.shape[-1], device=self.device)
            self.objects[target_name].set_state(value.detach())
            next_targets[sense] = target_name
        
        # 临时修改 observation aspects 的目标
        original_dst = {}
        for aspect in getattr(self.world_model_aspects, "observation_aspects", []):
            sense_name = aspect.dst_names[0]
            original_dst[aspect] = sense_name
            if sense_name in next_targets:
                aspect.dst_names[0] = next_targets[sense_name]
        
        # 设置目标状态（用于 preference）
        if target_state is not None and "target" in self.objects:
            self.objects["target"].set_state(target_state.detach())
            self.world_model_aspects.set_target_state(target_state.detach())
        
        # 计算自由能并更新参数
        self._world_model_optimizer.zero_grad()
        
        # 重新设置 internal 为可微的（用于计算梯度）
        self.objects["internal"].set_state(pred_next_internal)
        
        # 计算自由能：世界模型 Aspect + LinearGenerativeAspect（包括 touch Aspects）
        F = torch.tensor(0.0, device=self.device)
        # 世界模型 Aspect 的自由能
        for aspect in self.world_model_aspects.aspects:
            F = F + aspect.free_energy_contrib(self.objects)
        # LinearGenerativeAspect 的自由能（包括 touch Aspects）
        for aspect in self.aspects:
            if isinstance(aspect, LinearGenerativeAspect):
                F = F + aspect.free_energy_contrib(self.objects)
        
        F.backward(retain_graph=False)  # 世界模型学习后不需要保留图
        self._world_model_optimizer.step()
        
        # 清理梯度，避免影响后续的推理
        # 只清理叶子张量的梯度，避免警告
        for name in self.objects:
            state = self.objects[name].state
            if state.requires_grad and state.is_leaf and state.grad is not None:
                state.grad = None
        
        # 恢复 observation aspect 的 dst
        for aspect, dst in original_dst.items():
            aspect.dst_names[0] = dst
        
        # 恢复 internal 为 detach 状态
        self.objects["internal"].set_state(pred_next_internal.detach())
        
        # 更新 semantic_context（从 internal 状态提取语义部分）
        self.update_semantic_context(source="internal")
        
        self.sanitize_states()
    
    def set_world_model_target(self, target_state: torch.Tensor):
        """设置世界模型的目标状态（用于 preference）"""
        if self.world_model_aspects is not None:
            self.world_model_aspects.set_target_state(target_state)
            if "target" in self.objects:
                self.objects["target"].set_state(target_state)
    
    def forward(self, x: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        """
        if x is not None:
            for sense, value in x.items():
                if sense in self.objects:
                    self.objects[sense].set_state(value)
        
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
            "has_llm_aspect": self.llm_aspect is not None,
            "llm_aspect_name": self.llm_aspect.name if self.llm_aspect is not None else None,
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
                    "spec": self.pipeline_specs[i],
                }
                for i, p in enumerate(self.aspect_pipelines)
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

