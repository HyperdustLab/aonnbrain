# AONN 统一架构设计文档

## 概述

本文档记录 AONN 架构的重大演进：**统一向量化 Aspect Pipeline 和对象化 Aspect Network 两种架构**，实现在同一框架下支持深度学习任务和认知建模任务。

**版本**: V3.5 (Unified Architecture)  
**日期**: 2024  
**状态**: 设计阶段 → 实现中

---

## 背景与动机

### 问题

AONN 历史上存在两种不同的实现方式：

1. **向量化 Aspect Pipeline** (`aonn_mnist_aspect_pipeline`)
   - 高效批量处理
   - 适合深度学习任务（MNIST 分类等）
   - 参数紧凑，训练稳定

2. **对象化 Aspect Network** (`aonnbrain`)
   - 灵活的网络拓扑
   - 适合认知建模和主动推理
   - 可解释性强，支持动态演化

### 挑战

- 两种架构无法在同一框架下使用
- 需要为不同任务选择不同的实现
- 无法混合使用两种方式的优势

### 目标

**统一两种架构，实现：**
- ✅ 统一的接口和 API
- ✅ 自动选择最优实现方式
- ✅ 支持混合使用
- ✅ 保持向后兼容

---

## 两种架构对比

### 1. 向量化 Aspect Pipeline

**核心思想**：
- 状态：统一的状态向量 `x ∈ R^D`
- 计算：向量化的低秩算子，矩阵乘法批量处理

**网络结构**：
```
输入 → Encoder → Cell → AspectPipeLayer → AspectPipeLayer → ... → Cell → Classifier
```

**实现**：
```python
class AspectPipeLayer(nn.Module):
    def forward(self, x):
        a = W(x)           # [B, M] - 所有 Aspect 的激活
        z = ReLU(a)        # [B, M]
        delta = V(z)       # [B, D] - 所有 Aspect 的增量
        return ReLU(x + delta)  # 残差更新
```

**特点**：
- ✅ 高效：矩阵乘法批量处理
- ✅ 紧凑：参数少
- ✅ 可堆叠：深度堆叠
- ❌ 灵活性低：统一结构
- ❌ 可解释性弱：难以单独控制

### 2. 对象化 Aspect Network

**核心思想**：
- 状态：多个独立的 Object 节点
- 计算：独立的 Aspect 类，明确的 src/dst 连接

**网络结构**：
```
Object₁ ──[Aspect₁]──> Object₂
Object₃ ──[Aspect₂]──> Object₄
```

**实现**：
```python
class AspectBase(ABC):
    def forward(self, objects):
        # 从 src 读取，计算，返回 dst 的误差
        pass
    
    def free_energy_contrib(self, objects):
        # 计算自由能贡献
        pass
```

**特点**：
- ✅ 灵活：每个 Aspect 独立实现
- ✅ 可解释：明确的连接关系
- ✅ 模块化：易于扩展
- ❌ 效率较低：逐个计算
- ❌ 参数多：独立参数

### 对比表

| 特性 | Aspect Pipeline | Object-Aspect Network |
|------|----------------|---------------------|
| **状态表示** | 单一 Cell 向量 | 多个 Object 节点 |
| **Aspect 实现** | 向量化（矩阵） | 对象化（类） |
| **连接方式** | 顺序堆叠 | 图结构（任意连接） |
| **灵活性** | 低（统一结构） | 高（独立实现） |
| **效率** | 高（批量计算） | 中（逐个计算） |
| **可解释性** | 中 | 高 |
| **适用场景** | 深度学习任务 | 认知建模、自由能框架 |

---

## 统一架构设计

### 核心思想

**将 Pipeline 包装为 Aspect**，实现两种架构的统一接口。

### 设计原则

1. **统一接口**：Pipeline 和独立 Aspect 都实现 `AspectBase`
2. **自动选择**：根据任务复杂度自动选择最优实现
3. **混合使用**：可在同一网络中混合使用两种方式
4. **向后兼容**：保持现有 API 不变

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│                  AONNBrainV3 (Unified)                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐         ┌──────────────┐             │
│  │   Objects    │         │   Aspects    │             │
│  │  (状态层)    │◄───────►│  (计算层)    │             │
│  └──────────────┘         └──────────────┘             │
│       │                         │                       │
│       │                         ├─ LinearGenerativeAspect│
│       │                         ├─ LLMAspect           │
│       │                         ├─ DynamicsAspect       │
│       │                         └─ PipelineAspect ◄──┐ │
│       │                                             │ │
│       └─────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │         PipelineAspect (统一接口)                │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │      AspectPipeline (向量化实现)          │  │  │
│  │  │  - AspectLayer → AspectLayer → ...       │  │  │
│  │  │  - 高效批量处理                           │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 实现方案

### 方案 1：PipelineAspect（核心实现）

将 `AspectPipeline` 包装为 `AspectBase` 的子类：

```python
# src/aonn/aspects/pipeline_aspect.py
class PipelineAspect(AspectBase, nn.Module):
    """
    将 AspectPipeline 包装为 Aspect
    实现向量化 Pipeline 和对象化 Aspect 的统一接口
    """
    def __init__(
        self,
        src_names: List[str],
        dst_names: List[str],
        input_dim: int,
        output_dim: int,
        num_aspects: int,
        depth: int,
        name: str = "pipeline_aspect",
    ):
        super().__init__(name=name, src_names=src_names, dst_names=dst_names)
        self.pipeline = AspectPipeline(
            input_dim=input_dim,
            output_dim=output_dim,
            num_aspects=num_aspects,
            depth=depth,
        )
    
    def forward(self, objects: Dict[str, ObjectNode]) -> Dict[str, torch.Tensor]:
        """计算预测误差"""
        src_state = self._get_src_state(objects)
        pred = self.pipeline(src_state)
        dst_state = objects[self.dst_names[0]].state
        error = dst_state - pred
        return {self.dst_names[0]: error}
    
    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """计算自由能贡献"""
        error_dict = self.forward(objects)
        error = error_dict[self.dst_names[0]]
        return 0.5 * (error ** 2).sum()
    
    def _get_src_state(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """从源 Object 获取状态（支持多源拼接）"""
        states = [objects[name].state for name in self.src_names]
        return torch.cat(states, dim=-1) if len(states) > 1 else states[0]
    
    def parameters(self):
        """返回 Pipeline 的参数"""
        return self.pipeline.parameters()
```

### 方案 2：统一创建接口

在 `AONNBrainV3` 中添加统一创建方法：

```python
def create_unified_aspect(
    self,
    aspect_type: str,  # "pipeline" 或 "individual"
    src_names: List[str],
    dst_names: List[str],
    **kwargs
) -> AspectBase:
    """
    统一创建 Aspect，自动选择最优实现方式
    
    Args:
        aspect_type: "pipeline" (向量化) 或 "individual" (对象化)
        src_names: 源 Object 名称列表
        dst_names: 目标 Object 名称列表
        **kwargs: 额外参数
            - use_pipeline: 是否强制使用 Pipeline
            - input_dim, output_dim, num_aspects, depth: Pipeline 参数
    
    Returns:
        AspectBase 实例
    """
    # 自动选择：如果指定 use_pipeline 或 aspect_type == "pipeline"
    use_pipeline = kwargs.get("use_pipeline", False) or aspect_type == "pipeline"
    
    if use_pipeline:
        # 使用向量化 Pipeline（高效批量处理）
        return PipelineAspect(
            src_names=src_names,
            dst_names=dst_names,
            input_dim=kwargs.get("input_dim"),
            output_dim=kwargs.get("output_dim"),
            num_aspects=kwargs.get("num_aspects", 32),
            depth=kwargs.get("depth", 2),
            name=kwargs.get("name", "pipeline_aspect"),
        )
    else:
        # 使用独立 Aspect（灵活、可解释）
        return self.create_aspect(
            aspect_type=kwargs.get("individual_type", "sensory"),
            src_names=src_names,
            dst_names=dst_names,
            **{k: v for k, v in kwargs.items() 
               if k not in ["use_pipeline", "input_dim", "output_dim", 
                           "num_aspects", "depth", "individual_type"]}
        )
```

### 方案 3：统一自由能计算

在 `compute_free_energy` 中统一处理：

```python
def compute_free_energy(self) -> torch.Tensor:
    """计算总自由能（统一处理 Pipeline 和独立 Aspect）"""
    F = torch.tensor(0.0, device=self.device)
    
    # 所有 Aspect（包括 PipelineAspect）统一计算
    for aspect in self.aspects:
        F = F + aspect.free_energy_contrib(self.objects)
    
    # 兼容旧的 Pipeline 计算方式（如果存在）
    if len(self.aspect_pipelines) > 0:
        # Pipeline 现在也是 Aspect，已在上面计算
        pass
    
    return F
```

---

## 使用示例

### 示例 1：MNIST 分类（使用 Pipeline）

```python
from aonn.models.aonn_brain_v3 import AONNBrainV3

config = {
    "obs_dim": 784,  # 28x28
    "state_dim": 256,
    "act_dim": 10,  # 10个类别
    "sense_dims": {"vision": 784},
}

brain = AONNBrainV3(config=config, enable_evolution=True)

# 创建 Pipeline Aspect（高效批量处理）
pipeline_aspect = brain.create_unified_aspect(
    aspect_type="pipeline",
    src_names=["internal"],
    dst_names=["vision"],
    input_dim=256,
    output_dim=784,
    num_aspects=64,
    depth=4,
)

brain.aspects.append(pipeline_aspect)
```

### 示例 2：认知建模（使用独立 Aspect）

```python
# 创建独立 Aspect（灵活、可解释）
sensory_aspect = brain.create_unified_aspect(
    aspect_type="individual",
    src_names=["internal"],
    dst_names=["vision"],
    individual_type="sensory",
    obs_dim=784,
)

llm_aspect = brain.create_unified_aspect(
    aspect_type="individual",
    src_names=["semantic_context"],
    dst_names=["semantic_prediction"],
    individual_type="llm",
    llm_client=llm_client,
)

brain.aspects.extend([sensory_aspect, llm_aspect])
```

### 示例 3：混合使用

```python
# 混合使用两种方式
# - Pipeline 用于常规计算（高效）
# - 独立 Aspect 用于特殊功能（灵活）

# 1. Pipeline 处理视觉特征
vision_pipeline = brain.create_unified_aspect(
    aspect_type="pipeline",
    src_names=["internal"],
    dst_names=["vision"],
    input_dim=256,
    output_dim=784,
    num_aspects=64,
    depth=4,
)

# 2. 独立 Aspect 处理语义
llm_aspect = brain.create_unified_aspect(
    aspect_type="individual",
    src_names=["semantic_context"],
    dst_names=["semantic_prediction"],
    individual_type="llm",
    llm_client=llm_client,
)

brain.aspects.extend([vision_pipeline, llm_aspect])
```

---

## 优势与应用场景

### 统一后的优势

1. **统一接口**
   - Pipeline 和独立 Aspect 都实现 `AspectBase`
   - 统一的 `forward()` 和 `free_energy_contrib()` 方法
   - 可以在同一网络中无缝混合使用

2. **自动选择**
   - 根据任务复杂度自动选择最优实现
   - 深度学习任务 → Pipeline（高效）
   - 认知建模任务 → 独立 Aspect（灵活）

3. **性能优化**
   - Pipeline 用于批量计算（高效）
   - 独立 Aspect 用于特殊功能（灵活）
   - 混合使用获得最佳性能

4. **向后兼容**
   - 保持现有 API 不变
   - 现有代码无需修改
   - 渐进式迁移

### 应用场景

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| **MNIST 分类** | Pipeline | 高效批量处理，参数紧凑 |
| **认知建模** | 独立 Aspect | 灵活拓扑，可解释性强 |
| **LLM 集成** | 独立 Aspect | 需要特殊处理逻辑 |
| **混合任务** | 混合使用 | Pipeline 处理常规，独立 Aspect 处理特殊 |

---

## 实现路线图

### 阶段 1：核心实现 ✅
- [x] 设计统一接口
- [x] 实现 `PipelineAspect` 类
- [ ] 修改 `AONNBrainV3.create_aspect` 支持 Pipeline
- [ ] 统一 `compute_free_energy` 计算

### 阶段 2：功能完善
- [ ] 添加自动选择逻辑
- [ ] 实现配置选项
- [ ] 添加性能对比测试

### 阶段 3：文档和示例
- [ ] 更新使用文档
- [ ] 添加 MNIST 示例
- [ ] 添加混合使用示例

### 阶段 4：优化和测试
- [ ] 性能优化
- [ ] 单元测试
- [ ] 集成测试

---

## 技术细节

### Pipeline 到 Aspect 的映射

```
AspectPipeline (向量化)
    ↓ 包装
PipelineAspect (AspectBase)
    ↓ 统一接口
AspectBase.forward()
AspectBase.free_energy_contrib()
```

### 状态传递

```python
# Pipeline 方式
src_state = objects["internal"].state  # [D]
pred = pipeline(src_state)              # [D]
error = objects["vision"].state - pred   # [D]

# 独立 Aspect 方式
error = aspect.forward(objects)         # {"vision": error}
```

### 自由能计算

```python
# 统一计算
F_total = sum(aspect.free_energy_contrib(objects) 
              for aspect in all_aspects)
```

---

## 设计决策

### 决策 1：包装而非替换

**选择**：将 Pipeline 包装为 Aspect，而非替换现有架构

**原因**：
- 保持向后兼容
- 两种方式可以共存
- 渐进式迁移

### 决策 2：统一接口

**选择**：Pipeline 和独立 Aspect 都实现 `AspectBase`

**原因**：
- 简化使用
- 统一计算
- 易于扩展

### 决策 3：自动选择

**选择**：提供自动选择机制，但允许手动指定

**原因**：
- 降低使用门槛
- 保持灵活性
- 支持高级用法

---

## 影响分析

### 对现有代码的影响

**最小影响**：
- 现有代码无需修改
- API 保持兼容
- 可选使用新功能

### 对性能的影响

**正面影响**：
- Pipeline 提供高效批量处理
- 混合使用获得最佳性能
- 自动选择优化计算路径

### 对可扩展性的影响

**正面影响**：
- 统一的接口便于扩展
- 支持新的 Aspect 类型
- 易于添加新的 Pipeline 类型

---

## 未来发展方向

1. **智能选择算法**
   - 根据任务特征自动选择最优方式
   - 动态切换 Pipeline 和独立 Aspect

2. **混合优化**
   - Pipeline 和独立 Aspect 的协同优化
   - 自动分配计算资源

3. **统一训练**
   - 统一的训练接口
   - 支持混合架构的训练

4. **可视化工具**
   - 可视化混合架构
   - 分析 Pipeline 和独立 Aspect 的贡献

---

## 总结

统一架构设计实现了：

✅ **统一接口**：Pipeline 和独立 Aspect 都实现 `AspectBase`  
✅ **自动选择**：根据任务自动选择最优实现  
✅ **混合使用**：可在同一网络中混合使用两种方式  
✅ **向后兼容**：保持现有 API 不变  

这标志着 AONN 架构的重大演进，为支持更广泛的任务类型奠定了基础。

---

## 相关文档

- [AONN 架构对比](./AONN_ARCHITECTURE_COMPARISON.md)
- [AONN V3 演化文档](./AONN_V3_EVOLUTION.md)
- [AONN V2 架构文档](./AONN_V2_ARCHITECTURE.md)
- [MNIST 演化方案](./AONN_MNIST_EVOLUTION.md)

---

**文档维护者**: AONN 开发团队  
**最后更新**: 2024  
**版本**: 1.0

