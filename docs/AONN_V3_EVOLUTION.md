# AONN Brain V3 动态演化文档

AONN Brain V3 实现了从最小初始网络开始的动态演化，支持与世界模型交互，并观察自我模型网络结构的变化。

## 核心特性

### 1. 最小初始架构

V3 从最小网络开始：
- **Object**: 只有 `sensory` 和 `internal`
- **Aspect**: 无（或一个基础 identity aspect）
- **Pipeline**: 无

### 2. 动态演化机制

根据自由能动态创建：
- **Object**: 当自由能持续高时创建新 Object（如 `action`）
- **Aspect**: 当需要连接两个 Object 时创建 Aspect
- **Pipeline**: 当需要深度处理时创建 Pipeline

### 3. 世界模型交互

- **SimpleWorldModel**: 模拟环境，提供观察和奖励
- **WorldModelInterface**: 标准化的环境交互接口

### 4. 自我模型观察

- 记录网络结构快照
- 跟踪自由能变化
- 分析演化历史

## 设计哲学

1. **状态-计算解耦**  
   `ObjectNode` 只存储 μ 状态、`AspectBase` 只负责计算误差并回写增量（见 `src/aonn/core/object.py`、`src/aonn/core/aspect_base.py`）。这种“Object = 细胞（状态），Aspect = 神经元（计算）”的分层，是所有演化/学习规则的基础。

2. **自由能作为唯一驱动力**  
   `compute_total_free_energy` 将所有感官、先验、偏好项整合成单一标量，主动推理循环与世界模型学习都以“降低 F”为目标（`src/aonn/core/active_inference_loop.py`，`WorldModelAspectSet`）。因此网络拓扑调整与参数更新共享同一最优化准则。

3. **演化器 = 神经生成规则**  
   `NetworkEvolution`（`src/aonn/core/evolution.py`）像“基因程序”：只定义何时创建/剪枝 Object、Aspect、Pipeline，以及批量增殖策略。真正的结构实例在运行过程中自发长出，使网络能从最小骨架扩展到复杂拓扑，同时保持全局可控。

4. **Pipeline 充当介神经层**  
   `AspectPipeline` 串联多层低秩算子，形成 `internal→latent→...→action` 的“中枢通路”。其深度/宽度由演化器根据自由能自动扩张，承担多模态整合和动作规划，呼应线虫实验中的介神经网络。

5. **世界模型与自我模型共振**  
   世界模型提供可学习的 Dynamics/Observation/Preference 因子，自我模型负责结构演化，二者共享同一组 Object 状态。于是“环境统计”“内部拓扑”“主动推理”被统一在自由能框架内，真正形成可解释、可演化的自由能大脑。

## 架构设计

### 初始架构

```python
初始网络:
  Objects: {sensory, internal}
  Aspects: {}
  Pipelines: {}
```

### 演化过程

```
步骤 1: 创建基础 Aspect (sensory aspect)
  - internal → sensory

步骤 2: 创建 action Object (如果自由能高)
  - 添加 action Object

步骤 3: 创建 Pipeline (如果需要深度处理)
  - internal → [Pipeline] → action
```

## 使用示例

### 基本使用

```python
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.models.world_model import SimpleWorldModel, WorldModelInterface

# 配置
config = {
    "obs_dim": 16,
    "state_dim": 32,
    "act_dim": 8,
    "evolution": {
        "free_energy_threshold": 0.5,
        "prune_threshold": 0.01,
        "max_objects": 20,
        "max_aspects": 100,
    }
}

# 创建最小初始网络
brain = AONNBrainV3(config=config, enable_evolution=True)

# 创建世界模型
world_model = SimpleWorldModel(
    state_dim=32,
    action_dim=8,
    obs_dim=16
)
world_interface = WorldModelInterface(world_model)

# 交互循环
for step in range(num_steps):
    # 获取观察
    obs = world_interface.get_observation()
    
    # 网络演化
    brain.evolve_network(obs)
    
    # 主动推理
    # ...
    
    # 观察自我模型
    if step % 10 == 0:
        snapshot = brain.observe_self_model()
```

### 观察自我模型

```python
# 获取网络结构
structure = brain.get_network_structure()
print(f"Objects: {structure['num_objects']}")
print(f"Aspects: {structure['num_aspects']}")
print(f"Pipelines: {structure['num_pipelines']}")

# 获取演化历史
history = brain.get_evolution_history()
for event in history:
    print(f"步骤 {event.step}: {event.event_type} - {event.trigger_condition}")

# 获取演化摘要
summary = brain.evolution.get_evolution_summary()
print(f"总事件: {summary['total_events']}")
print(f"Objects 创建: {summary['stats']['objects_created']}")
```

## 演化机制

### 创建条件

1. **Object 创建**:
   - 自由能持续高于阈值
   - 某个 Object 的误差持续高（需要分解）

2. **Aspect 创建**:
   - 两个 Object 之间需要连接
   - 创建 Aspect 能显著降低自由能

3. **Pipeline 创建**:
   - 需要深度处理两个 Object Layer
   - 现有 Aspect 不足以处理

### 剪枝机制

- 自由能贡献 < 阈值
- 权重 < 阈值
- 自由能贡献为负（噪声）

## 世界模型

### SimpleWorldModel

模拟环境特性：
- **状态空间**: 连续状态
- **动作空间**: 连续动作
- **观察空间**: 部分可观察
- **奖励**: 基于状态和动作

### 接口方法

```python
world_interface.get_observation()  # 获取观察
world_interface.get_reward(action)  # 获取奖励
world_interface.step(action)       # 执行动作
world_interface.reset()            # 重置环境
```

## 自我模型观察

### 快照内容

```python
snapshot = {
    "step": 10,
    "free_energy": 0.0967,
    "structure": {
        "num_objects": 2,
        "num_aspects": 1,
        "num_pipelines": 0,
        "objects": {...},
        "aspects": [...],
        "pipelines": [...]
    },
    "evolution_stats": {...}
}
```

### 演化日志

演化日志保存在 `data/evolution_log.json`，包含：
- 每个快照的网络结构
- 自由能变化
- 演化统计

## 运行演示

```bash
python scripts/demo_v3_evolution.py
```

演示会：
1. 创建最小初始网络
2. 与世界模型交互
3. 观察网络结构演化
4. 记录自我模型变化
5. 保存演化日志

## 与 V2 的区别

| 特性 | V2 | V3 |
|------|----|----|
| **初始架构** | 固定配置 | 最小初始 |
| **网络结构** | 静态 | 动态演化 |
| **Object 创建** | 初始化时创建 | 动态创建 |
| **Aspect 创建** | 初始化时创建 | 动态创建 |
| **Pipeline 创建** | 初始化时创建 | 动态创建 |
| **世界模型** | 无 | 支持 |
| **自我模型观察** | 无 | 支持 |

## 未来扩展

1. **更智能的演化策略**: 基于自由能梯度的演化决策
2. **更复杂的世界模型**: 支持多智能体、动态环境
3. **演化可视化**: 实时可视化网络结构变化
4. **演化分析工具**: 分析演化模式和趋势

