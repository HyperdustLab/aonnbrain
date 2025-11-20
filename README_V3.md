# AONN Brain V3 - 动态演化版本

## 概述

AONN Brain V3 实现了从最小初始网络开始的动态演化，支持与世界模型交互，并能够观察自我模型网络结构的变化。这是第一个支持网络拓扑动态演化的 AONN 实现。

## 核心特性

### ✅ 1. 最小初始架构
- 从只有 2 个 Object（sensory, internal）开始
- 无 Aspect，无 Pipeline
- 完全空白的状态，等待演化

### ✅ 2. 动态演化机制
- **Object 动态创建**: 根据自由能自动创建新 Object（如 action）
- **Aspect 动态创建**: 根据连接需求自动创建 Aspect
- **Pipeline 动态创建**: 根据深度处理需求自动创建 Pipeline
- **智能剪枝**: 自动移除不重要的连接

### ✅ 3. 世界模型交互
- **SimpleWorldModel**: 模拟连续状态空间环境
- **WorldModelInterface**: 标准化的环境交互接口
- 支持观察、动作、奖励的完整循环

### ✅ 4. 自我模型观察
- 记录网络结构快照
- 跟踪自由能变化趋势
- 分析演化历史事件
- 保存演化日志到 JSON

## 快速开始

### 运行演示

```bash
python scripts/demo_v3_evolution.py
```

演示会展示：
1. 最小初始网络的创建
2. 与世界模型的交互
3. 网络结构的动态演化
4. 自我模型的变化观察

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
world_model = SimpleWorldModel(state_dim=32, action_dim=8, obs_dim=16)
world_interface = WorldModelInterface(world_model)

# 交互循环
for step in range(num_steps):
    obs = world_interface.get_observation()
    brain.evolve_network(obs)
    
    # 观察自我模型
    if step % 10 == 0:
        snapshot = brain.observe_self_model()
```

## 架构对比

| 特性 | V1 | V2 | V3 |
|------|----|----|----|
| **初始架构** | 固定配置 | 固定配置 | **最小初始** |
| **网络结构** | 静态 | 静态 | **动态演化** |
| **Object 创建** | 初始化 | 初始化 | **动态创建** |
| **Aspect 创建** | 初始化 | 初始化 | **动态创建** |
| **Pipeline 创建** | 初始化 | 初始化 | **动态创建** |
| **世界模型** | ❌ | ❌ | ✅ |
| **自我模型观察** | ❌ | ❌ | ✅ |
| **演化历史** | ❌ | ❌ | ✅ |

## 演化示例

### 初始状态
```
Objects: {sensory, internal}
Aspects: {}
Pipelines: {}
自由能: 0.0
```

### 演化后（步骤 10）
```
Objects: {sensory, internal}
Aspects: {linear_int2sens: internal → sensory}
Pipelines: {}
自由能: 0.1147
```

### 演化后（步骤 30）
```
Objects: {sensory, internal}
Aspects: {linear_int2sens: internal → sensory}
Pipelines: {}
自由能: 0.0967
```

## 文件结构

```
src/aonn/
├── core/
│   └── evolution.py          # 演化机制
├── models/
│   ├── aonn_brain_v3.py     # V3 主类
│   └── world_model.py       # 世界模型
scripts/
└── demo_v3_evolution.py     # 演示脚本
docs/
└── AONN_V3_EVOLUTION.md     # 详细文档
data/
└── evolution_log.json        # 演化日志
```

## 关键设计

### 1. 遵循自由能原理
- 所有演化决策基于自由能
- 最小化自由能驱动网络增长
- 符合 Friston 的 Active Inference 框架

### 2. 马尔可夫毯结构
- Object = Vertical MB（垂直马尔可夫毯）
- Aspect = Cross-MB（横切马尔可夫毯）
- 符合贝叶斯推理网络

### 3. 动态演化
- 类似人脑神经网络的发育
- 从简单到复杂
- 根据任务需求自适应

## 未来扩展

1. **更智能的演化策略**: 基于自由能梯度的演化决策
2. **更复杂的世界模型**: 支持多智能体、动态环境
3. **演化可视化**: 实时可视化网络结构变化
4. **演化分析工具**: 分析演化模式和趋势
5. **参数学习**: 结合自由能最小化的参数学习

## 相关文档

- [AONN_V3_EVOLUTION.md](docs/AONN_V3_EVOLUTION.md) - 详细演化文档
- [AONN_V2_ARCHITECTURE.md](docs/AONN_V2_ARCHITECTURE.md) - V2 架构文档
- [AONN_ARCHITECTURE_COMPARISON.md](docs/AONN_ARCHITECTURE_COMPARISON.md) - 架构对比

## 总结

AONN Brain V3 是第一个支持动态演化的 AONN 实现，它：
- ✅ 从最小初始网络开始
- ✅ 根据自由能动态演化
- ✅ 支持世界模型交互
- ✅ 观察自我模型变化
- ✅ 遵循自由能原理和贝叶斯推理
- ✅ 符合马尔可夫毯结构

这为构建真正自适应的、类似人脑神经网络的 AI 系统奠定了基础。

