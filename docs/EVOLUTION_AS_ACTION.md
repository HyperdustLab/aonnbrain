# 演化作为行动：主动推理的统一框架

## 概述

在主动推理框架中，**演化（网络结构变化）应该作为行动的一部分**，通过最小化预期自由能来选择。这符合主动推理的核心原理：所有能影响未来观察和状态的操作都应该通过预期自由能最小化来选择。

## 理论背景

### 主动推理中的行动选择

在主动推理中，行动选择的目标是**最小化预期自由能**（Expected Free Energy）：

```
E[F] = E[F(obs_{t+1}, state_{t+1}) | state_t, action]
```

行动应该包括所有能影响未来观察和状态的操作：
- **连续行动**：分类预测、状态更新（可微优化）
- **离散行动**：网络演化决策（结构变化）

### 网络演化作为行动

网络演化（创建/删除结构）确实会影响未来的自由能：
- 创建新 Aspect/Pipeline 可能降低未来自由能
- 删除无用结构可能降低计算成本
- 演化决策应基于预期自由能评估

## 实现方案

### 1. 评估演化选项（实际测试）

`evaluate_evolution_options` 函数**实际创建临时结构并测试**自由能变化：

```python
def evaluate_evolution_options(
    brain,
    observation_aspect,
    dynamics_aspect,
    classification_aspect,
    current_state,
    current_obs,
    target,
    config,
    device=None,
) -> Dict:
    """
    评估不同演化选项的预期自由能（实际测试）
    
    演化选项：
    1. no_change: 不改变结构
    2. add_sensory_aspect: 添加感官 Aspect（实际创建并测试）
    3. add_pipeline: 添加 Pipeline（实际创建并测试）
    4. prune: 剪枝无用 Aspect（暂不实现）
    """
    # 计算当前自由能（作为基准）
    F_current = brain.compute_free_energy().item()
    
    # 1. 测试添加感官 Aspect
    # 创建临时 aspect，添加到 brain，计算自由能，然后移除
    temp_sensory_aspect = LinearGenerativeAspect(...)
    brain.aspects.append(temp_sensory_aspect)
    F_with_sensory = brain.compute_free_energy().item()
    brain.aspects.remove(temp_sensory_aspect)
    
    # 2. 测试添加 Pipeline
    # 创建临时 pipeline，添加到 brain，计算自由能，然后移除
    temp_pipeline = PipelineAspect(...)
    brain.aspects.append(temp_pipeline)
    F_with_pipeline = brain.compute_free_energy().item()
    brain.aspects.remove(temp_pipeline)
    
    options = {
        "no_change": F_current,
        "add_sensory_aspect": F_with_sensory,
        "add_pipeline": F_with_pipeline,
        "prune": F_current * 0.95,  # 暂不实现
    }
    
    return options
```

**关键改进**：
- **实际测试**：不再使用估算值，而是实际创建临时结构并测量自由能
- **准确评估**：基于真实自由能变化决定是否演化
- **自动清理**：测试后自动移除临时结构，不影响原始网络

### 2. 统一行动选择

`optimize_action_with_evolution` 函数将演化决策纳入行动选择：

```python
def optimize_action_with_evolution(
    brain,
    observation_aspect,
    dynamics_aspect,
    classification_aspect,
    current_state,
    current_obs,
    target,
    config,
    ...
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    通过优化自由能选择行动（包括演化决策）
    
    行动 = 分类预测（连续） + 演化决策（离散）
    """
    # 1. 评估演化选项的预期自由能
    evolution_options = evaluate_evolution_options(...)
    
    # 2. 选择最优演化选项（最小预期自由能）
    best_evolution_option = min(evolution_options.items(), key=lambda x: x[1])
    
    # 3. 如果预期收益足够大，执行演化
    if (best_evolution_option[0] != "no_change" and 
        expected_reduction > threshold):
        evolution_decision = {
            "option": best_evolution_option[0],
            "expected_F": best_evolution_option[1],
            "expected_reduction": ...,
        }
    
    # 4. 优化分类预测（连续行动）
    action_logits = optimize_classification_action(...)
    
    return action_logits, evolution_decision
```

### 3. 执行演化决策

在实验主循环中，如果演化决策建议演化，执行相应的结构变化：

```python
# 行动选择
action, evolution_decision = optimize_action_with_evolution(...)

# 如果演化决策建议演化，执行演化
if evolution_decision is not None:
    evolution_option = evolution_decision["option"]
    
    if evolution_option == "add_sensory_aspect":
        brain.evolve_network(obs, target=target)
    elif evolution_option == "add_pipeline":
        brain.evolve_network(obs, target=target)
    elif evolution_option == "prune":
        brain.prune_network(...)
```

## 配置参数

新增配置参数：

```python
config = {
    "evolution_action_threshold": 0.1,  # 演化行动阈值：预期自由能至少降低 10% 才执行演化
    ...
}
```

## 优势

1. **理论一致性**：所有行动（包括结构变化）都基于预期自由能最小化
2. **统一框架**：连续行动和离散行动在同一个框架下选择
3. **自适应演化**：演化决策基于实际预期收益，而非固定阈值
4. **可解释性**：演化决策有明确的预期自由能评估

## 实验记录

演化决策会被记录在实验结果中：

```json
{
    "evolution_decisions": [
        {
            "step": 10,
            "option": "add_pipeline",
            "expected_F": 2.5,
            "current_F": 3.0,
            "expected_reduction": 0.5
        },
        ...
    ],
    "snapshots": [
        {
            "step": 10,
            "evolution_decision": {...},
            ...
        },
        ...
    ]
}
```

## 未来改进

1. **更精确的预期自由能评估**：使用生成模型预测未来状态和观察
2. **演化成本建模**：考虑计算复杂度、内存占用等成本
3. **多步前瞻**：评估演化对多步未来自由能的影响
4. **演化策略学习**：学习哪些演化选项在哪些情况下更有效

## 参考

- [MNIST 主动推理实验 V2](./MNIST_ACTIVE_INFERENCE_V2.md)
- [AONN 统一架构](./AONN_UNIFIED_ARCHITECTURE.md)

