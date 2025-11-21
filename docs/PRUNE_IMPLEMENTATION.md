# Prune 功能实现：反向抑制与自由能停滞检测

## 概述

实现了完整的 prune（剪枝）功能，作为演化行动的一部分。当网络长时间无法通过其他方式降低自由能时，自动触发剪枝，移除不重要的 aspect，改变网络结构。

## 核心功能

### 1. 取消演化条件限制

**之前**：演化需要满足阈值条件（如降低 1% 以上）

**现在**：取消阈值限制，只要预期自由能更低就执行演化

```python
# 判断是否执行演化：取消阈值限制，只要不是"不改变"就执行
should_evolve = (
    best_evolution_option[0] != "no_change" and 
    F_best < F_current  # 只要预期自由能更低就执行
)
```

### 2. Prune 评估：实际测试移除 aspect 后的自由能

**实现**：实际测试移除贡献最小的 aspect 后的自由能变化

```python
# 找出贡献最小的 aspect
aspect_contributions = []
for asp in brain.aspects:
    contrib = asp.free_energy_contrib(brain.objects).item()
    aspect_type = type(asp).__name__
    # 只考虑可剪枝的 aspect（排除核心生成模型）
    if aspect_type not in ["ObservationAspect", "DynamicsAspect", "ClassificationAspect", "EncoderAspect"]:
        aspect_contributions.append((asp, contrib))

# 找出贡献最小的 aspect
aspect_contributions.sort(key=lambda x: x[1])
weakest_aspect, weakest_contrib = aspect_contributions[0]

# 临时移除最弱的 aspect，测试自由能
brain.aspects.remove(weakest_aspect)
F_without_weakest = brain.compute_free_energy().item()
brain.aspects.append(weakest_aspect)  # 恢复

# 如果移除后自由能降低，说明这个 aspect 是噪声
if F_without_weakest < F_current:
    options["prune"] = F_without_weakest
else:
    options["prune"] = F_current + 1.0  # 不应该剪枝
```

### 3. Prune 执行：移除不重要的 aspect

**实现**：当 prune 被选为最优选项时，真正移除贡献最小的 aspect

```python
elif evolution_option == "prune":
    # 执行剪枝：移除贡献最小的 aspect
    aspect_contributions = []
    for asp in brain.aspects:
        try:
            contrib = asp.free_energy_contrib(brain.objects).item()
            aspect_type = type(asp).__name__
            # 只剪枝非核心 aspect
            if aspect_type not in ["ObservationAspect", "DynamicsAspect", "ClassificationAspect", "EncoderAspect"]:
                aspect_contributions.append((asp, contrib))
        except:
            continue
    
    if len(aspect_contributions) > 0:
        aspect_contributions.sort(key=lambda x: x[1])
        weakest_aspect, weakest_contrib = aspect_contributions[0]
        
        # 移除最弱的 aspect
        brain.aspects.remove(weakest_aspect)
        if hasattr(brain, 'aspect_modules'):
            try:
                brain.aspect_modules.remove(weakest_aspect)
            except:
                pass
```

### 4. 自由能停滞检测：长时间无改善时强制剪枝

**实现**：检测自由能是否长时间无法降低，如果停滞超过阈值，强制触发剪枝

```python
# 自由能停滞检测
free_energy_stagnation = {
    "best_F": float('inf'),
    "stagnation_steps": 0,
    "stagnation_threshold": 50,  # 50 步无改善则剪枝
    "min_improvement": 0.01,  # 至少改善 1% 才算改善
}

# 每个步骤检查
F_current_step = F_total.item()
improvement = (free_energy_stagnation["best_F"] - F_current_step) / free_energy_stagnation["best_F"]

if improvement > free_energy_stagnation["min_improvement"]:
    # 有改善，重置计数器
    free_energy_stagnation["best_F"] = F_current_step
    free_energy_stagnation["stagnation_steps"] = 0
else:
    # 无改善，增加停滞步数
    free_energy_stagnation["stagnation_steps"] += 1

# 如果停滞时间过长，强制触发剪枝
if (free_energy_stagnation["stagnation_steps"] >= free_energy_stagnation["stagnation_threshold"] and
    len(brain.aspects) > 4):  # 至少保留 4 个核心 aspect
    
    # 找出并移除贡献最小的 aspect
    # ...（剪枝逻辑）
    
    # 重置停滞计数器
    free_energy_stagnation["stagnation_steps"] = 0
    free_energy_stagnation["best_F"] = float('inf')
```

## 配置参数

```python
config = {
    "evolution_action_threshold": 0.0,  # 已取消，只要预期自由能更低就执行
    "prune_stagnation_steps": 50,  # 自由能停滞多少步后触发强制剪枝
    "prune_min_improvement": 0.01,  # 至少改善 1% 才算改善
    ...
}
```

## 保护机制

1. **核心 Aspect 保护**：不剪枝核心生成模型
   - `ObservationAspect`
   - `DynamicsAspect`
   - `ClassificationAspect`
   - `EncoderAspect`

2. **最小结构保护**：至少保留 4 个核心 aspect

3. **剪枝后重置**：剪枝后重置停滞计数器，给网络重新学习的机会

## 实验结果

从测试实验（50 步）中观察到：

- **演化决策触发**：1 次强制剪枝（step 49）
- **剪枝的 Aspect**：`preference` aspect（贡献=10.56）
- **最终结构**：从 7 Aspects 减少到 6 Aspects
- **最终自由能**：215.00
- **最终准确率**：14.00%

## 优势

1. **自适应结构优化**：网络能够自动移除不重要的组件
2. **防止过拟合**：移除噪声 aspect，简化网络结构
3. **反向抑制**：当无法通过增加结构降低自由能时，通过减少结构来优化
4. **长期优化**：通过停滞检测，确保网络持续优化

## 未来改进

1. **更智能的剪枝策略**：考虑 aspect 之间的依赖关系
2. **剪枝后的恢复机制**：如果剪枝后自由能增加，考虑恢复
3. **批量剪枝**：一次移除多个不重要的 aspect
4. **剪枝历史记录**：记录被剪枝的 aspect，避免重复创建

## 参考

- [演化作为行动](./EVOLUTION_AS_ACTION.md)
- [演化评估逻辑优化](./EVOLUTION_OPTIMIZATION_SUMMARY.md)

