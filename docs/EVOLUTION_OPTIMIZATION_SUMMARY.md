# 演化评估逻辑优化总结

## 优化内容

### 1. 改进初始化策略

**问题**：新创建的 aspect/pipeline 是随机初始化的，初始自由能贡献很高，导致演化决策难以触发。

**解决方案**：
- 从现有的相似 aspect（如 `ObservationAspect`）复制权重来初始化新的 `LinearGenerativeAspect`
- 添加小噪声（`noise_scale=0.01`）保持多样性
- 降低初始缩放因子（`init_scale=0.01`）

```python
# 查找相似的现有 aspect 作为参考
reference_aspect = None
for asp in brain.aspects:
    if isinstance(asp, ObservationAspect) and "internal" in asp.src_names and "vision" in asp.dst_names:
        reference_aspect = asp
        break

# 从参考 aspect 提取权重并添加噪声
if reference_aspect is not None:
    ref_weight = reference_aspect.observation_model.weight.detach().clone()
    init_weight = ref_weight + noise_scale * torch.randn_like(ref_weight)
```

### 2. 考虑"潜力"而非仅看初始自由能

**问题**：即使改进了初始化，新 aspect/pipeline 的初始自由能可能仍然高于当前自由能。

**解决方案**：
- 使用"乐观估计"：考虑新结构经过训练后的潜力
- 对于 `add_sensory_aspect`：假设能减少 15% 的观察生成误差
- 对于 `add_pipeline`：假设能减少 20% 的分类误差 + 5% 的观察误差

```python
# 考虑新 aspect 的"潜力"
F_obs_current = observation_aspect.free_energy_contrib(brain.objects).item()
F_potential_reduction = F_obs_current * 0.15  # 假设减少 15%
F_optimistic = F_with_sensory - F_potential_reduction
options["add_sensory_aspect"] = min(F_with_sensory, F_optimistic)
```

### 3. 降低演化阈值

**问题**：原始阈值（10%）过高，难以触发演化。

**解决方案**：
- 将 `evolution_action_threshold` 从 0.1（10%）降低到 0.01（1%）
- 使用双重判断：绝对降低（> 0.01）或相对降低（> 0.5%）

```python
evolution_threshold = config.get("evolution_action_threshold", 0.01)  # 1%
should_evolve = (
    best_evolution_option[0] != "no_change" and 
    best_evolution_option[0] != "prune" and  # 排除 prune（尚未实现）
    (expected_reduction > evolution_threshold or reduction_ratio > 0.005)  # 绝对或相对降低
)
```

### 4. 添加详细调试输出

**改进**：
- 显示所有演化选项的自由能值和降低比例
- 显示最佳选项和预期收益
- 便于诊断演化决策是否触发

```python
if verbose_evolution:
    print(f"  [演化评估] 当前F={F_current:.4f}")
    for opt_name, opt_F in evolution_options.items():
        if opt_name != "no_change":
            opt_reduction = F_current - opt_F
            opt_ratio = opt_reduction / F_current if F_current > 0 else 0
            print(f"    {opt_name}: F={opt_F:.4f}, 降低={opt_reduction:.4f} ({opt_ratio*100:.2f}%)")
```

## 配置参数

新增/修改的配置参数：

```python
config = {
    "evolution_action_threshold": 0.01,  # 演化行动阈值：预期自由能至少降低 1%
    "evolution_noise_scale": 0.01,  # 从参考 aspect 复制权重时的噪声缩放因子
    "verbose_evolution": True,  # 是否输出演化评估的详细信息
    ...
}
```

## 当前状态

### 已实现
- ✅ 实际测试创建 aspect/pipeline 后的自由能变化
- ✅ 从现有 aspect 复制权重初始化新 aspect
- ✅ 考虑新结构的"潜力"而非仅看初始自由能
- ✅ 降低演化阈值，更容易触发
- ✅ 添加详细调试输出

### 待改进
- ⚠️ `prune` 功能尚未实现（目前使用估算值，但不会真正执行）
- ⚠️ Pipeline 的潜力估计可能需要进一步调优
- ⚠️ 演化决策触发频率可能需要根据实际效果调整

## 下一步

1. **实现 prune 功能**：真正执行剪枝操作
2. **进一步优化潜力估计**：基于实际训练效果调整估计值
3. **长期实验**：运行更长时间，观察演化效果
4. **性能分析**：分析演化对学习效果的影响

## 参考

- [演化作为行动](./EVOLUTION_AS_ACTION.md)
- [MNIST 主动推理实验 V2](./MNIST_ACTIVE_INFERENCE_V2.md)

