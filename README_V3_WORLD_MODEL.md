# AONN Brain V3 - 世界模型学习版本

## 完成的三点改进

### ✅ 1. 在 AONN 内部建立可学习的生成模型

实现了三个世界模型 Aspect：

- **DynamicsAspect**: 学习状态转移模型 `p(s_{t+1} | s_t, a_t)`
- **ObservationAspect**: 学习观察生成模型 `p(o_t | s_t)`
- **PreferenceAspect**: 学习偏好/目标模型（先验）

所有 Aspect 都是可学习的神经网络，通过自由能最小化学习参数。

### ✅ 2. 将目标/奖励转化为先验项

**PreferenceAspect** 将目标状态转化为先验项：

```python
F_preference = 0.5 * ||internal - target||²
```

这相当于 `-log p(internal | target)`，成为自由能的一部分。

### ✅ 3. 参数学习机制

实现了 `learn_world_model()` 方法：

```python
# 通过自由能最小化学习参数
F = F_dynamics + F_observation + F_preference
F.backward()
optimizer.step()
```

## 实验结果

### 自由能下降趋势

```
初始: 4.2483
步骤 5:  0.4249
步骤 10: 0.3470
步骤 15: 0.3005
步骤 20: 0.2583
步骤 25: 0.2065
步骤 30: 0.1892
```

**自由能下降了 95.5%**，说明世界模型正在学习。

### 世界模型自由能贡献下降

```
步骤 5:  0.3336
步骤 10: 0.2592
步骤 15: 0.2131
步骤 20: 0.1696
步骤 25: 0.1268
```

**世界模型预测误差下降了 62%**，说明生成模型正在逼近真实世界。

## 架构对比

| 特性 | 之前 | 现在 |
|------|------|------|
| **世界模型** | 固定环境模拟器 | ✅ 可学习的生成模型 |
| **生成模型** | ❌ | ✅ Dynamics + Observation |
| **先验项** | ❌ | ✅ PreferenceAspect |
| **参数学习** | ❌ | ✅ 自由能最小化 |
| **符合 FEP** | 部分 | ✅ 完全符合 |

## 自由能组成

```
F_total = F_dynamics + F_observation + F_preference + F_other

其中：
- F_dynamics: 状态转移误差（学习动力学）
- F_observation: 观察生成误差（学习观察模型）
- F_preference: 先验项（目标约束）
```

## 使用方法

```python
# 配置（启用世界模型学习）
config = {
    "obs_dim": 16,
    "state_dim": 32,
    "act_dim": 8,
    "enable_world_model_learning": True,  # 关键！
}

# 创建 brain
brain = AONNBrainV3(config=config, enable_evolution=True)

# 学习循环
for step in range(num_steps):
    obs = world_interface.get_observation()
    action = generate_action()
    next_obs, reward = world_interface.step(action)
    
    # 学习世界模型
    target_state = world_model.get_true_state()
    brain.learn_world_model(obs, action, next_obs, target_state)
```

## 关键文件

- `src/aonn/aspects/world_model_aspects.py` - 世界模型 Aspect
- `src/aonn/models/aonn_brain_v3.py` - V3 主类（包含学习机制）
- `scripts/demo_v3_evolution.py` - 完整演示
- `docs/WORLD_MODEL_LEARNING.md` - 详细文档

## 总结

AONN Brain V3 现在完全符合自由能原理：

1. ✅ **可学习的生成模型**: Dynamics、Observation、Preference
2. ✅ **先验项**: PreferenceAspect 将目标转化为先验
3. ✅ **参数学习**: 通过自由能最小化学习参数
4. ✅ **动态演化**: 网络结构可以动态演化
5. ✅ **自我模型观察**: 可以观察学习过程

这为构建真正自适应的、符合自由能原理的 AI 系统奠定了基础。

