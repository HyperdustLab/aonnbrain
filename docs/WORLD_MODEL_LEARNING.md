# 世界模型学习文档

## 概述

AONN Brain V3 实现了符合自由能原理的世界模型学习机制。世界模型不再是固定的环境模拟器，而是 AONN 内部可学习的生成模型。

## 三点改进

### 1. 在 AONN 内部建立可学习的生成模型

实现了三个世界模型 Aspect：

#### DynamicsAspect（状态转移模型）
```python
# 学习：p(s_{t+1} | s_t, a_t)
internal_t + action → internal_{t+1}
```

- **功能**: 预测下一状态
- **自由能**: F = 0.5 * ||internal_{t+1} - pred(internal_t, action)||²
- **参数**: 可学习的神经网络（Linear → ReLU → Linear）

#### ObservationAspect（观察生成模型）
```python
# 学习：p(o_t | s_t)
internal → vision/olfactory/proprio
```

- **功能**: 从状态生成多模态观察（视觉、嗅觉、躯体感知）
- **自由能**: F = 0.5 * ||sensory - pred(internal)||²
- **参数**: 可学习的神经网络
- **扩展**: 支持任意感官组合。通过配置 `sense_dims`，可以变为 `chemo/thermo/touch` 等线虫级通道

#### PreferenceAspect（偏好/目标模型）
```python
# 先验项：p(internal | target)
internal → target
```

- **功能**: 将目标状态转化为先验项
- **自由能**: F = 0.5 * weight * ||internal - target||²
- **参数**: 可学习的目标状态

### 2. 将目标/奖励转化为先验项

#### PreferenceAspect 作为先验

```python
# 自由能 = 预测误差 + 先验项
F_total = F_dynamics + F_observation + F_preference

# PreferenceAspect 提供先验项
F_preference = 0.5 * ||internal - target||²
```

这相当于：
```
-log p(internal | target) ≈ 0.5 * ||internal - target||²
```

#### 目标状态设置

```python
# 从世界模型获取目标状态
target_state = world_model.get_true_state()

# 设置到 PreferenceAspect
brain.set_world_model_target(target_state)
```

### 3. 参数学习机制

#### 自由能最小化学习

```python
def learn_world_model(
    observation,      # 当前观察
    action,          # 执行的动作
    next_observation, # 下一观察
    target_state,    # 目标状态
):
    # 1. 设置 Object 状态
    objects["sensory"].set_state(observation)
    objects["action"].set_state(action)
    
    # 2. 预测下一状态
    pred_next = dynamics_aspect.predict_next_state(internal, action)
    
    # 3. 计算自由能（只计算世界模型 Aspect）
    F = F_dynamics + F_observation + F_preference
    
    # 4. 反向传播更新参数
    F.backward()
    optimizer.step()
```

## 完整的学习循环

```
for step in range(num_steps):
    # 1. 获取观察
    obs = world_model.get_observation()
    
    # 2. 网络演化
    brain.evolve_network(obs)
    
    # 3. 主动推理（状态更新）
    infer_states()
    
    # 4. 生成动作
    action = generate_action()
    
    # 5. 执行动作，获取下一观察
    next_obs, reward = world_model.step(action)
    
    # 6. 学习世界模型（参数更新）
    target_state = world_model.get_true_state()
    brain.learn_world_model(obs, action, next_obs, target_state)
```

## 自由能组成

### 总自由能

```
F_total = F_dynamics + F_observation + F_preference + F_other
```

### 各组件贡献

1. **F_dynamics**: 状态转移误差
   - 衡量预测的下一状态与实际的差异
   - 学习真实世界的动力学

2. **F_observation**: 观察生成误差
   - 衡量从状态预测的观察与实际的差异
   - 学习观察模型

3. **F_preference**: 先验项
   - 衡量当前状态与目标状态的差异
   - 提供偏好/目标约束

## 实验结果

从演示运行可以看到：

### 自由能下降

```
初始自由能: 4.2483
步骤 5:  0.4249
步骤 10: 0.3470
步骤 15: 0.3005
步骤 20: 0.2583
步骤 25: 0.2065
```

### 世界模型自由能贡献下降

```
步骤 5:  0.3336
步骤 10: 0.2592
步骤 15: 0.2131
步骤 20: 0.1696
步骤 25: 0.1268
```

这表明世界模型正在学习，预测误差在减小。

## 与自由能原理的对应

### Friston 的 Active Inference 框架

```
F = -log p(obs|model) - log p(model)
  = 预测误差 + 先验项
```

### AONN 的实现

```
F = F_dynamics + F_observation + F_preference
  = ||pred_next - actual_next||² + ||pred_obs - actual_obs||² + ||state - target||²
```

- **F_dynamics + F_observation**: 对应 -log p(obs|model)（生成模型）
- **F_preference**: 对应 -log p(model)（先验）

## 关键特性

1. ✅ **可学习的生成模型**: Dynamics、Observation、Preference 都是可学习的
2. ✅ **先验项**: PreferenceAspect 将目标转化为先验
3. ✅ **自由能最小化**: 通过梯度下降学习参数
4. ✅ **动态演化**: 网络结构可以动态演化
5. ✅ **自我模型观察**: 可以观察学习过程

## 使用示例

```python
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.models.world_model import SimpleWorldModel, WorldModelInterface

# 配置（启用世界模型学习）
config = {
    "obs_dim": 16,
    "state_dim": 32,
    "act_dim": 8,
    "enable_world_model_learning": True,  # 关键！
}

# 创建 brain
brain = AONNBrainV3(config=config, enable_evolution=True)

# 创建世界模型
world_model = SimpleWorldModel(...)
world_interface = WorldModelInterface(world_model)

# 学习循环
for step in range(num_steps):
    obs = world_interface.get_observation()
    action = generate_action()
    next_obs, reward = world_interface.step(action)
    
    # 学习世界模型
    target_state = world_model.get_true_state()
    brain.learn_world_model(obs, action, next_obs, target_state)
```

## 批量 Aspect 演化实践

为了让 AONN 在几十/几百步内生成“数百个神经元”并自动停止在目标规模，可在配置中添加 `batch_growth`：

```python
"evolution": {
    "free_energy_threshold": 0.04,
    "max_aspects": 360,
    "error_ema_alpha": 0.4,
    "batch_growth": {
        "base": 8,
        "max_per_step": 32,
        "max_total": 96,
        "min_per_sense": 4,
        "error_threshold": 0.04,
        "error_multiplier": 0.7
    }
}
```

演化循环会：
1. 统计各感官 `LinearGenerativeAspect` 的自由能贡献；
2. 通过 EMA 平滑高误差感官；
3. 一次性创建 `base * error_ratio` 个新 Aspect，直到达到 `max_per_step` 或 `max_total`。

### Pipeline 深度扩展

为获得 “latent → latent → action” 的多级动作路径，可在根配置添加：

```python
"pipeline_growth": {
    "initial_depth": 2,
    "initial_width": 24,
    "depth_increment": 1,
    "width_increment": 8,
    "max_stages": 3,
    "min_interval": 80,
    "free_energy_trigger": None,
    "max_depth": 6
}
```

网络会先创建 `internal → action` 的动作头，然后每经过 `min_interval` 步就在末端前插入新的 `internal → internal` Pipeline，使动作生成链条越变越深。

### 300 个神经元级 Aspect（稳定）

```
python scripts/run_long_evolution.py \
  --steps 400 \
  --state-dim 64 --action-dim 16 --obs-dim 48 \
  --free-energy-threshold 0.04 \
  --state-noise 0.05 --obs-noise 0.03 --target-drift 0.02 \
  --output data/long_run_400.json
```

- 400 步后拥有 `293` 个 Aspect（绝大多数是感官神经元），并自动形成 3 条 Pipeline（2 条 latent + 终端动作头）
- 自由能降低到 `19.47`，对象范数维持在 `[0.16, 0.68]`
- 结果保存在 `data/long_run_400.json`，可作为“稳定 300 神经元 + 深度 pipeline”案例

### 线虫模式（多模态 + 350 神经元）

`scripts/run_lineworm_experiment.py` 默认：

- `max_aspects=420`，`batch_growth.max_total=120`，让感官神经元稳定在 350±30
- `pipeline_growth` 允许最多 4 条管线，初始深度 3
- `ActiveInferenceLoop(infer_lr=0.02)` 以减少高噪声世界带来的数值发散
- `LineWormWorldModel(noise_config=...)` 使用时间相关 + 空间相关噪声（默认 `chemo_std=0.04, temporal_corr=0.97, spatial_scale=3.0`；`thermo_std=0.03, temporal_corr=0.95, spatial_scale=4.5`），模拟真实实验中化学/温度场的波动。可以为 `noise_config["chemo" or "thermo"]` 提供 `std/temporal_corr/spatial_scale/basis_size/amplitude` 以贴近不同环境。

若仍观察到 `NaN`，可进一步降低 `infer_lr` 或缩短演化步数，尤其是在 `LineWormWorldModel` 的高噪声场景中。

## 脚本与场景

- `scripts/run_long_evolution.py`：适用于果蝇/高复杂度设置，可配置状态/动作/观察维度以及噪声、目标漂移
- `scripts/run_lineworm_experiment.py`：线虫世界模型（化学梯度、温度场、触觉障碍），观察是否自动演化出多感官 → 中枢 → 动作的结构
- `scripts/plot_long_run.py`：从任意结果 JSON 中绘制自由能曲线

## 总结

AONN Brain V3 现在完全符合自由能原理：

1. ✅ **可学习的生成模型**: Dynamics、Observation、Preference Aspect
2. ✅ **先验项**: PreferenceAspect 将目标转化为先验
3. ✅ **参数学习**: 通过自由能最小化学习参数
4. ✅ **动态演化**: 网络结构可以动态演化
5. ✅ **自我模型观察**: 可以观察学习过程

这为构建真正自适应的、符合自由能原理的 AI 系统奠定了基础。

