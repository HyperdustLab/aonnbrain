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

## 批量感官 Aspect 演化

AONN Brain V3 支持在演化阶段一次性生成一整组感官 `LinearGenerativeAspect`。通过在配置里添加 `evolution.batch_growth` 字段，可以精细控制“神经元”扩张节奏：

```python
"evolution": {
    "free_energy_threshold": 0.05,
    "max_aspects": 600,
    "error_ema_alpha": 0.35,
    "batch_growth": {
        "base": 16,              # 每次最少新增 16 个 Aspect
        "max_per_step": 96,      # 单步最多新增 96 个
        "max_total": 256,        # 每个感官的上限
        "min_per_sense": 4,      # 每个感官至少持有 4 个
        "error_threshold": 0.025,# EMA 误差低于该值时不增殖
        "error_multiplier": 1.0  # 可放大/缩小误差
    }
}
```

演化管理器会统计每个感官 Aspect 的自由能贡献，计算 EMA 后的误差，再调用批量创建接口。这样只需几十个演化步就能扩展到“数百个神经元级 Aspect”，同时仍由自由能驱动。

### 动作 Pipeline 深度控制

想要在几百步之后形成“深度管线”，可以启用 `pipeline_growth`：

```python
"pipeline_growth": {
    "initial_depth": 2,
    "initial_width": 24,
    "depth_increment": 1,
    "width_increment": 8,
    "max_stages": 3,
    "min_interval": 80,   # 至少间隔多少演化步再扩一层
    "free_energy_trigger": None,  # 设为 None 表示只按步数
    "max_depth": 6
}
```

默认会先创建 `internal → action` 的动作头，然后在满足间隔条件时在末端前插入新的 `internal → internal` Pipeline，自动形成“latent → latent → action”多级结构。脚本(`run_long_evolution.py` / `run_lineworm_experiment.py`)都会连续调用 `pipeline` 列表，从而真正得到深度处理链。

## 长周期演化实验

```bash
# 运行更长步数并提高任务难度
python scripts/run_long_evolution.py \
  --steps 200 500 1000 \
  --state-dim 256 \
  --action-dim 64 \
  --obs-dim 704 \
  --free-energy-threshold 0.1 \
  --state-noise 0.1 \
  --obs-noise 0.05 \
  --target-drift 0.02

# 绘制自由能曲线
python scripts/plot_long_run.py
```

- 输出：
  - `data/long_run_results.json`：记录各步数的演化日志、自由能、世界模型贡献
  - `data/long_run_free_energy.png`：自由能曲线
- 参数含义：
  - `state/action/obs dim`：控制内部状态与多模态感官维度（默认拆成 vision/olfactory/proprio）
  - `free-energy-threshold`：越小越容易触发新结构
  - `state/obs noise`：提高真实世界噪声
  - `target-drift`：目标状态漂移，模拟动态偏好

### 百级神经元演化记录

- 命令示例：

  ```bash
  python scripts/run_long_evolution.py \
    --steps 60 \
    --state-dim 64 \
    --action-dim 16 \
    --obs-dim 48 \
    --free-energy-threshold 0.05 \
    --state-noise 0.05 \
    --obs-noise 0.03 \
    --target-drift 0.02 \
    --output data/test_long_small.json
  ```

- 稳定示例：`python scripts/run_long_evolution.py --steps 400 ... --output data/long_run_400.json`
  - 400 步后 `293` 个 Aspect（3 条 Pipeline，包含 2 条 latent pipeline + 终端动作头）
  - 自由能降至 `19.47`，结构长期停留在 280~310 区间，满足“数百个神经元且稳定”的预期
  - 快照保存在 `data/long_run_400.json`，可用 `scripts/plot_long_run.py --input data/long_run_400.json` 绘制自由能曲线

## 线虫级世界模型实验

```bash
python scripts/run_lineworm_experiment.py \
  --steps 200 500 1000 \
  --device cpu \
  --output data/lineworm_results.json
```

- 世界模型：包含化学梯度、温度场和触觉障碍物，动作是推进和转向；默认感官维度 `{"chemo":128,"thermo":32,"touch":64}`。  
- AONN 初始结构仅包含 `internal + 感官 + action`；默认阈值 0.045、`max_aspects=420`，借助 `batch_growth` 自动停留在 350±30 区间。若需要更激进的增长，可调大 `batch_growth.max_total` 与 `max_aspects`。  
- 线虫环境噪声更大，脚本默认将主动推理学习率降到 `infer_lr=0.02`，若仍出现 `NaN` 可进一步降低或减少演化步数。  
- `LineWormWorldModel` 现在提供真实感的化学/温度波动：内部使用时间相关（Ornstein-Uhlenbeck）噪声和高斯平滑的空间相关噪声来驱动场强变化，可通过 `noise_config` 覆盖默认的 `std/temporal_corr/spatial_scale/basis_size` 等参数；想进一步增加难度，可以把 `scripts/run_lineworm_experiment.py` 里的 `noise_config` 放大或设置 `max_aspects=280`、`pipeline_growth.max_stages=6` 等，推动动作管线在 400 步后仍持续扩张。  
- 输出：
  - `data/lineworm_results.json`：记录 200/500/1000 步演化过程
  - 可使用 `scripts/plot_long_run.py --input data/lineworm_results.json --output data/lineworm_free_energy.png` 绘制自由能曲线

- 观察指标：
  - 是否出现新的感官/动作 Pipeline
  - 感官对象的自由能贡献
  - 结构是否向线虫脑的多通路拓扑演化

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

