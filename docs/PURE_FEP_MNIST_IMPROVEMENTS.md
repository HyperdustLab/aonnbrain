# 纯 FEP MNIST 改进方案

## 问题诊断

### 核心问题：编码器和解码器无法收敛

在原始的纯 FEP MNIST 实验中，发现了以下问题：

1. **F_obs 占主导（98.4%）**：观察重建是主要挑战
2. **F_encoder 反而增加**：从 0.18 增加到 0.61，说明编码器没有收敛
3. **验证准确率低（50.40%）**：远低于理想值

### 根本原因

**学习循环中的矛盾**：

```python
# 步骤1：状态推理（优化 internal 状态）
internal = infer_states(...)  # 通过最小化 F_obs + F_encoder + F_pref 得到

# 步骤2：参数学习
F_encoder = ||internal - Encoder(vision)||²  # 编码器期望 internal = Encoder(vision)
F_obs = ||vision - Observation(internal)||²  # 解码器期望 vision = Observation(internal)
```

**问题**：
- `internal` 是通过多目标优化得到的，不是编码器的直接输出
- 如果 `F_obs` 占主导，`internal` 主要满足解码器的需求
- 编码器无法收敛，因为 `internal` 不是它的输出

## 改进方案

### 方案2：使用编码器输出作为 internal 的初始值

**核心思想**：在状态推理前，使用编码器输出初始化 `internal`，然后只进行少量迭代优化。

**实现**：
```python
# 使用编码器输出作为初始值
with torch.no_grad():
    internal_init = encoder.encoder(vision)
    fep_system.objects["internal"].set_state(internal_init.requires_grad_(True))

# 进行少量迭代优化（从 5 次减少到 2 次）
fep_system.infer_loop.infer_states(
    target_objects=("internal",),
    num_iters=2,  # 减少迭代次数
)
```

**优点**：
- 编码器输出占主导，确保编码器能够收敛
- 仍然允许状态推理进行微调，满足其他约束（如分类先验）

### 方案5：直接使用编码器输出（跳过状态推理）

**核心思想**：对于静态分类任务，可能不需要复杂的状态推理，直接使用编码器输出。

**实现**：
```python
# 跳过状态推理，直接使用编码器输出
with torch.no_grad():
    internal = encoder.encoder(vision)
    fep_system.objects["internal"].set_state(internal)
```

**优点**：
- 最简单直接
- 编码器能够完全收敛
- 适合静态分类任务

### 方案3：调整自由能权重

**核心思想**：降低 `F_obs` 的权重，提高 `F_pref` 的权重，平衡不同组件的学习。

**实现**：
```python
F_total = (
    0.1 * F_obs +      # 降低观察重建权重（从 1.0 降到 0.1）
    1.0 * F_encoder +  # 保持编码器权重
    10.0 * F_pref +    # 提高分类先验权重（从 1.0 提高到 10.0）
    F_dyn
)
```

**优点**：
- 平衡不同组件的学习
- 提高分类任务的重要性

### 方案4：使用分离优化器

**核心思想**：为不同组件使用不同的优化器和学习率。

**实现**：
```python
encoder_optimizer = Adam(encoder.parameters(), lr=0.001)
observation_optimizer = Adam(observation.parameters(), lr=0.0001)  # 更低的学习率
preference_optimizer = Adam(preference.parameters(), lr=0.01)  # 更高的学习率
classifier_optimizer = Adam(classifier.parameters(), lr=0.001)
```

**优点**：
- 针对不同组件使用不同的学习率
- 解码器使用更低的学习率，避免过度优化观察重建

## 改进版实验脚本

创建了 `scripts/run_pure_fep_mnist_improved.py`，包含以下改进：

1. **方案2**：使用编码器输出作为 `internal` 的初始值（默认启用）
2. **方案5**：提供选项跳过状态推理（`--skip-inference`）
3. **方案3**：调整自由能权重（`obs_weight=0.1`, `pref_weight=10.0`）
4. **方案4**：使用分离优化器，为不同组件设置不同学习率

### 使用方法

```bash
# 使用编码器初始化（默认）
python3 scripts/run_pure_fep_mnist_improved.py --steps 1000

# 跳过状态推理（直接使用编码器输出）
python3 scripts/run_pure_fep_mnist_improved.py --steps 1000 --skip-inference

# 长时训练
python3 scripts/run_pure_fep_mnist_improved.py --steps 60000 --output data/pure_fep_mnist_improved_60000steps.json
```

## 预期效果

1. **编码器收敛**：`F_encoder` 应该持续降低，而不是增加
2. **平衡的自由能**：`F_obs` 和 `F_encoder` 应该更平衡
3. **提高准确率**：验证准确率应该显著提高（目标 > 80%）

## 下一步

如果改进版实验效果良好，可以考虑：

1. **进一步优化**：调整学习率、权重等超参数
2. **移除 DynamicsAspect**：在静态分类任务中不需要状态转移模型
3. **改进网络架构**：使用更强大的编码器/解码器架构

