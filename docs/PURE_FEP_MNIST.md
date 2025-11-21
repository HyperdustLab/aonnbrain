# 纯 FEP MNIST 实验

## 概述

这是一个纯自由能原理（Free Energy Principle, FEP）的 MNIST 手写数字识别实验，**不使用 AONN 框架**，只使用生成模型和主动推理学习。

## 核心设计

### 1. 不使用 AONN Brain

- 直接管理 `ObjectNode` 字典
- 不使用 `AONNBrainV3` 的演化机制
- 简化为纯生成模型 + 主动推理

### 2. 生成模型组件

#### 2.1 EncoderAspect: `p(state | obs)`
- **功能**：从观察推断内部状态
- **实现**：`vision -> internal`
- **自由能贡献**：`F_encoder = 0.5 * ||internal - Encoder(vision)||²`

#### 2.2 ObservationAspect: `p(obs | state)`
- **功能**：从状态生成观察
- **实现**：`internal -> vision`
- **自由能贡献**：`F_obs = 0.5 * ||vision - Observation(internal)||²`

#### 2.3 DynamicsAspect: `p(state_{t+1} | state_t, action)`
- **功能**：状态转移模型
- **实现**：`internal + action -> internal_{t+1}`
- **自由能贡献**：`F_dyn = 0.5 * ||state_{t+1} - Dynamics(state_t, action)||²`

#### 2.4 PreferenceAspect: `p(target | state)`
- **功能**：将分类目标转化为先验约束
- **实现**：`internal -> target`
- **自由能贡献**：`F_pref = 0.5 * weight * ||internal - target_state||²`
- **作用**：在纯 FEP 中，分类目标通过先验约束来引导学习

### 3. 分类实现

在纯 FEP 中，分类通过以下方式实现：

1. **PreferenceAspect**：将分类目标（one-hot 标签）转化为先验约束，引导 `internal` 状态学习分类相关的表示
2. **独立分类器**：用于评估准确率，不参与自由能计算，但参与参数学习

### 4. 主动推理学习

#### 4.1 状态推理
```python
# 给定观察，推断 internal 状态（最小化自由能）
fep_system.infer_loop.infer_states(
    target_objects=("internal",),
    num_iters=5,
    sanitize_callback=fep_system.sanitize_states,
)
```

#### 4.2 行动选择
```python
# 通过优化自由能选择 action（分类预测）
action_logits = fep_system.objects["action"].state.clone().detach().requires_grad_(True)
for action_iter in range(num_action_iters):
    F = fep_system.compute_free_energy()
    F.backward()
    action_logits = action_logits - action_lr * action_logits.grad
action = torch.softmax(action_logits, dim=-1)
```

#### 4.3 参数学习
```python
# 更新生成模型参数（最小化自由能）
F_total = F_obs + F_encoder + F_dyn + F_pref
F_class = cross_entropy(classifier(internal), target)
total_loss = F_total + classification_weight * F_class
total_loss.backward()
optimizer.step()
```

## 完整自由能

总自由能 = 观察生成误差 + 编码误差 + 状态转移误差 + 先验约束：

```
F_total = F_obs + F_encoder + F_dyn + F_pref
```

其中：
- `F_obs`：观察生成误差（`ObservationAspect`）
- `F_encoder`：编码误差（`EncoderAspect`）
- `F_dyn`：状态转移误差（`DynamicsAspect`）
- `F_pref`：先验约束（`PreferenceAspect`）

## 使用方法

### 基本运行

```bash
# 运行 1000 步实验（使用卷积编码器/解码器）
python3 scripts/run_pure_fep_mnist.py --steps 1000

# 使用线性编码器/解码器
python3 scripts/run_pure_fep_mnist.py --steps 1000 --no-conv

# 详细输出
python3 scripts/run_pure_fep_mnist.py --steps 1000 --verbose

# 指定状态维度
python3 scripts/run_pure_fep_mnist.py --steps 1000 --state-dim 256

# 使用 GPU
python3 scripts/run_pure_fep_mnist.py --steps 1000 --device cuda
```

### 参数说明

- `--steps`: 训练步数（默认：1000）
- `--state-dim`: 内部状态维度（默认：128）
- `--verbose`: 详细输出
- `--output`: 输出文件路径（默认：`data/pure_fep_mnist.json`）
- `--device`: 设备（默认：`cpu`）
- `--save-interval`: 快照保存间隔（默认：100）
- `--no-conv`: 不使用卷积编码器/解码器

## 实验流程

每个训练步骤：

1. **设置观察**：从世界模型获取图像观察
2. **设置目标**：设置 one-hot 标签
3. **状态推理**：通过最小化自由能推断 `internal` 状态
4. **行动选择**：通过优化自由能选择 `action`（分类预测）
5. **执行行动**：在世界模型中执行行动，获取新观察
6. **参数学习**：更新生成模型参数（最小化自由能 + 分类损失）

## 与 AONN 版本的区别

| 特性 | AONN 版本 | 纯 FEP 版本 |
|------|-----------|-------------|
| 框架 | 使用 AONN Brain | 直接管理 Objects |
| 演化机制 | 支持网络演化 | 不支持演化 |
| 分类实现 | ClassificationAspect | PreferenceAspect + 独立分类器 |
| 复杂度 | 较高 | 较低 |
| 灵活性 | 支持动态结构 | 固定结构 |

## 预期效果

- **目标**：实现高识别率的 MNIST FEP 系统
- **方法**：通过生成模型和主动推理学习，不使用显式分类器
- **优势**：更符合 FEP 原理，结构更简洁

## 结果分析

实验结果保存在 JSON 文件中，包含：
- `free_energy_history`: 自由能历史
- `F_obs_history`: 观察生成误差历史
- `F_encoder_history`: 编码误差历史
- `F_dyn_history`: 状态转移误差历史
- `F_pref_history`: 先验约束历史
- `accuracy_history`: 准确率历史
- `snapshots`: 定期快照

## 改进方向

1. **优化 PreferenceAspect**：改进先验约束的实现方式
2. **调整权重**：平衡不同自由能组件的权重
3. **增加训练步数**：观察长期学习效果
4. **超参数调优**：优化学习率、推理迭代次数等

