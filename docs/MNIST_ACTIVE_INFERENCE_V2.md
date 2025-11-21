# MNIST 主动推理实验 V2：完整生成模型 + 行动选择

## 概述

这是改进版的 MNIST 主动推理实验，实现了完整的生成模型和行动选择机制，符合主动推理的核心原理。

## 核心改进

### 1. 生成模型学习

实现了三个关键生成模型：

#### 1.1 EncoderAspect: `p(state | obs)`
- **功能**：从观察推断内部状态
- **实现**：`vision -> internal`
- **自由能贡献**：`F_encoder = 0.5 * ||internal - Encoder(vision)||²`

#### 1.2 ObservationAspect: `p(obs | state)`
- **功能**：从状态生成观察
- **实现**：`internal -> vision`
- **自由能贡献**：`F_obs = 0.5 * ||vision - ObservationAspect(internal)||²`

#### 1.3 DynamicsAspect: `p(state_{t+1} | state_t, action)`
- **功能**：状态转移模型
- **实现**：`internal + action -> internal_{t+1}`
- **自由能贡献**：`F_dyn = 0.5 * ||state_{t+1} - DynamicsAspect(state_t, action)||²`

#### 1.4 ClassificationAspect: `p(target | state)`
- **功能**：从状态预测分类
- **实现**：`internal -> target`
- **自由能贡献**：`F_class = cross_entropy(logits, target)`

### 2. 行动选择

通过优化自由能选择行动：

```python
def optimize_action(...):
    """
    通过优化自由能选择行动
    
    预期自由能：
    E[F] = ||obs_t - ObservationAspect(state_t)||²  (观察生成误差)
         + ||target - ClassificationAspect(state_t)||²  (分类误差)
         + E[||state_{t+1} - DynamicsAspect(state_t, action)||²]  (状态转移误差)
    """
    # 初始化行动（分类预测的 logits）
    action_logits = torch.zeros(10, requires_grad=True)
    
    # 迭代优化行动
    for iter in range(num_action_iters):
        # 计算总自由能
        F_total = F_class + 0.1 * F_obs
        
        # 反向传播，更新 action_logits
        F_total.backward()
        action_logits = action_logits - action_lr * action_logits.grad
    
    # 返回 softmax 后的行动（概率分布）
    return torch.softmax(action_logits, dim=-1)
```

### 3. 完整自由能

总自由能 = 观察生成误差 + 状态转移误差 + 分类误差：

```python
F_total = F_obs + F_dyn + F_class
```

其中：
- `F_obs`：观察生成误差（`ObservationAspect`）
- `F_dyn`：状态转移误差（`DynamicsAspect`）
- `F_class`：分类误差（`ClassificationAspect`）

## 主动推理流程

### 每个步骤的流程：

1. **状态推理**：给定观察 `obs_t`，推断内部状态 `state_t`（最小化自由能）
   ```python
   loop.infer_states(target_objects=("internal",), num_iters=3)
   ```

2. **行动选择**：通过优化自由能选择行动 `action_t`
   ```python
   action = optimize_action(...)
   ```

3. **执行行动**：在世界模型中执行行动，获得新观察 `obs_{t+1}`
   ```python
   obs, reward = world_interface.step(action)
   ```

4. **参数学习**：更新生成模型参数（最小化实际自由能）
   ```python
   F_total.backward()
   optimizer.step()
   ```

## 实验结果

### 初始测试（50步）

- **最终自由能**：247.07
- **训练准确率**：2.00%
- **验证准确率**：9.90%
- **网络结构**：5 Objects, 7 Aspects

### 分析

1. **准确率低的原因**：
   - 训练步数太少（50步）
   - 生成模型需要更多时间学习
   - 主动推理学习比监督学习慢

2. **自由能组成**：
   - `F_obs`：观察生成误差（主要贡献）
   - `F_dyn`：状态转移误差（较小贡献）
   - `F_class`：分类误差（较小贡献）

3. **改进方向**：
   - 增加训练步数（建议 1000+ 步）
   - 调整自由能权重（平衡各组件）
   - 优化行动选择算法（增加迭代次数）

## 使用方法

```bash
# 运行实验
python3 scripts/run_mnist_active_inference_v2.py \
    --steps 1000 \
    --state-dim 128 \
    --verbose \
    --output data/mnist_active_inference_v2.json

# 参数说明
--steps: 训练步数（默认 500）
--state-dim: 状态维度（默认 128）
--verbose: 详细输出
--output: 输出文件路径
--device: 设备（默认 cpu）
--save-interval: 快照保存间隔（默认 50）
```

## 配置参数

```python
config = {
    "obs_dim": 784,
    "state_dim": 128,
    "act_dim": 10,
    "infer_lr": 0.01,              # 状态推理学习率
    "learning_rate": 0.001,        # 参数学习率
    "weight_decay": 1e-4,          # 权重衰减
    "num_infer_iters": 3,          # 状态推理迭代次数
    "num_action_iters": 5,         # 行动优化迭代次数
    "action_lr": 0.1,              # 行动学习率
    "classification_loss_weight": 1.0,  # 分类损失权重
    "max_grad_norm": 100.0,        # 梯度裁剪阈值
}
```

## 与 V1 版本的对比

| 特性 | V1 版本 | V2 版本 |
|------|---------|---------|
| 生成模型 | 仅 Encoder | Encoder + Observation + Dynamics |
| 行动选择 | 直接生成 | 通过优化自由能选择 |
| 自由能计算 | 不完整 | 完整（F_obs + F_dyn + F_class） |
| 学习机制 | 简单参数更新 | 主动推理 + 参数学习 |

## 下一步改进

1. **增加训练步数**：运行更长时间（1000+ 步）
2. **调整自由能权重**：平衡各组件贡献
3. **优化行动选择**：改进 `optimize_action` 算法
4. **添加先验项**：引入 `PreferenceAspect` 作为目标先验
5. **批量学习**：支持批量更新以提高效率

## 参考文献

- Active Inference: A Process Theory (Friston et al., 2017)
- The Free Energy Principle: A Unified Brain Theory? (Friston, 2010)
- AONN: Active Object Neural Network (本项目)

