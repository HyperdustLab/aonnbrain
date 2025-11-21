# MNIST 过拟合问题分析

## 问题描述

训练准确率：**100%**  
验证准确率：**12.5%**  
差异：**87.5%**

这是典型的**严重过拟合**问题。

## 原因分析

### 1. 训练准确率计算方式

```python
# 当前代码
avg_acc = sum(accuracy_history[-100:]) / min(100, len(accuracy_history))
```

**问题**：
- 只计算最近 100 步的平均准确率
- 如果最后 100 步都正确，准确率就是 100%
- 但这可能只是模型记住了最近看到的样本，而不是真正学会了泛化

### 2. 验证准确率计算

```python
# 当前代码
num_samples=min(1000, len(val_world.dataset))  # 只用 1000 个样本
```

**问题**：
- 验证样本数太少（1000 个，而测试集有 10000 个）
- 12.5% 意味着在 1000 个样本中只对了 125 个
- 说明模型完全没有泛化能力

### 3. 学习率可能太高

```python
"learning_rate": 0.001,  # 可能太高
```

**问题**：
- 学习率 0.001 可能导致参数更新太快
- 模型快速适应训练样本，但无法泛化到新样本

### 4. 没有正则化

**问题**：
- 没有权重衰减（weight decay）
- 没有 dropout
- 没有其他正则化技术

### 5. 状态推理问题

**问题**：
- 训练时：使用主动推理更新 `internal` 状态，然后学习参数
- 验证时：也使用主动推理更新 `internal` 状态，但参数不更新
- 如果训练时的状态被过度优化到特定样本，验证时可能无法正确推理

### 6. 样本顺序问题

**问题**：
- 训练时使用 `step()` 顺序遍历数据集
- 模型可能记住了样本的顺序模式，而不是真正的特征

## 改进建议

### 1. 降低学习率

```python
"learning_rate": 0.0001,  # 降低 10 倍
```

### 2. 添加权重衰减

```python
aspect_optimizer = Adam(
    aspect_params,
    lr=config.get("learning_rate", 0.0001),
    weight_decay=1e-4,  # 添加权重衰减
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

### 3. 改进训练准确率计算

```python
# 计算整个训练集的准确率（定期评估）
if (step + 1) % 1000 == 0:
    train_acc = evaluate_on_dataset(brain, train_interface, num_samples=1000)
```

### 4. 增加验证样本数

```python
num_samples=min(10000, len(val_world.dataset))  # 使用全部测试集
```

### 5. 减少验证时的推理迭代次数

```python
# 验证时使用更少的推理迭代
loop.infer_states(
    target_objects=("internal",),
    num_iters=1,  # 从 3 降到 1
    sanitize_callback=brain.sanitize_states
)
```

### 6. 添加早停机制

```python
# 如果验证准确率不再提升，停止训练
if val_acc < best_val_acc:
    patience += 1
    if patience > 10:
        break
```

### 7. 使用随机采样

```python
# 训练时随机选择样本，而不是顺序遍历
self.current_idx = torch.randint(0, len(self.dataset), (1,)).item()
```

## 预期改进效果

实施这些改进后，预期：
- 训练准确率：从 100% 降到 80-90%（更真实）
- 验证准确率：从 12.5% 提升到 60-80%（更好的泛化）
- 差异：从 87.5% 降到 10-20%（更合理）

