# MNIST AONN 数字识别演示指南

本指南介绍如何使用演示工具观察 AONN 如何识别手写数字，以及做了哪些 action。

## 工具说明

### `scripts/demo_mnist_simple.py`

这是一个演示工具，展示 AONN 识别 MNIST 手写数字的完整过程，包括：

1. **状态推理过程**：从观察推断内部状态
2. **Action 选择过程**：通过优化自由能选择 action
3. **预测演化**：预测结果如何随时间变化
4. **可视化输出**：生成详细的识别过程图表

## 使用方法

### 基本用法

```bash
# 识别3个样本并生成可视化
python3 scripts/demo_mnist_simple.py --num-samples 3 --output-dir data/recognition_demos
```

### 参数说明

- `--num-samples`: 要识别的样本数量（默认：10）
- `--output-dir`: 输出目录（默认：data/recognition_demos）
- `--state-dim`: 状态维度（默认：128）
- `--num-infer-iters`: 状态推理迭代次数（默认：10）
- `--num-action-iters`: Action选择迭代次数（默认：5）

### 示例

```bash
# 识别10个样本，使用更多迭代
python3 scripts/demo_mnist_simple.py \
    --num-samples 10 \
    --num-infer-iters 20 \
    --num-action-iters 10 \
    --output-dir data/my_demos
```

## 输出说明

### 可视化图表

每个识别样本会生成一个详细的图表，包含以下8个子图：

1. **输入图像**：显示原始 MNIST 手写数字图像和真实标签
2. **预测结果**：显示对10个数字的概率分布，高亮预测结果
3. **最终 Action**：显示 AONN 选择的 action（数字概率分布）
4. **Action 演化历史**：显示 action 在迭代过程中的变化
5. **自由能演化**：显示整个识别过程中自由能的变化
6. **Action 选择详情**：显示 action 选择过程中各自由能组件的变化
7. **预测演化**：显示预测类别和置信度在状态推理过程中的变化
8. **预测 vs Action 对比**：对比最终预测概率和 action 概率

### 控制台输出

```
样本 1/3: 真实标签 = 0
  预测: 6, 置信度: 10.7%
  Action选择: 6
  结果: ✗ 错误
```

## 识别过程详解

### 步骤1：状态推理（State Inference）

AONN 通过 `ActiveInferenceLoop` 从观察（图像）推断内部状态：

- **输入**：vision (784维图像)
- **处理**：通过 EncoderAspect 编码为 internal (128维)
- **迭代**：多次迭代优化 internal 状态，最小化自由能
- **输出**：优化后的 internal 状态

### 步骤2：Action 选择（Action Selection）

AONN 通过优化预期自由能选择 action：

- **目标**：最小化预期自由能 `E[F] = F_class + F_obs + F_dyn`
- **方法**：梯度下降优化 action logits
- **迭代**：多次迭代，逐步优化 action 概率分布
- **输出**：最终 action（10维概率分布，表示对每个数字的"选择"）

### Action 的含义

在 MNIST 识别任务中，action 表示：
- **分类预测**：AONN 对每个数字的"选择"概率
- **主动推理**：通过最小化自由能来选择最可能的数字
- **与预测的关系**：action 和预测通常应该一致，但可能略有不同（因为 action 考虑了更多因素）

## 观察要点

### 1. 预测准确性

- **正确识别**：预测标签 = 真实标签
- **置信度**：预测概率的最大值（越高越好）
- **错误分析**：观察错误识别的样本，分析原因

### 2. Action 选择

- **Action 演化**：观察 action 在迭代过程中如何变化
- **最终选择**：最终 action 选择的数字
- **与预测一致性**：action 和预测是否一致

### 3. 自由能变化

- **下降趋势**：自由能应该逐渐下降
- **组件分析**：观察 F_class, F_obs, F_dyn 的相对大小
- **收敛速度**：自由能下降的速度

### 4. 预测演化

- **稳定性**：预测是否稳定（不频繁变化）
- **收敛性**：预测是否收敛到最终结果
- **置信度变化**：置信度是否逐渐提高

## 使用训练好的模型

**注意**：当前演示工具使用的是未训练的模型，准确率会很低。

要使用训练好的模型，需要：

1. **保存模型权重**：在训练脚本中保存模型权重
2. **加载权重**：在演示工具中加载保存的权重
3. **使用相同配置**：确保演示工具使用与训练时相同的配置

### 示例：保存和加载权重

```python
# 在训练脚本中保存
torch.save({
    'encoder': encoder.state_dict(),
    'classification': classification_aspect.state_dict(),
    # ... 其他组件
}, 'model_weights.pth')

# 在演示工具中加载
checkpoint = torch.load('model_weights.pth')
encoder.load_state_dict(checkpoint['encoder'])
classification_aspect.load_state_dict(checkpoint['classification'])
```

## 常见问题

### Q: 为什么准确率很低？

**A**: 演示工具默认使用未训练的模型。要获得高准确率，需要使用训练好的模型权重。

### Q: Action 和预测有什么区别？

**A**: 
- **预测**：ClassificationAspect 直接从 internal 状态预测类别
- **Action**：通过优化预期自由能选择的"行动"，考虑了更多因素（观察生成、状态转移等）

### Q: 如何理解 Action 选择过程？

**A**: Action 选择是主动推理的核心：
1. 计算预期自由能（考虑分类、观察生成、状态转移）
2. 通过梯度下降优化 action
3. 选择最小化预期自由能的 action

### Q: 自由能组件的作用？

**A**:
- **F_class**：分类误差（主要目标）
- **F_obs**：观察生成误差（正则化）
- **F_dyn**：状态转移误差（正则化）

## 下一步

1. **改进模型保存/加载**：实现完整的模型权重保存和加载
2. **批量识别**：识别更多样本并统计准确率
3. **错误分析**：分析错误识别的样本特征
4. **参数调优**：调整迭代次数、学习率等参数

