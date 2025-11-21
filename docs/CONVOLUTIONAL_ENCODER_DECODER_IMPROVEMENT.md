# 卷积编码器和解码器改进

## 概述

将 `EncoderAspect` 和 `ObservationAspect` 从简单的线性网络升级为卷积编码器和卷积解码器，以更好地处理 MNIST 图像数据。

## 改进内容

### 1. EncoderAspect（卷积编码器）

**之前**：简单的线性编码器
```python
nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
)
```

**现在**：卷积编码器（标准 VAE 架构）
```python
nn.Sequential(
    # 28x28 -> 14x14
    nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
    nn.ReLU(),
    # 14x14 -> 7x7
    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
    nn.ReLU(),
    # 7x7 -> 3x3
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
    nn.ReLU(),
    # Flatten: 128 * 3 * 3 = 1152
    nn.Flatten(),
    # 1152 -> 128
    nn.Linear(1152, 128),
)
```

**优势**：
- 更好地捕捉图像的空间结构
- 参数共享，减少参数量
- 更适合图像数据

### 2. ObservationAspect（卷积解码器）

**之前**：简单的线性解码器
```python
nn.Sequential(
    nn.Linear(128, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 784),
)
```

**现在**：卷积解码器（标准 VAE 架构）
```python
nn.Sequential(
    # 128 -> 1152
    nn.Linear(128, 128 * 3 * 3),
    nn.ReLU(),
    # Reshape: 1152 -> 128 x 3 x 3
    View((128, 3, 3)),
    # 3x3 -> 7x7
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
    nn.ReLU(),
    # 7x7 -> 14x14
    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
    nn.ReLU(),
    # 14x14 -> 28x28
    nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
    nn.Sigmoid(),  # 输出像素值在 [0, 1]
)
```

**优势**：
- 更好地生成图像的空间结构
- 使用转置卷积进行上采样
- 输出通过 Sigmoid 归一化到 [0, 1]

## 架构细节

### 编码器路径
```
输入: 1 x 28 x 28 (784 维向量，reshape 为图像)
  ↓ Conv2d(1->32)  28x28 -> 14x14
  ↓ Conv2d(32->64) 14x14 -> 7x7
  ↓ Conv2d(64->128) 7x7 -> 3x3
  ↓ Flatten: 128 * 3 * 3 = 1152
  ↓ Linear: 1152 -> 128
输出: 128 维向量
```

### 解码器路径
```
输入: 128 维向量
  ↓ Linear: 128 -> 1152
  ↓ Reshape: 1152 -> 128 x 3 x 3
  ↓ ConvTranspose2d(128->64) 3x3 -> 7x7
  ↓ ConvTranspose2d(64->32)  7x7 -> 14x14
  ↓ ConvTranspose2d(32->1)   14x14 -> 28x28
  ↓ Flatten: 1 x 28 x 28 -> 784
输出: 784 维向量 (28x28 图像)
```

## 参数统计

- **EncoderAspect**: 312,160 参数
- **ObservationAspect**: 313,057 参数
- **总计**: 625,217 参数

相比之前的线性网络（约 100,000 参数），参数量增加了，但这是合理的，因为：
1. 卷积网络更适合图像数据
2. 参数共享使得实际有效参数更少
3. 能够学习更复杂的特征表示

## 使用方式

在 `run_mnist_active_inference_v2.py` 中，已经自动启用卷积编码器和解码器：

```python
# EncoderAspect 自动使用卷积编码器
encoder = EncoderAspect(
    sensory_name="vision",
    internal_name="internal",
    input_dim=784,
    output_dim=128,
    use_conv=True,  # 启用卷积编码器
    image_size=28,  # MNIST 图像尺寸
)

# ObservationAspect 自动使用卷积解码器
observation_aspect = ObservationAspect(
    internal_name="internal",
    sensory_name="vision",
    state_dim=128,
    obs_dim=784,
    use_conv=True,  # 启用卷积解码器
    image_size=28,  # MNIST 图像尺寸
)
```

## 预期效果

1. **更好的特征提取**：卷积编码器能够更好地提取图像的空间特征
2. **更好的图像生成**：卷积解码器能够生成更真实的图像
3. **降低 F_obs**：由于生成模型更强大，观察生成误差应该降低
4. **提高准确率**：更好的特征表示应该提高分类准确率

## 兼容性

- 对于非图像数据（向量数据），自动回退到线性编码器/解码器
- 通过 `use_conv` 参数控制是否使用卷积
- 自动推断图像尺寸（如果未指定）

## 测试

所有测试已通过：
- ✅ EncoderAspect forward 和 free_energy_contrib
- ✅ ObservationAspect forward 和 free_energy_contrib
- ✅ 维度匹配验证
- ✅ 参数统计

## 下一步

1. 运行 MNIST 实验，验证改进效果
2. 对比线性网络和卷积网络的性能
3. 可能需要调整学习率（卷积网络可能需要不同的学习率）
4. 监控 F_obs 的变化，应该显著降低

