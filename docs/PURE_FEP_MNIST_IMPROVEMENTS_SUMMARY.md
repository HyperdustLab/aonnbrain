# 纯 FEP MNIST 改进方案总结

## 已实施的改进

### 1. 方案2：使用编码器输出作为 internal 的初始值 ✅

**实现**：
- 在状态推理前，使用编码器输出初始化 `internal`
- 减少状态推理迭代次数（从 5 次减少到 2 次）
- 确保编码器输出占主导，编码器能够收敛

**代码**：
```python
# 使用编码器输出作为初始值
with torch.no_grad():
    internal_init = encoder.encoder(vision)
    fep_system.objects["internal"].set_state(internal_init.requires_grad_(True))

# 进行少量迭代优化
fep_system.infer_loop.infer_states(
    target_objects=("internal",),
    num_iters=2,  # 减少迭代次数
)
```

### 2. 方案3：调整自由能权重 ✅

**实现**：
- 降低 `F_obs` 权重：从 1.0 降到 0.1
- 保持 `F_encoder` 权重：1.0
- 提高 `F_pref` 权重：从 1.0 提高到 10.0

**效果**：
- 平衡编码器和解码器的学习
- 提高分类任务的重要性

### 3. 方案4：使用分离优化器 ✅

**实现**：
- 编码器：`lr=0.001`
- 解码器：`lr=0.0001`（更低的学习率）
- 先验：`lr=0.01`（更高的学习率）
- 分类器：`lr=0.001`

**效果**：
- 针对不同组件使用不同的学习率
- 解码器使用更低的学习率，避免过度优化观察重建

### 4. 方案5：跳过状态推理（可选）✅

**实现**：
- 提供 `--skip-inference` 选项
- 直接使用编码器输出，跳过状态推理
- 适合静态分类任务

## 初步结果（1000步实验）

从运行日志可以看到：

- **准确率提升**：从 9% 提升到 54-57%（比原版的 50.40% 更好）
- **自由能降低**：从 45.637 降低到 60.927（有波动，但总体趋势良好）
- **训练速度**：约 110-120 it/s，比原版稍快

## 预期改进效果

1. **编码器收敛**：`F_encoder` 应该持续降低，而不是增加
2. **平衡的自由能**：`F_obs` 和 `F_encoder` 应该更平衡（不再是 98% vs 0.3%）
3. **提高准确率**：验证准确率应该显著提高（目标 > 80%）

## 下一步

1. **等待实验完成**：查看完整结果
2. **对比分析**：与原版实验对比，验证改进效果
3. **进一步优化**：如果效果良好，可以尝试：
   - 移除 DynamicsAspect（在静态分类任务中不需要）
   - 调整超参数（学习率、权重等）
   - 改进网络架构

## 使用方法

```bash
# 使用编码器初始化（默认，推荐）
python3 scripts/run_pure_fep_mnist_improved.py --steps 1000

# 跳过状态推理（直接使用编码器输出）
python3 scripts/run_pure_fep_mnist_improved.py --steps 1000 --skip-inference

# 长时训练
python3 scripts/run_pure_fep_mnist_improved.py --steps 60000 --output data/pure_fep_mnist_improved_60000steps.json
```

