# MNIST AONN 观察指南

本指南介绍如何观察和监控 MNIST AONN 的工作情况。

## 工具概览

### 1. 可视化工具 (`scripts/visualize_mnist_aonn.py`)

用于生成实验结果的可视化图表。

#### 基本用法

```bash
# 生成所有图表
python3 scripts/visualize_mnist_aonn.py --input data/mnist_evolution_1000steps_relaxed_prune.json --all

# 只生成特定图表
python3 scripts/visualize_mnist_aonn.py --input <结果文件> --free-energy --accuracy --structure --events

# 打印实验摘要
python3 scripts/visualize_mnist_aonn.py --input <结果文件> --summary
```

#### 生成的图表

1. **自由能演化图** (`free_energy_evolution.png`)
   - 总自由能曲线
   - 自由能组件（F_obs, F_dyn, F_class）
   - 移动平均线

2. **准确率演化图** (`accuracy_evolution.png`)
   - 分类准确率随时间变化
   - 初始/最终/最高准确率统计

3. **网络结构演化图** (`network_structure_evolution.png`)
   - Aspects 和 Objects 数量变化
   - Pipelines 数量变化

4. **演化事件时间线** (`evolution_events_timeline.png`)
   - 各种演化事件（add_pipeline, prune等）的时间分布
   - 事件类型统计

### 2. 实时监控工具 (`scripts/monitor_mnist_aonn.py`)

用于实时监控正在运行的实验。

#### 基本用法

```bash
# 监控实验（默认每2秒更新）
python3 scripts/monitor_mnist_aonn.py --json data/mnist_evolution_test.json --log data/mnist_evolution.log

# 自定义更新间隔
python3 scripts/monitor_mnist_aonn.py --json <结果文件> --log <日志文件> --interval 1.0

# 同时显示日志输出
python3 scripts/monitor_mnist_aonn.py --json <结果文件> --log <日志文件> --show-log
```

#### 监控信息

- 当前步数
- 自由能
- 准确率
- 网络结构（Objects, Aspects, Pipelines）
- 最新日志（可选）

### 3. 分析工具 (`scripts/analyze_mnist_v2_results.py`)

详细分析实验结果。

```bash
python3 scripts/analyze_mnist_v2_results.py --input data/mnist_evolution_test.json
```

## 观察要点

### 1. 自由能变化

**正常情况：**
- 自由能应该逐渐下降
- 下降速度在初期较快，后期变慢
- 最终自由能应该比初始值低很多

**异常情况：**
- 自由能持续上升：可能学习率过高或网络结构有问题
- 自由能波动剧烈：可能梯度爆炸或数值不稳定
- 自由能停滞：可能需要调整演化参数

### 2. 准确率变化

**正常情况：**
- 准确率应该逐渐提升
- 训练准确率可能高于验证准确率（正常过拟合）
- 准确率应该稳定在某个水平

**异常情况：**
- 准确率始终很低（<20%）：可能网络结构不足或学习率过低
- 准确率突然下降：可能发生了网络结构变化（剪枝）
- 准确率波动大：可能样本选择或评估方法有问题

### 3. 网络结构演化

**正常情况：**
- Aspects 数量应该逐渐增加（通过演化）
- Pipelines 应该被创建并增长
- 结构变化应该与自由能降低相关

**异常情况：**
- Aspects 数量爆炸式增长：可能演化条件太宽松
- 没有演化发生：可能演化阈值设置过高
- 频繁剪枝：可能需要放宽剪枝条件

### 4. 演化事件

**观察重点：**
- `add_pipeline`: 创建新的 Pipeline（表示学习增强）
- `prune`: 剪枝操作（移除低效 Aspects）
- `add_sensory_aspect`: 添加感官处理 Aspects

**正常模式：**
- 初期：频繁创建新结构（add_pipeline, add_sensory_aspect）
- 中期：结构稳定，主要进行参数学习
- 后期：偶尔剪枝优化结构

## 典型工作流程

### 1. 运行实验

```bash
# 运行实验并保存结果
python3 scripts/run_mnist_active_inference_v2.py \
    --steps 1000 \
    --verbose \
    --output data/mnist_experiment.json \
    --save-interval 50 \
    2>&1 | tee data/mnist_experiment.log
```

### 2. 实时监控（另一个终端）

```bash
# 在另一个终端运行监控
python3 scripts/monitor_mnist_aonn.py \
    --json data/mnist_experiment.json \
    --log data/mnist_experiment.log \
    --interval 2.0 \
    --show-log
```

### 3. 实验完成后分析

```bash
# 生成所有可视化图表
python3 scripts/visualize_mnist_aonn.py \
    --input data/mnist_experiment.json \
    --all

# 查看图表
open data/plots/free_energy_evolution.png
open data/plots/accuracy_evolution.png
open data/plots/network_structure_evolution.png
open data/plots/evolution_events_timeline.png
```

## 关键指标解读

### 自由能组件

- **F_obs**: 观察生成误差（越小越好）
- **F_dyn**: 状态转移误差（越小越好）
- **F_class**: 分类误差（越小越好，这是MNIST的主要目标）

### 网络结构指标

- **Aspects 数量**: 计算单元数量，反映网络复杂度
- **Pipelines 数量**: 深度结构数量
- **Pipeline 深度**: 每层 Pipeline 的层数（深度）
- **Pipeline 宽度**: 每层 Pipeline 的 Aspects 数量（宽度）

### 演化决策

- **add_pipeline**: 创建新 Pipeline（通常降低自由能）
- **prune**: 剪枝（移除低效组件）
- **no_change**: 不进行演化（当前结构最优）

## 常见问题排查

### Q: 自由能不下降？

**可能原因：**
1. 学习率过低
2. 网络结构不足
3. 演化阈值过高

**解决方案：**
- 检查学习率设置
- 降低演化阈值
- 增加初始网络结构

### Q: 准确率很低？

**可能原因：**
1. 网络结构不足
2. 学习率设置不当
3. 自由能权重不平衡

**解决方案：**
- 检查 `ClassificationAspect` 是否正常工作
- 调整 `classification_loss_weight`
- 增加网络复杂度

### Q: 网络结构不演化？

**可能原因：**
1. 演化阈值过高
2. 自由能已经很低
3. 达到最大结构限制

**解决方案：**
- 降低 `free_energy_threshold`
- 检查 `max_aspects` 设置
- 增加任务复杂度

## 最佳实践

1. **实验前**：设置合理的参数，确保初始网络能工作
2. **实验中**：使用监控工具实时观察，及时发现问题
3. **实验后**：使用可视化工具全面分析，找出改进方向
4. **对比实验**：保存不同参数配置的结果，进行对比分析

## 示例：完整观察流程

```bash
# 1. 运行实验（后台）
python3 scripts/run_mnist_active_inference_v2.py \
    --steps 1000 \
    --output data/exp1.json \
    2>&1 | tee data/exp1.log &

# 2. 实时监控
python3 scripts/monitor_mnist_aonn.py \
    --json data/exp1.json \
    --log data/exp1.log

# 3. 实验完成后生成报告
python3 scripts/visualize_mnist_aonn.py \
    --input data/exp1.json \
    --all

# 4. 查看结果
ls -lh data/plots/*.png
```

## 相关文件

- `scripts/run_mnist_active_inference_v2.py`: 主实验脚本
- `scripts/visualize_mnist_aonn.py`: 可视化工具
- `scripts/monitor_mnist_aonn.py`: 实时监控工具
- `scripts/analyze_mnist_v2_results.py`: 详细分析工具
- `data/plots/`: 图表输出目录

