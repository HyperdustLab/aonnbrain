# 长时间演化实验参数配置

本文档详细说明长时间演化实验（8小时）的所有参数配置。

## 一、运行参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **默认步数** | 15,000 步 | 约 8 小时（假设每步 2 秒） |
| **保存间隔** | 100 步 | 每 100 步保存一次快照 |
| **检查点间隔** | 1,000 步 | 每 1,000 步自动保存检查点 |
| **LLM** | Ollama cogito:32b | 默认使用本地 Ollama 模型（LLMAspect 已启用） |
| **设备** | CPU | 默认使用 CPU（可改为 GPU） |

## 二、世界模型配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **状态维度** | 1,024 | 内部状态维度 |
| **动作维度** | 256 | 动作空间维度 |
| **观察维度** | 1,408 | 总观察维度 |
| **语义维度** | 512 | 语义状态维度 |

### 感官维度分解

- **vision**: 512 维
- **language**: 512 维
- **audio**: 128 维
- **multimodal**: 256 维

## 三、网络演化配置

### 3.1 基本限制

| 参数 | 值 | 说明 |
|------|-----|------|
| **max_aspects** | 10,000 | Aspect 数量上限（大幅提升） |
| **max_objects** | 100 | Object 数量上限 |
| **free_energy_threshold** | 0.1 | 自由能阈值（触发演化） |
| **prune_threshold** | 0.01 | 剪枝阈值（低于此值会被剪枝） |

### 3.2 批量增长配置（新建 Aspect 速率）

| 参数 | 值 | 说明 | 变化 |
|------|-----|------|------|
| **base** | 2 | 基础批量 | 8 → 2（减少 75%） |
| **max_per_step** | 8 | 每步最大创建数量 | 32 → 8（减少 75%） |
| **max_total** | 2,500 | 每个感官上限 | 150 → 2,500（提升） |
| **min_per_sense** | 4 | 每个感官最小数量 | 8 → 4（减少） |
| **error_threshold** | 0.15 | 错误阈值 | 0.1 → 0.15（提高） |
| **error_multiplier** | 0.5 | 错误倍数 | 0.7 → 0.5（降低） |

**设计理念**: 大幅降低新建速率，确保每个 Aspect 有充分时间学习后再创建新的。

## 四、Pipeline 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **enable** | True | 启用 Pipeline 增长 |
| **initial_depth** | 3 | 初始深度 |
| **initial_width** | 32 | 初始宽度 |
| **depth_increment** | 1 | 深度增量 |
| **width_increment** | 8 | 宽度增量 |
| **max_stages** | 100 | 最大 Pipeline 数量 |
| **min_interval** | 1 步 | 最小扩展间隔 |
| **free_energy_trigger** | 0.1 | 自由能触发阈值 |
| **max_depth** | 12 | 最大深度 |

## 五、学习参数

| 参数 | 值 | 说明 | 变化 |
|------|-----|------|------|
| **infer_lr** | 0.005 | 推理学习率 | 0.01 → 0.005（降低 50%） |
| **learning_rate** | 0.005 | 世界模型学习率 | 0.01 → 0.005（降低 50%） |
| **num_infer_iters** | 3 | 推理迭代次数 | 2 → 3（增加） |
| **max_grad_norm** | 100.0 | 梯度裁剪阈值 | 268.0 → 100.0（更保守） |
| **state_clip_value** | 5.0 | 状态裁剪值 | 保持不变 |

**设计理念**: 降低学习率，增加迭代次数，确保充分学习和稳定更新。

## 六、LLM 配置（LLMAspect）

| 参数 | 值 | 说明 |
|------|-----|------|
| **默认模型** | Ollama cogito:32b | 本地 Ollama 模型 |
| **启用状态** | **已启用** | LLMAspect 自动创建并添加到网络 |
| **连接** | semantic_context → semantic_prediction | 语义预测路径 |
| **base_url** | http://localhost:11434 | Ollama API 地址 |
| **summary_size** | 8 | 语义摘要大小 |
| **max_tokens** | 120 | 最大生成 token 数 |
| **temperature** | 0.7 | 生成温度 |

### LLMAspect 的作用

1. **语义预测**: 从 `semantic_context` 预测 `semantic_prediction`
2. **自由能贡献**: `F_llm = 0.5 * ||semantic_prediction - llm_prediction||²`
3. **降低复杂度**: 提供语义先验，减少感官 Aspects 需求（5-10x）
4. **加速演化**: 引导网络向语义正确的方向演化（3-10x）

### 验证 LLMAspect 启用

运行实验时，会看到：
- 初始化输出: `[LLMAspect] ✓ 已启用`
- 进度条显示: `LLM=LLM✓`
- 快照包含: `has_llm_aspect: true`

## 七、配置对比（vs 之前配置）

### 7.1 Aspect 上限

- **之前**: 400
- **现在**: 10,000
- **变化**: 提升 25 倍

### 7.2 新建速率

| 参数 | 之前 | 现在 | 变化 |
|------|------|------|------|
| base | 8 | 2 | ↓ 75% |
| max_per_step | 32 | 8 | ↓ 75% |
| error_threshold | 0.1 | 0.15 | ↑ 50% |

### 7.3 学习参数

| 参数 | 之前 | 现在 | 变化 |
|------|------|------|------|
| infer_lr | 0.01 | 0.005 | ↓ 50% |
| learning_rate | 0.01 | 0.005 | ↓ 50% |
| num_infer_iters | 2 | 3 | ↑ 50% |
| max_grad_norm | 268.0 | 100.0 | ↓ 63% |

## 八、预期演化结果

基于当前配置，预期演化结果：

- **Aspect 数量**: 1,000 - 5,000 个（充分演化）
- **Pipeline 深度**: 8 - 12 层
- **自由能**: 稳定下降（从初始 ~1000 降至 < 100）
- **演化时间**: 8 - 12 小时
- **最终网络规模**: 
  - Objects: 10 - 20 个
  - Aspects: 1,000 - 5,000 个
  - Pipelines: 50 - 100 个

## 九、设计理念总结

### 9.1 充分学习

- **降低学习率**: 从 0.01 降至 0.005，确保每个参数更新更充分
- **增加迭代**: 从 2 次增至 3 次，确保状态推理更充分
- **降低梯度裁剪**: 从 268.0 降至 100.0，更保守的更新

### 9.2 稳定演化

- **减少新建速率**: base 和 max_per_step 都减少 75%
- **提高错误阈值**: 从 0.1 提高至 0.15，只有错误较大时才创建
- **降低错误倍数**: 从 0.7 降至 0.5，更保守的批量增长

### 9.3 充分扩展

- **提高上限**: max_aspects 从 400 提升至 10,000
- **允许深度演化**: max_depth 保持 12，允许深层 Pipeline
- **长期运行**: 15,000 步，充分时间演化

### 9.4 数据安全

- **自动检查点**: 每 1,000 步自动保存
- **定期快照**: 每 100 步保存快照
- **时间戳目录**: 每次运行创建独立目录

## 十、运行命令

### 10.1 默认运行

```bash
# 使用所有默认配置（15,000 步，Ollama cogito:32b）
python scripts/run_long_evolution.py
```

### 10.2 自定义步数

```bash
# 运行 20,000 步（约 10-11 小时）
python scripts/run_long_evolution.py --steps 20000
```

### 10.3 后台运行

```bash
# 后台运行并保存日志
nohup python scripts/run_long_evolution.py > evolution.log 2>&1 &

# 查看日志
tail -f evolution.log
```

### 10.4 使用其他 LLM

```bash
# 使用其他 Ollama 模型
python scripts/run_long_evolution.py --ollama-model llama3:70b

# 使用 OpenAI
python scripts/run_long_evolution.py --use-openai

# 禁用 LLM（使用 Mock）
python scripts/run_long_evolution.py --disable-ollama
```

## 十一、监控和检查

### 11.1 查看最新检查点

```bash
# 查看最新的检查点
ls -lht data/checkpoints_*/checkpoint_step_*.json | head -5
```

### 11.2 查看自由能趋势

```python
import json
from pathlib import Path
import glob

checkpoints = sorted(glob.glob('data/checkpoints_*/checkpoint_step_*.json'))
if checkpoints:
    with open(checkpoints[-1]) as f:
        data = json.load(f)
        print(f'最新检查点: {checkpoints[-1]}')
        print(f'步数: {data.get("step", "N/A")}')
        print(f'自由能: {data.get("free_energy", "N/A"):.4f}')
        structure = data.get('structure', {})
        print(f'Aspects: {structure.get("num_aspects", 0)}')
        print(f'Pipelines: {structure.get("num_pipelines", 0)}')
```

## 十二、注意事项

1. **Ollama 服务**: 确保 Ollama 服务正在运行
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **模型下载**: 确保 cogito:32b 模型已下载
   ```bash
   ollama pull cogito:32b
   ```

3. **资源需求**:
   - CPU: 建议多核
   - 内存: 建议至少 8GB（随着 Aspect 数量增加，内存需求会增加）
   - 磁盘: 建议至少 10GB 可用空间

4. **长时间运行**:
   - 确保计算机不会休眠
   - 使用后台运行（nohup/screen/tmux）
   - 定期检查日志和检查点

## 十三、配置修改

如需修改配置，编辑 `scripts/run_general_ai_experiment.py` 中的配置字典（约第 342-400 行）。

