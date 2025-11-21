# AONN 长时间演化实验运行指南

## 快速开始

### 方法 1: 使用 Python 启动脚本（推荐）

```bash
# 使用默认配置（15000 步，约 8 小时，使用 Ollama cogito:32b）
python scripts/run_long_evolution.py

# 自定义步数（20000 步，约 10-11 小时）
python scripts/run_long_evolution.py --steps 20000

# 使用其他 Ollama 模型
python scripts/run_long_evolution.py --ollama-model llama3:70b

# 禁用 Ollama，使用 Mock LLM（仅用于测试）
python scripts/run_long_evolution.py --disable-ollama

# 使用 OpenAI LLM
python scripts/run_long_evolution.py --use-openai

# 禁用 LLM（使用 Mock）
python scripts/run_long_evolution.py --disable-llm

# 自定义保存间隔
python scripts/run_long_evolution.py --save-interval 50

# 详细输出
python scripts/run_long_evolution.py --verbose
```

### 方法 2: 使用 Shell 脚本

```bash
# 使用默认配置
bash scripts/run_long_evolution.sh

# 自定义步数和保存间隔
bash scripts/run_long_evolution.sh 20000 50
```

### 方法 3: 直接运行实验脚本

```bash
# 使用默认配置（15000 步）
python scripts/run_general_ai_experiment.py

# 自定义参数
python scripts/run_general_ai_experiment.py \
    --steps 20000 \
    --save-interval 100 \
    --checkpoint-dir data/my_checkpoints \
    --use-ollama-llm \
    --ollama-model cogito:32b
```

## 后台运行

### 使用 nohup（推荐）

```bash
# 后台运行并保存日志
nohup python scripts/run_long_evolution.py > evolution.log 2>&1 &

# 查看进程
ps aux | grep run_long_evolution

# 查看日志
tail -f evolution.log

# 停止实验（找到进程 ID 后）
kill <PID>
```

### 使用 screen

```bash
# 创建新的 screen 会话
screen -S aonn_evolution

# 运行实验
python scripts/run_long_evolution.py

# 按 Ctrl+A 然后 D 来分离会话

# 重新连接会话
screen -r aonn_evolution

# 查看所有会话
screen -ls
```

### 使用 tmux

```bash
# 创建新的 tmux 会话
tmux new -s aonn_evolution

# 运行实验
python scripts/run_long_evolution.py

# 按 Ctrl+B 然后 D 来分离会话

# 重新连接会话
tmux attach -t aonn_evolution

# 查看所有会话
tmux ls
```

## 配置说明

### 当前配置（已优化）

- **Aspect 上限**: 10000
- **新建 Aspect 速率**: 非常保守（base=2, max_per_step=8）
- **学习率**: 0.005（充分学习）
- **推理迭代**: 3 次（充分推理）
- **梯度裁剪**: 100.0（保守更新）
- **默认步数**: 15000（约 8 小时）

### 预期结果

- 网络有充足时间学习和适应
- Aspect 创建更慢但更稳定
- 最终可能演化出 **1000-5000 个 Aspects**
- 自由能应该能稳定下降
- 每 1000 步自动保存检查点

## 输出文件

实验运行后，会在以下位置生成文件：

```
data/
├── long_evolution_YYYYMMDD_HHMMSS.json  # 最终结果
└── checkpoints_YYYYMMDD_HHMMSS/        # 检查点目录
    ├── checkpoint_step_1000.json        # 每 1000 步的检查点
    ├── checkpoint_step_2000.json
    └── checkpoint_final_step_15000.json
```

## 监控实验进度

### 查看实时进度

```bash
# 如果使用 nohup，查看日志
tail -f evolution.log

# 查看最新的检查点
ls -lht data/checkpoints_*/checkpoint_step_*.json | head -5

# 查看最新检查点的自由能
python -c "
import json
from pathlib import Path
import glob

checkpoints = sorted(glob.glob('data/checkpoints_*/checkpoint_step_*.json'))
if checkpoints:
    with open(checkpoints[-1]) as f:
        data = json.load(f)
        print(f'最新检查点: {checkpoints[-1]}')
        print(f'步数: {data.get(\"step\", \"N/A\")}')
        print(f'自由能: {data.get(\"free_energy\", \"N/A\"):.4f}')
        structure = data.get('structure', {})
        print(f'Aspects: {structure.get(\"num_aspects\", 0)}')
        print(f'Pipelines: {structure.get(\"num_pipelines\", 0)}')
"
```

### 绘制演化曲线

```bash
# 使用 Python 分析结果
python -c "
import json
import matplotlib.pyplot as plt
from pathlib import Path

# 读取最终结果
with open('data/long_evolution_*.json') as f:
    data = json.load(f)

steps = [s['step'] for s in data['snapshots']]
free_energy = [s['free_energy'] for s in data['snapshots']]
aspects = [s['structure']['num_aspects'] for s in data['snapshots']]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(steps, free_energy)
plt.xlabel('步数')
plt.ylabel('自由能')
plt.title('自由能演化')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(steps, aspects)
plt.xlabel('步数')
plt.ylabel('Aspect 数量')
plt.title('Aspect 数量演化')
plt.grid(True)

plt.tight_layout()
plt.savefig('evolution_curve.png')
print('演化曲线已保存到 evolution_curve.png')
"
```

## 故障排除

### 如果实验中断

检查点会自动保存，可以从最新的检查点恢复：

```python
# 恢复脚本（需要实现）
# python scripts/resume_experiment.py --checkpoint data/checkpoints_*/checkpoint_step_5000.json
```

### 如果内存不足

减少保存间隔或使用更少的步数：

```bash
python scripts/run_long_evolution.py --steps 10000 --save-interval 200
```

### 如果运行太慢

- 检查是否使用了 GPU（如果可用）
- 考虑减少 `num_infer_iters`（在配置中）
- 考虑减少 `max_aspects`（在配置中）

## 注意事项

1. **长时间运行**: 实验可能需要 8-12 小时，确保：
   - 计算机不会休眠
   - 有足够的磁盘空间（检查点文件可能较大）
   - 网络连接稳定（如果使用 OpenAI/Ollama API）

2. **资源需求**:
   - CPU: 建议多核
   - 内存: 建议至少 8GB（随着 Aspect 数量增加，内存需求会增加）
   - 磁盘: 建议至少 10GB 可用空间

3. **中断处理**: 
   - 实验会自动保存检查点
   - 可以随时中断（Ctrl+C）
   - 部分结果已保存在检查点中

## 更多帮助

查看完整参数列表：

```bash
python scripts/run_long_evolution.py --help
python scripts/run_general_ai_experiment.py --help
```

