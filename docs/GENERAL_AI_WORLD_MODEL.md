# 通用AI智能体世界模型

本文档介绍通用AI智能体世界模型的实现和使用方法。

## 概述

通用AI智能体世界模型（`GeneralAIWorldModel`）是一个高复杂度的世界模型，设计用于模拟通用AI智能体的环境。相比线虫世界模型，它具有：

- **更大的状态空间**：2000-8000 维（语义、记忆、上下文、物理、目标）
- **多模态感官**：视觉、语言、音频、多模态融合
- **复杂动作空间**：语言生成、工具调用、多模态输出
- **长期记忆机制**：工作记忆 + 长期记忆库
- **多目标奖励**：任务完成度、知识获取、社交反馈、安全性等

## 架构设计

### 状态空间

```python
总状态维度 = semantic_dim + memory_dim + context_dim + physical_dim + goal_dim

- semantic_dim (1024): 语义状态（语言/概念表示）
- memory_dim (512): 记忆状态（工作记忆 + 长期记忆索引）
- context_dim (256): 上下文状态（对话历史、任务上下文）
- physical_dim (64): 物理状态（位置、姿态、工具状态）
- goal_dim (256): 目标状态（当前任务目标、子目标栈）
```

### 感官空间

```python
总观察维度 = vision_dim + language_dim + audio_dim + multimodal_dim

- vision_dim (512): 视觉特征
- language_dim (512): 语言嵌入
- audio_dim (128): 音频特征
- multimodal_dim (256): 多模态融合表示
```

### 动作空间

```python
action_dim (256): 混合动作空间
- 语言生成：离散 token 序列（词汇表 ~50K）
- 工具调用：结构化动作（API 调用、函数执行）
- 多模态输出：文本 + 图像 + 代码
```

## 使用方法

### 基本使用

```python
from aonn.models.general_ai_world_model import GeneralAIWorldModel, GeneralAIWorldInterface
import torch

device = torch.device("cpu")

# 创建世界模型
world_model = GeneralAIWorldModel(
    semantic_dim=1024,
    memory_dim=512,
    context_dim=256,
    physical_dim=64,
    goal_dim=256,
    vision_dim=512,
    language_dim=512,
    audio_dim=128,
    multimodal_dim=256,
    action_dim=256,
    device=device,
)

world_interface = GeneralAIWorldInterface(world_model)

# 重置环境
obs = world_interface.reset()

# 执行动作
action = torch.randn(256, device=device) * 0.1
obs, reward = world_interface.step(action)
```

### 与 AONN 集成

```python
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.clients.mock_llm_client import MockLLMClient

# 创建 LLM 客户端
llm_client = MockLLMClient(
    input_dim=512,
    output_dim=512,
    hidden_dim=256,
    device=device,
)

# 配置 AONN
config = {
    "state_dim": 1024,
    "act_dim": 256,
    "obs_dim": 1408,  # 512+512+128+256
    "sem_dim": 512,
    "sense_dims": {
        "vision": 512,
        "language": 512,
        "audio": 128,
        "multimodal": 256,
    },
    "enable_world_model_learning": True,
    "evolution": {
        "free_energy_threshold": 0.05,
        "max_aspects": 2000,
        # ...
    },
    # ...
}

# 创建 AONN Brain
brain = AONNBrainV3(
    config=config,
    llm_client=llm_client,
    device=device,
    enable_evolution=True,
)
```

## 运行实验

使用提供的实验脚本：

```bash
python scripts/run_general_ai_experiment.py \
  --steps 500 \
  --device cpu \
  --output data/general_ai_results.json \
  --verbose
```

### 启用 OpenAI LLM

```bash
export OPENAI_API_KEY="sk-..."  # 或使用 --openai-api-key 传参

python scripts/run_general_ai_experiment.py \
  --steps 500 \
  --use-openai-llm \
  --openai-api-key "$OPENAI_API_KEY"
```

参数说明：

- `--use-openai-llm`：使用 `OpenAILLMClient`，把 LLMAspect 接到 OpenAI Chat/Embedding API
- `--openai-api-key`：可选，默认读取 `OPENAI_API_KEY`
- 配置项 `config["llm"]` 可调整 `model`（默认 `gpt-4o-mini`）、`embedding_model`（默认 `text-embedding-3-small`）、`summary_size`、`max_tokens` 等

### 参数说明

- `--steps`: 演化步数（默认 500）
- `--device`: 设备（cpu/cuda，默认 cpu）
- `--output`: 输出文件路径
- `--verbose`: 实时输出演化快照
- `--disable-llm`: 禁用 LLMAspect（用于对比实验）
- `--use-openai-llm`: 启用真实 OpenAI LLM，实现语义先验与语义因子

## 配置参数

### 世界模型配置

```python
"world_model": {
    "semantic_dim": 1024,      # 语义状态维度
    "memory_dim": 512,          # 记忆状态维度
    "context_dim": 256,         # 上下文状态维度
    "physical_dim": 64,         # 物理状态维度
    "goal_dim": 256,            # 目标状态维度
    "vision_dim": 512,          # 视觉感官维度
    "language_dim": 512,        # 语言感官维度
    "audio_dim": 128,           # 音频感官维度
    "multimodal_dim": 256,      # 多模态融合维度
    "state_noise_std": 0.01,    # 状态噪声标准差
    "observation_noise_std": 0.01,  # 观察噪声标准差
    "enable_tools": True,       # 是否启用工具调用
}
```

### AONN 演化配置

```python
"evolution": {
    "free_energy_threshold": 0.05,  # 自由能阈值（比线虫更严格）
    "max_aspects": 2000,            # 最大 Aspect 数量（比线虫更多）
    "batch_growth": {
        "base": 16,                 # 批量增长基数
        "max_per_step": 64,         # 单步最大增长
        "max_total": 400,           # 每个感官的最大总数
        "min_per_sense": 8,         # 每个感官的最小数量
        "error_threshold": 0.05,    # 误差阈值
    },
}

"pipeline_growth": {
    "max_stages": 8,                # 最大 Pipeline 阶段数（比线虫更深）
    "max_depth": 12,                # 单个 Pipeline 的最大深度
    "min_interval": 100,            # 扩展间隔
}
```

## 预期演化结果

在通用AI智能体世界模型下，AONN 预期演化特征：

- **Aspect 数量**: 1000-2000 个（vs 线虫的 280 个）
- **Pipeline 深度**: 8-12 层（vs 线虫的 6 层）
- **Object 类型**: 语义、记忆、上下文、目标等
- **演化时间**: 需要更长的演化周期（数千步）

## LLMAspect 的作用

在通用AI智能体世界模型中，`LLMAspect` 发挥关键作用：

1. **语义压缩**: 将高维语言/知识空间压缩成语义潜变量
2. **先验约束**: 提供语言/概念级的"正确性"指导
3. **结构指引**: 动态建议网络拓扑演化方向

**有效复杂度降低**:
- 感官 Aspect 数量：**5-10x 减少**
- Pipeline 深度：**2x 减少**
- 演化步数：**3-10x 加速**

详见 `docs/WORLD_MODEL_COMPLEXITY.md`。

## 与线虫世界模型对比

| 维度 | 线虫世界模型 | 通用AI智能体世界模型 | 复杂度提升 |
|------|------------|-------------------|----------|
| **状态维度** | 260 | 2000-8000 | **8-30x** |
| **动作维度** | 2 (实际) | 100-1000 | **50-500x** |
| **感官维度** | 224 | 1500-5000 | **7-22x** |
| **Aspect 数量** | ~280 | 1000-2000 | **3-7x** |
| **Pipeline 深度** | 6 | 8-12 | **1.3-2x** |

## 未来扩展

1. **真实 LLM 集成**: 替换 MockLLMClient 为真实 LLM API
2. **工具调用实现**: 实现真实的工具调用接口
3. **长期记忆优化**: 实现更高效的记忆检索机制
4. **多智能体支持**: 支持多智能体交互
5. **可视化工具**: 实时可视化网络结构演化

