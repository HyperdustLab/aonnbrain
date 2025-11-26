# Office AI 架构更新文档

## 概述

本文档描述了 Office AI 实验的最新架构更新，包括 prompt 观察变量的添加、世界模型维度更新、以及推理网络的逐步扩展实现。

## 主要更新

### 1. Prompt 观察变量

**更新内容**：
- 将 `prompt` 作为独立的观察变量添加到 Office AI 世界模型
- `prompt` 同时作为世界模型状态的一部分（`prompt_dim`）
- 创建了对应的观察网络和推理网络

**维度配置**：
- `prompt_dim`: 128（世界模型状态维度）
- `prompt_obs_dim`: 128（观察维度）

### 2. 世界模型状态维度更新

**更新前**：
```
总状态维度 = document_dim (256) + task_dim (128) + schedule_dim (64) + context_dim (128) = 576
```

**更新后**：
```
总状态维度 = document_dim (256) + task_dim (128) + schedule_dim (64) + context_dim (128) + prompt_dim (128) = 704
```

**状态组成**：
- `document_state`: 256 维
- `task_state`: 128 维
- `schedule_state`: 64 维
- `context_state`: 128 维
- `prompt_state`: 128 维（新增）

### 3. 观察变量更新

**更新前**：
- `document`: 256 维
- `table`: 128 维
- `calendar`: 64 维
- 总观察维度: 448

**更新后**：
- `document`: 256 维
- `table`: 128 维
- `calendar`: 64 维
- `prompt`: 128 维（新增）
- 总观察维度: 576

### 4. AONN Brain 维度更新

**配置更新**：
- `state_dim`: 576 → 704
- `obs_dim`: 448 → 576
- `sense_dims`: 添加 `prompt: 128`
- `world_model`: 添加 `prompt_dim: 128`

**网络维度匹配**：
- `internal` 维度: 704（匹配世界模型总状态维度）
- 观察网络输入维度: 704
- 推理网络输出维度: 704
- 动态模型状态维度: 704

## 网络结构

### 观察网络（Decoder）

使用 `PipelineAspect` 实现，从 `internal` 状态预测观察：

1. `world_pipeline_document`: internal (704) → document (256)
2. `world_pipeline_table`: internal (704) → table (128)
3. `world_pipeline_calendar`: internal (704) → calendar (64)
4. `world_pipeline_prompt`: internal (704) → prompt (128)

**结构**：
- 深度: 3 层 AspectLayer
- 宽度: 256/128/64/128 个 aspects/层
- 使用 AspectPipeline 实现

### 推理网络（Encoder）

使用 `PipelineAspect` 实现，从观察推断 `internal` 状态：

1. `inference_pipeline_document`: document (256) → internal (704)
2. `inference_pipeline_table`: table (128) → internal (704)
3. `inference_pipeline_calendar`: calendar (64) → internal (704)
4. `inference_pipeline_prompt`: prompt (128) → internal (704)

**结构**：
- 深度: 5 层 AspectLayer（可配置）
- 宽度: 256 个 aspects/层
- **逐步扩展维度** (`progressive_expansion=True`)
  - 避免第一层就扩展到输出维度的信息瓶颈
  - 每层逐步扩展，更平滑的维度过渡
  - 例如 calendar: 64 → 166 → 268 → 371 → 473 → 576

### 动态模型（Dynamics）

使用 `DynamicsAspect` 实现状态转移：

- `dynamics`: internal (704) + action (128) → internal (704)
- 结构: 2 层 MLP（Linear + ReLU + Linear）

### 语义网络

使用 `LLMAspect` 实现语义预测：

- `llm_aspect`: semantic_context (128) → semantic_prediction (128)

## 世界模型观察生成

所有观察模型都使用 `AspectPipeline` 实现：

1. `document_obs_model`: AspectPipeline (704 → 256)
2. `table_obs_model`: AspectPipeline (704 → 128)
3. `calendar_obs_model`: AspectPipeline (704 → 64)
4. `prompt_obs_model`: AspectPipeline (128 → 128)

**从 prompt_state 生成 prompt_obs**：
- `prompt_state` 是世界模型状态的一部分（128 维）
- `prompt_obs_model` 从 `prompt_state` 生成 `prompt_obs`（128 维）

## 配置示例

```python
config = {
    "state_dim": 704,  # 总状态维度（包含 prompt_dim）
    "act_dim": 128,
    "obs_dim": 576,  # 总观察维度（包含 prompt_obs_dim）
    "sem_dim": 128,
    "sense_dims": {
        "document": 256,
        "table": 128,
        "calendar": 64,
        "prompt": 128,  # 新增
    },
    "world_model": {
        "document_dim": 256,
        "task_dim": 128,
        "schedule_dim": 64,
        "context_dim": 128,
        "prompt_dim": 128,  # 新增
    },
    "world_model_pipeline_map": {
        "document": "document_dim",
        "table": "task_dim",
        "calendar": "schedule_dim",
        "prompt": "prompt_dim",  # 新增
    },
    "use_world_model_pipelines": True,
    "sensory_pipeline": {
        "depth": 3,
        "width": 256,
        "use_gate": False,
    },
    "inference_pipeline": {
        "depth": 5,
        "width": 256,
        "use_gate": False,
        "progressive_expansion": True,  # 启用逐步扩展
    },
    "enable_evolution": False,
}
```

## 关键改进

### 1. 逐步扩展维度

推理网络使用逐步扩展，避免了信息瓶颈：
- **修改前**: 第一层 64 → 576（扩展 9x，信息瓶颈）
- **修改后**: 64 → 166 → 268 → 371 → 473 → 576（每层扩展 1.2-2.6x）

### 2. 统一的 Aspect Pipeline 架构

所有网络都使用 Aspect Pipeline：
- 观察网络: `PipelineAspect`
- 推理网络: `PipelineAspect`
- 世界模型观察生成: `AspectPipeline`

### 3. 维度一致性

所有网络维度与世界模型完全匹配：
- 世界模型总状态维度: 704
- AONN Brain internal 维度: 704
- 观察网络输入维度: 704
- 推理网络输出维度: 704
- 动态模型状态维度: 704

## 更新的脚本

以下脚本已更新以反映新的架构：

1. `scripts/run_office_ai_event_injection.py`
2. `scripts/run_office_ai_experiment.py`
3. `scripts/save_office_ai_weights.py`
4. `scripts/profile_inference_loop.py`

## 验证

运行以下命令验证维度一致性：

```python
from aonn.models.office_ai_world_model import OfficeAIWorldModel
from aonn.models.aonn_brain_v3 import AONNBrainV3

# 创建世界模型
world = OfficeAIWorldModel(
    document_dim=256,
    task_dim=128,
    schedule_dim=64,
    context_dim=128,
    prompt_dim=128,  # 新增
    document_obs_dim=256,
    table_obs_dim=128,
    calendar_obs_dim=64,
    prompt_obs_dim=128,  # 新增
    action_dim=128,
)

# 验证维度
assert world.total_state_dim == 704
assert world.total_obs_dim == 576

# 创建 AONN Brain（使用更新后的配置）
brain = AONNBrainV3(config=config, ...)

# 验证维度匹配
assert brain.objects['internal'].dim == world.total_state_dim
```

## 总结

本次更新实现了：
1. ✅ Prompt 作为世界模型状态的一部分
2. ✅ 世界模型总状态维度更新到 704
3. ✅ 推理网络使用逐步扩展维度
4. ✅ 所有网络使用 Aspect Pipeline 架构
5. ✅ 所有维度完全匹配

AONN Brain 的观察、推理和动态模型现在与世界模型维度完全一致。

