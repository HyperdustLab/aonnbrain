# LLMAspect 使用与实现指南

## 1. 背景概述
- **位置**：`src/aonn/aspects/llm_aspect.py`
- **作用**：把外部大语言模型（LLM）当作“语义预测因子”，在自由能框架中提供 **语义先验约束** 与 **语义预测**，而不是把 LLM 当作黑盒大脑。
- **典型适用场景**：通用 AI 智能体、Office AI 办公助手、混合语义/多模态任务。

LLMAspect 通过增加自由能项 `F_llm = 0.5 * weight * ||semantic_prediction - llm_prediction||^2`，迫使内部语义表示与 LLM 的预测保持一致，从而加速语义学习、降低演化复杂度。

---

## 2. 关键对象
| Object | 维度 | 含义 | 来源 |
| --- | --- | --- | --- |
| `semantic_context` | `sem_dim`（默认 512） | 系统内部对外部世界语义上下文的表示 | 由 `update_semantic_context()` 从世界模型语义状态或 `internal` 截取 |
| `semantic_prediction` | `sem_dim` | LLM 期望的语义输出，通常与 `internal` 的语义部分相连 | 可独立 Object，或由 Aspect 与 `internal` 连接 |
| `internal` | `state_dim` | 主动推理的核心状态，负责融合所有模态与先验 | 由状态推理动态更新 |

```text
semantic_context  --(LLMAspect读取)-->  LLM → 语义预测向量
      ↓                                           ↓
semantic_prediction (由 internal 或独立 Object 提供)
```

---

## 3. 工作流程
1. **同步语义上下文** (`AONNBrainV3.update_semantic_context`)
   - 优先使用世界模型给出的语义状态（如通用 AI 场景里的 `world_model.semantic_state`）。
   - 若无外部语义状态，则截取 `internal` 的前 `sem_dim` 维。

2. **语义预测** (`LLMAspect._call_llm`)
   - 从 `semantic_context` 读取语义向量，构造 prompt/特征。
   - 调用外部 LLM（OpenAI、Ollama、Mock），得到 `llm_prediction`。
   - 支持调用频率控制：`every_iter` / `last_iter_only` / `every_n_steps`。

3. **自由能贡献** (`LLMAspect.free_energy_contrib`)
   - 计算 `F_llm = 0.5 * weight * ||semantic_prediction - llm_prediction||^2`。
   - 在 `compute_total_free_energy()` 中与其他 Aspect 一同累加。

4. **状态推理** (`ActiveInferenceLoop.infer_states`)
   - 对 `internal`（以及可能的 `action`）执行若干次梯度下降。
   - `F_llm` 的梯度会通过 `semantic_prediction` 回传到 `internal`，引导语义部分向 LLM 预测靠拢。

---

## 4. 数学关系
- **总自由能**（含 LLM）：
  ```
  F_total = F_obs + F_encoder + F_dyn + F_pref + F_llm
  ```
- **LLMAspect 项**：
  ```
  F_llm = 0.5 * weight * ||semantic_prediction - llm_prediction||^2
  ```
- **梯度传播（示例：semantic_prediction = internal 的前 `sem_dim` 维）**：
  ```
  ∇F_llm / ∇internal = internal[:sem_dim] - llm_prediction   # 其他维度梯度为 0
  ```
- **状态更新**（推理学习率 `η`）
  ```
  internal ← internal - η * (∇F_obs + ∇F_encoder + ∇F_dyn + ∇F_pref + ∇F_llm)
  ```
  `∇F_llm` 会把 `internal` 的语义部分拉向 LLM 预测，使语义理解更精准。

---

## 5. 场景示例：会议安排任务
1. **事件**：世界模型接收到文本“明天下午3点与张总开会，讨论 Q4 项目计划”。
2. **世界模型状态**：
   - 文档状态：`"Q4 项目计划.docx"` 的语义特征。
   - 任务状态：`["准备会议材料", "发送会议邀请", ...]` 及进度。
   - 日程状态：`"明天下午3点"` 的时间编码。
   - 上下文状态：对话历史“用户提出会议安排请求”。
3. **观察**：文档/表格/日程多模态向量被写入 `brain.objects`。
4. **语义同步**：`semantic_context ← world_model.semantic_state[:sem_dim]`，编码“会议安排任务”的语义。
5. **LLMAspect**：
   - 输入：`semantic_context`。
   - 输出：`llm_prediction = "创建会议事件，时间 15:00，参与者张总，主题 Q4 计划"` → 语义向量。
   - 生成 `F_llm`，约束 `semantic_prediction` 接近该语义。
6. **状态推理**：
   - `F_llm` + 其他自由能项的梯度共同更新 `internal`。
   - 更新后的 `internal` 更准确地表示“会议安排任务”。
7. **动作生成**：
   - `internal` 通过 Pipeline 生成动作：创建日历事件、发送邀请、准备材料。
8. **世界模型执行**：
   - 日历状态更新为“会议已创建”。
   - 任务进度提升。

---

## 6. 配置与集成
- **启用 LLMAspect**：在脑模型构造时传入 `llm_client`（OpenAI、Ollama、Mock）。
  ```python
  brain = AONNBrainV3(config=config, llm_client=llm_client, device=device)
  ```
- **语义维度**：`config["sem_dim"]`（默认 512）。
- **调用频率**：`llm_config = {"call_frequency": "last_iter_only", ...}`。
- **常用脚本**：
  - `scripts/run_general_ai_experiment.py`
  - `scripts/run_office_ai_experiment.py`
  - `scripts/profile_inference_loop.py`

---

## 7. 性能与收益
摘自 `docs/WORLD_MODEL_COMPLEXITY.md`：

| 指标 | 无 LLMAspect | 有 LLMAspect | 提升 |
| --- | --- | --- | --- |
| 感官 Aspect 数量 | 1000-5000 | 100-500 | ↓5-10x |
| Pipeline 深度 | 10-20 层 | 5-10 层 | ↓2x |
| 演化步数 | 5000-20000 | 500-2000 | ↓3-10x |
| 自由能阈值 | 0.01-0.05 | 0.02-0.08 | 更宽松 |
| 语义理解能力 | 需演化学习 | 预训练提供 | 质的提升 |

LLMAspect 将语义理解“外包”给预训练的 LLM，大幅降低 AONN 在高维语义任务上的有效复杂度。

---

## 8. 总结
- LLMAspect 把 LLM 的语义能力以“自由能因子”的形式整合到主动推理框架中。
- 通过 `F_llm`，它在状态推理阶段直接拉动 `internal` 的语义部分，使系统在少量迭代内获得语义一致性。
- 适合所有需要语义理解、语义压缩、结构指引的复杂世界模型，尤其是通用 AI、办公 AI、多模态助手等应用。 
