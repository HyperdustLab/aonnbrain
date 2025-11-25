# 语义匹配与马尔可夫毯原则

## 问题背景

在实现 LLMAspect 的语义相似度匹配功能时，最初的设计是从 `semantic_context` 的 `metadata` 中读取期望关键词来验证 LLM 输出。这违反了**马尔可夫毯（Markov Blanket）**原则。

## 马尔可夫毯原则

在自由能原理（Free Energy Principle, FEP）中，马尔可夫毯是系统与外部环境之间的边界：

- **内部状态（Internal States）**：系统的隐藏状态
- **感觉状态（Sensory States）**：系统感知环境的方式
- **主动状态（Active States）**：系统影响环境的方式

系统只能通过感觉状态和主动状态与环境交互，**不应该直接访问外部世界的真实状态（ground truth）**。

## 问题分析

如果系统内部（LLMAspect）直接从 `metadata` 中读取期望关键词来验证输出：

1. **违反马尔可夫毯**：系统"知道"了应该期望什么（ground truth），这是外部世界的真实状态
2. **破坏自主性**：系统不再需要通过感觉状态推断环境状态，而是直接获得了答案
3. **影响学习**：系统无法通过预测误差来学习，因为已经知道了正确答案

## 解决方案

### 1. 语义匹配作为外部评估工具

语义相似度匹配应该作为**外部评估工具**，而不是系统内部的一部分：

```python
# ✅ 正确：在系统外部进行语义匹配
llm_aspect.forward(objects)  # 系统内部推理，不访问期望关键词
generated_text = llm_client._last_generated_text

# 外部评估（不破坏马尔可夫毯）
coverage, matched, missing, similarities = llm_aspect.compute_semantic_similarity(
    llm_text=generated_text,
    expected_keywords=expected_keywords,  # 来自外部测试数据
    context_description=context_description,
    expectations=expectations,
)
```

### 2. 系统内部不使用期望关键词

系统内部的自由能计算不应该使用语义匹配结果：

```python
# ❌ 错误：在自由能计算中使用期望关键词
if self.enable_semantic_matching and self.semantic_matching_weight > 0:
    coverage = self._last_semantic_match_result.get("coverage", 0.0)
    semantic_penalty = (1.0 - coverage) * self.semantic_matching_weight
    free_energy = free_energy + semantic_penalty  # 这违反了马尔可夫毯原则
```

### 3. 正确的架构

```
┌─────────────────────────────────────────┐
│         系统内部（马尔可夫毯内）          │
│                                         │
│  semantic_context → LLMAspect →        │
│  semantic_prediction                    │
│                                         │
│  （只使用系统内部状态，不访问 ground truth）│
└─────────────────────────────────────────┘
                    │
                    │ LLM 输出文本
                    ▼
┌─────────────────────────────────────────┐
│         系统外部（评估工具）              │
│                                         │
│  compute_semantic_similarity(           │
│    llm_text,                            │
│    expected_keywords  ← 来自外部测试数据  │
│  )                                      │
│                                         │
│  （用于评估和验证，不影响系统内部推理）    │
└─────────────────────────────────────────┘
```

## 实现细节

### LLMAspect 的设计

1. **`compute_semantic_similarity()`**：用于外部评估，需要显式传入期望关键词
2. **`set_semantic_match_result()`**：允许外部评估工具设置匹配结果
3. **`get_semantic_match_result()`**：用于外部查询，不影响系统内部推理
4. **自由能计算**：不使用语义匹配结果，只基于系统内部状态

### 验证脚本的使用

在 `validate_office_events_with_llm.py` 中：

```python
# 系统内部推理（不访问期望关键词）
llm_aspect.forward(objects)

# 外部评估（从测试数据中获取期望关键词）
expected_keywords = payload.get("keywords", [])  # 来自外部测试数据
coverage, matched, missing, similarities = llm_aspect.compute_semantic_similarity(
    llm_text=generated_text,
    expected_keywords=expected_keywords,
    ...
)
```

## 总结

- ✅ **语义匹配是外部评估工具**，用于验证和调试
- ✅ **系统内部不访问期望关键词**，保持马尔可夫毯完整性
- ✅ **自由能计算只基于系统内部状态**，不依赖外部 ground truth
- ✅ **系统通过感觉状态和主动状态与环境交互**，符合 FEP 原则

这样的设计既保留了语义匹配的评估功能，又维护了系统的自主性和学习能力。

