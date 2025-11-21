# Ollama LLM 客户端使用指南

## 安装和配置

### 1. 安装 Ollama

访问 [Ollama 官网](https://ollama.com) 下载并安装 Ollama。

### 2. 下载模型

```bash
# 推荐使用较小的模型（更快）
ollama pull llama3:8b

# 或者使用其他模型
ollama pull mistral
ollama pull qwen2:7b
```

### 3. 启动 Ollama 服务

Ollama 服务默认在 `http://localhost:11434` 运行。确保服务正在运行：

```bash
# 检查服务状态
curl http://localhost:11434/api/tags

# 或者运行一个模型测试
ollama run llama3:8b
```

## 在实验中使用 Ollama

### 基本用法

```bash
# 使用默认模型 (llama3)
python scripts/run_general_ai_experiment.py \
    --steps 10 \
    --use-ollama-llm \
    --verbose

# 指定模型
python scripts/run_general_ai_experiment.py \
    --steps 10 \
    --use-ollama-llm \
    --ollama-model cogito:32b \
    --verbose

# 指定自定义 API 地址
python scripts/run_general_ai_experiment.py \
    --steps 10 \
    --use-ollama-llm \
    --ollama-base-url http://localhost:11434 \
    --ollama-model llama3:8b \
    --verbose
```

### 参数说明

- `--use-ollama-llm`: 启用 Ollama LLM 客户端
- `--ollama-model`: 指定模型名称（默认: `llama3`）
- `--ollama-base-url`: Ollama API 地址（默认: `http://localhost:11434`）

### 模型选择建议

- **快速测试**: `llama3:8b`, `mistral:7b`, `qwen2:7b`
- **高质量**: `llama3:70b`, `cogito:32b`, `gpt-oss:20b`
- **注意**: 大模型可能需要更长的响应时间（已设置 120 秒超时）

## 配置示例

在 `config` 中也可以配置：

```python
config = {
    "llm": {
        "base_url": "http://localhost:11434",
        "model": "llama3:8b",
        "embedding_model": "llama3:8b",  # 如果为 None，使用 model
        "summary_size": 8,
        "max_tokens": 120,
        "temperature": 0.7,
    },
}
```

## 故障排除

### 1. 连接超时

如果遇到超时，尝试：
- 使用更小的模型
- 增加超时时间（在代码中修改 `timeout` 参数）
- 检查 Ollama 服务是否正常运行

### 2. 模型不存在

确保模型已下载：
```bash
ollama list
ollama pull <model_name>
```

### 3. API 错误

检查 Ollama 服务状态：
```bash
curl http://localhost:11434/api/tags
```

## 与 OpenAI 对比

| 特性 | Ollama | OpenAI |
|------|--------|--------|
| 成本 | 免费（本地） | 按使用量收费 |
| 速度 | 取决于硬件 | 通常较快 |
| 隐私 | 完全本地 | 数据发送到云端 |
| 模型选择 | 有限 | 丰富 |
| 设置复杂度 | 需要安装 | 只需 API Key |

## 注意事项

1. **性能**: 本地模型通常比云端 API 慢，特别是大模型
2. **内存**: 确保有足够的 RAM 运行模型
3. **GPU**: 如果有 GPU，Ollama 会自动使用以加速推理
4. **超时**: 大模型可能需要更长的响应时间，已设置 120 秒超时

