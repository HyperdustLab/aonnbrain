# AONN Free Energy Brain

AONN (Active Object Neural Network) - 自由能大脑框架，集成 LLMAspect 用于语义推理。

## 项目结构

```
aonn-free-energy-brain/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/              # 配置文件
├── data/                 # 数据目录
├── scripts/             # 训练和推理脚本
├── src/aonn/            # 核心代码
│   ├── core/            # 核心抽象
│   ├── aspects/         # Aspect 实现
│   ├── models/          # 模型组装
│   ├── pipeline/        # 训练和推理流程
│   └── utils/           # 工具函数
├── tests/               # 测试文件
└── notebooks/           # Jupyter 笔记本
```

## 核心概念

### ObjectNode
存储状态 μ 的节点，不做计算。

### AspectBase
Aspect 抽象基类，负责：
- 读取 ObjectNode 的状态
- 输出预测误差
- 提供自由能贡献

### ActiveInferenceLoop
主动推理循环，通过梯度下降最小化自由能。

### LLMAspect
将 LLM 作为语义预测因子，集成到自由能框架中。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据预处理

```bash
python scripts/preprocess_dataset.py --input data/raw/dialogs.jsonl --output data/processed/aonn_dataset.pt
```

### 训练

```bash
python scripts/train_aonn_brain.py
```

### 主动推理演示

```bash
python scripts/run_active_inference.py
```

### 测试 LLMAspect

```bash
python scripts/benchmark_llm_aspect.py
```

## 运行测试

```bash
pytest tests/
```

## 配置说明

- `configs/brain_default.yaml`: 大脑结构配置
- `configs/training_default.yaml`: 训练参数
- `configs/llm_aspect_openai.yaml`: LLM 对接配置

## 开发计划

- [ ] 实现完整的 LLM 客户端接口（OpenAI / 本地 LLM）
- [ ] 添加文本编码/解码器
- [ ] 实现更多 Aspect（dynamics, intent, preference, action）
- [ ] 添加数据加载和预处理管道
- [ ] 实现完整的训练和评估流程

## 许可证

MIT
