# AONN Brain V2 架构文档

按照 `AONN网络.txt` 设计的正确 AONN 架构。

## 核心设计原则

### 1. 状态与计算完全分离

- **Object Layer（状态层）**：只存储状态，不执行计算
  - 类似"神经状态膜"（细胞膜电位分布）
  - 每个 Object 是 Vertical MB（有自己的 S/A/internal）
  
- **Aspect Layer（计算层）**：只执行计算，不存储状态
  - 横切的神经元组织（Cross-MB）
  - 每个 Aspect 是一个神经元/因子

### 2. 隐层只有 Aspect，没有 Object

```
Object Layer (k)  —— 上一层状态（Vertical MB）
  ┌─────────────────────────────────────────┐
  │   C_k_1   C_k_2   ...             C_k_N │
  └─────────────────────────────────────────┘
             │        │                │
             ▼        ▼                ▼
  ┌─────────────────────────────────────────┐
  │      Aspect Layer (k→k+1)              │
  │ A1    A2    A3   ...              A_M  │
  │ (每一个 A_j 就是一个隐层神经元/Aspect)   │
  └─────────────────────────────────────────┘
             │        │                │
             ▼        ▼                ▼
   Object Layer (k+1) —— 下一层状态（Vertical MB）
  ┌─────────────────────────────────────────┐
  │  C_{k+1,1}  C_{k+1,2} ...        C_{k+1,N} │
  └─────────────────────────────────────────┘
```

**关键**：中间那一层只有 Aspect，完全没有"普通 object 点"。

### 3. 深度网络结构

深度网络 = Object Layer + Aspect Pipeline 交替：

```
Object → Aspect → Aspect → Aspect → Aspect → Object
```

- 所有"有状态的东西"都落在 Object Layer 上
- 所有"计算/神经元/权重"都集中在 Aspect Layer
- Aspect Pipeline 可以连续堆叠多个 Aspect Layer

## 实现架构

### ObjectLayer

```python
class ObjectLayer:
    """状态层：只存储状态，不执行计算"""
    - get_state_vector(): 将所有 Object 状态拼接成向量
    - set_state_vector(): 从向量设置所有 Object 状态
```

### AspectLayer

```python
class AspectLayer(nn.Module):
    """Aspect 层：横切的神经元组织"""
    - W: Linear(input_dim, num_aspects)  # 读取方向
    - V: Linear(num_aspects, output_dim) # 写回方向
    - forward: x' = ReLU(proj(x) + V(ReLU(Wx)))
```

### AspectPipeline

```python
class AspectPipeline(nn.Module):
    """Aspect Pipeline：连续的 Aspect Layer"""
    - 结构：Aspect → Aspect → Aspect → ...
    - 中间完全没有 Object Layer
```

### AONNBrainV2

```python
class AONNBrainV2(nn.Module):
    """正确的深度 AONN 架构"""
    - Object Layers: 输入层、隐藏层、输出层
    - Aspect Pipelines: 连接相邻 Object Layers
    - 前向传播：Object → Aspect → Aspect → ... → Object
```

## 网络结构示例

```
Object Layers（状态层）：
  - input: 1 objects, dim=128
  - hidden_1: 1 objects, dim=256
  - hidden_2: 1 objects, dim=256
  - output: 1 objects, dim=10

Aspect Pipelines（计算层，纯神经元）：
  - Pipeline 1: depth=4, aspects=32, 128→256
  - Pipeline 2: depth=4, aspects=32, 256→256
  - Pipeline 3: depth=4, aspects=32, 256→10

网络流程：
  input
    ↓ [Aspect Pipeline 1: 4层 Aspect，每层32个神经元]
  hidden_1
    ↓ [Aspect Pipeline 2: 4层 Aspect，每层32个神经元]
  hidden_2
    ↓ [Aspect Pipeline 3: 4层 Aspect，每层32个神经元]
  output
```

## 关键特点

1. ✅ **状态与计算完全分离**
   - Object Layer 只存储状态
   - Aspect Layer 只执行计算

2. ✅ **隐层只有 Aspect**
   - Aspect Pipeline 中完全没有 Object
   - 只有神经元/Aspect

3. ✅ **深度来自函数复合**
   - Object → Aspect → Aspect → ... → Object
   - 深度不仅来自维度，更来自函数复合

4. ✅ **因子图结构**
   - Object = 变量节点（Variable nodes）
   - Aspect = 因子节点（Factor nodes）
   - 整个网络 = 深度因子图

5. ✅ **天然稳定**
   - 残差结构：x' = x + V(σ(Wx))
   - 低秩分解
   - 训练稳定

## 使用方法

```python
from aonn.models.aonn_brain_v2 import AONNBrainV2

config = {
    "input_dim": 128,
    "hidden_dims": [256, 256],
    "output_dim": 10,
    "num_aspects": 32,
    "aspect_depth": 4,
    "use_gate": False,
}

brain = AONNBrainV2(config=config)

# 前向传播
x = torch.randn(4, 128)
y = brain(x)  # [4, 10]

# 可视化网络
print(brain.visualize_network())
```

## 与 V1 的区别

| 特性 | AONNBrain V1 | AONNBrain V2 |
|------|-------------|--------------|
| **架构** | Object 和 Aspect 混合 | Object Layer 和 Aspect Layer 分离 |
| **隐层** | 可能有 Object | 只有 Aspect |
| **深度** | 通过添加 Aspect | 通过 Aspect Pipeline |
| **结构** | 图结构（灵活但复杂） | 层次结构（清晰且高效） |
| **适用** | 认知建模、自由能框架 | 深度学习任务、高效训练 |

## 测试

```bash
python scripts/test_aonn_v2.py
```

测试验证：
- ✅ 状态与计算完全分离
- ✅ Object Layer 只存储状态
- ✅ Aspect Pipeline 只有神经元，没有 Object
- ✅ 深度网络：Object → Aspect → Aspect → ... → Object

