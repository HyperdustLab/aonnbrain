# AONN 架构实现对比

本文档对比两种 AONN 实现方式：

1. **aonn_mnist_aspect_pipeline** - 向量化 Aspect Pipeline
2. **aonnbrain** - 对象化 Aspect 网络

## 1. aonn_mnist_aspect_pipeline 架构

### 核心思想
- **状态（Cell）**：统一的状态向量 `x ∈ R^D`
- **计算（Aspect）**：向量化的低秩算子，通过矩阵乘法批量处理

### 网络结构
```
输入图像 → Encoder → Cell → AspectPipeLayer → AspectPipeLayer → ... → Cell → Classifier
```

### AspectPipeLayer 实现
```python
class AspectPipeLayer(nn.Module):
    """
    向量化的 Aspect 层：
    - W: Linear(D, M)  # 所有 M 个 Aspect 的读取方向
    - V: Linear(M, D)  # 所有 M 个 Aspect 的写回方向
    """
    def forward(self, x):
        a = W(x)           # [B, M] - 所有 Aspect 的激活
        z = ReLU(a)        # [B, M]
        delta = V(z)       # [B, D] - 所有 Aspect 的增量
        return ReLU(x + delta)  # 残差更新
```

### 特点
- ✅ **高效**：矩阵乘法批量处理所有 Aspect
- ✅ **紧凑**：参数少，适合深度学习任务
- ✅ **可堆叠**：可以深度堆叠多个 AspectPipeLayer
- ❌ **灵活性低**：所有 Aspect 共享相同的结构
- ❌ **可解释性弱**：难以单独控制每个 Aspect

## 2. aonnbrain 架构

### 核心思想
- **状态（Object）**：多个独立的 Object 节点，每个存储特定状态
- **计算（Aspect）**：独立的 Aspect 类，每个有明确的 src/dst 连接

### 网络结构
```
Object₁ ──[Aspect₁]──> Object₂
Object₃ ──[Aspect₂]──> Object₄
...
```

### Aspect 实现
```python
class AspectBase(ABC):
    def __init__(self, src_names, dst_names):
        self.src_names = src_names  # 源 Object
        self.dst_names = dst_names  # 目标 Object
    
    def forward(self, objects):
        # 从 src 读取，计算，返回 dst 的误差
        pass
    
    def free_energy_contrib(self, objects):
        # 计算自由能贡献
        pass
```

### 特点
- ✅ **灵活**：每个 Aspect 可以有不同的实现
- ✅ **可解释**：明确的 Object-Aspect-Object 连接关系
- ✅ **模块化**：易于添加新的 Aspect 类型
- ✅ **网络拓扑**：明确的图结构，支持复杂连接
- ❌ **效率较低**：每个 Aspect 独立计算
- ❌ **参数多**：每个 Aspect 可能有独立参数

## 3. 两种架构的对比

| 特性 | Aspect Pipeline | Object-Aspect Network |
|------|----------------|---------------------|
| **状态表示** | 单一 Cell 向量 | 多个 Object 节点 |
| **Aspect 实现** | 向量化（矩阵） | 对象化（类） |
| **连接方式** | 顺序堆叠 | 图结构（任意连接） |
| **灵活性** | 低（统一结构） | 高（独立实现） |
| **效率** | 高（批量计算） | 中（逐个计算） |
| **可解释性** | 中 | 高 |
| **适用场景** | 深度学习任务 | 认知建模、自由能框架 |

## 4. 整合方案

可以创建两种架构的桥接：

### 方案 A：在 aonnbrain 中添加 AspectPipeLayer
```python
class AspectPipeLayerAspect(AspectBase, nn.Module):
    """
    将 AspectPipeLayer 包装为 Aspect
    用于连接两个 Object
    """
    def __init__(self, src_name, dst_name, dim, num_aspects):
        super().__init__(src_name, dst_name)
        self.pipe_layer = AspectPipeLayer(dim, num_aspects)
    
    def forward(self, objects):
        src_state = objects[self.src_names[0]].state
        dst_state = objects[self.dst_names[0]].state
        pred = self.pipe_layer(src_state)
        error = dst_state - pred
        return {self.dst_names[0]: error}
```

### 方案 B：在 Aspect Pipeline 中添加自由能框架
```python
class AONN_MNIST_FE(AONN_MNIST):
    """
    在 Aspect Pipeline 基础上添加自由能计算
    """
    def compute_free_energy(self, x, target):
        # 计算预测误差的自由能
        logits = self.forward(x)
        pred_probs = F.softmax(logits, dim=1)
        target_probs = F.one_hot(target, 10).float()
        # 自由能 = -log(p(target|x))
        fe = -torch.sum(target_probs * torch.log(pred_probs + 1e-8), dim=1)
        return fe.mean()
```

## 5. 推荐使用场景

### 使用 Aspect Pipeline（aonn_mnist_aspect_pipeline）
- ✅ 深度学习任务（分类、回归）
- ✅ 需要高效批量处理
- ✅ 标准的前向传播训练
- ✅ 资源受限环境

### 使用 Object-Aspect Network（aonnbrain）
- ✅ 认知建模和主动推理
- ✅ 需要明确的自由能框架
- ✅ 需要灵活的网络拓扑
- ✅ 需要可解释的 Aspect 连接
- ✅ 需要集成外部组件（如 LLM）

## 6. 未来发展方向

1. **混合架构**：结合两种方式的优点
2. **统一接口**：提供统一的 API 支持两种架构
3. **自动转换**：在两种架构间自动转换
4. **性能优化**：为 Object-Aspect 网络添加批量计算优化

