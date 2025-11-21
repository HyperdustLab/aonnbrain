# 统一架构下的 MNIST 演化方案

## 概述

本文档描述如何在 AONN 统一架构下，通过动态网络演化解决 MNIST 手写数字识别问题。这展示了统一架构如何同时支持深度学习任务（MNIST）和认知建模任务（LineWorm、OfficeAI、GeneralAI）。

**版本**: 1.0  
**日期**: 2024  
**状态**: 设计完成 → 实现中

---

## MNIST 任务适配

### 1. MNISTWorldModel 设计

将 MNIST 数据集包装为世界模型，使其符合 AONN 的演化框架：

```python
# src/aonn/models/mnist_world_model.py
class MNISTWorldModel:
    """
    MNIST 世界模型：将 MNIST 数据集包装为 AONN 可交互的环境
    
    状态空间：
    - 图像特征表示：256 维（内部状态）
    
    观察空间：
    - 图像像素：784 维（28x28）
    
    动作空间：
    - 分类输出：10 维（10个数字类别）
    
    目标：
    - 正确分类标签：10 维 one-hot 编码
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 10,  # 10个类别
        obs_dim: int = 784,    # 28x28
        device=None,
        dataset=None,  # MNIST 数据集
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.device = device or torch.device("cpu")
        self.dataset = dataset
        
        # 当前样本索引
        self.current_idx = 0
        self.current_image = None
        self.current_label = None
        
        # 加载数据集
        if self.dataset is None:
            from torchvision import datasets, transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.dataset = datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """重置到随机样本，返回观察"""
        self.current_idx = torch.randint(0, len(self.dataset), (1,)).item()
        image, label = self.dataset[self.current_idx]
        
        # 展平图像 [1, 28, 28] -> [784]
        self.current_image = image.flatten().to(self.device)
        self.current_label = label
        
        return {"vision": self.current_image}
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool]:
        """
        执行一步（分类预测）
        
        Args:
            action: [10] 分类预测（logits 或概率）
        
        Returns:
            (observation, reward, done)
        """
        # 计算奖励（分类正确性）
        pred_class = action.argmax().item()
        correct = (pred_class == self.current_label)
        reward = 1.0 if correct else -0.1
        
        # 移动到下一个样本
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        image, label = self.dataset[self.current_idx]
        self.current_image = image.flatten().to(self.device)
        self.current_label = label
        
        # 返回新观察
        obs = {"vision": self.current_image}
        done = False
        
        return obs, reward, done
    
    def get_true_state(self) -> torch.Tensor:
        """获取真实状态（用于学习）"""
        # 返回图像特征表示（可以是从图像编码的）
        # 这里简化，返回图像的某种编码
        return self.current_image[:self.state_dim] if self.current_image is not None \
            else torch.zeros(self.state_dim, device=self.device)
    
    def get_target(self) -> torch.Tensor:
        """获取目标（one-hot 编码的标签）"""
        target = torch.zeros(10, device=self.device)
        if self.current_label is not None:
            target[self.current_label] = 1.0
        return target
```

### 2. ClassificationAspect 设计

创建专门用于分类的 Aspect：

```python
# src/aonn/aspects/classification_aspect.py
class ClassificationAspect(AspectBase, nn.Module):
    """
    分类 Aspect：从 internal 状态预测类别
    
    自由能 = 交叉熵损失
    F = -log(p(target|internal))
    """
    
    def __init__(
        self,
        internal_name: str = "internal",
        target_name: str = "target",
        state_dim: int = 256,
        num_classes: int = 10,
        name: str = "classification",
    ):
        super().__init__(name=name, src_names=[internal_name], dst_names=[target_name])
        
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, objects: Dict[str, ObjectNode]) -> Dict[str, torch.Tensor]:
        """计算预测误差"""
        internal = objects[self.src_names[0]].state
        logits = self.classifier(internal)
        
        target = objects[self.dst_names[0]].state  # one-hot
        pred_probs = torch.softmax(logits, dim=-1)
        
        # 误差 = 目标概率 - 预测概率
        error = target - pred_probs
        return {self.dst_names[0]: error}
    
    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """计算自由能贡献（交叉熵）"""
        internal = objects[self.src_names[0]].state
        logits = self.classifier(internal)
        
        target = objects[self.dst_names[0]].state  # one-hot
        target_class = target.argmax(dim=-1)
        
        # 交叉熵损失
        F = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), 
            target_class.unsqueeze(0),
            reduction='none'
        )
        return F.squeeze(0)
    
    def parameters(self):
        return list(self.classifier.parameters())
```

---

## 网络演化流程

### 初始架构

```python
初始网络:
  Objects: 
    - vision (784维): 图像输入
    - internal (256维): 内部状态
    - target (10维): 分类目标（one-hot）
  Aspects: {}
  Pipelines: {}
```

### 演化步骤

#### 步骤 1: 创建视觉编码 Pipeline

当自由能高时（图像无法正确编码），创建 Pipeline：

```python
# 在 evolve_network 中
if F_current > threshold:
    # 创建 vision -> internal 的 Pipeline
    vision_pipeline = brain.create_unified_aspect(
        aspect_type="pipeline",
        src_names=["vision"],
        dst_names=["internal"],
        input_dim=784,
        output_dim=256,
        num_aspects=64,
        depth=3,
    )
    brain.aspects.append(vision_pipeline)
```

**演化触发条件**：
- 自由能 > `free_energy_threshold`
- `vision` Object 的误差高

#### 步骤 2: 创建分类 Aspect

当需要预测类别时，创建分类 Aspect：

```python
# 创建分类 Aspect
classification_aspect = ClassificationAspect(
    internal_name="internal",
    target_name="target",
    state_dim=256,
    num_classes=10,
)
brain.aspects.append(classification_aspect)
```

**演化触发条件**：
- `target` Object 存在
- 分类误差高

#### 步骤 3: 扩展 Pipeline 深度

当分类精度不足时，增加 Pipeline 深度：

```python
# 在 maybe_expand_action_pipeline 中
# 如果分类误差高，增加 internal -> internal 的 Pipeline
if classification_error > threshold:
    brain.maybe_expand_action_pipeline(F_current)
```

**演化触发条件**：
- 分类自由能 > `pipeline_growth.free_energy_trigger`
- 达到最小间隔步数

### 完整演化示例

```python
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.models.mnist_world_model import MNISTWorldModel
from aonn.aspects.classification_aspect import ClassificationAspect

# 配置
config = {
    "obs_dim": 784,
    "state_dim": 256,
    "act_dim": 10,
    "sense_dims": {"vision": 784},
    "evolution": {
        "free_energy_threshold": 2.0,  # 分类任务需要更高的阈值
        "max_aspects": 500,
        "batch_growth": {
            "base": 8,
            "max_per_step": 32,
            "error_threshold": 1.5,
        },
        "pipeline_growth": {
            "free_energy_trigger": 1.0,
            "min_interval": 10,
            "max_stages": 5,
            "max_depth": 6,
            "initial_width": 64,
        },
    },
}

# 创建 AONN Brain
brain = AONNBrainV3(config=config, enable_evolution=True)

# 创建 MNIST 世界模型
world_model = MNISTWorldModel(
    state_dim=256,
    action_dim=10,
    obs_dim=784,
)

# 创建分类 Aspect（初始）
classification_aspect = ClassificationAspect(
    internal_name="internal",
    target_name="target",
    state_dim=256,
    num_classes=10,
)
brain.aspects.append(classification_aspect)

# 创建 target Object
brain.create_object("target", dim=10)

# 演化循环
for step in range(num_steps):
    # 1. 获取观察和目标
    obs = world_model.reset()
    target = world_model.get_target()
    
    # 2. 设置观察和目标
    brain.objects["vision"].set_state(obs["vision"])
    brain.objects["target"].set_state(target)
    
    # 3. 网络演化
    brain.evolve_network(obs, target=target)
    
    # 4. 主动推理（更新 internal 状态）
    loop = ActiveInferenceLoop(
        brain.objects,
        brain.aspects,
        infer_lr=0.01,
    )
    loop.infer_states(target_objects=("internal",), num_iters=3)
    
    # 5. 世界模型学习（学习分类器参数）
    brain.learn_world_model(
        observation=obs,
        action=None,  # 分类任务不需要动作
        next_observation=obs,  # 简化
        target_state=world_model.get_true_state(),
        learning_rate=0.001,
    )
    
    # 6. 获取预测
    action = brain.objects["action"].state if "action" in brain.objects \
        else classification_aspect.classifier(brain.objects["internal"].state)
    
    # 7. 评估
    pred_class = action.argmax().item()
    true_class = target.argmax().item()
    accuracy = 1.0 if pred_class == true_class else 0.0
```

---

## 演化策略

### 1. 初始阶段（步骤 0-100）

**目标**：建立基础编码能力

- 创建 `vision -> internal` Pipeline（3-4层，64个Aspect）
- 创建分类 Aspect
- 自由能阈值：2.0（允许较高误差）

**预期**：
- 网络开始学习图像特征
- 自由能从高（~10）降到中等（~3-5）

### 2. 优化阶段（步骤 100-500）

**目标**：提高分类精度

- 扩展 Pipeline 深度（增加到 5-6 层）
- 增加 Pipeline 宽度（64 -> 128 个Aspect）
- 创建额外的 `internal -> internal` Pipeline（增加深度）
- 自由能阈值：1.5

**预期**：
- 分类准确率从随机（10%）提升到 70-80%
- 自由能降到 1.0-2.0

### 3. 精炼阶段（步骤 500+）

**目标**：达到高精度

- 进一步优化 Pipeline 结构
- 剪枝不重要的 Aspect
- 自由能阈值：0.5

**预期**：
- 分类准确率达到 90-95%
- 自由能降到 0.5 以下

---

## 关键设计点

### 1. 自由能定义

对于分类任务，自由能 = 交叉熵损失：

```python
F_classification = -log(p(target_class | internal_state))
```

### 2. 演化触发

- **Pipeline 创建**：当 `vision` 误差高时创建编码 Pipeline
- **Pipeline 扩展**：当分类误差高时扩展深度
- **Aspect 创建**：当需要新功能时创建（如分类 Aspect）

### 3. 学习机制

- **主动推理**：更新 `internal` 状态，使其更好地编码图像
- **参数学习**：学习 Pipeline 和分类器的参数
- **结构演化**：根据自由能动态调整网络结构

### 4. 混合使用 Pipeline 和独立 Aspect

- **Pipeline**：用于 `vision -> internal` 编码（高效批量处理）
- **独立 Aspect**：用于分类（灵活，易于解释）

---

## 预期演化轨迹

```
步骤 0:   Objects=3,  Aspects=1,  Pipelines=0,  F=10.0,  Acc=10%
步骤 50:  Objects=3,  Aspects=1,  Pipelines=1,  F=5.0,   Acc=30%
步骤 100: Objects=3,  Aspects=1,  Pipelines=1,  F=3.0,   Acc=50%
步骤 200: Objects=3,  Aspects=1,  Pipelines=2,  F=2.0,   Acc=70%
步骤 500: Objects=3,  Aspects=1,  Pipelines=3,  F=1.0,   Acc=85%
步骤 1000:Objects=3,  Aspects=1,  Pipelines=4,  F=0.5,   Acc=92%
```

---

## 优势

1. **自动演化**：网络根据任务自动调整结构
2. **统一框架**：Pipeline 和独立 Aspect 统一使用
3. **可解释性**：可以观察网络如何演化来解决分类问题
4. **灵活性**：可以根据任务复杂度自动调整

---

## 实现检查清单

- [ ] 创建 `MNISTWorldModel`
- [ ] 创建 `ClassificationAspect`
- [ ] 实现 `PipelineAspect`（统一接口）
- [ ] 修改 `evolve_network` 支持分类任务
- [ ] 添加分类任务的演化策略
- [ ] 实现 MNIST 实验脚本
- [ ] 添加分类准确率评估
- [ ] 可视化演化过程

---

## 总结

在统一架构下，MNIST 任务可以通过以下方式演化：

1. **初始**：最小网络（只有 Objects）
2. **演化**：根据分类误差（自由能）创建 Pipeline 和 Aspect
3. **学习**：通过自由能最小化学习参数
4. **优化**：动态调整网络结构以提高精度

这展示了 AONN 统一架构的强大能力：**同一个框架可以处理从简单分类到复杂认知建模的各种任务**。

---

## 相关文档

- [AONN 统一架构设计](./AONN_UNIFIED_ARCHITECTURE.md)
- [AONN V3 演化文档](./AONN_V3_EVOLUTION.md)
- [AONN 架构对比](./AONN_ARCHITECTURE_COMPARISON.md)

---

**文档维护者**: AONN 开发团队  
**最后更新**: 2024  
**版本**: 1.0

