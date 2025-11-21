# src/aonn/aspects/llm_aspect.py
from typing import Dict, Any
import torch

from aonn.core.object import ObjectNode
from aonn.core.aspect_base import AspectBase


class LLMAspect(AspectBase):
    """
    LLMAspect：把 LLM 当作"语义预测因子"，不当黑盒大脑。

    - 读取若干 Object state（比如 semantic_context, intent）
    - 构造 prompt / 输入向量，调用外部 LLM 推理
    - 生成"预测输出"或"期望语义"，与 Object 中某个语义 Object 比较，形成自由能
    """

    def __init__(
        self,
        name: str = "llm_aspect",
        src_names=("semantic_context",),
        dst_names=("semantic_prediction",),
        llm_client: Any = None,
        llm_config: Dict[str, Any] = None,
        loss_weight: float = 1.0,
        call_frequency: str = "last_iter_only",  # "every_iter", "last_iter_only", "every_n_steps"
        call_every_n_steps: int = 1,  # 当 call_frequency="every_n_steps" 时使用
    ):
        super().__init__(name=name, src_names=src_names, dst_names=dst_names)
        self.llm_client = llm_client   # 比如 OpenAI client / HTTP client / MockLLMClient
        self.llm_config = llm_config or {}
        self.loss_weight = loss_weight
        self.call_frequency = call_frequency  # 控制 LLM 调用频率
        self.call_every_n_steps = call_every_n_steps
        self._last_prediction = None  # 缓存最后一次预测
        self._step_counter = 0  # 步数计数器
        self._last_iteration_idx = None  # 最后一次迭代的索引

    def _call_llm(self, objects: Dict[str, ObjectNode], iteration_idx: int = None, is_last_iter: bool = False) -> torch.Tensor:
        """
        调用 LLM 客户端进行语义预测：
        - 从 semantic_context 里抽特征（或直接用文本）
        - 调 LLM 拿到一个"预测语义向量" or logits
        
        Args:
            objects: Object 字典
            iteration_idx: 当前迭代索引（用于判断是否调用 LLM）
            is_last_iter: 是否是最后一次迭代
        """
        context_vec = objects[self.src_names[0]].state  # e.g. [d_model]

        # 如果没有提供客户端，使用 identity 作为占位
        if self.llm_client is None:
            return context_vec.clone()

        # 根据调用频率策略决定是否调用 LLM
        should_call_llm = False
        if self.call_frequency == "every_iter":
            should_call_llm = True
        elif self.call_frequency == "last_iter_only":
            should_call_llm = is_last_iter
        elif self.call_frequency == "every_n_steps":
            should_call_llm = (self._step_counter % self.call_every_n_steps == 0)
        else:
            should_call_llm = True  # 默认每次都调用
        
        # 如果不需要调用 LLM，使用缓存的预测
        if not should_call_llm and self._last_prediction is not None:
            return self._last_prediction.to(context_vec.device)

        # 检查客户端是否有 semantic_predict 方法（MockLLMClient 或真实 LLM 客户端）
        if hasattr(self.llm_client, 'semantic_predict'):
            # 调用客户端的 semantic_predict 方法
            pred_vec = self.llm_client.semantic_predict(context_vec, **self.llm_config)
            # 确保返回的是 torch.Tensor
            if not isinstance(pred_vec, torch.Tensor):
                pred_vec = torch.tensor(pred_vec, device=context_vec.device, dtype=context_vec.dtype)
            # 缓存预测结果
            self._last_prediction = pred_vec.detach().clone()
            return pred_vec
        elif hasattr(self.llm_client, '__call__'):
            # 如果客户端是可调用的（比如 nn.Module），直接调用
            pred_vec = self.llm_client(context_vec, **self.llm_config)
            if not isinstance(pred_vec, torch.Tensor):
                pred_vec = torch.tensor(pred_vec, device=context_vec.device, dtype=context_vec.dtype)
            # 缓存预测结果
            self._last_prediction = pred_vec.detach().clone()
            return pred_vec
        else:
            raise NotImplementedError(
                "LLM 客户端必须实现 semantic_predict 方法或可调用。"
                "请使用 MockLLMClient 或实现兼容的客户端接口。"
            )
    
    def set_iteration_info(self, iteration_idx: int, is_last_iter: bool, step_counter: int = None):
        """
        设置迭代信息，用于决定是否调用 LLM
        
        Args:
            iteration_idx: 当前迭代索引
            is_last_iter: 是否是最后一次迭代
            step_counter: 步数计数器（可选）
        """
        self._last_iteration_idx = iteration_idx
        if step_counter is not None:
            self._step_counter = step_counter

    def forward(self, objects: Dict[str, ObjectNode], iteration_idx: int = None, is_last_iter: bool = False):
        """
        返回：对 semantic_prediction Object 的误差
        
        Args:
            objects: Object 字典
            iteration_idx: 当前迭代索引（可选）
            is_last_iter: 是否是最后一次迭代（可选）
        """
        pred_sem = self._call_llm(objects, iteration_idx=iteration_idx, is_last_iter=is_last_iter)  # [d]
        target_sem = objects[self.dst_names[0]].state
        error = target_sem - pred_sem
        return {self.dst_names[0]: error}

    def free_energy_contrib(self, objects: Dict[str, ObjectNode], iteration_idx: int = None, is_last_iter: bool = False) -> torch.Tensor:
        """
        计算自由能贡献
        
        Args:
            objects: Object 字典
            iteration_idx: 当前迭代索引（可选）
            is_last_iter: 是否是最后一次迭代（可选）
        """
        pred_sem = self._call_llm(objects, iteration_idx=iteration_idx, is_last_iter=is_last_iter)
        target_sem = objects[self.dst_names[0]].state
        error = target_sem - pred_sem
        return 0.5 * self.loss_weight * (error ** 2).sum()
    
    def parameters(self):
        """
        返回 LLMAspect 的可训练参数
        如果 llm_client 是可训练的（如 MockLLMClient），包含其参数
        """
        params = []
        if self.llm_client is not None:
            if hasattr(self.llm_client, 'parameters'):
                params.extend(self.llm_client.parameters())
        return params

