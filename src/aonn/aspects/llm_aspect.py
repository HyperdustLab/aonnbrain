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
    ):
        super().__init__(name=name, src_names=src_names, dst_names=dst_names)
        self.llm_client = llm_client   # 比如 OpenAI client / HTTP client / MockLLMClient
        self.llm_config = llm_config or {}
        self.loss_weight = loss_weight

    def _call_llm(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        """
        调用 LLM 客户端进行语义预测：
        - 从 semantic_context 里抽特征（或直接用文本）
        - 调 LLM 拿到一个"预测语义向量" or logits
        """
        context_vec = objects[self.src_names[0]].state  # e.g. [d_model]

        # 如果没有提供客户端，使用 identity 作为占位
        if self.llm_client is None:
            return context_vec.clone()

        # 检查客户端是否有 semantic_predict 方法（MockLLMClient 或真实 LLM 客户端）
        if hasattr(self.llm_client, 'semantic_predict'):
            # 调用客户端的 semantic_predict 方法
            pred_vec = self.llm_client.semantic_predict(context_vec, **self.llm_config)
            # 确保返回的是 torch.Tensor
            if not isinstance(pred_vec, torch.Tensor):
                pred_vec = torch.tensor(pred_vec, device=context_vec.device, dtype=context_vec.dtype)
            return pred_vec
        elif hasattr(self.llm_client, '__call__'):
            # 如果客户端是可调用的（比如 nn.Module），直接调用
            pred_vec = self.llm_client(context_vec, **self.llm_config)
            if not isinstance(pred_vec, torch.Tensor):
                pred_vec = torch.tensor(pred_vec, device=context_vec.device, dtype=context_vec.dtype)
            return pred_vec
        else:
            raise NotImplementedError(
                "LLM 客户端必须实现 semantic_predict 方法或可调用。"
                "请使用 MockLLMClient 或实现兼容的客户端接口。"
            )

    def forward(self, objects: Dict[str, ObjectNode]):
        """
        返回：对 semantic_prediction Object 的误差
        """
        pred_sem = self._call_llm(objects)  # [d]
        target_sem = objects[self.dst_names[0]].state
        error = target_sem - pred_sem
        return {self.dst_names[0]: error}

    def free_energy_contrib(self, objects: Dict[str, ObjectNode]) -> torch.Tensor:
        pred_sem = self._call_llm(objects)
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

