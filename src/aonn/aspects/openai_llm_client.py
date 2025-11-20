# src/aonn/aspects/openai_llm_client.py
"""
OpenAI LLM 客户端：把 OpenAI Chat/Embedding API 封装成 LLMAspect 可调用的接口
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from openai import OpenAI

# 尝试加载 .env 文件（如果可用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未安装时忽略


class OpenAILLMClient(nn.Module):
    """
    使用 OpenAI API 作为语义先验 / 语义预测器

    工作流程：
    1. 将语义向量摘要转换为自然语言提示词
    2. 调用 Chat Completion 生成语义描述
    3. 调用 Embedding API 将文本嵌入到向量空间
    4. 通过线性投影映射到 AONN 所需的 `output_dim`
    """

    _EMBED_DIM_HINTS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: Optional[int] = None,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        summary_size: int = 8,
        max_tokens: int = 120,
        temperature: float = 0.7,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = model
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.summary_size = summary_size

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise RuntimeError(
                "OPENAI_API_KEY 未设置，无法初始化 OpenAILLMClient。\n"
                "请执行以下步骤：\n"
                "1. 复制 .env.example 为 .env: cp .env.example .env\n"
                "2. 编辑 .env 文件，将 OPENAI_API_KEY 替换为你的真实 API Key\n"
                "3. 或者通过环境变量设置: export OPENAI_API_KEY='sk-...'"
            )
        self._client = OpenAI(api_key=api_key)

        self.prompt_template = prompt_template or (
            "You are a semantic state predictor. The latent state summary is [{summary}]. "
            "Describe the next semantic embedding in one short sentence."
        )
        self.system_prompt = system_prompt or (
            "You convert latent state summaries into brief semantic descriptions. "
            "Keep the response concise (<= 30 tokens)."
        )

        self.embedding_dim = embedding_dim or self._infer_embedding_dim(embedding_model)
        self.output_projector = nn.Linear(self.embedding_dim, output_dim).to(self.device)
        nn.init.xavier_uniform_(self.output_projector.weight, gain=0.8)
        nn.init.zeros_(self.output_projector.bias)
        
        # 缓存机制：避免频繁调用 API
        self._cache = {}
        self._cache_max_size = 100
        self._call_count = 0
        self._cache_hits = 0

    @staticmethod
    def _infer_embedding_dim(model_name: str) -> int:
        for key, value in OpenAILLMClient._EMBED_DIM_HINTS.items():
            if key in model_name:
                return value
        # 默认使用 text-embedding-3-small 的维度
        return 1536

    def _vector_to_prompt(self, context_vec: torch.Tensor) -> str:
        values = context_vec.detach().cpu().tolist()
        if not values:
            summary = "0"
        else:
            summary = ", ".join(
                f"{values[i]:.3f}"
                for i in range(min(self.summary_size, len(values)))
            )
        return self.prompt_template.format(summary=summary)

    def _chat_completion(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                timeout=30.0,  # 30秒超时
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            # 如果 API 调用失败，返回默认文本
            import warnings
            warnings.warn(f"OpenAI API 调用失败: {e}，使用默认响应")
            return "Default semantic description due to API error."

    def _text_to_embedding(self, text: str, dtype: torch.dtype) -> torch.Tensor:
        try:
            embedding = self._client.embeddings.create(
                model=self.embedding_model,
                input=text,
                timeout=30.0,  # 30秒超时
            ).data[0].embedding
        except Exception as e:
            # 如果 API 调用失败，返回零向量
            import warnings
            warnings.warn(f"OpenAI Embedding API 调用失败: {e}，使用零向量")
            embedding = [0.0] * self.embedding_dim
        emb_tensor = torch.tensor(embedding, dtype=dtype, device=self.device)
        if emb_tensor.shape[0] != self.embedding_dim:
            # 调整 projector 输入维度
            self.embedding_dim = emb_tensor.shape[0]
            self.output_projector = nn.Linear(self.embedding_dim, self.output_dim).to(self.device)
            nn.init.xavier_uniform_(self.output_projector.weight, gain=0.8)
            nn.init.zeros_(self.output_projector.bias)
        return emb_tensor

    def semantic_predict(self, context_vec: torch.Tensor, use_cache: bool = True, **kwargs) -> torch.Tensor:
        """
        Args:
            context_vec: [input_dim] 或 [batch, input_dim] 的语义向量
        """
        if context_vec.dim() == 1:
            return self._predict_single(context_vec, **kwargs)
        preds = []
        for row in context_vec:
            preds.append(self._predict_single(row, **kwargs))
        return torch.stack(preds, dim=0)

    def _predict_single(self, vec: torch.Tensor, use_cache: bool = True, **kwargs) -> torch.Tensor:
        # 使用向量哈希作为缓存键（只使用前几个值）
        vec_key = tuple(vec[:8].detach().cpu().tolist()) if use_cache else None
        
        if use_cache and vec_key in self._cache:
            self._cache_hits += 1
            return self._cache[vec_key].to(vec.device)
        
        prompt = self._vector_to_prompt(vec)
        try:
            text = self._chat_completion(prompt)
            if not text:
                raise ValueError("Empty completion")
            embedding_tensor = self._text_to_embedding(text, dtype=vec.dtype)
            projected = self.output_projector(embedding_tensor)
            
            # 更新缓存
            if use_cache and vec_key is not None:
                self._call_count += 1
                if len(self._cache) >= self._cache_max_size:
                    # 删除最旧的缓存项（简单策略：删除第一个）
                    self._cache.pop(next(iter(self._cache)))
                self._cache[vec_key] = projected.detach().clone()
            
            return projected
        except Exception:
            # 发生错误时退化为恒等映射，避免中断训练/推理
            return vec.detach().clone()

    def forward(self, context_vec: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.semantic_predict(context_vec, **kwargs)


