# src/aonn/aspects/ollama_llm_client.py
"""
Ollama LLM 客户端：把本地 Ollama API 封装成 LLMAspect 可调用的接口
"""
from __future__ import annotations

import json
import os
from typing import Optional
import warnings

import torch
import torch.nn as nn
import requests

# 尝试加载 .env 文件（如果可用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未安装时忽略


class OllamaLLMClient(nn.Module):
    """
    使用本地 Ollama API 作为语义先验 / 语义预测器

    工作流程：
    1. 将语义向量摘要转换为自然语言提示词
    2. 调用 Ollama Chat API 生成语义描述
    3. 调用 Ollama Embedding API 将文本嵌入到向量空间
    4. 通过线性投影映射到 AONN 所需的 `output_dim`
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        embedding_model: Optional[str] = None,  # 如果为 None，使用 model
        embedding_dim: Optional[int] = None,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        summary_size: int = 8,
        max_tokens: int = 120,
        temperature: float = 0.7,
        verbose: bool = False,
        timeout: float = 120.0,  # Ollama 本地调用可能需要更长时间（大模型）
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = model
        self.embedding_model = embedding_model or model  # 默认使用相同的模型
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.summary_size = summary_size
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

        # 检查 Ollama 服务是否可用
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code != 200:
                warnings.warn(f"Ollama API 可能不可用 (状态码: {response.status_code})")
        except Exception as e:
            warnings.warn(f"无法连接到 Ollama API ({self.base_url}): {e}")

        self.prompt_template = prompt_template or (
            "You are a semantic state predictor. The latent state summary is [{summary}]. "
            "Describe the next semantic embedding in one short sentence."
        )
        self.system_prompt = system_prompt or (
            "You convert latent state summaries into brief semantic descriptions. "
            "Keep the response concise (<= 30 tokens)."
        )

        # Ollama embedding 维度通常是固定的（取决于模型）
        # 常见模型：llama3=4096, mistral=4096, qwen=4096
        self.embedding_dim = embedding_dim or self._infer_embedding_dim(self.embedding_model)
        self.output_projector = nn.Linear(self.embedding_dim, output_dim).to(self.device)
        nn.init.xavier_uniform_(self.output_projector.weight, gain=0.8)
        nn.init.zeros_(self.output_projector.bias)
        
        # 缓存机制：避免频繁调用 API
        self._cache = {}
        self._cache_max_size = 100
        self._call_count = 0
        self._cache_hits = 0
        
        # 输出控制
        self.verbose = verbose
        self._last_generated_text = None

    @staticmethod
    def _infer_embedding_dim(model_name: str) -> int:
        """
        根据模型名称推断 embedding 维度
        """
        # Ollama 常见模型的 embedding 维度
        model_dims = {
            "llama3": 4096,
            "llama2": 4096,
            "mistral": 4096,
            "qwen": 4096,
            "phi": 2048,
            "gemma": 2560,
        }
        
        for key, dim in model_dims.items():
            if key in model_name.lower():
                return dim
        
        # 默认值（大多数现代模型使用 4096）
        return 4096

    def _vector_to_prompt(self, context_vec: torch.Tensor) -> str:
        """
        将语义向量转换为 prompt 字符串
        """
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
        """
        调用 Ollama Chat API 生成文本
        """
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,  # Ollama 使用 num_predict 而不是 max_tokens
                },
                "stream": False,  # 非流式响应
            }
            
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            import warnings
            warnings.warn(f"Ollama Chat API 调用失败: {e}，使用默认响应")
            return "Default semantic description due to API error."
        except Exception as e:
            import warnings
            warnings.warn(f"Ollama Chat API 调用发生未知错误: {e}，使用默认响应")
            return "Default semantic description due to API error."

    def _text_to_embedding(self, text: str, dtype: torch.dtype) -> torch.Tensor:
        """
        调用 Ollama Embedding API 将文本转换为向量
        如果模型不支持 embedding，使用文本特征降级方案
        """
        try:
            url = f"{self.base_url}/api/embeddings"
            payload = {
                "model": self.embedding_model,
                "prompt": text,
            }
            
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("embedding", [])
            
            if not embedding:
                raise ValueError("Empty embedding from Ollama API")
            
            # 更新 embedding_dim（如果实际维度与预期不同）
            actual_dim = len(embedding)
            if actual_dim != self.embedding_dim:
                if self.verbose:
                    print(f"  [Ollama] Embedding 维度从 {self.embedding_dim} 调整为 {actual_dim}")
                self.embedding_dim = actual_dim
                self.output_projector = nn.Linear(self.embedding_dim, self.output_dim).to(self.device)
                nn.init.xavier_uniform_(self.output_projector.weight, gain=0.8)
                nn.init.zeros_(self.output_projector.bias)
            
            return torch.tensor(embedding, dtype=dtype, device=self.device)
        except requests.exceptions.HTTPError as e:
            # HTTP 错误（如 404, 500 等）
            if e.response.status_code == 404:
                # 404 可能是模型不支持 embedding，或者模型正在使用中
                if self.verbose:
                    import warnings
                    warnings.warn(f"Ollama Embedding API 返回 404，可能是模型不支持或正在使用中，使用文本特征降级方案")
            else:
                if self.verbose:
                    import warnings
                    warnings.warn(f"Ollama Embedding API HTTP 错误 ({e.response.status_code}): {e}，使用文本特征降级方案")
            return self._text_to_features_fallback(text, dtype)
        except (requests.exceptions.RequestException, ValueError) as e:
            # 其他错误（超时、网络错误等）
            if self.verbose:
                import warnings
                warnings.warn(f"Ollama Embedding 不可用 ({e})，使用文本特征降级方案")
            return self._text_to_features_fallback(text, dtype)
        except Exception as e:
            import warnings
            warnings.warn(f"Ollama Embedding API 调用发生未知错误: {e}，使用文本特征降级方案")
            return self._text_to_features_fallback(text, dtype)
    
    def _text_to_features_fallback(self, text: str, dtype: torch.dtype) -> torch.Tensor:
        """
        文本特征降级方案：当 embedding 不可用时，使用简单的文本特征
        基于字符频率、长度等特征生成固定维度的向量
        """
        import hashlib
        import numpy as np
        
        # 方法1: 使用文本的哈希值生成伪随机向量
        text_hash = hashlib.sha256(text.encode()).digest()
        
        # 将哈希转换为固定维度的向量
        # 使用多个哈希值来填充 embedding_dim
        num_hashes = (self.embedding_dim // 32) + 1
        features = []
        for i in range(num_hashes):
            # 使用不同的哈希算法或偏移
            hash_input = text_hash + bytes([i])
            h = hashlib.sha256(hash_input).digest()
            # 将 32 字节转换为 8 个 float32 (每个 4 字节)
            for j in range(0, min(32, self.embedding_dim - len(features)), 4):
                if len(features) >= self.embedding_dim:
                    break
                val = int.from_bytes(h[j:j+4], 'big', signed=False)
                # 归一化到 [-1, 1]
                normalized = (val / (2**31)) - 1.0
                features.append(normalized)
        
        # 如果还不够，用文本特征补充
        while len(features) < self.embedding_dim:
            # 使用文本长度、字符多样性等特征
            text_len = len(text)
            char_diversity = len(set(text)) / max(len(text), 1)
            features.append(text_len / 1000.0)  # 归一化长度
            features.append(char_diversity)
            if len(features) >= self.embedding_dim:
                break
        
        # 截断或填充到 exact embedding_dim
        features = features[:self.embedding_dim]
        if len(features) < self.embedding_dim:
            features.extend([0.0] * (self.embedding_dim - len(features)))
        
        return torch.tensor(features, dtype=dtype, device=self.device)

    def semantic_predict(self, context_vec: torch.Tensor, use_cache: bool = True, **kwargs) -> torch.Tensor:
        """
        语义预测：从语义上下文向量生成预测语义向量
        
        Args:
            context_vec: 语义上下文向量 [d] 或 [batch, d]
            use_cache: 是否使用缓存
            **kwargs: 其他参数
        
        Returns:
            预测的语义向量 [d] 或 [batch, d]
        """
        if context_vec.dim() == 1:
            return self._predict_single(context_vec, use_cache=use_cache, **kwargs)
        
        # 批量处理
        preds = []
        for row in context_vec:
            preds.append(self._predict_single(row, use_cache=use_cache, **kwargs))
        return torch.stack(preds, dim=0)

    def _predict_single(self, vec: torch.Tensor, use_cache: bool = True, **kwargs) -> torch.Tensor:
        """
        单个向量的预测
        """
        # 简单的缓存键（使用前几个值的哈希）
        vec_key = None
        if use_cache:
            import hashlib
            vec_hash = hashlib.sha256(vec[:8].detach().cpu().numpy().tobytes()).hexdigest()
            vec_key = (vec_hash, tuple(vec[:8].detach().cpu().tolist()))
            
            if vec_key in self._cache:
                self._cache_hits += 1
                cached_val = self._cache.pop(vec_key)
                self._cache[vec_key] = cached_val  # 移到末尾（LRU）
                return cached_val.to(vec.device)
            
            self._call_count += 1
        
        prompt = self._vector_to_prompt(vec)
        try:
            text = self._chat_completion(prompt)
            if not text or text.strip() == "":
                # 如果 completion 为空，使用默认文本
                text = "Default semantic state description."
                if self.verbose:
                    import warnings
                    warnings.warn("Ollama 返回空 completion，使用默认文本")
            
            # 保存生成的文本（用于输出）
            self._last_generated_text = text
            
            # 如果启用 verbose，输出生成的文本
            if self.verbose:
                print(f"  [Ollama] 生成的语义描述: {text}")
            
            embedding_tensor = self._text_to_embedding(text, dtype=vec.dtype)
            projected = self.output_projector(embedding_tensor)
            
            # 更新缓存
            if use_cache and vec_key is not None:
                if len(self._cache) >= self._cache_max_size:
                    # 删除最旧的缓存项
                    self._cache.pop(next(iter(self._cache)))
                self._cache[vec_key] = projected.detach().clone()
            
            return projected
        except Exception as e:
            import warnings
            warnings.warn(f"Ollama 预测失败: {e}，使用恒等映射")
            # 发生错误时退化为恒等映射
            if self.output_dim == vec.shape[-1]:
                return vec.detach().clone()
            else:
                return torch.zeros(self.output_dim, device=vec.device, dtype=vec.dtype)

    def forward(self, context_vec: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        nn.Module 接口：直接调用 semantic_predict
        """
        return self.semantic_predict(context_vec, **kwargs)

