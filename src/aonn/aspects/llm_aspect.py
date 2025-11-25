# src/aonn/aspects/llm_aspect.py
from typing import Dict, Any, List, Tuple, Optional
import torch

from aonn.core.object import ObjectNode
from aonn.core.aspect_base import AspectBase

# 尝试导入 sentence-transformers（可选依赖）
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore


class LLMAspect(AspectBase):
    """
    LLMAspect：把 LLM 当作"语义预测因子"，不当黑盒大脑。

    - 读取若干 Object state（比如 semantic_context, intent）
    - 构造 prompt / 输入向量，调用外部 LLM 推理
    - 生成"预测输出"或"期望语义"，与 Object 中某个语义 Object 比较，形成自由能
    
    马尔可夫毯原则：
    - 系统只能通过感觉状态（sensory states）和主动状态（active states）与环境交互
    - 系统不应该直接访问外部世界的真实状态（ground truth）
    - compute_semantic_similarity() 方法仅用于外部评估，不应在系统内部的自由能计算中使用
    - 语义匹配结果（通过 set_semantic_match_result() 设置）仅用于外部查询，不影响系统内部推理
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
        enable_semantic_matching: bool = True,  # 是否启用语义相似度匹配
        similarity_threshold: float = 0.3,  # 语义相似度阈值
        semantic_matching_weight: float = 0.0,  # 语义匹配对自由能的贡献权重（0表示不加入自由能）
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
        
        # 语义相似度匹配相关
        self.enable_semantic_matching = enable_semantic_matching
        self.similarity_threshold = similarity_threshold
        self.semantic_matching_weight = semantic_matching_weight
        self._similarity_model = None  # 延迟加载 sentence-transformers 模型
        self._last_semantic_match_result = None  # 存储最后一次语义匹配结果

    def _get_similarity_model(self):
        """延迟加载 sentence-transformers 模型"""
        if not HAS_SENTENCE_TRANSFORMERS:
            return None
        if self._similarity_model is None:
            try:
                self._similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                import warnings
                warnings.warn(f"无法加载 sentence-transformers 模型: {e}")
                return None
        return self._similarity_model
    
    def compute_semantic_similarity(
        self,
        llm_text: str,
        expected_keywords: List[str],
        context_description: str = "",
        expectations: Optional[List[str]] = None,
    ) -> Tuple[float, List[str], List[str], Dict[str, float]]:
        """
        计算 LLM 输出文本与期望关键词的语义相似度（仅用于外部评估）
        
        注意：此方法需要访问期望关键词（ground truth），因此只能在系统外部使用，
        不应在系统内部的自由能计算中调用，否则会破坏马尔可夫毯原则。
        
        Args:
            llm_text: LLM 生成的文本描述
            expected_keywords: 期望的关键词列表（来自外部测试数据，不是系统内部状态）
            context_description: 上下文描述（用于增强匹配）
            expectations: 期望列表（用于增强匹配）
        
        Returns:
            (coverage_ratio, matched_keywords, missing_keywords, keyword_similarities)
        """
        if not expected_keywords:
            return 0.0, [], [], {}
        
        expectations = expectations or []
        keyword_similarities = {}
        
        model = self._get_similarity_model()
        if model is None:
            # 降级到精确字符串匹配
            tlower = llm_text.lower()
            matched = []
            missing = []
            for kw in expected_keywords:
                if kw.lower() in tlower:
                    matched.append(kw)
                    keyword_similarities[kw] = 1.0
                else:
                    missing.append(kw)
                    keyword_similarities[kw] = 0.0
            coverage = len(matched) / len(expected_keywords) if expected_keywords else 0.0
            return coverage, matched, missing, keyword_similarities
        
        try:
            # 使用语义相似度匹配
            # 嵌入 LLM 文本
            text_embedding = model.encode(llm_text, convert_to_tensor=True)
            
            matched = []
            missing = []
            
            for kw in expected_keywords:
                # 嵌入关键词
                kw_embedding = model.encode(kw, convert_to_tensor=True)
                
                # 计算余弦相似度
                from torch.nn.functional import cosine_similarity
                similarity = cosine_similarity(
                    text_embedding.unsqueeze(0), 
                    kw_embedding.unsqueeze(0)
                ).item()
                keyword_similarities[kw] = similarity
                
                # 如果提供了上下文描述和期望，可以增强匹配
                context_text = f"{context_description} {' '.join(expectations)}"
                if context_text.strip():
                    context_embedding = model.encode(context_text, convert_to_tensor=True)
                    context_sim = cosine_similarity(
                        text_embedding.unsqueeze(0), 
                        context_embedding.unsqueeze(0)
                    ).item()
                    # 如果上下文相似度高，提升关键词相似度
                    if context_sim > 0.4:
                        similarity = max(similarity, context_sim * 0.7)
                        keyword_similarities[kw] = similarity
                
                if similarity >= self.similarity_threshold:
                    matched.append(kw)
                else:
                    missing.append(kw)
            
            coverage = len(matched) / len(expected_keywords) if expected_keywords else 0.0
            return coverage, matched, missing, keyword_similarities
            
        except Exception as e:
            import warnings
            warnings.warn(f"语义相似度计算失败: {e}，降级到精确匹配")
            # 降级到精确匹配
            tlower = llm_text.lower()
            matched = []
            missing = []
            for kw in expected_keywords:
                if kw.lower() in tlower:
                    matched.append(kw)
                    keyword_similarities[kw] = 1.0
                else:
                    missing.append(kw)
                    keyword_similarities[kw] = 0.0
            coverage = len(matched) / len(expected_keywords) if expected_keywords else 0.0
            return coverage, matched, missing, keyword_similarities
    
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
        src_object = objects[self.src_names[0]]
        context_vec = src_object.state  # e.g. [d_model]
        context_metadata = getattr(src_object, "metadata", None)

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
            pred_vec = self.llm_client.semantic_predict(
                context_vec,
                context_metadata=context_metadata,
                **self.llm_config,
            )
            # 确保返回的是 torch.Tensor
            if not isinstance(pred_vec, torch.Tensor):
                pred_vec = torch.tensor(pred_vec, device=context_vec.device, dtype=context_vec.dtype)
            # 缓存预测结果
            self._last_prediction = pred_vec.detach().clone()
            
            # 注意：语义相似度匹配不应该在系统内部使用期望关键词（ground truth），
            # 这会破坏马尔可夫毯原则。语义匹配应该作为外部评估工具使用。
            # 如果需要评估，应该在系统外部调用 compute_semantic_similarity() 方法。
            
            return pred_vec
        elif hasattr(self.llm_client, '__call__'):
            # 如果客户端是可调用的（比如 nn.Module），尝试传入 metadata
            try:
                pred_vec = self.llm_client(
                    context_vec,
                    context_metadata=context_metadata,
                    **self.llm_config,
                )
            except TypeError:
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
        free_energy = 0.5 * self.loss_weight * (error ** 2).sum()
        
        # 注意：语义匹配不应该影响自由能计算，因为它需要访问外部世界的真实状态（ground truth），
        # 这会破坏马尔可夫毯原则。系统应该只通过感觉状态和主动状态与环境交互。
        
        return free_energy
    
    def get_semantic_match_result(self) -> Optional[Dict[str, Any]]:
        """
        获取最后一次语义匹配结果（仅用于外部评估，不应在系统内部使用）
        
        注意：此方法返回的结果是通过外部调用 compute_semantic_similarity() 得到的，
        不应该在系统内部的自由能计算中使用，因为这会破坏马尔可夫毯原则。
        
        Returns:
            包含 coverage, matched_keywords, missing_keywords, keyword_similarities 的字典
            如果没有进行过匹配，返回 None
        """
        return self._last_semantic_match_result
    
    def set_semantic_match_result(self, result: Optional[Dict[str, Any]]):
        """
        设置语义匹配结果（仅用于外部评估）
        
        此方法允许外部评估工具设置匹配结果，但不会影响系统内部的自由能计算。
        
        Args:
            result: 语义匹配结果字典，包含 coverage, matched_keywords, missing_keywords, keyword_similarities
        """
        self._last_semantic_match_result = result
    
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

