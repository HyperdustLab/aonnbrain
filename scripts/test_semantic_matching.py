#!/usr/bin/env python3
"""
测试 LLMAspect 的语义相似度匹配功能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aonn.aspects.llm_aspect import LLMAspect
from aonn.aspects.mock_llm_client import MockLLMClient
from aonn.core.object import ObjectNode

def test_semantic_matching():
    """测试语义匹配功能"""
    device = torch.device("cpu")
    sem_dim = 128
    
    # 创建 Mock LLM 客户端
    llm_client = MockLLMClient(
        input_dim=sem_dim,
        output_dim=sem_dim,
        device=device,
    )
    
    # 创建 LLMAspect，启用语义匹配
    llm_aspect = LLMAspect(
        name="test_llm_aspect",
        src_names=("semantic_context",),
        dst_names=("semantic_prediction",),
        llm_client=llm_client,
        enable_semantic_matching=True,
        similarity_threshold=0.3,
    )
    
    # 创建测试对象
    context_obj = ObjectNode("semantic_context", sem_dim, device=device)
    pred_obj = ObjectNode("semantic_prediction", sem_dim, device=device)
    
    # 设置上下文状态
    context_vec = torch.randn(sem_dim, device=device) * 0.1
    context_obj.set_state(context_vec)
    pred_obj.set_state(torch.zeros(sem_dim, device=device))
    
    # 设置 metadata（包含期望的关键词）
    metadata = {
        "text": "A team meeting is scheduled to discuss project progress and deadlines.",
        "keywords": ["meeting", "team", "project", "deadline", "progress"],
        "expectations": ["Discuss project status", "Review deadlines"],
    }
    context_obj.set_metadata(metadata)
    
    objects = {
        "semantic_context": context_obj,
        "semantic_prediction": pred_obj,
    }
    
    # 运行 LLMAspect（系统内部，不访问期望关键词）
    llm_aspect.set_iteration_info(iteration_idx=0, is_last_iter=True, step_counter=0)
    with torch.no_grad():
        result = llm_aspect.forward(objects, iteration_idx=0, is_last_iter=True)
    
    # 在系统外部进行语义匹配（作为评估工具，不破坏马尔可夫毯）
    generated_text = llm_client._last_generated_text
    expected_keywords = metadata["keywords"]
    context_description = metadata["text"]
    expectations = metadata["expectations"]
    
    # 使用 LLMAspect 的语义相似度计算方法（外部评估）
    coverage, matched, missing, keyword_similarities = llm_aspect.compute_semantic_similarity(
        llm_text=generated_text,
        expected_keywords=expected_keywords,
        context_description=context_description,
        expectations=expectations,
    )
    
    # 将结果存储到 LLMAspect（仅用于外部查询）
    match_result = {
        "coverage": coverage,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "keyword_similarities": keyword_similarities,
        "llm_text": generated_text,
    }
    llm_aspect.set_semantic_match_result(match_result)
    
    print("=" * 80)
    print("语义匹配测试结果")
    print("=" * 80)
    print(f"LLM 生成的文本: {llm_client._last_generated_text}")
    print()
    
    if match_result:
        print(f"关键词覆盖率: {match_result['coverage'] * 100:.1f}%")
        print(f"匹配的关键词: {', '.join(match_result['matched_keywords']) or '无'}")
        print(f"未匹配的关键词: {', '.join(match_result['missing_keywords']) or '无'}")
        print()
        print("关键词相似度详情:")
        for kw, sim in match_result['keyword_similarities'].items():
            status = "✓" if kw in match_result['matched_keywords'] else "✗"
            print(f"  {status} {kw}: {sim:.3f}")
    else:
        print("未获取到语义匹配结果")
    
    print("=" * 80)
    
    return match_result is not None

if __name__ == "__main__":
    success = test_semantic_matching()
    sys.exit(0 if success else 1)

