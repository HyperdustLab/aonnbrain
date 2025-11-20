#!/usr/bin/env python3
"""
运行所有测试
"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_basic_tests():
    """运行基本功能测试"""
    print("=" * 60)
    print("运行基本功能测试")
    print("=" * 60)
    
    # 测试 1: 验证设置
    print("\n[1/5] 验证项目设置...")
    try:
        from aonn.core.object import ObjectNode
        from aonn.core.aspect_base import AspectBase
        from aonn.core.free_energy import compute_total_free_energy
        from aonn.core.active_inference_loop import ActiveInferenceLoop
        from aonn.aspects.sensory_aspect import LinearGenerativeAspect
        from aonn.aspects.llm_aspect import LLMAspect
        from aonn.aspects.mock_llm_client import MockLLMClient
        from aonn.models.aonn_brain import AONNBrain
        print("   ✓ 所有模块导入成功")
    except Exception as e:
        print(f"   ✗ 导入失败: {e}")
        return False
    
    # 测试 2: ObjectNode
    print("\n[2/5] 测试 ObjectNode...")
    try:
        obj = ObjectNode("test", dim=10)
        assert obj.dim == 10
        assert obj.state.shape == (10,)
        print("   ✓ ObjectNode 测试通过")
    except Exception as e:
        print(f"   ✗ ObjectNode 测试失败: {e}")
        return False
    
    # 测试 3: LinearGenerativeAspect
    print("\n[3/5] 测试 LinearGenerativeAspect...")
    try:
        import torch
        dim = 32
        internal = ObjectNode("internal", dim=dim, init="normal")
        sensory = ObjectNode("sensory", dim=dim, init="normal")
        objects = {"internal": internal, "sensory": sensory}
        
        aspect = LinearGenerativeAspect(
            internal_name="internal",
            sensory_name="sensory",
            state_dim=dim,
            obs_dim=dim,
        )
        
        F = aspect.free_energy_contrib(objects)
        assert F.item() >= 0
        print(f"   ✓ LinearGenerativeAspect 测试通过 (F={F.item():.4f})")
    except Exception as e:
        print(f"   ✗ LinearGenerativeAspect 测试失败: {e}")
        return False
    
    # 测试 4: MockLLMClient
    print("\n[4/5] 测试 MockLLMClient...")
    try:
        import torch
        client = MockLLMClient(input_dim=64, output_dim=64)
        test_input = torch.randn(64)
        output = client.semantic_predict(test_input)
        assert output.shape == (64,)
        print(f"   ✓ MockLLMClient 测试通过 (参数数: {sum(p.numel() for p in client.parameters()):,})")
    except Exception as e:
        print(f"   ✗ MockLLMClient 测试失败: {e}")
        return False
    
    # 测试 5: LLMAspect 集成
    print("\n[5/5] 测试 LLMAspect 集成...")
    try:
        import torch
        dim = 64
        context = ObjectNode("semantic_context", dim=dim, init="normal")
        prediction = ObjectNode("semantic_prediction", dim=dim, init="normal")
        objects = {"semantic_context": context, "semantic_prediction": prediction}
        
        client = MockLLMClient(input_dim=dim, output_dim=dim)
        llm_aspect = LLMAspect(
            src_names=("semantic_context",),
            dst_names=("semantic_prediction",),
            llm_client=client,
        )
        
        F = llm_aspect.free_energy_contrib(objects)
        assert F.item() >= 0
        print(f"   ✓ LLMAspect 集成测试通过 (F={F.item():.4f})")
    except Exception as e:
        print(f"   ✗ LLMAspect 集成测试失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有基本测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)

