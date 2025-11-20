#!/usr/bin/env python3
"""
验证项目设置是否正确
"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from aonn.core.object import ObjectNode
    from aonn.core.aspect_base import AspectBase
    from aonn.core.free_energy import compute_total_free_energy
    from aonn.core.active_inference_loop import ActiveInferenceLoop
    from aonn.aspects.sensory_aspect import LinearGenerativeAspect
    from aonn.aspects.llm_aspect import LLMAspect
    from aonn.models.aonn_brain import AONNBrain
    print("✓ 所有核心模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试基本功能
try:
    obj = ObjectNode("test", dim=10)
    assert obj.dim == 10
    print("✓ ObjectNode 创建成功")
except Exception as e:
    print(f"✗ ObjectNode 测试失败: {e}")
    sys.exit(1)

print("\n项目设置验证完成！")

