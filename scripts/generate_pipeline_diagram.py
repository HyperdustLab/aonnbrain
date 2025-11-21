#!/usr/bin/env python3
"""
生成 Pipeline 结构图（ASCII 艺术）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import argparse


def generate_pipeline_diagram(result_file: str):
    """生成 Pipeline 结构图"""
    
    # 读取结果文件
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    structure = data.get('final_structure', {})
    objects = structure.get('objects', {})
    aspects = structure.get('aspects', [])
    pipelines = structure.get('pipelines', [])
    
    print("=" * 80)
    print("AONN Pipeline 结构图")
    print("=" * 80)
    print()
    
    # 找到 vision_encoder_pipeline
    vision_pipeline = None
    for aspect in aspects:
        if aspect.get('name') == 'vision_encoder_pipeline':
            vision_pipeline = aspect
            break
    
    # 绘制完整的网络结构图
    print("完整网络结构:")
    print()
    
    # 输入层
    print("  [vision] (784D)")
    print("     │")
    if vision_pipeline:
        print("     │ PipelineAspect: vision_encoder_pipeline")
        print("     │   depth=4, num_aspects=64")
        print("     │   ┌─────────────────────────────────┐")
        print("     │   │ AspectLayer 1 (64 aspects)      │")
        print("     │   │ AspectLayer 2 (64 aspects)      │")
        print("     │   │ AspectLayer 3 (64 aspects)      │")
        print("     │   │ AspectLayer 4 (64 aspects)      │")
        print("     │   └─────────────────────────────────┘")
    print("     ↓")
    print("  [internal] (256D)")
    print("     │")
    print("     ├───[DynamicsAspect]───┐")
    print("     │   (internal+action→internal)")
    print("     │")
    print("     ├───[ObservationAspect]───→ [vision] (预测)")
    print("     │")
    print("     ├───[PreferenceAspect]───→ [target] (先验)")
    print("     │")
    print("     ├───[ClassificationAspect]───→ [target] (分类)")
    print("     │")
    
    # 内部 Pipeline
    if pipelines:
        print("     │ 内部 Pipeline:")
        for i, pipeline in enumerate(pipelines, 1):
            spec = pipeline.get('spec', {})
            stage = spec.get('metadata', {}).get('stage', 'unknown')
            depth = pipeline.get('depth', 0)
            num_aspects = pipeline.get('num_aspects', 0)
            output_obj = spec.get('output', 'unknown')
            
            print(f"     │   Pipeline {i} ({stage}):")
            print(f"     │     internal [{pipeline.get('input_dim', 0)}D]")
            for layer_idx in range(depth):
                print(f"     │       ↓ AspectLayer {layer_idx+1} ({num_aspects} aspects)")
            print(f"     │     {output_obj} [{pipeline.get('output_dim', 0)}D]")
            if i < len(pipelines):
                print("     │")
    
    print("     │")
    print("     ↓")
    print("  [action] (10D) - 分类输出")
    print()
    
    # 详细统计
    print("=" * 80)
    print("Pipeline 统计")
    print("=" * 80)
    
    total_aspects = 0
    total_params = 0
    
    # vision_encoder_pipeline
    if vision_pipeline:
        depth = 4
        num_aspects = 64
        input_dim = 784
        output_dim = 256
        # 第一层: 784 -> 256
        params_layer1 = (input_dim * num_aspects + num_aspects) + (num_aspects * output_dim) + (input_dim * output_dim)
        # 后续层: 256 -> 256
        params_per_layer = (output_dim * num_aspects + num_aspects) + (num_aspects * output_dim)
        vision_params = params_layer1 + params_per_layer * (depth - 1)
        vision_total_aspects = num_aspects * depth
        
        print(f"\n1. vision_encoder_pipeline (PipelineAspect):")
        print(f"   - 深度: {depth} 层")
        print(f"   - 每层 Aspect 数: {num_aspects}")
        print(f"   - 总 Aspect 数: {vision_total_aspects}")
        print(f"   - 输入: vision (784D)")
        print(f"   - 输出: internal (256D)")
        print(f"   - 估算参数数: {vision_params:,}")
        
        total_aspects += vision_total_aspects
        total_params += vision_params
    
    # 其他 Pipeline
    for i, pipeline in enumerate(pipelines, 1):
        depth = pipeline.get('depth', 0)
        num_aspects = pipeline.get('num_aspects', 0)
        input_dim = pipeline.get('input_dim', 0)
        output_dim = pipeline.get('output_dim', 0)
        spec = pipeline.get('spec', {})
        stage = spec.get('metadata', {}).get('stage', 'unknown')
        
        # 计算参数数
        if input_dim == output_dim:
            params_per_layer = (input_dim * num_aspects + num_aspects) + (num_aspects * output_dim)
        else:
            params_per_layer = (input_dim * num_aspects + num_aspects) + (num_aspects * output_dim) + (input_dim * output_dim)
        pipeline_params = params_per_layer * depth
        pipeline_total_aspects = num_aspects * depth
        
        print(f"\n{i+1 if vision_pipeline else i}. Pipeline {i} ({stage}):")
        print(f"   - 深度: {depth} 层")
        print(f"   - 每层 Aspect 数: {num_aspects}")
        print(f"   - 总 Aspect 数: {pipeline_total_aspects}")
        print(f"   - 输入: {spec.get('input', 'unknown')} ({input_dim}D)")
        print(f"   - 输出: {spec.get('output', 'unknown')} ({output_dim}D)")
        print(f"   - 估算参数数: {pipeline_params:,}")
        
        total_aspects += pipeline_total_aspects
        total_params += pipeline_params
    
    print()
    print("=" * 80)
    print("总计")
    print("=" * 80)
    print(f"Pipeline 总数: {len(pipelines) + (1 if vision_pipeline else 0)}")
    print(f"总 Aspect 数（所有 Pipeline）: {total_aspects}")
    print(f"总参数数（估算）: {total_params:,}")
    print()
    
    # 数据流图
    print("=" * 80)
    print("数据流图")
    print("=" * 80)
    print()
    print("  输入图像 (28×28 = 784)")
    print("        ↓")
    print("   [vision] Object")
    print("        ↓")
    print("   vision_encoder_pipeline (4层, 64 aspects/层)")
    print("        ↓")
    print("   [internal] Object")
    print("        ↓")
    if pipelines:
        for i, pipeline in enumerate(pipelines, 1):
            spec = pipeline.get('spec', {})
            stage = spec.get('metadata', {}).get('stage', 'unknown')
            depth = pipeline.get('depth', 0)
            num_aspects = pipeline.get('num_aspects', 0)
            output_obj = spec.get('output', 'unknown')
            
            if output_obj == 'action':
                print(f"   Pipeline {i} ({stage}, {depth}层, {num_aspects} aspects/层)")
                print(f"        ↓")
                print(f"   [action] Object (10D) → 分类预测")
            else:
                print(f"   Pipeline {i} ({stage}, {depth}层, {num_aspects} aspects/层)")
                print(f"        ↓")
                print(f"   [internal] Object (循环处理)")
    else:
        print("   [classification] Aspect")
        print("        ↓")
        print("   [target] Object → 分类预测")
    print()


def main():
    parser = argparse.ArgumentParser(description="生成 Pipeline 结构图")
    parser.add_argument("--result-file", type=str, default="data/mnist_results.json",
                       help="实验结果 JSON 文件路径")
    
    args = parser.parse_args()
    
    result_file = Path(__file__).parent.parent / args.result_file
    if not result_file.exists():
        print(f"错误: 文件不存在: {result_file}")
        return
    
    generate_pipeline_diagram(str(result_file))


if __name__ == "__main__":
    main()

