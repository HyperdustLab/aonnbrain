#!/usr/bin/env python3
"""
可视化 Pipeline 结构
从实验结果中提取并显示详细的 Pipeline 结构信息
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import argparse
from typing import Dict, List


def visualize_pipeline_structure(result_file: str):
    """可视化 Pipeline 结构"""
    
    # 读取结果文件
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("AONN Pipeline 结构可视化")
    print("=" * 80)
    print()
    
    # 基本信息
    print(f"实验步数: {data.get('num_steps', 'N/A')}")
    print(f"最终自由能: {data.get('final_free_energy', 'N/A'):.4f}")
    print(f"最终准确率: {data.get('final_accuracy', 'N/A')*100:.2f}%")
    print()
    
    # 网络结构
    structure = data.get('final_structure', {})
    
    print("=" * 80)
    print("网络结构概览")
    print("=" * 80)
    print(f"Objects: {structure.get('num_objects', 0)}")
    print(f"Aspects: {structure.get('num_aspects', 0)}")
    print(f"Pipelines: {structure.get('num_pipelines', 0)}")
    print()
    
    # Objects 信息
    print("=" * 80)
    print("Object 节点")
    print("=" * 80)
    objects = structure.get('objects', {})
    for name, info in objects.items():
        print(f"  {name:15s} | 维度: {info.get('dim', 'N/A'):4d} | 状态范数: {info.get('state_norm', 0):.4f}")
    print()
    
    # Aspects 信息
    print("=" * 80)
    print("Aspects（计算单元）")
    print("=" * 80)
    aspects = structure.get('aspects', [])
    for i, aspect in enumerate(aspects, 1):
        src_str = " + ".join(aspect.get('src', []))
        dst_str = " + ".join(aspect.get('dst', []))
        print(f"  {i}. {aspect.get('name', 'N/A'):30s}")
        print(f"     类型: {aspect.get('type', 'N/A')}")
        print(f"     连接: [{src_str}] → [{dst_str}]")
    print()
    
    # Pipelines 详细信息
    print("=" * 80)
    print("Pipeline 结构（详细）")
    print("=" * 80)
    pipelines = structure.get('pipelines', [])
    
    if not pipelines:
        print("  无 Pipeline")
    else:
        for i, pipeline in enumerate(pipelines, 1):
            print(f"\n  Pipeline {i}:")
            print(f"    - 深度 (depth): {pipeline.get('depth', 'N/A')}")
            print(f"    - 每层 Aspect 数: {pipeline.get('num_aspects', 'N/A')}")
            print(f"    - 输入维度: {pipeline.get('input_dim', 'N/A')}")
            print(f"    - 输出维度: {pipeline.get('output_dim', 'N/A')}")
            
            spec = pipeline.get('spec', {})
            if spec:
                print(f"    - 输入 Object: {spec.get('input', 'N/A')}")
                print(f"    - 输出 Object: {spec.get('output', 'N/A')}")
                metadata = spec.get('metadata', {})
                if metadata:
                    print(f"    - 阶段: {metadata.get('stage', 'N/A')}")
            
            # 计算总参数数（估算）
            depth = pipeline.get('depth', 0)
            num_aspects = pipeline.get('num_aspects', 0)
            input_dim = pipeline.get('input_dim', 0)
            output_dim = pipeline.get('output_dim', 0)
            
            # 每个 AspectLayer 的参数：
            # - W: (input_dim * num_aspects + num_aspects) [bias]
            # - V: (num_aspects * output_dim) [no bias]
            # - proj (if input_dim != output_dim): (input_dim * output_dim) [no bias]
            params_per_layer = 0
            if input_dim == output_dim:
                # W + V
                params_per_layer = (input_dim * num_aspects + num_aspects) + (num_aspects * output_dim)
            else:
                # W + V + proj
                params_per_layer = (input_dim * num_aspects + num_aspects) + (num_aspects * output_dim) + (input_dim * output_dim)
            
            total_params = params_per_layer * depth
            print(f"    - 估算参数数: {total_params:,}")
            
            # 绘制 Pipeline 结构图
            print(f"    - 结构图:")
            print(f"        {spec.get('input', 'input')} [{input_dim}D]")
            for layer_idx in range(depth):
                print(f"          ↓ AspectLayer {layer_idx+1} ({num_aspects} aspects)")
            print(f"        {spec.get('output', 'output')} [{output_dim}D]")
    
    print()
    
    # PipelineAspect 信息（如果有）
    print("=" * 80)
    print("PipelineAspect（包装的 Pipeline）")
    print("=" * 80)
    pipeline_aspects = [a for a in aspects if a.get('type') == 'PipelineAspect']
    if pipeline_aspects:
        for i, aspect in enumerate(pipeline_aspects, 1):
            print(f"\n  {i}. {aspect.get('name', 'N/A')}")
            src_list = aspect.get('src', [])
            dst_list = aspect.get('dst', [])
            src_str = " + ".join(src_list) if src_list else 'N/A'
            dst_str = " + ".join(dst_list) if dst_list else 'N/A'
            print(f"     连接: [{src_str}] → [{dst_str}]")
            
            # 尝试从 pipelines 中找到对应的 pipeline
            src_name = src_list[0] if src_list else ''
            dst_name = dst_list[0] if dst_list else ''
            matching_pipeline = None
            for p in pipelines:
                spec = p.get('spec', {})
                if spec.get('input') == src_name and spec.get('output') == dst_name:
                    matching_pipeline = p
                    break
            
            if matching_pipeline:
                depth = matching_pipeline.get('depth', 0)
                num_aspects = matching_pipeline.get('num_aspects', 0)
                input_dim = matching_pipeline.get('input_dim', 0)
                output_dim = matching_pipeline.get('output_dim', 0)
                
                print(f"     对应 Pipeline 详情:")
                print(f"       - 深度: {depth} 层")
                print(f"       - 每层 Aspect 数: {num_aspects}")
                print(f"       - 输入维度: {input_dim}")
                print(f"       - 输出维度: {output_dim}")
                
                # 计算参数数
                if input_dim == output_dim:
                    params_per_layer = (input_dim * num_aspects + num_aspects) + (num_aspects * output_dim)
                else:
                    params_per_layer = (input_dim * num_aspects + num_aspects) + (num_aspects * output_dim) + (input_dim * output_dim)
                total_params = params_per_layer * depth
                print(f"       - 估算参数数: {total_params:,}")
                
                # 结构图
                print(f"       - 结构图:")
                print(f"           {src_name} [{input_dim}D]")
                for layer_idx in range(depth):
                    print(f"             ↓ AspectLayer {layer_idx+1} ({num_aspects} aspects)")
                print(f"           {dst_name} [{output_dim}D]")
            else:
                # 如果没有找到匹配的 pipeline，可能是 vision_encoder_pipeline
                # 它是在实验开始时创建的，作为 PipelineAspect，不在 aspect_pipelines 中
                if aspect.get('name') == 'vision_encoder_pipeline':
                    print(f"     注意: 这是初始创建的 vision → internal PipelineAspect")
                    print(f"     配置（从实验脚本）: depth=4, num_aspects=64, 784→256")
                    print(f"     结构图:")
                    print(f"         vision [784D]")
                    for layer_idx in range(4):
                        print(f"           ↓ AspectLayer {layer_idx+1} (64 aspects)")
                    print(f"         internal [256D]")
                    
                    # 计算参数数
                    input_dim = 784
                    output_dim = 256
                    num_aspects = 64
                    depth = 4
                    # 第一层: 784 -> 256, 后续层: 256 -> 256
                    # 第一层参数
                    params_layer1 = (input_dim * num_aspects + num_aspects) + (num_aspects * output_dim) + (input_dim * output_dim)
                    # 后续层参数（256 -> 256）
                    params_per_layer = (output_dim * num_aspects + num_aspects) + (num_aspects * output_dim)
                    total_params = params_layer1 + params_per_layer * (depth - 1)
                    print(f"     估算参数数: {total_params:,}")
    else:
        print("  无 PipelineAspect（Pipeline 可能直接存储在 aspect_pipelines 中）")
    
    print()
    
    # 网络拓扑图
    print("=" * 80)
    print("网络拓扑图")
    print("=" * 80)
    print()
    
    # 绘制简化的拓扑图
    print("  Objects:")
    for name in objects.keys():
        print(f"    [{name}]")
    print()
    
    print("  Connections (via Aspects):")
    for aspect in aspects:
        src_str = " + ".join(aspect.get('src', []))
        dst_str = " + ".join(aspect.get('dst', []))
        aspect_type = aspect.get('type', 'Unknown')
        if aspect_type == 'PipelineAspect':
            print(f"    [{src_str}] ──[Pipeline: {aspect.get('name')}]──> [{dst_str}]")
        else:
            print(f"    [{src_str}] ──[{aspect.get('name')} ({aspect_type})]──> [{dst_str}]")
    
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="可视化 Pipeline 结构")
    parser.add_argument("--result-file", type=str, default="data/mnist_results.json",
                       help="实验结果 JSON 文件路径")
    
    args = parser.parse_args()
    
    result_file = Path(__file__).parent.parent / args.result_file
    if not result_file.exists():
        print(f"错误: 文件不存在: {result_file}")
        return
    
    visualize_pipeline_structure(str(result_file))


if __name__ == "__main__":
    main()

