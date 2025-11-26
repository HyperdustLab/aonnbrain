#!/usr/bin/env python3
"""
从已完成的 Office AI 实验结果中提取并保存模型权重
注意：由于实验脚本之前没有保存权重，这个脚本只能保存当前模型架构的权重
（权重是随机初始化的，不是训练后的权重）
"""

import argparse
import json
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.models.office_ai_world_model import OfficeAIWorldModel
from aonn.aspects.llm_aspect import LLMAspect
from aonn.aspects.mock_llm_client import MockLLMClient

try:
    from aonn.aspects.ollama_llm_client import OllamaLLMClient
except Exception:
    OllamaLLMClient = None

try:
    from aonn.aspects.openai_llm_client import OpenAILLMClient
except Exception:
    OpenAILLMClient = None


def main():
    parser = argparse.ArgumentParser(description="Save Office AI model weights from experiment result")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment result JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for weights file (default: experiment_file.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda)",
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load experiment result
    experiment_path = Path(args.experiment)
    if not experiment_path.exists():
        print(f"Error: Experiment file not found: {experiment_path}")
        return
    
    print("=" * 80)
    print("Save Office AI Model Weights")
    print("=" * 80)
    print(f"Loading experiment: {experiment_path}")
    
    with open(experiment_path, 'r') as f:
        result = json.load(f)
    
    # Extract config from result or use default
    # Note: The result file might not have full config, so we'll use a default
    config = {
        "state_dim": 704,  # 更新：576 → 704 (包含 prompt_dim 128)
        "act_dim": 128,
        "obs_dim": 576,  # 更新：448 → 576 (包含 prompt_obs_dim 128)
        "auto_classification_aspect": False,
        "sem_dim": 128,
        "sense_dims": {
            "document": 256,
            "table": 128,
            "calendar": 64,
            "prompt": 128,
        },
        "enable_evolution": False,
        "use_world_model_pipelines": True,
        "world_model": {
            "document_dim": 256,
            "task_dim": 128,
            "schedule_dim": 64,
            "context_dim": 128,
            "prompt_dim": 128,
        },
        "sensory_pipeline_cfg": {
            "depth": 2,
            "width": 4,
            "use_gate": True,
        },
    }
    
    # Initialize LLM Client (use Mock since we don't need real LLM for saving weights)
    llm_client = MockLLMClient(
        input_dim=config["sem_dim"],
        output_dim=config["sem_dim"],
        hidden_dims=[256, 512, 256],
        device=device,
    )
    
    # Create Brain (this will have random weights, not trained weights)
    print("\n⚠️  WARNING: Creating brain with random weights")
    print("   The experiment script did not save trained weights.")
    print("   This will save the model architecture with random initialization.")
    print("   To get trained weights, re-run the experiment with the updated script.\n")
    
    brain = AONNBrainV3(
        config=config,
        llm_client=llm_client,
        device=device,
        enable_evolution=config.get("enable_evolution", False),
    )
    
    # Determine output path
    if args.output:
        weights_path = Path(args.output)
    else:
        weights_path = experiment_path.with_suffix('.pth')
    
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect weights
    checkpoint = {
        "config": config,
        "brain_state_dict": brain.state_dict(),
        "num_steps": result.get("num_steps", 0),
        "final_free_energy": result.get("final_free_energy", 0.0),
        "experiment_file": str(experiment_path),
    }
    
    # Save aspect pipeline weights if they exist
    if hasattr(brain, 'aspect_pipelines') and len(brain.aspect_pipelines) > 0:
        pipeline_weights = {}
        for i, pipeline in enumerate(brain.aspect_pipelines):
            if hasattr(pipeline, 'state_dict'):
                pipeline_weights[f"pipeline_{i}"] = pipeline.state_dict()
        checkpoint["pipeline_weights"] = pipeline_weights
    
    # Save aspect module weights
    if hasattr(brain, 'aspect_modules') and len(brain.aspect_modules) > 0:
        aspect_weights = {}
        for i, aspect in enumerate(brain.aspect_modules):
            if hasattr(aspect, 'state_dict'):
                aspect_weights[f"aspect_{i}"] = aspect.state_dict()
        checkpoint["aspect_weights"] = aspect_weights
    
    # Save weights
    torch.save(checkpoint, weights_path)
    
    print(f"✓ Model weights saved to: {weights_path}")
    print(f"  File size: {weights_path.stat().st_size / (1024 * 1024):.2f} MB")
    print("\n" + "=" * 80)
    print("NOTE: These are random weights, not trained weights!")
    print("To save trained weights, re-run the experiment with the updated script.")
    print("=" * 80)


if __name__ == "__main__":
    main()

