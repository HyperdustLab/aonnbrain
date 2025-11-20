#!/usr/bin/env python3
"""
通用AI智能体世界模型实验脚本

测试 AONN 在通用AI智能体世界模型下的演化能力
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        # 也尝试从当前目录加载
        load_dotenv(override=True)
except ImportError:
    pass  # python-dotenv 未安装时忽略
except Exception as e:
    import warnings
    warnings.warn(f"加载 .env 文件时出错: {e}")

import json
import sys
from typing import Dict, Optional

import torch
from tqdm import tqdm

from aonn.models.general_ai_world_model import GeneralAIWorldModel, GeneralAIWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.aspects.mock_llm_client import MockLLMClient

try:
    from aonn.aspects.openai_llm_client import OpenAILLMClient
except Exception:  # pragma: no cover - optional dependency
    OpenAILLMClient = None


def run_experiment(
    num_steps: int,
    config: Dict,
    device: torch.device,
    *,
    verbose: bool = False,
    use_openai_llm: bool = False,
    openai_api_key: Optional[str] = None,
):
    """
    运行通用AI智能体实验
    
    Args:
        num_steps: 演化步数
        config: 配置字典
        device: 设备
        verbose: 是否实时输出
    """
    # 创建世界模型
    world_model = GeneralAIWorldModel(
        semantic_dim=config["world_model"]["semantic_dim"],
        memory_dim=config["world_model"]["memory_dim"],
        context_dim=config["world_model"]["context_dim"],
        physical_dim=config["world_model"]["physical_dim"],
        goal_dim=config["world_model"]["goal_dim"],
        vision_dim=config["world_model"]["vision_dim"],
        language_dim=config["world_model"]["language_dim"],
        audio_dim=config["world_model"]["audio_dim"],
        multimodal_dim=config["world_model"]["multimodal_dim"],
        action_dim=config["act_dim"],
        device=device,
        state_noise_std=config["world_model"].get("state_noise_std", 0.01),
        observation_noise_std=config["world_model"].get("observation_noise_std", 0.01),
        enable_tools=config["world_model"].get("enable_tools", True),
    )
    world_interface = GeneralAIWorldInterface(world_model)
    
    llm_client = None
    sem_dim = config.get("sem_dim", 128)
    if config.get("disable_llm", False):
        llm_client = None
    elif use_openai_llm and OpenAILLMClient is not None:
        llm_cfg = config.get("llm", {})
        llm_client = OpenAILLMClient(
            input_dim=sem_dim,
            output_dim=sem_dim,
            api_key=openai_api_key or llm_cfg.get("api_key"),
            model=llm_cfg.get("model", "gpt-4o-mini"),
            embedding_model=llm_cfg.get("embedding_model", "text-embedding-3-small"),
            summary_size=llm_cfg.get("summary_size", 8),
            max_tokens=llm_cfg.get("max_tokens", 120),
            temperature=llm_cfg.get("temperature", 0.7),
            device=device,
        )
    else:
        llm_client = MockLLMClient(
            input_dim=sem_dim,
            output_dim=sem_dim,
            hidden_dims=config.get("llm", {}).get("hidden_dims", [256, 512, 256]),
            device=device,
        )
    
    # 创建 AONN Brain
    brain = AONNBrainV3(
        config=config,
        llm_client=llm_client,
        device=device,
        enable_evolution=True,
    )
    
    # 初始化
    obs = world_interface.reset()
    prev_obs = None
    prev_action = None
    
    # 演化历史
    snapshots = []
    
    progress = tqdm(range(num_steps), desc=f"GeneralAI {num_steps}", mininterval=0.5, file=sys.stdout)
    for step in progress:
        if verbose and step == 0:
            print(f"开始步骤 {step}...")
        
        if step > 0:
            obs, reward = world_interface.step(action)
        
        # 设置观察到 brain
        for sense, value in obs.items():
            if sense in brain.objects:
                brain.objects[sense].set_state(value)
        
        # 网络演化
        # 使用完整状态的前 state_dim 维作为 target（匹配 internal 状态维度）
        full_state = world_model.get_true_state()
        if full_state.shape[-1] >= config["state_dim"]:
            target_state = full_state[:config["state_dim"]]
        else:
            # 如果状态维度小于 state_dim，用零填充
            padding = torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)
            target_state = torch.cat([full_state, padding], dim=-1)
        try:
            brain.evolve_network(obs, target=target_state)
        except Exception as e:
            if verbose:
                print(f"Step {step}: Evolution error: {e}")
            pass
        
        # 主动推理（在学习之后，确保所有状态都是 detached）
        if len(brain.aspects) > 0:
            try:
                # 确保所有 Object 状态都是 detached，避免图冲突
                # 只检查叶子张量，避免警告
                for obj_name, obj in brain.objects.items():
                    state = obj.state
                    if state.requires_grad and state.is_leaf and state.grad is not None:
                        obj.set_state(state.detach())
                    elif not state.is_leaf:
                        # 如果不是叶子张量，直接 detach
                        obj.set_state(state.detach())
                
                loop = ActiveInferenceLoop(
                    brain.objects,
                    brain.aspects,
                    infer_lr=config.get("infer_lr", 0.02),
                    device=device,
                )
                loop.infer_states(target_objects=("internal",), num_iters=1)  # 减少迭代次数，更稳定
                brain.sanitize_states()
            except Exception as e:
                if verbose:
                    print(f"Step {step}: Inference error: {e}")
                pass
        
        # 生成动作（通过 pipeline）
        if "action" in brain.objects and len(brain.aspect_pipelines) > 0:
            action = brain.objects["internal"].state
            for pipeline in brain.aspect_pipelines:
                action = pipeline(action)
            brain.objects["action"].set_state(action)
        else:
            action = torch.randn(config["act_dim"], device=device) * 0.1
            if "action" in brain.objects:
                brain.objects["action"].set_state(action)
        
        # 学习世界模型
        if prev_obs is not None and prev_action is not None:
            # 使用完整状态的前 state_dim 维作为 target
            full_state = world_model.get_true_state()
            if full_state.shape[-1] >= config["state_dim"]:
                target_state = full_state[:config["state_dim"]]
            else:
                padding = torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)
                target_state = torch.cat([full_state, padding], dim=-1)
            try:
                brain.learn_world_model(
                    observation=prev_obs,
                    action=prev_action,
                    next_observation=obs,
                    target_state=target_state,
                    learning_rate=config.get("learning_rate", 0.0005),
                )
                # 学习后也清理状态
                brain.sanitize_states()
            except Exception as e:
                if verbose:
                    print(f"Step {step}: Learning error: {e}")
                pass
        
        prev_obs = {sense: value.clone() for sense, value in obs.items()}
        prev_action = action.clone()
        
        # 定期清理状态，防止累积误差
        if step % 10 == 0:
            brain.sanitize_states()
        
        # 记录快照
        if step % 50 == 0 or step == num_steps - 1:
            F = brain.compute_free_energy().item()
            self_model_snapshot = brain.observe_self_model()
            snapshot = {
                "step": step,
                "free_energy": F,
                "structure": self_model_snapshot.get("structure", {}),
            }
            snapshots.append(snapshot)
            
            if verbose:
                structure = snapshot.get("structure", {})
                print(f"Step {step}: F={F:.4f}, "
                      f"Objects={structure.get('num_objects', 0)}, "
                      f"Aspects={structure.get('num_aspects', 0)}, "
                      f"Pipelines={structure.get('num_pipelines', 0)}")
    
    # 最终结果
    final_snapshot = brain.observe_self_model()
    final_F = brain.compute_free_energy().item()
    
    result = {
        "num_steps": num_steps,
        "final_free_energy": final_F,
        "final_structure": final_snapshot.get("structure", {}),
        "snapshots": snapshots,
        "evolution_summary": brain.evolution.get_evolution_summary() if brain.evolution else {},
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="通用AI智能体世界模型演化实验")
    parser.add_argument("--steps", type=int, default=500, help="演化步数")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--output", type=Path, default=Path("data/general_ai_results.json"), help="输出文件")
    parser.add_argument("--verbose", action="store_true", help="实时输出演化快照")
    parser.add_argument("--disable-llm", action="store_true", help="禁用 LLMAspect")
    parser.add_argument("--use-openai-llm", action="store_true", help="使用 OpenAI 作为 LLMAspect 客户端")
    parser.add_argument("--openai-api-key", type=str, default=None, help="OpenAI API Key（默认读取 OPENAI_API_KEY）")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 配置
    config = {
        "state_dim": 1024,  # 内部状态维度（对应通用AI的总状态维度）
        "act_dim": 256,  # 动作维度
        "obs_dim": 1408,  # 总观察维度（512+512+128+256）
        "sem_dim": 512,  # 语义维度
        "sense_dims": {
            "vision": 512,
            "language": 512,
            "audio": 128,
            "multimodal": 256,
        },
        "enable_world_model_learning": True,
        "world_model": {
            "semantic_dim": 1024,
            "memory_dim": 512,
            "context_dim": 256,
            "physical_dim": 64,
            "goal_dim": 256,
            "vision_dim": 512,
            "language_dim": 512,
            "audio_dim": 128,
            "multimodal_dim": 256,
            "state_noise_std": 0.01,
            "observation_noise_std": 0.01,
            "enable_tools": True,
        },
        "evolution": {
            "free_energy_threshold": 0.05,
            "prune_threshold": 0.01,
            "max_objects": 100,
            "max_aspects": 2000,  # 通用AI需要更多Aspect
            "error_ema_alpha": 0.4,
            "batch_growth": {
                "base": 16,
                "max_per_step": 64,
                "max_total": 400,
                "min_per_sense": 8,
                "error_threshold": 0.05,
                "error_multiplier": 0.7,
            },
        },
        "pipeline_growth": {
            "enable": True,
            "initial_depth": 3,
            "initial_width": 32,
            "depth_increment": 1,
            "width_increment": 8,
            "max_stages": 8,  # 通用AI需要更深pipeline
            "min_interval": 100,
            "free_energy_trigger": None,
            "max_depth": 12,
        },
        "state_clip_value": 5.0,  # 降低裁剪值，防止状态爆炸
        "infer_lr": 0.01,  # 降低推理学习率，更稳定
        "learning_rate": 0.0001,  # 降低世界模型学习率
        "llm": {
            "model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "summary_size": 8,
            "max_tokens": 120,
            "temperature": 0.7,
        },
    }
    
    config["disable_llm"] = args.disable_llm
    if args.use_openai_llm and OpenAILLMClient is None:
        raise RuntimeError("OpenAILLMClient 无法导入，请确认已安装 openai==1.x 并成功初始化。")
    
    # 运行实验
    result = run_experiment(
        num_steps=args.steps,
        config=config,
        device=device,
        verbose=args.verbose,
        use_openai_llm=args.use_openai_llm,
        openai_api_key=args.openai_api_key,
    )
    
    # 保存结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n实验完成！结果保存到: {args.output}")
    print(f"最终自由能: {result['final_free_energy']:.4f}")
    print(f"最终结构: {result['final_structure']['num_objects']} Objects, "
          f"{result['final_structure']['num_aspects']} Aspects, "
          f"{result['final_structure']['num_pipelines']} Pipelines")


if __name__ == "__main__":
    main()

