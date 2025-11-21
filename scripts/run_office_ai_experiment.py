#!/usr/bin/env python3
"""
Office AI 世界模型演化实验

测试 AONN 在 Office AI 世界模型下的演化能力
Office AI 世界模型复杂度介于 LineWorm 和 GeneralAI 之间
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
        load_dotenv(override=True)
except ImportError:
    pass  # python-dotenv 未安装时忽略
except Exception as e:
    import warnings
    warnings.warn(f"加载 .env 文件时出错: {e}")

import json
import math
import time
from typing import Dict, Optional

import torch
from tqdm import tqdm

from aonn.models.office_ai_world_model import OfficeAIWorldModel, OfficeAIWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.aspects.mock_llm_client import MockLLMClient

try:
    from aonn.aspects.openai_llm_client import OpenAILLMClient
except Exception:  # pragma: no cover - optional dependency
    OpenAILLMClient = None

try:
    from aonn.aspects.ollama_llm_client import OllamaLLMClient
except Exception:  # pragma: no cover - optional dependency
    OllamaLLMClient = None


def run_experiment(
    num_steps: int,
    config: Dict,
    device: torch.device,
    *,
    verbose: bool = False,
    use_openai_llm: bool = False,
    use_ollama_llm: bool = False,
    openai_api_key: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
    save_interval: int = 100,
    checkpoint_dir: str = "data/checkpoints",
):
    """运行 Office AI 世界模型演化实验"""
    
    # 创建世界模型
    world_model = OfficeAIWorldModel(
        document_dim=config.get("world_model", {}).get("document_dim", 256),
        task_dim=config.get("world_model", {}).get("task_dim", 128),
        schedule_dim=config.get("world_model", {}).get("schedule_dim", 64),
        context_dim=config.get("world_model", {}).get("context_dim", 128),
        document_obs_dim=config["sense_dims"]["document"],
        table_obs_dim=config["sense_dims"]["table"],
        calendar_obs_dim=config["sense_dims"]["calendar"],
        action_dim=config["act_dim"],
        device=device,
        state_noise_std=config.get("world_model", {}).get("state_noise_std", 0.01),
        observation_noise_std=config.get("world_model", {}).get("observation_noise_std", 0.01),
    )
    world_interface = OfficeAIWorldInterface(world_model)
    
    # 初始化 LLM 客户端（可选）
    llm_client = None
    sem_dim = config.get("sem_dim", 128)
    if config.get("disable_llm", False):
        llm_client = None
    elif use_ollama_llm and OllamaLLMClient is not None:
        llm_cfg = config.get("llm", {})
        llm_client = OllamaLLMClient(
            input_dim=sem_dim,
            output_dim=sem_dim,
            base_url=ollama_base_url or llm_cfg.get("base_url", "http://localhost:11434"),
            model=ollama_model or llm_cfg.get("model", "llama3"),
            embedding_model=llm_cfg.get("embedding_model"),
            summary_size=llm_cfg.get("summary_size", 8),
            max_tokens=llm_cfg.get("max_tokens", 120),
            temperature=llm_cfg.get("temperature", 0.7),
            timeout=llm_cfg.get("timeout", 120.0),
            verbose=False,
            device=device,
        )
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
            verbose=False,
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
    brain = AONNBrainV3(config=config, llm_client=llm_client, device=device, enable_evolution=True)
    
    # 初始化环境
    obs = world_interface.reset()
    for sense, value in obs.items():
        if sense in brain.objects:
            brain.objects[sense].set_state(value)
    
    prev_obs = None
    prev_action = None
    snapshots = []
    
    # 初始化 action（避免在第一次迭代时未定义）
    action = torch.randn(config["act_dim"], device=device) * 0.1
    
    progress = tqdm(range(num_steps), desc=f"OfficeAI {num_steps}")
    
    try:
        for step in progress:
            step_start_time = time.perf_counter()
            
            if verbose and step == 0:
                print(f"开始步骤 {step}...")
            
            if step > 0:
                obs, reward = world_interface.step(action)
            
            # 设置观察到 brain
            for sense, value in obs.items():
                if sense in brain.objects:
                    brain.objects[sense].set_state(value)
            
            # 网络演化
            full_state = world_model.get_true_state()
            if full_state.shape[-1] >= config["state_dim"]:
                target_state = full_state[:config["state_dim"]]
            else:
                padding = torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)
                target_state = torch.cat([full_state, padding], dim=-1)
            
            # 从世界模型的上下文状态同步到 semantic_context（如果存在）
            if "semantic_context" in brain.objects and hasattr(world_model, "context_state"):
                sem_dim = config.get("sem_dim", 128)
                world_context = world_model.context_state[:sem_dim].detach().clone()
                brain.update_semantic_context(world_semantic_state=world_context)
            
            try:
                brain.evolve_network(obs, target=target_state)
            except Exception as e:
                if verbose:
                    print(f"Step {step}: Evolution error: {e}")
                pass
            
            # 主动推理
            if len(brain.aspects) > 0:
                try:
                    # 确保所有 Object 状态都是 detached
                    for obj_name, obj in brain.objects.items():
                        state = obj.state
                        if state.requires_grad and state.is_leaf and state.grad is not None:
                            obj.set_state(state.detach())
                        elif not state.is_leaf:
                            obj.set_state(state.detach())
                    
                    # 在推理前，如果启用了 LLMAspect，输出 semantic_context 的状态摘要（总是输出）
                    if "semantic_context" in brain.objects and llm_client is not None:
                        sem_ctx = brain.objects["semantic_context"].state
                        print(f"  [Step {step}] semantic_context 状态摘要: {sem_ctx[:8].detach().cpu().tolist()}")
                    
                    loop = ActiveInferenceLoop(
                        brain.objects,
                        brain.aspects,
                        infer_lr=config.get("infer_lr", 0.02),
                        max_grad_norm=config.get("max_grad_norm", None),
                        device=device,
                    )
                    num_iters = config.get("num_infer_iters", 2)
                    loop.infer_states(target_objects=("internal",), num_iters=num_iters, sanitize_callback=brain.sanitize_states)
                    brain.sanitize_states()
                    
                    # 在推理后，如果启用了 LLMAspect，输出生成的语义描述（总是输出）
                    if llm_client is not None and hasattr(llm_client, '_last_generated_text'):
                        if llm_client._last_generated_text:
                            print(f"  [Step {step}] LLM 语义描述: {llm_client._last_generated_text}")
                        else:
                            print(f"  [Step {step}] LLM 语义描述: (空)")
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
            
            if verbose:
                print(f"  [Step {step}] Action norm: {action.norm().item():.4f}, range: [{action.min().item():.4f}, {action.max().item():.4f}]")
            
            # 学习世界模型
            if prev_obs is not None and prev_action is not None:
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
                        learning_rate=config.get("learning_rate", 0.0015),
                    )
                    brain.sanitize_states()
                except Exception as e:
                    if verbose:
                        print(f"Step {step}: Learning error: {e}")
                    pass
            
            prev_obs = {sense: value.clone() for sense, value in obs.items()}
            prev_action = action.clone()
            
            # 定期清理状态
            if step % 10 == 0:
                brain.sanitize_states()
            
            # 计算并记录自由能
            F = brain.compute_free_energy().item()
            if not math.isfinite(F):
                brain.sanitize_states()
                F = 1e-6
            self_model_snapshot = brain.observe_self_model()
            structure = self_model_snapshot.get("structure", {})
            
            # 记录快照
            if step % save_interval == 0 or step == num_steps - 1:
                snapshot = {
                    "step": step,
                    "free_energy": F,
                    "structure": structure,
                    "llm_description": llm_client._last_generated_text if llm_client and hasattr(llm_client, '_last_generated_text') else None,
                }
                snapshots.append(snapshot)
                
                # 定期保存检查点
                should_save_checkpoint = (
                    (step > 0 and step % (save_interval * 10) == 0) or 
                    step == num_steps - 1
                )
                if should_save_checkpoint:
                    checkpoint_path = Path(__file__).parent.parent / checkpoint_dir
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    checkpoint_file = checkpoint_path / f"checkpoint_step_{step}.json"
                    interim_result = {
                        "step": step,
                        "free_energy": F,
                        "structure": structure,
                        "snapshots": snapshots,
                        "num_steps": num_steps,
                        "is_final": step == num_steps - 1,
                    }
                    try:
                        with open(checkpoint_file, "w") as f:
                            json.dump(interim_result, f, indent=2)
                        if verbose or step == num_steps - 1:
                            print(f"  检查点已保存: {checkpoint_file}")
                    except Exception as e:
                        print(f"  ⚠️  保存检查点失败: {e}")
            
            # 输出进度
            if verbose:
                print(f"Step {step}: F={F:.4f}, "
                      f"Objects={structure.get('num_objects', 0)}, "
                      f"Aspects={structure.get('num_aspects', 0)}, "
                      f"Pipelines={structure.get('num_pipelines', 0)}, "
                      f"LLM={'LLM✓' if structure.get('has_llm_aspect', False) else 'LLM✗'}")
            else:
                progress.set_postfix(
                    F=f"{F:.2f}",
                    Obj=structure.get('num_objects', 0),
                    Asp=structure.get('num_aspects', 0),
                    Pipe=structure.get('num_pipelines', 0),
                    LLM='LLM✓' if structure.get('has_llm_aspect', False) else 'LLM✗'
                )
            step_end_time = time.perf_counter()
            progress.set_postfix(
                F=f"{F:.4f}",
                time=f"{step_end_time - step_start_time:.2f}s",
                Obj=structure.get('num_objects', 0),
                Asp=structure.get('num_aspects', 0),
                Pipe=structure.get('num_pipelines', 0),
                LLM='LLM✓' if structure.get('has_llm_aspect', False) else 'LLM✗'
            )
    
    except KeyboardInterrupt:
        print("\n\n⚠️  实验被用户中断，正在保存当前状态...")
        final_step = step if 'step' in locals() else 0
        checkpoint_path = Path(__file__).parent.parent / checkpoint_dir
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        interrupt_checkpoint = checkpoint_path / f"checkpoint_interrupted_step_{final_step}.json"
        try:
            F = brain.compute_free_energy().item()
            self_model_snapshot = brain.observe_self_model()
            structure = self_model_snapshot.get("structure", {})
            interrupt_result = {
                "step": final_step,
                "free_energy": F,
                "structure": structure,
                "snapshots": snapshots,
                "num_steps": num_steps,
                "is_final": False,
                "interrupted": True,
            }
            with open(interrupt_checkpoint, "w") as f:
                json.dump(interrupt_result, f, indent=2)
            print(f"✓ 中断检查点已保存: {interrupt_checkpoint}")
        except Exception as e:
            print(f"  ⚠️  保存中断检查点失败: {e}")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ 实验发生错误: {e}，正在保存当前状态...")
        final_step = step if 'step' in locals() else 0
        checkpoint_path = Path(__file__).parent.parent / checkpoint_dir
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        error_checkpoint = checkpoint_path / f"checkpoint_error_step_{final_step}.json"
        try:
            F = brain.compute_free_energy().item()
            self_model_snapshot = brain.observe_self_model()
            structure = self_model_snapshot.get("structure", {})
            error_result = {
                "step": final_step,
                "free_energy": F,
                "structure": structure,
                "snapshots": snapshots,
                "num_steps": num_steps,
                "is_final": False,
                "error": str(e),
            }
            with open(error_checkpoint, "w") as f:
                json.dump(error_result, f, indent=2)
            print(f"✓ 错误检查点已保存: {error_checkpoint}")
        except Exception as save_e:
            print(f"  ⚠️  保存错误检查点失败: {save_e}")
        sys.exit(1)
    
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
    parser = argparse.ArgumentParser(description="Office AI 世界模型演化实验")
    parser.add_argument("--steps", type=int, default=500, help="演化步数")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--output", type=Path, default=Path("data/office_ai_results.json"), help="输出文件")
    parser.add_argument("--verbose", action="store_true", help="实时输出演化快照")
    parser.add_argument("--disable-llm", action="store_true", help="禁用 LLMAspect")
    parser.add_argument("--use-openai-llm", action="store_true", help="使用 OpenAI 作为 LLMAspect 客户端")
    parser.add_argument("--use-ollama-llm", action="store_true", help="使用本地 Ollama 作为 LLMAspect 客户端")
    parser.add_argument("--openai-api-key", type=str, default=None, help="OpenAI API Key")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434", help="Ollama API 基础 URL")
    parser.add_argument("--ollama-model", type=str, default="llama3", help="Ollama 模型名称")
    parser.add_argument("--save-interval", type=int, default=50, help="保存快照的间隔（步数）")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints", help="检查点保存目录")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 配置（使用 LineWorm 的参数配置，因为 Office AI 复杂度介于 LineWorm 和 GeneralAI 之间）
    config = {
        "state_dim": 576,  # Office AI 总状态维度（256+128+64+128）
        "act_dim": 128,    # Office AI 动作维度
        "obs_dim": 448,    # Office AI 总观察维度（256+128+64）
        "sem_dim": 128,    # 语义维度
        "sense_dims": {
            "document": 256,  # 文档观察
            "table": 128,     # 表格观察
            "calendar": 64,   # 日历观察
        },
        "enable_world_model_learning": True,
        "llm": {
            "call_frequency": "last_iter_only",  # "every_iter", "last_iter_only", "every_n_steps"
            "call_every_n_steps": 1,  # 当 call_frequency="every_n_steps" 时使用
        },
        "world_model": {
            "document_dim": 256,
            "task_dim": 128,
            "schedule_dim": 64,
            "context_dim": 128,
            "state_noise_std": 0.01,
            "observation_noise_std": 0.01,
        },
        "evolution": {
            "free_energy_threshold": 0.08,  # 使用 LineWorm 的配置
            "prune_threshold": 0.01,
            "max_objects": 80,
            "max_aspects": 500,
            "error_ema_alpha": 0.5,
            "batch_growth": {
                "base": 8,
                "max_per_step": 32,
                "max_total": 200,
                "min_per_sense": 6,
                "error_threshold": 0.07,
                "error_multiplier": 0.7,
            },
        },
        "pipeline_growth": {
            "enable": True,
            "initial_depth": 3,
            "initial_width": 32,
            "depth_increment": 1,
            "width_increment": 12,
            "max_stages": 6,
            "min_interval": 80,
            "free_energy_trigger": None,
            "max_depth": 10,
        },
        "state_clip_value": 5.0,
        "infer_lr": 0.02,
        "learning_rate": 0.0015,
        "num_infer_iters": 5,
        "max_grad_norm": 100.0,
        "llm": {
            "model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "summary_size": 8,
            "max_tokens": 120,
            "temperature": 0.7,
        },
    }
    
    print("=" * 80)
    print("Office AI 世界模型演化实验")
    print("=" * 80)
    print(f"实验步数: {args.steps}")
    print(f"状态维度: {config['state_dim']}")
    print(f"观察维度: {config['obs_dim']}")
    print(f"动作维度: {config['act_dim']}")
    print(f"感官: {list(config['sense_dims'].keys())}")
    print(f"LLM: {'禁用' if args.disable_llm else ('Ollama' if args.use_ollama_llm else ('OpenAI' if args.use_openai_llm else 'Mock'))}")
    print("=" * 80)
    print()
    
    result = run_experiment(
        num_steps=args.steps,
        config=config,
        device=device,
        verbose=args.verbose,
        use_openai_llm=args.use_openai_llm,
        use_ollama_llm=args.use_ollama_llm,
        openai_api_key=args.openai_api_key,
        ollama_base_url=args.ollama_base_url,
        ollama_model=args.ollama_model,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # 保存结果
    output_path = Path(__file__).parent.parent / args.output if not args.output.is_absolute() else args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print()
    print("=" * 80)
    print("实验完成！")
    print("=" * 80)
    print(f"结果保存到: {output_path}")
    print(f"最终检查点: {Path(__file__).parent.parent / args.checkpoint_dir / f'checkpoint_final_step_{args.steps}.json'}")
    print(f"最终自由能: {result['final_free_energy']:.4f}")
    print(f"最终结构: {result['final_structure'].get('num_objects', 0)} Objects, "
          f"{result['final_structure'].get('num_aspects', 0)} Aspects, "
          f"{result['final_structure'].get('num_pipelines', 0)} Pipelines")
    print("=" * 80)


if __name__ == "__main__":
    main()

