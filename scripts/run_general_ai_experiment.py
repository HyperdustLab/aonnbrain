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
    elif use_ollama_llm and OllamaLLMClient is not None:
        llm_cfg = config.get("llm", {})
        llm_client = OllamaLLMClient(
            input_dim=sem_dim,
            output_dim=sem_dim,
            base_url=ollama_base_url or llm_cfg.get("base_url", "http://localhost:11434"),
            model=ollama_model or llm_cfg.get("model", "llama3"),
            embedding_model=llm_cfg.get("embedding_model"),  # 如果为 None，使用 model
            summary_size=llm_cfg.get("summary_size", 8),
            max_tokens=llm_cfg.get("max_tokens", 120),
            temperature=llm_cfg.get("temperature", 0.7),
            verbose=False,  # 关闭 LLM 客户端的 verbose，只在实验脚本中统一输出
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
            verbose=False,  # 关闭 LLM 客户端的 verbose，只在实验脚本中统一输出
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
    
    # 验证 LLMAspect 是否启用
    if verbose:
        if brain.llm_aspect is not None:
            print(f"\n[LLMAspect] ✓ 已启用")
            print(f"  - 名称: {brain.llm_aspect.name}")
            print(f"  - 源 Object: {brain.llm_aspect.src_names}")
            print(f"  - 目标 Object: {brain.llm_aspect.dst_names}")
            if hasattr(llm_client, 'model'):
                print(f"  - LLM 模型: {llm_client.model}")
            if hasattr(llm_client, 'base_url'):
                print(f"  - LLM 服务: {llm_client.base_url}")
        else:
            print(f"\n[LLMAspect] ✗ 未启用（llm_client 为 None）")
    
    # 初始化
    obs = world_interface.reset()
    prev_obs = None
    prev_action = None
    
    # 初始化 action（避免在第一次迭代时未定义）
    action = torch.randn(config["act_dim"], device=device) * 0.1
    
    # 演化历史
    snapshots = []
    
    progress = tqdm(range(num_steps), desc=f"GeneralAI {num_steps}", mininterval=0.5, file=sys.stdout)
    try:
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
            
            # 从世界模型的语义状态同步到 semantic_context（如果存在）
            if "semantic_context" in brain.objects and hasattr(world_model, "semantic_state"):
                sem_dim = config.get("sem_dim", 128)
                world_semantic = world_model.semantic_state[:sem_dim].detach().clone()
                brain.update_semantic_context(world_semantic_state=world_semantic)
            
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
                    
                    # 在推理前，如果启用了 LLMAspect，输出 semantic_context 的状态摘要（总是输出）
                    if "semantic_context" in brain.objects and llm_client is not None:
                        sem_ctx = brain.objects["semantic_context"].state
                        print(f"  [Step {step}] semantic_context 状态摘要: {sem_ctx[:8].detach().cpu().tolist()}")
                    
                    loop = ActiveInferenceLoop(
                        brain.objects,
                        brain.aspects,
                        infer_lr=config.get("infer_lr", 0.02),
                        max_grad_norm=config.get("max_grad_norm", None),  # 梯度裁剪阈值
                        device=device,
                    )
                    num_iters = config.get("num_infer_iters", 2)  # 从配置读取推理迭代次数
                    # 在每次迭代后裁剪状态，防止状态在迭代过程中爆炸
                    loop.infer_states(target_objects=("internal",), num_iters=num_iters, sanitize_callback=brain.sanitize_states)
                    brain.sanitize_states()  # 最后再清理一次
                    
                    # 在推理后，如果启用了 LLMAspect，输出生成的语义描述（总是输出，不依赖 verbose）
                    if llm_client is not None and hasattr(llm_client, '_last_generated_text'):
                        if llm_client._last_generated_text:
                            print(f"  [Step {step}] LLM 语义描述: {llm_client._last_generated_text}")
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
            
            # 每一步都计算并输出自由能和网络结构
            F = brain.compute_free_energy().item()
            # 如果自由能是 NaN/Inf，清理状态并跳过记录
            import math
            if not math.isfinite(F):
                brain.sanitize_states()
                F = 1e-6  # 使用一个小的非零值
            self_model_snapshot = brain.observe_self_model()
            structure = self_model_snapshot.get("structure", {})
            
            # 记录快照（按间隔记录，减少内存占用）
            if step % save_interval == 0 or step == num_steps - 1:
                snapshot = {
                    "step": step,
                    "free_energy": F,
                    "structure": structure,
                    "llm_description": llm_client._last_generated_text if llm_client and hasattr(llm_client, '_last_generated_text') else None,
                }
                snapshots.append(snapshot)
                
                # 定期保存检查点（防止长时间运行丢失数据）
                # 每 save_interval * 10 步保存一次，或者在最后一步保存
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
            
            # 每一步都输出自由能和网络结构
            llm_status = "LLM✓" if structure.get('has_llm_aspect', False) else "LLM✗"
            if verbose:
                print(f"Step {step}: F={F:.4f}, "
                      f"Objects={structure.get('num_objects', 0)}, "
                      f"Aspects={structure.get('num_aspects', 0)}, "
                      f"Pipelines={structure.get('num_pipelines', 0)}, "
                      f"{llm_status}")
            else:
                # 即使不 verbose，也在 progress bar 中显示
                progress.set_postfix(
                    F=f"{F:.2f}",
                    Obj=structure.get('num_objects', 0),
                    Asp=structure.get('num_aspects', 0),
                    Pipe=structure.get('num_pipelines', 0),
                    LLM=llm_status
                )
    
    except KeyboardInterrupt:
        # 用户中断时保存当前状态
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
            print(f"✗ 保存中断检查点失败: {e}")
        raise
    except Exception as e:
        # 其他异常时也尝试保存
        print(f"\n\n⚠️  实验发生异常: {e}")
        print("正在尝试保存当前状态...")
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
        except Exception as save_error:
            print(f"✗ 保存错误检查点失败: {save_error}")
        raise
    
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
    parser.add_argument("--steps", type=int, default=15000, help="演化步数（默认 15000 步，约 8 小时）")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--output", type=Path, default=Path("data/general_ai_results.json"), help="输出文件")
    parser.add_argument("--verbose", action="store_true", help="实时输出演化快照")
    parser.add_argument("--disable-llm", action="store_true", help="禁用 LLMAspect")
    parser.add_argument("--use-openai-llm", action="store_true", help="使用 OpenAI 作为 LLMAspect 客户端")
    parser.add_argument("--use-ollama-llm", action="store_true", help="使用本地 Ollama 作为 LLMAspect 客户端")
    parser.add_argument("--openai-api-key", type=str, default=None, help="OpenAI API Key（默认读取 OPENAI_API_KEY）")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434", help="Ollama API 基础 URL")
    parser.add_argument("--ollama-model", type=str, default="llama3", help="Ollama 模型名称")
    parser.add_argument("--save-interval", type=int, default=100, help="保存快照的间隔（步数）")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints", help="检查点保存目录")
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
            "free_energy_threshold": 0.08,  # 使用 LineWorm 的配置
            "prune_threshold": 0.01,
            "max_objects": 80,  # 使用 LineWorm 的配置
            "max_aspects": 500,  # 使用 LineWorm 的配置（从 10000 降低到 500）
            "error_ema_alpha": 0.5,  # 使用 LineWorm 的配置
            "batch_growth": {
                "base": 8,  # 使用 LineWorm 的配置（从 2 提高到 8）
                "max_per_step": 32,  # 使用 LineWorm 的配置（从 8 提高到 32）
                "max_total": 200,  # 使用 LineWorm 的配置（从 2500 降低到 200）
                "min_per_sense": 6,  # 使用 LineWorm 的配置（从 4 提高到 6）
                "error_threshold": 0.07,  # 使用 LineWorm 的配置（从 0.15 降低到 0.07）
                "error_multiplier": 0.7,  # 使用 LineWorm 的配置（从 0.5 提高到 0.7）
            },
        },
        "pipeline_growth": {
            "enable": True,
            "initial_depth": 3,
            "initial_width": 32,  # 使用 LineWorm 的配置（从 1 提高到 32）
            "depth_increment": 1,
            "width_increment": 12,  # 使用 LineWorm 的配置（从 0 提高到 12）
            "max_stages": 6,  # 使用 LineWorm 的配置（从 200 降低到 6）
            "min_interval": 80,  # 使用 LineWorm 的配置（从 10 提高到 80）
            "free_energy_trigger": None,  # 使用 LineWorm 的配置（从 0.1 改为 None）
            "max_depth": 10,  # 使用 LineWorm 的配置（从 50 降低到 10）
        },
        "state_clip_value": 5.0,
        "infer_lr": 0.02,  # 使用 LineWorm 的配置（从 0.005 提高到 0.02）
        "learning_rate": 0.0015,  # 使用 LineWorm 的配置（从 0.005 降低到 0.0015）
        "num_infer_iters": 5,  # 使用 LineWorm 的配置（从 3 提高到 5）
        "max_grad_norm": 100.0,
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
    
    # 检查 Ollama 是否可用
    if args.use_ollama_llm and OllamaLLMClient is None:
        raise RuntimeError("OllamaLLMClient 无法导入，请确认已安装 requests 库。")
    
    # 运行实验
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
    
    # 保存最终结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    # 保存最终检查点
    checkpoint_path = Path(__file__).parent.parent / args.checkpoint_dir
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    final_checkpoint = checkpoint_path / f"checkpoint_final_step_{args.steps}.json"
    with open(final_checkpoint, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n实验完成！")
    print(f"结果保存到: {args.output}")
    print(f"最终检查点: {final_checkpoint}")
    print(f"最终自由能: {result['final_free_energy']:.4f}")
    print(f"最终结构: {result['final_structure']['num_objects']} Objects, "
          f"{result['final_structure']['num_aspects']} Aspects, "
          f"{result['final_structure']['num_pipelines']} Pipelines")


if __name__ == "__main__":
    main()

