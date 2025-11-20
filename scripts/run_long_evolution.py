#!/usr/bin/env python3
"""
AONN Brain V3 长周期演化实验

运行不同步数的演化实验，记录自由能和网络结构的演化趋势。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from typing import Dict, List

import torch
from tqdm import tqdm

from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.models.world_model import SimpleWorldModel, WorldModelInterface
from aonn.core.active_inference_loop import ActiveInferenceLoop


def run_single_experiment(
    num_steps: int,
    device: torch.device,
    *,
    state_dim: int,
    action_dim: int,
    obs_dim: int,
    free_energy_threshold: float,
    state_noise_std: float,
    obs_noise_std: float,
    target_drift_std: float,
    verbose: bool = False,
    pipeline_enable: bool = True,
) -> Dict:
    sense_dims = {
        "vision": max(obs_dim * 3 // 4, 8),
        "olfactory": max(obs_dim // 5, 4),
        "proprio": max(obs_dim - (max(obs_dim * 3 // 4, 8) + max(obs_dim // 5, 4)), 4),
    }
    total_obs_dim = sum(sense_dims.values())

    batch_error_threshold = max(free_energy_threshold * 0.8, 0.04)
    pipeline_interval = max(60, num_steps // 6)

    config = {
        "obs_dim": total_obs_dim,
        "state_dim": state_dim,
        "act_dim": action_dim,
        "enable_world_model_learning": True,
        "vision_dim": sense_dims["vision"],
        "olfactory_dim": sense_dims["olfactory"],
        "proprio_dim": sense_dims["proprio"],
        "sense_dims": sense_dims,
        "evolution": {
            "free_energy_threshold": free_energy_threshold,
            "prune_threshold": 0.01,
            "max_objects": 60,
            "max_aspects": 360,
            "error_ema_alpha": 0.4,
            "batch_growth": {
                "base": 8,
                "max_per_step": 32,
                "max_total": 96,
                "min_per_sense": 4,
                "error_threshold": batch_error_threshold,
                "error_multiplier": 0.7,
            },
        },
        "pipeline_growth": {
            "enable": pipeline_enable,
            "initial_depth": 2,
            "initial_width": 24,
            "depth_increment": 1,
            "width_increment": 8,
            "max_stages": 3,
            "min_interval": pipeline_interval,
            "free_energy_trigger": None,
            "max_depth": 6,
        },
    }

    brain = AONNBrainV3(config=config, device=device, enable_evolution=True)
    world_model = SimpleWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        obs_dim=total_obs_dim,
        device=device,
        state_noise_std=state_noise_std,
        observation_noise_std=obs_noise_std,
        target_drift_std=target_drift_std,
        vision_dim=sense_dims["vision"],
        olfactory_dim=sense_dims["olfactory"],
        proprio_dim=sense_dims["proprio"],
    )
    world_interface = WorldModelInterface(world_model)

    obs = world_interface.reset()
    for sense, value in obs.items():
        if sense in brain.objects:
            brain.objects[sense].set_state(value)

    prev_obs = None
    prev_action = None

    snapshot_interval = max(5, num_steps // 10)
    snapshots: List[Dict] = []

    progress_iter = tqdm(range(num_steps), desc=f"Steps {num_steps}")
    for step in progress_iter:
        if step > 0:
            obs, reward = world_interface.step(action)
        for sense, value in obs.items():
            if sense in brain.objects:
                brain.objects[sense].set_state(value)

        # 演化和推理
        brain.evolve_network(obs)
        if len(brain.aspects) > 0:
            try:
                infer_loop = ActiveInferenceLoop(
                    objects=brain.objects,
                    aspects=brain.aspects,
                    infer_lr=0.1,
                    device=device,
                )
                infer_loop.infer_states(target_objects=("internal",), num_iters=1)
                brain.sanitize_states()
            except Exception:
                pass

        # 动作
        if "action" in brain.objects and len(brain.aspect_pipelines) > 0:
            action = brain.objects["internal"].state
            for pipeline in brain.aspect_pipelines:
                action = pipeline(action)
            brain.objects["action"].set_state(action)
        else:
            action = torch.randn(config["act_dim"], device=device) * 0.1
            if "action" in brain.objects:
                brain.objects["action"].set_state(action)

        # 世界模型学习
        if prev_obs is not None and prev_action is not None and brain.world_model_aspects is not None:
            target_state = world_model.get_true_state()
            brain.learn_world_model(
                observation=prev_obs,
                action=prev_action,
                next_observation=obs,
                target_state=target_state,
                learning_rate=0.001,
            )

        prev_obs = {sense: value.clone() for sense, value in obs.items()}
        prev_action = action.clone()

        if (step + 1) % snapshot_interval == 0 or step == num_steps - 1:
            snapshot = brain.observe_self_model()
            world_F = 0.0
            if brain.world_model_aspects is not None:
                world_F = sum(
                    aspect.free_energy_contrib(brain.objects).item()
                    for aspect in brain.world_model_aspects.aspects
                )
            snapshot_info = {
                "step": step + 1,
                "free_energy": snapshot["free_energy"],
                "structure": snapshot["structure"],
                "world_model_F": world_F,
            }
            snapshots.append(snapshot_info)
            if verbose:
                progress_iter.write(
                    f"[Step {step+1}] F={snapshot_info['free_energy']:.3f} | "
                    f"Aspects={snapshot_info['structure']['num_aspects']} "
                    f"Pipelines={snapshot_info['structure']['num_pipelines']} "
                    f"WorldF={snapshot_info['world_model_F']:.3f}"
                )

    evolution_history = brain.get_evolution_history()
    summary = {
        "num_steps": num_steps,
        "final_free_energy": brain.compute_free_energy().item(),
        "final_structure": brain.get_network_structure(),
        "total_events": len(evolution_history),
        "event_stats": {
            "create_object": sum(1 for e in evolution_history if e.event_type == "create_object"),
            "create_aspect": sum(1 for e in evolution_history if e.event_type == "create_aspect"),
            "create_pipeline": sum(1 for e in evolution_history if e.event_type == "create_pipeline"),
            "prune": sum(1 for e in evolution_history if e.event_type == "prune"),
        },
        "snapshots": snapshots,
    }
    return summary


def run_long_evolution():
    parser = argparse.ArgumentParser(description="运行 AONN 长周期演化实验")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[100, 300, 500, 1000],
        help="演化步数列表",
    )
    parser.add_argument(
        "--free-energy-threshold",
        type=float,
        default=0.2,
        help="触发演化的自由能阈值",
    )
    parser.add_argument(
        "--state-noise",
        type=float,
        default=0.05,
        help="世界模型状态噪声 std",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=0.02,
        help="世界模型观察噪声 std",
    )
    parser.add_argument(
        "--target-drift",
        type=float,
        default=0.01,
        help="目标漂移噪声 std",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=32,
        help="内部状态维度",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=8,
        help="动作维度",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=16,
        help="观察维度",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备 (cpu 或 cuda)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_run_results.json"),
        help="输出结果文件路径",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="实时输出演化快照信息",
    )
    parser.add_argument(
        "--disable-pipeline",
        action="store_true",
        help="关闭 pipeline 增长，仅保留基础动作头",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    steps_list = args.steps
    results = []
    pipeline_enable = not args.disable_pipeline

    for steps in steps_list:
        print("\n" + "=" * 70)
        print(f"运行 {steps} 步演化实验")
        print("=" * 70)
        summary = run_single_experiment(
            steps,
            device,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            obs_dim=args.obs_dim,
            free_energy_threshold=args.free_energy_threshold,
            state_noise_std=args.state_noise,
            obs_noise_std=args.obs_noise,
            target_drift_std=args.target_drift,
            verbose=args.verbose,
            pipeline_enable=pipeline_enable,
        )
        results.append(summary)
        print(f"完成 {steps} 步: 自由能 {summary['final_free_energy']:.4f}, 总事件 {summary['total_events']}")

    # 保存结果
    output_path = Path(__file__).parent.parent / args.output if not args.output.is_absolute() else args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n长周期演化结果已保存到: {output_path}")

    # 打印概览
    print("\n演化总结:")
    for summary in results:
        print(
            f"  步数 {summary['num_steps']}: F_final={summary['final_free_energy']:.4f}, "
            f"Objects={summary['final_structure']['num_object_layers'] if 'num_object_layers' in summary['final_structure'] else summary['final_structure']['num_objects']}, "
            f"Pipelines={summary['final_structure']['num_aspect_pipelines'] if 'num_aspect_pipelines' in summary['final_structure'] else summary['final_structure']['num_pipelines']}"
        )


if __name__ == "__main__":
    run_long_evolution()
