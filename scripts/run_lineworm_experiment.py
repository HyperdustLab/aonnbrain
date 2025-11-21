#!/usr/bin/env python3
"""
线虫世界模型演化实验
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from typing import Dict

import torch
from tqdm import tqdm

from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.models.lineworm_world_model import LineWormWorldModel, LineWormWorldInterface
from aonn.core.active_inference_loop import ActiveInferenceLoop


def run_experiment(num_steps: int, config: Dict, device: torch.device, verbose: bool = False):
    brain = AONNBrainV3(config=config, device=device, enable_evolution=True)
    world = LineWormWorldModel(
        state_dim=config["state_dim"],
        action_dim=config["act_dim"],
        chemo_dim=config["sense_dims"]["chemo"],
        thermo_dim=config["sense_dims"]["thermo"],
        touch_dim=config["sense_dims"]["touch"],
        plane_size=12.0,
        preferred_temp=config.get("world_model", {}).get("preferred_temp", 0.2),
        noise_config=config.get("world_model", {}).get("noise_config"),
        device=device,
    )
    world_interface = LineWormWorldInterface(world)

    obs = world_interface.reset()
    for sense, value in obs.items():
        if sense in brain.objects:
            brain.objects[sense].set_state(value)

    prev_obs = None
    prev_action = None

    snapshots = []
    snapshot_interval = max(5, num_steps // 10)

    progress = tqdm(range(num_steps), desc=f"LineWorm {num_steps}")
    for step in progress:
        if step > 0:
            obs, reward = world_interface.step(action)
        for sense, value in obs.items():
            if sense in brain.objects:
                brain.objects[sense].set_state(value)

        brain.evolve_network(obs)
        if len(brain.aspects) > 0:
            try:
                loop = ActiveInferenceLoop(
                    brain.objects,
                    brain.aspects,
                    infer_lr=0.01,  # 降低推理学习率，更稳定
                    max_grad_norm=100.0,  # 梯度裁剪
                    device=device,
                )
                loop.infer_states(target_objects=("internal",), num_iters=5, sanitize_callback=brain.sanitize_states)  # 保持5次迭代
                brain.sanitize_states()
            except Exception:
                pass

        if "action" in brain.objects and len(brain.aspect_pipelines) > 0:
            action = brain.objects["internal"].state
            for pipeline in brain.aspect_pipelines:
                action = pipeline(action)
            brain.objects["action"].set_state(action)
        else:
            action = torch.randn(config["act_dim"], device=device) * 0.1
            if "action" in brain.objects:
                brain.objects["action"].set_state(action)

        if prev_obs is not None and prev_action is not None:
            brain.learn_world_model(
                observation=prev_obs,
                action=prev_action,
                next_observation=obs,
                target_state=world.get_true_state(),
                learning_rate=0.0015,  # 适度提高学习率，加快学习
            )

        prev_obs = {sense: value.clone() for sense, value in obs.items()}
        prev_action = action.clone()

        if (step + 1) % snapshot_interval == 0 or step == num_steps - 1:
            snapshot = brain.observe_self_model()
            snapshots.append(snapshot)
            if verbose:
                progress.write(
                    f"[Step {step+1}] F={snapshot['free_energy']:.3f} | "
                    f"Aspects={snapshot['structure']['num_aspects']} "
                    f"Pipelines={snapshot['structure']['num_pipelines']} | "
                    f"Energy={brain.objects['internal'].state.norm().item():.3f}"
                )

    return {
        "num_steps": num_steps,
        "final_free_energy": brain.compute_free_energy().item(),
        "final_structure": brain.get_network_structure(),
        "snapshots": snapshots,
        "evolution_events": [
            {
                "step": event.step,
                "type": event.event_type,
                "details": event.details,
                "trigger": event.trigger_condition,
                "free_energy": (event.free_energy_before, event.free_energy_after),
            }
            for event in brain.get_evolution_history()
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="线虫世界模型演化实验")
    parser.add_argument("--steps", nargs="+", type=int, default=[200, 500, 1000])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=Path("data/lineworm_results.json"))
    parser.add_argument("--verbose", action="store_true", help="实时输出演化快照")
    parser.add_argument("--disable-pipeline", action="store_true", help="关闭 pipeline 演化")
    args = parser.parse_args()

    device = torch.device(args.device)
    sense_dims = {"chemo": 128, "thermo": 32, "touch": 64}
    config = {
        "state_dim": 256,
        "act_dim": 32,
        "obs_dim": sum(sense_dims.values()),
        "sense_dims": sense_dims,
        "enable_world_model_learning": True,
        "world_model": {
            "noise_config": {
                "chemo": {"std": 0.04, "amplitude": 0.4, "spatial_scale": 2.5},  # 降低噪声：std 0.08→0.04, amplitude 0.8→0.4
                "thermo": {"std": 0.03, "amplitude": 0.25, "spatial_scale": 3.5},  # 降低噪声：std 0.06→0.03, amplitude 0.5→0.25
            },
            "preferred_temp": 0.1,
        },
        "evolution": {
            "free_energy_threshold": 0.08,
            "prune_threshold": 0.01,
            "max_objects": 80,
            "max_aspects": 500,  # 增加 Aspect 数量上限，提供更多学习容量
            "error_ema_alpha": 0.5,
            "batch_growth": {
                "base": 8,
                "max_per_step": 32,
                "max_total": 200,  # 增加每个感官的最大 Aspect 数量
                "min_per_sense": 6,
                "error_threshold": 0.07,
                "error_multiplier": 0.7,
            },
        },
        "pipeline_growth": {
            "initial_depth": 3,
            "initial_width": 32,
            "depth_increment": 1,
            "width_increment": 12,
            "max_stages": 6,
            "min_interval": 80,
            "free_energy_trigger": None,
            "max_depth": 10,
        },
    }
    config["pipeline_growth"]["enable"] = not args.disable_pipeline

    results = []
    for steps in args.steps:
        print("\n" + "=" * 70)
        print(f"线虫实验：{steps} 步")
        print("=" * 70)
        summary = run_experiment(steps, config, device, verbose=args.verbose)
        results.append(summary)
        print(
            f"完成 {steps} 步: 自由能 {summary['final_free_energy']:.4f}, "
            f"Objects={summary['final_structure']['num_objects']}, "
            f"Aspects={summary['final_structure']['num_aspects']}"
        )

    output_path = Path(__file__).parent.parent / args.output if not args.output.is_absolute() else args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

