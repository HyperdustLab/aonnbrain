#!/usr/bin/env python3
"""
详细分析各 Aspect 的贡献，找出瓶颈 Aspects
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import json
from typing import Dict, List, Tuple
from aonn.models.lineworm_world_model import LineWormWorldModel, LineWormWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.core.active_inference_loop import ActiveInferenceLoop

def analyze_aspect_contributions(config: Dict, device: torch.device, num_steps: int = 100):
    """详细分析各 Aspect 的贡献"""
    # 创建世界模型
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
    
    # 创建 AONN Brain
    brain = AONNBrainV3(config=config, device=device, enable_evolution=True)
    
    # 运行几步让网络演化
    obs = world_interface.reset()
    for sense, value in obs.items():
        if sense in brain.objects:
            brain.objects[sense].set_state(value)
    
    prev_obs = None
    prev_action = None
    
    print("=" * 80)
    print("Aspect 贡献详细分析")
    print("=" * 80)
    print()
    
    for step in range(num_steps):
        if step > 0:
            action = torch.randn(config["act_dim"], device=device) * 0.1
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
                    infer_lr=config.get("infer_lr", 0.01),
                    max_grad_norm=config.get("max_grad_norm", 100.0),
                    device=device,
                )
                loop.infer_states(target_objects=("internal",), num_iters=config.get("num_infer_iters", 5), sanitize_callback=brain.sanitize_states)
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
                learning_rate=config.get("learning_rate", 0.001),
            )
        
        prev_obs = {sense: value.clone() for sense, value in obs.items()}
        prev_action = action.clone()
        
        # 每10步分析一次
        if step % 10 == 0 or step == num_steps - 1:
            total_F = brain.compute_free_energy().item()
            
            # 收集所有 Aspects 的贡献
            aspect_details = []
            for aspect in brain.aspects:
                contrib = aspect.free_energy_contrib(brain.objects).item()
                aspect_type = type(aspect).__name__
                
                details = {
                    "name": getattr(aspect, "name", "unknown"),
                    "type": aspect_type,
                    "contrib": contrib,
                    "contrib_percent": (contrib / total_F * 100) if total_F > 0 else 0,
                }
                
                # 对于 LinearGenerativeAspect，添加更多信息
                if aspect_type == "LinearGenerativeAspect":
                    details["src"] = aspect.src_names
                    details["dst"] = aspect.dst_names
                    if hasattr(aspect, "W"):
                        W = aspect.W
                        details["weight_norm"] = torch.norm(W).item()
                        details["weight_max"] = torch.max(torch.abs(W)).item()
                        details["weight_mean"] = torch.mean(torch.abs(W)).item()
                
                aspect_details.append(details)
            
            # 按贡献排序
            aspect_details.sort(key=lambda x: x["contrib"], reverse=True)
            
            print(f"Step {step}: 总自由能 = {total_F:.2f}")
            print(f"  Aspects 总数: {len(brain.aspects)}")
            print()
            
            # 显示 Top 20 贡献最大的 Aspects
            print("  Top 20 贡献最大的 Aspects:")
            print("-" * 80)
            for i, details in enumerate(aspect_details[:20], 1):
                print(f"  {i:2d}. {details['name']:40s} | "
                      f"类型: {details['type']:25s} | "
                      f"贡献: {details['contrib']:8.2f} ({details['contrib_percent']:5.2f}%)")
                if details['type'] == "LinearGenerativeAspect":
                    print(f"      src={details.get('src', [])}, dst={details.get('dst', [])}, "
                          f"权重norm={details.get('weight_norm', 0):.3f}")
            print()
            
            # 按类型汇总
            type_summary = {}
            for details in aspect_details:
                aspect_type = details["type"]
                if aspect_type not in type_summary:
                    type_summary[aspect_type] = {
                        "count": 0,
                        "total_contrib": 0,
                        "max_contrib": 0,
                        "min_contrib": float('inf'),
                    }
                type_summary[aspect_type]["count"] += 1
                type_summary[aspect_type]["total_contrib"] += details["contrib"]
                type_summary[aspect_type]["max_contrib"] = max(type_summary[aspect_type]["max_contrib"], details["contrib"])
                type_summary[aspect_type]["min_contrib"] = min(type_summary[aspect_type]["min_contrib"], details["contrib"])
            
            print("  按类型汇总:")
            print("-" * 80)
            for aspect_type, summary in sorted(type_summary.items(), key=lambda x: x[1]["total_contrib"], reverse=True):
                avg_contrib = summary["total_contrib"] / summary["count"] if summary["count"] > 0 else 0
                print(f"  {aspect_type:30s}: "
                      f"数量={summary['count']:4d}, "
                      f"总贡献={summary['total_contrib']:8.2f}, "
                      f"平均={avg_contrib:6.2f}, "
                      f"最大={summary['max_contrib']:6.2f}, "
                      f"最小={summary['min_contrib']:6.2f}")
            print()
            
            # 分析 LinearGenerativeAspect 的分布
            linear_aspects = [d for d in aspect_details if d["type"] == "LinearGenerativeAspect"]
            if linear_aspects:
                print("  LinearGenerativeAspect 详细分析:")
                print("-" * 80)
                
                # 按感官类型分组
                sense_groups = {}
                for details in linear_aspects:
                    dst = details.get("dst", [])
                    if dst:
                        sense_name = dst[0] if isinstance(dst, (list, tuple)) else dst
                        if sense_name not in sense_groups:
                            sense_groups[sense_name] = []
                        sense_groups[sense_name].append(details)
                
                for sense_name, aspects in sorted(sense_groups.items(), key=lambda x: sum(a["contrib"] for a in x[1]), reverse=True):
                    total_contrib = sum(a["contrib"] for a in aspects)
                    avg_contrib = total_contrib / len(aspects) if aspects else 0
                    print(f"    {sense_name:15s}: {len(aspects):3d} 个 Aspects, "
                          f"总贡献={total_contrib:8.2f}, 平均={avg_contrib:6.2f}")
                
                # 找出贡献最大的感官
                max_sense = max(sense_groups.items(), key=lambda x: sum(a["contrib"] for a in x[1]))
                print(f"    最大贡献感官: {max_sense[0]} (总贡献={sum(a['contrib'] for a in max_sense[1]):.2f})")
                print()
            
            print("=" * 80)
            print()
            
            # 只分析最后一步
            if step == num_steps - 1:
                break

if __name__ == "__main__":
    device = torch.device("cpu")
    sense_dims = {"chemo": 128, "thermo": 32, "touch": 64}
    config = {
        "state_dim": 256,
        "act_dim": 32,
        "obs_dim": sum(sense_dims.values()),
        "sense_dims": sense_dims,
        "enable_world_model_learning": True,
        "world_model": {
            "noise_config": {
                "chemo": {"std": 0.04, "amplitude": 0.4, "spatial_scale": 2.5},
                "thermo": {"std": 0.03, "amplitude": 0.25, "spatial_scale": 3.5},
            },
            "preferred_temp": 0.1,
        },
        "evolution": {
            "free_energy_threshold": 0.08,
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
        "infer_lr": 0.01,
        "learning_rate": 0.0015,
        "num_infer_iters": 5,
        "max_grad_norm": 100.0,
    }
    
    analyze_aspect_contributions(config, device, num_steps=200)

