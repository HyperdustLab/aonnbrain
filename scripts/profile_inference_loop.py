#!/usr/bin/env python3
"""
主动推理循环性能分析脚本
分析各个环节的耗时分布
"""
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
from tqdm import tqdm

from aonn.models.office_ai_world_model import OfficeAIWorldModel, OfficeAIWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.aspects.mock_llm_client import MockLLMClient

try:
    from aonn.aspects.ollama_llm_client import OllamaLLMClient
except Exception:
    OllamaLLMClient = None


class TimingProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.current_step = None
        self.step_timings: Dict[int, Dict[str, float]] = {}
    
    def start(self, name: str):
        """开始计时"""
        self.current_step = (name, time.perf_counter())
    
    def end(self, name: str = None):
        """结束计时"""
        if self.current_step is None:
            return
        step_name, start_time = self.current_step
        if name is None:
            name = step_name
        elapsed = time.perf_counter() - start_time
        self.timings[name].append(elapsed)
        self.current_step = None
        return elapsed
    
    def record(self, name: str, duration: float):
        """直接记录时间"""
        self.timings[name].append(duration)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取统计摘要"""
        summary = {}
        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times),
                }
        return summary
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        print("\n" + "=" * 80)
        print("性能分析摘要")
        print("=" * 80)
        print(f"{'环节':<30} {'总耗时(秒)':<15} {'平均(秒)':<15} {'最小(秒)':<15} {'最大(秒)':<15} {'调用次数':<10}")
        print("-" * 80)
        
        # 按总耗时排序
        sorted_items = sorted(summary.items(), key=lambda x: x[1]["total"], reverse=True)
        
        for name, stats in sorted_items:
            print(f"{name:<30} {stats['total']:<15.4f} {stats['mean']:<15.4f} "
                  f"{stats['min']:<15.4f} {stats['max']:<15.4f} {stats['count']:<10}")
        
        total_time = sum(s["total"] for s in summary.values())
        print("-" * 80)
        print(f"{'总计':<30} {total_time:<15.4f}")
        print("=" * 80)
        
        # 计算百分比
        print("\n各环节耗时占比:")
        print("-" * 80)
        for name, stats in sorted_items:
            percentage = (stats["total"] / total_time * 100) if total_time > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"{name:<30} {percentage:>6.2f}% {bar}")
        print("=" * 80)


def profile_experiment(num_steps: int, config: Dict, device: torch.device, use_llm: bool = False):
    """运行性能分析实验"""
    
    profiler = TimingProfiler()
    
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
    )
    world_interface = OfficeAIWorldInterface(world_model)
    
    # 初始化 LLM 客户端（可选）
    llm_client = None
    sem_dim = config.get("sem_dim", 128)
    if use_llm and OllamaLLMClient is not None:
        llm_cfg = config.get("llm", {})
        llm_client = OllamaLLMClient(
            input_dim=sem_dim,
            output_dim=sem_dim,
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            model=llm_cfg.get("model", "cogito:32b"),
            embedding_model=llm_cfg.get("embedding_model"),
            summary_size=llm_cfg.get("summary_size", 8),
            max_tokens=llm_cfg.get("max_tokens", 120),
            temperature=llm_cfg.get("temperature", 0.7),
            timeout=llm_cfg.get("timeout", 120.0),
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
    profiler.start("brain_init")
    brain = AONNBrainV3(config=config, llm_client=llm_client, device=device, enable_evolution=True)
    profiler.end("brain_init")
    
    # 初始化环境
    obs = world_interface.reset()
    for sense, value in obs.items():
        if sense in brain.objects:
            brain.objects[sense].set_state(value)
    
    prev_obs = None
    prev_action = None
    action = torch.randn(config["act_dim"], device=device) * 0.1
    
    print(f"开始性能分析实验（{num_steps} 步）...")
    print(f"LLM: {'启用' if use_llm else '禁用（Mock）'}")
    print()
    
    for step in tqdm(range(num_steps), desc="性能分析"):
        step_start = time.perf_counter()
        step_timings = {}
        
        # 1. 世界模型步骤
        profiler.start("world_model_step")
        if prev_obs is not None and prev_action is not None:
            step_result = world_interface.step(action)
            if isinstance(step_result, tuple) and len(step_result) == 3:
                obs, reward, done = step_result
            elif isinstance(step_result, tuple) and len(step_result) == 2:
                obs, reward = step_result
                done = False
            else:
                obs = step_result
                reward = 0.0
                done = False
        else:
            obs = world_interface.reset()
            reward = 0.0
            done = False
        profiler.end("world_model_step")
        
        # 2. 设置观察
        profiler.start("set_observation")
        for sense, value in obs.items():
            if sense in brain.objects:
                brain.objects[sense].set_state(value)
        profiler.end("set_observation")
        
        # 3. 网络演化
        profiler.start("evolve_network")
        target_state = world_model.get_true_state()
        if target_state.shape[-1] >= config["state_dim"]:
            target_state = target_state[:config["state_dim"]]
        else:
            padding = torch.zeros(config["state_dim"] - target_state.shape[-1], device=device)
            target_state = torch.cat([target_state, padding], dim=-1)
        brain.evolve_network(obs, target=target_state)
        profiler.end("evolve_network")
        
        # 4. 世界模型学习
        profiler.start("learn_world_model")
        if prev_obs is not None and prev_action is not None:
            brain.learn_world_model(
                observation=prev_obs,
                action=prev_action,
                next_observation=obs,
                target_state=target_state,
                learning_rate=config.get("learning_rate", 0.001),
            )
        profiler.end("learn_world_model")
        
        # 5. 主动推理（详细分析）
        if len(brain.aspects) > 0:
            # 5.1 准备状态（detach）
            profiler.start("prepare_states")
            for obj_name, obj in brain.objects.items():
                state = obj.state
                if state.requires_grad and state.is_leaf and state.grad is not None:
                    obj.set_state(state.detach())
                elif not state.is_leaf:
                    obj.set_state(state.detach())
            profiler.end("prepare_states")
            
            # 5.2 创建推理循环
            profiler.start("create_inference_loop")
            loop = ActiveInferenceLoop(
                brain.objects,
                brain.aspects,
                infer_lr=config.get("infer_lr", 0.02),
                max_grad_norm=config.get("max_grad_norm", None),
                device=device,
            )
            profiler.end("create_inference_loop")
            
            # 5.3 推理迭代（详细分析）
            num_iters = config.get("num_infer_iters", 2)
            for iter_idx in range(num_iters):
                iter_start = time.perf_counter()
                
                # 5.3.1 重置状态为可微
                profiler.start("reset_states_for_grad")
                for name in ("internal",):
                    mu = brain.objects[name].clone_detached(requires_grad=True)
                    brain.objects[name].state = mu
                profiler.end("reset_states_for_grad")
                
                # 5.3.2 计算自由能
                profiler.start("compute_free_energy")
                from aonn.core.free_energy import compute_total_free_energy
                is_last_iter = (iter_idx == num_iters - 1)
                F = compute_total_free_energy(
                    brain.objects,
                    brain.aspects,
                    iteration_idx=iter_idx,
                    is_last_iter=is_last_iter,
                )
                profiler.end("compute_free_energy")
                
                # 5.3.3 反向传播
                profiler.start("backward")
                retain_graph = (iter_idx < num_iters - 1)
                try:
                    F.backward(retain_graph=retain_graph)
                except RuntimeError as e:
                    if "backward through the graph a second time" in str(e) or "freed" in str(e):
                        for name in ("internal",):
                            mu = brain.objects[name].clone_detached(requires_grad=False)
                            brain.objects[name].state = mu
                        continue
                    else:
                        raise
                profiler.end("backward")
                
                # 5.3.4 更新状态
                profiler.start("update_states")
                with torch.no_grad():
                    for name in ("internal",):
                        mu = brain.objects[name].state
                        if mu.requires_grad and mu.is_leaf and mu.grad is not None:
                            grad = mu.grad
                            if loop.max_grad_norm is not None and loop.max_grad_norm > 0:
                                grad_norm = torch.norm(grad)
                                if grad_norm > loop.max_grad_norm:
                                    grad = grad * (loop.max_grad_norm / grad_norm)
                            mu = mu - loop.infer_lr * grad
                        brain.objects[name].state = mu.detach()
                profiler.end("update_states")
                
                # 5.3.5 状态清理
                profiler.start("sanitize_states")
                brain.sanitize_states()
                profiler.end("sanitize_states")
                
                iter_time = time.perf_counter() - iter_start
                profiler.record(f"inference_iter_{iter_idx}", iter_time)
            
            # 5.4 最终状态清理
            profiler.start("final_sanitize")
            brain.sanitize_states()
            profiler.end("final_sanitize")
        
        # 6. 更新语义上下文
        profiler.start("update_semantic_context")
        brain.update_semantic_context(source="internal")
        profiler.end("update_semantic_context")
        
        # 7. 获取动作
        profiler.start("get_action")
        action = brain.objects["action"].state.clone()
        profiler.end("get_action")
        
        # 记录步骤总时间
        step_time = time.perf_counter() - step_start
        profiler.record("total_step_time", step_time)
        
        prev_obs = {sense: value.clone() for sense, value in obs.items()}
        prev_action = action.clone()
    
    return profiler


def main():
    parser = argparse.ArgumentParser(description="主动推理循环性能分析")
    parser.add_argument("--steps", type=int, default=10, help="实验步数")
    parser.add_argument("--use-ollama-llm", action="store_true", help="使用 Ollama LLM")
    parser.add_argument("--ollama-model", type=str, default="cogito:32b", help="Ollama 模型名称")
    parser.add_argument("--output", type=str, default="data/profile_results.json", help="输出文件路径")
    
    args = parser.parse_args()
    
    device = torch.device("cpu")
    
    # 配置
    config = {
        "state_dim": 576,
        "act_dim": 128,
        "obs_dim": 448,
        "sem_dim": 128,
        "sense_dims": {
            "document": 256,
            "table": 128,
            "calendar": 64,
        },
        "enable_world_model_learning": True,
        "llm": {
            "call_frequency": "last_iter_only",
            "call_every_n_steps": 1,
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
            "free_energy_threshold": 0.08,
            "prune_threshold": 0.01,
            "max_objects": 80,
            "max_aspects": 500,
            "error_ema_alpha": 0.5,
            "batch_growth": {
                "base": 8,
                "max_per_step": 32,
                "max_total": 200,
                "min_per_sense": 8,
                "error_threshold": 0.1,
                "error_multiplier": 0.7,
            },
            "pipeline_growth": {
                "free_energy_trigger": 0.1,
                "min_interval": 10,
                "max_stages": 200,
                "max_depth": 50,
                "initial_width": 1,
                "width_increment": 0,
            },
        },
        "infer_lr": 0.02,
        "learning_rate": 0.001,
        "num_infer_iters": 2,
        "max_grad_norm": 100.0,
        "state_clip_value": 5.0,
    }
    
    # 运行性能分析
    profiler = profile_experiment(
        num_steps=args.steps,
        config=config,
        device=device,
        use_llm=args.use_ollama_llm,
    )
    
    # 打印摘要
    profiler.print_summary()
    
    # 保存结果
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "summary": profiler.get_summary(),
        "all_timings": {k: v for k, v in profiler.timings.items()},
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

