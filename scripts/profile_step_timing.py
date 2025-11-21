#!/usr/bin/env python3
"""
性能分析脚本：分析每一步的耗时分布
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aonn.models.general_ai_world_model import GeneralAIWorldModel, GeneralAIWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.aspects.ollama_llm_client import OllamaLLMClient

def profile_step():
    device = torch.device("cpu")
    
    config = {
        "state_dim": 1024,
        "act_dim": 256,
        "obs_dim": 1408,
        "sem_dim": 128,
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
        },
        "evolution": {
            "free_energy_threshold": 0.05,
            "max_aspects": 2000,
        },
    }
    
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
    )
    world_interface = GeneralAIWorldInterface(world_model)
    
    # 创建 LLM 客户端
    llm_client = OllamaLLMClient(
        input_dim=128,
        output_dim=128,
        model="cogito:32b",
        verbose=False,  # 关闭 verbose 以减少输出
        device=device,
    )
    
    # 创建 Brain
    brain = AONNBrainV3(config, llm_client=llm_client, device=device, enable_evolution=True)
    
    # 初始化
    obs = world_interface.reset()
    for sense, value in obs.items():
        if sense in brain.objects:
            brain.objects[sense].set_state(value)
    
    # 运行一步并分析耗时
    print("=" * 70)
    print("性能分析：单步耗时分布")
    print("=" * 70)
    
    total_start = time.time()
    
    # 1. 世界模型步进
    t0 = time.time()
    action = torch.randn(config["act_dim"], device=device) * 0.1
    obs, reward = world_interface.step(action)
    t1 = time.time()
    print(f"1. 世界模型步进: {t1-t0:.3f}秒")
    
    # 2. 设置观察
    t0 = time.time()
    for sense, value in obs.items():
        if sense in brain.objects:
            brain.objects[sense].set_state(value)
    t1 = time.time()
    print(f"2. 设置观察: {t1-t0:.3f}秒")
    
    # 3. 更新 semantic_context
    t0 = time.time()
    if "semantic_context" in brain.objects and hasattr(world_model, "semantic_state"):
        sem_dim = config.get("sem_dim", 128)
        world_semantic = world_model.semantic_state[:sem_dim].detach().clone()
        brain.update_semantic_context(world_semantic_state=world_semantic)
    t1 = time.time()
    print(f"3. 更新 semantic_context: {t1-t0:.3f}秒")
    
    # 4. 网络演化
    t0 = time.time()
    full_state = world_model.get_true_state()
    if full_state.shape[-1] >= config["state_dim"]:
        target_state = full_state[:config["state_dim"]]
    else:
        padding = torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)
        target_state = torch.cat([full_state, padding], dim=-1)
    brain.evolve_network(obs, target=target_state)
    t1 = time.time()
    print(f"4. 网络演化: {t1-t0:.3f}秒")
    print(f"   当前 Aspects 数量: {len(brain.aspects)}")
    
    # 5. 主动推理（这里会调用 LLM）
    t0 = time.time()
    if len(brain.aspects) > 0:
        for obj_name, obj in brain.objects.items():
            state = obj.state
            if state.requires_grad and state.is_leaf and state.grad is not None:
                obj.set_state(state.detach())
            elif not state.is_leaf:
                obj.set_state(state.detach())
        
        # 测试 LLM 调用耗时
        llm_start = time.time()
        if "semantic_context" in brain.objects and llm_client is not None:
            test_vec = brain.objects["semantic_context"].state
            _ = llm_client.semantic_predict(test_vec, use_cache=False)  # 不使用缓存，测试真实耗时
        llm_end = time.time()
        print(f"5. LLM API 调用（单次）: {llm_end-llm_start:.3f}秒")
        
        # 主动推理循环
        infer_start = time.time()
        loop = ActiveInferenceLoop(
            brain.objects,
            brain.aspects,
            infer_lr=0.02,
            device=device,
        )
        loop.infer_states(target_objects=("internal",), num_iters=1)
        infer_end = time.time()
        print(f"6. 主动推理循环: {infer_end-infer_start:.3f}秒")
        print(f"   包含 LLM 调用次数: 1次（每次 infer_states 会调用 compute_free_energy -> LLMAspect.free_energy_contrib）")
        
        brain.sanitize_states()
    t1 = time.time()
    
    # 6. 世界模型学习
    t0 = time.time()
    prev_obs = {sense: value.clone() for sense, value in obs.items()}
    prev_action = action.clone()
    if prev_obs is not None and prev_action is not None:
        full_state = world_model.get_true_state()
        if full_state.shape[-1] >= config["state_dim"]:
            target_state = full_state[:config["state_dim"]]
        else:
            padding = torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)
            target_state = torch.cat([full_state, padding], dim=-1)
        brain.learn_world_model(
            observation=prev_obs,
            action=prev_action,
            next_observation=obs,
            target_state=target_state,
            learning_rate=0.0001,
        )
    t1 = time.time()
    print(f"7. 世界模型学习: {t1-t0:.3f}秒")
    
    total_end = time.time()
    print("=" * 70)
    print(f"总耗时: {total_end-total_start:.3f}秒")
    print("=" * 70)
    print("\n分析：")
    print("- LLM API 调用是主要瓶颈（cogito:32b 是32B参数大模型）")
    print("- 每次 infer_states 都会调用 LLM（通过 free_energy_contrib）")
    print("- 如果网络中有很多 Aspects，计算自由能也会较慢")
    print("\n优化建议：")
    print("1. 增加 LLM 缓存命中率（use_cache=True）")
    print("2. 减少 infer_states 的迭代次数（已设置为1）")
    print("3. 使用更小的模型（如 llama3:8b）")
    print("4. 减少 Aspects 数量（降低演化阈值）")

if __name__ == "__main__":
    profile_step()

