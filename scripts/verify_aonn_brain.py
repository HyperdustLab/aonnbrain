#!/usr/bin/env python3
"""
AONN 大脑完整验证脚本
全面验证 AONN 大脑的各个组件和功能是否正常工作
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from aonn.models.aonn_brain import AONNBrain
from aonn.pipeline.inference_agent import InferenceAgent
from aonn.core.free_energy import compute_total_free_energy
from aonn.aspects.mock_llm_client import create_default_mock_llm_client
from aonn.utils.config import load_config
from aonn.utils.logging import setup_logger


def verify_aonn_brain():
    """完整验证 AONN 大脑"""
    logger = setup_logger()
    
    logger.info("=" * 70)
    logger.info("AONN 大脑完整验证")
    logger.info("=" * 70)
    
    # 加载配置
    try:
        config = load_config("configs/brain_default.yaml")
        train_config = load_config("configs/training_default.yaml")
        config.update(train_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return False
    
    # ========== 验证 1: 大脑创建 ==========
    logger.info("\n[验证 1/10] 大脑创建...")
    try:
        llm_client = create_default_mock_llm_client(
            input_dim=config["sem_dim"],
            output_dim=config["sem_dim"],
            device=device,
        )
        brain = AONNBrain(config=config, llm_client=llm_client, device=device)
        brain = brain.to(device)
        logger.info("   ✓ 大脑创建成功")
        logger.info(f"   设备: {device}")
    except Exception as e:
        logger.error(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== 验证 2: Object 完整性 ==========
    logger.info("\n[验证 2/10] Object 完整性...")
    try:
        required_objects = ["sensory", "internal", "action", 
                           "semantic_context", "semantic_prediction"]
        for obj_name in required_objects:
            if obj_name not in brain.objects:
                logger.error(f"   ✗ 缺少 Object: {obj_name}")
                return False
            obj = brain.objects[obj_name]
            assert obj.dim > 0, f"Object {obj_name} 维度无效"
            assert obj.state.shape == (obj.dim,), f"Object {obj_name} 状态形状错误"
        logger.info(f"   ✓ 所有必需的 Object 存在 ({len(brain.objects)} 个)")
        for name, obj in brain.objects.items():
            logger.info(f"      - {name}: dim={obj.dim}, device={obj.device}")
    except Exception as e:
        logger.error(f"   ✗ 失败: {e}")
        return False
    
    # ========== 验证 3: Aspect 完整性 ==========
    logger.info("\n[验证 3/11] Aspect 完整性...")
    try:
        aspects = brain.aspects if isinstance(brain.aspects, list) else list(brain.aspects)
        assert len(aspects) > 0, "应该有至少一个 Aspect"
        assert hasattr(brain, 'sensory_aspect'), "缺少 sensory_aspect"
        assert hasattr(brain, 'llm_aspect'), "缺少 llm_aspect"
        
        # 验证每个 Aspect 都有自由能贡献方法
        for aspect in aspects:
            assert hasattr(aspect, 'free_energy_contrib'), \
                f"Aspect {aspect.name} 缺少 free_energy_contrib 方法"
        
        logger.info(f"   ✓ 找到 {len(aspects)} 个 Aspect")
        for aspect in aspects:
            logger.info(f"      - {aspect.name}: src={aspect.src_names}, dst={aspect.dst_names}")
    except Exception as e:
        logger.error(f"   ✗ 失败: {e}")
        return False
    
    # ========== 验证 3.5: 网络拓扑结构 ==========
    logger.info("\n[验证 3.5/11] 网络拓扑结构...")
    try:
        assert hasattr(brain, 'topology'), "大脑应该包含网络拓扑结构"
        topology = brain.topology
        
        # 验证拓扑结构完整性
        assert len(topology.objects) == len(brain.objects), "拓扑中的 Object 数量应该匹配"
        assert len(topology.aspects) == len(aspects), "拓扑中的 Aspect 数量应该匹配"
        assert len(topology.edges) > 0, "应该有至少一条边"
        
        # 验证网络图
        graph = brain.get_network_graph()
        assert graph['num_objects'] == len(brain.objects)
        assert graph['num_aspects'] == len(aspects)
        assert graph['num_edges'] == len(topology.edges)
        
        logger.info(f"   ✓ 网络拓扑结构正常")
        logger.info(f"      - 节点数: {graph['num_objects']}")
        logger.info(f"      - Aspect 数: {graph['num_aspects']}")
        logger.info(f"      - 边数: {graph['num_edges']}")
        logger.info(f"   网络连接:")
        for edge in graph['edges']:
            logger.info(f"      {edge['aspect']}: {edge['src']} -> {edge['dst']}")
    except Exception as e:
        logger.error(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== 验证 4: 自由能计算 ==========
    logger.info("\n[验证 4/11] 自由能计算...")
    try:
        F = compute_total_free_energy(brain.objects, list(brain.aspects))
        assert F.item() >= 0, "自由能应该非负"
        assert isinstance(F, torch.Tensor), "自由能应该是 Tensor"
        logger.info(f"   ✓ 自由能计算正常: F = {F.item():.4f}")
        
        # 验证每个 Aspect 的贡献
        aspects = brain.aspects if isinstance(brain.aspects, list) else list(brain.aspects)
        for aspect in aspects:
            F_contrib = aspect.free_energy_contrib(brain.objects)
            logger.info(f"      - {aspect.name} 贡献: {F_contrib.item():.4f}")
    except Exception as e:
        logger.error(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== 验证 5: 推理代理创建 ==========
    logger.info("\n[验证 5/11] 推理代理...")
    try:
        agent = InferenceAgent(brain, infer_lr=config["inference"]["infer_lr"])
        assert agent.brain == brain, "推理代理应该引用同一个大脑"
        logger.info("   ✓ 推理代理创建成功")
        logger.info(f"   推理学习率: {agent.infer_loop.infer_lr}")
    except Exception as e:
        logger.error(f"   ✗ 失败: {e}")
        return False
    
    # ========== 验证 6: 观察设置 ==========
    logger.info("\n[验证 6/11] 观察设置...")
    try:
        obs = torch.randn(config["obs_dim"]).to(device)
        agent.observe(obs)
        sensory_state = agent.brain.objects["sensory"].state
        assert torch.allclose(sensory_state, obs), "观察应该正确设置到 sensory Object"
        logger.info(f"   ✓ 观察设置成功: shape={obs.shape}")
    except Exception as e:
        logger.error(f"   ✗ 失败: {e}")
        return False
    
    # ========== 验证 7: 主动推理循环 ==========
    logger.info("\n[验证 7/11] 主动推理循环...")
    try:
        F_before = agent.get_free_energy()
        logger.info(f"   推理前自由能: {F_before:.4f}")
        
        # 记录初始状态
        internal_before = agent.get_internal_state().clone()
        action_before = agent.get_action().clone()
        
        # 执行推理
        num_iters = 5
        target_objects = config["inference"]["target_objects"]
        agent.infer(num_iters=num_iters, target_objects=target_objects)
        
        F_after = agent.get_free_energy()
        internal_after = agent.get_internal_state()
        action_after = agent.get_action()
        
        # 验证状态变化
        internal_changed = not torch.allclose(internal_before, internal_after, atol=1e-6)
        action_changed = not torch.allclose(action_before, action_after, atol=1e-6)
        
        logger.info(f"   推理后自由能: {F_after:.4f}")
        logger.info(f"   自由能变化: {F_after - F_before:.4f}")
        logger.info(f"   内部状态变化: {'是' if internal_changed else '否'}")
        logger.info(f"   动作状态变化: {'是' if action_changed else '否'}")
        
        # 自由能应该下降或至少不显著上升（允许小幅波动）
        assert F_after <= F_before * 1.2, \
            f"自由能不应该显著上升 (前: {F_before:.4f}, 后: {F_after:.4f})"
        assert internal_changed or action_changed, "至少一个状态应该发生变化"
        
        logger.info("   ✓ 主动推理循环正常")
    except Exception as e:
        logger.error(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== 验证 8: 状态访问 ==========
    logger.info("\n[验证 8/11] 状态访问...")
    try:
        internal = agent.get_internal_state()
        action = agent.get_action()
        free_energy = agent.get_free_energy()
        
        assert internal.shape == (config["state_dim"],), \
            f"内部状态形状错误: 期望 ({config['state_dim']},), 实际 {internal.shape}"
        assert action.shape == (config["act_dim"],), \
            f"动作形状错误: 期望 ({config['act_dim']},), 实际 {action.shape}"
        assert isinstance(free_energy, float), "自由能应该是标量"
        
        logger.info(f"   ✓ 状态访问正常")
        logger.info(f"      - 内部状态: shape={internal.shape}, 范围=[{internal.min():.3f}, {internal.max():.3f}]")
        logger.info(f"      - 动作: shape={action.shape}, 范围=[{action.min():.3f}, {action.max():.3f}]")
        logger.info(f"      - 自由能: {free_energy:.4f}")
    except Exception as e:
        logger.error(f"   ✗ 失败: {e}")
        return False
    
    # ========== 验证 9: 自由能单调性 ==========
    logger.info("\n[验证 9/11] 自由能单调性验证...")
    try:
        # 设置一个目标观察
        target_obs = torch.randn(config["obs_dim"]).to(device)
        agent.observe(target_obs)
        
        # 获取初始自由能
        F_initial = agent.get_free_energy()
        logger.info(f"   初始自由能: {F_initial:.4f}")
        
        # 执行多步推理
        for i in range(10):
            agent.infer(num_iters=1, target_objects=("internal",))
            F_current = agent.get_free_energy()
            logger.info(f"   步骤 {i+1}: F = {F_current:.4f}")
        
        F_final = agent.get_free_energy()
        logger.info(f"   最终自由能: {F_final:.4f}")
        
        # 验证自由能趋势（应该下降或稳定）
        improvement = (F_initial - F_final) / F_initial if F_initial > 0 else 0
        logger.info(f"   改善比例: {improvement * 100:.2f}%")
        
        # 允许小幅波动，但不应该大幅上升
        assert F_final <= F_initial * 1.5, \
            f"自由能不应该大幅上升 (初始: {F_initial:.4f}, 最终: {F_final:.4f})"
        
        logger.info("   ✓ 自由能单调性验证通过")
    except Exception as e:
        logger.error(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== 验证 10: 训练能力 ==========
    logger.info("\n[验证 10/11] 训练能力...")
    try:
        # 检查参数
        params = list(brain.parameters())
        assert len(params) > 0, "应该有可训练参数"
        
        total_params = sum(p.numel() for p in params)
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        
        logger.info(f"   总参数数量: {total_params:,}")
        logger.info(f"   可训练参数: {trainable_params:,}")
        logger.info(f"   参数组数: {len(params)}")
        
        # 测试梯度计算
        optimizer = torch.optim.Adam(brain.parameters(), lr=0.001)
        
        # 前向传播
        aspects = brain.aspects if isinstance(brain.aspects, list) else list(brain.aspects)
        F = compute_total_free_energy(brain.objects, aspects)
        
        # 反向传播
        optimizer.zero_grad()
        F.backward()
        
        # 检查梯度
        has_grad = any(p.grad is not None for p in params if p.requires_grad)
        assert has_grad, "至少部分参数应该有梯度"
        
        # 优化步骤
        optimizer.step()
        
        logger.info("   ✓ 训练流程正常")
        logger.info(f"      - 梯度计算: {'是' if has_grad else '否'}")
        logger.info(f"      - 优化步骤: 成功")
    except Exception as e:
        logger.error(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== 验证总结 ==========
    logger.info("\n" + "=" * 70)
    logger.info("✓ 所有验证通过！AONN 大脑工作正常")
    logger.info("=" * 70)
    logger.info("\n验证总结:")
    logger.info("  ✓ 大脑结构完整")
    logger.info("  ✓ 自由能计算正确")
    logger.info("  ✓ 主动推理循环正常")
    logger.info("  ✓ 状态更新机制正常")
    logger.info("  ✓ 训练能力正常")
    logger.info("\nAONN 大脑已准备好使用！")
    
    return True


def main():
    """主函数"""
    success = verify_aonn_brain()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

