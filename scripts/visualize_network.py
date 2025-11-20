#!/usr/bin/env python3
"""
可视化 AONN 网络架构
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aonn.models.aonn_brain import AONNBrain
from aonn.aspects.mock_llm_client import create_default_mock_llm_client
from aonn.utils.config import load_config


def main():
    # 加载配置
    config = load_config("configs/brain_default.yaml")
    device = "cpu"
    
    # 创建 LLM 客户端
    llm_client = create_default_mock_llm_client(
        input_dim=config["sem_dim"],
        output_dim=config["sem_dim"],
        device=device,
    )
    
    # 创建大脑
    brain = AONNBrain(config=config, llm_client=llm_client, device=device)
    
    # 可视化网络
    print(brain.visualize_network())
    
    # 获取网络图结构
    graph = brain.get_network_graph()
    print("\n网络图统计:")
    print(f"  - Object 节点数: {graph['num_objects']}")
    print(f"  - Aspect 数量: {graph['num_aspects']}")
    print(f"  - 边数量: {graph['num_edges']}")
    
    print("\n详细边信息:")
    for edge in graph['edges']:
        print(f"  {edge['aspect']}: {edge['src']} -> {edge['dst']}")


if __name__ == "__main__":
    main()

