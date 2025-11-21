#!/usr/bin/env python3
"""
MNIST AONN 实时监控工具
用于实时观察正在运行的实验
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import time
import argparse
import os
from typing import Optional


def load_latest_snapshot(json_file: str) -> Optional[dict]:
    """加载最新的快照"""
    if not os.path.exists(json_file):
        return None
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        snapshots = data.get('snapshots', [])
        if snapshots:
            return snapshots[-1]
        return data
    except:
        return None


def monitor_log_file(log_file: str, last_lines: int = 20):
    """监控日志文件的最后几行"""
    if not os.path.exists(log_file):
        return []
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return lines[-last_lines:]
    except:
        return []


def format_snapshot_info(snapshot: dict) -> str:
    """格式化快照信息"""
    if not snapshot:
        return "等待数据..."
    
    step = snapshot.get('step', 0)
    structure = snapshot.get('structure', {})
    F = snapshot.get('F', 0)
    accuracy = snapshot.get('accuracy', 0)
    
    info = f"""
╔══════════════════════════════════════════════════════════════╗
║ Step: {step:<60} ║
║ Free Energy: {F:<55.4f} ║
║ Accuracy: {accuracy*100:<56.2f}% ║
║ Objects: {structure.get('num_objects', 0):<58} ║
║ Aspects: {structure.get('num_aspects', 0):<59} ║
║ Pipelines: {structure.get('num_pipelines', 0):<57} ║
╚══════════════════════════════════════════════════════════════╝
"""
    return info


def main():
    parser = argparse.ArgumentParser(description="MNIST AONN 实时监控工具")
    parser.add_argument("--json", type=str, default="data/mnist_evolution_test_1000steps.json",
                        help="实验结果JSON文件路径")
    parser.add_argument("--log", type=str, default="data/mnist_evolution_1000steps.log",
                        help="日志文件路径")
    parser.add_argument("--interval", type=float, default=2.0, help="更新间隔（秒）")
    parser.add_argument("--show-log", action="store_true", help="显示日志输出")
    
    args = parser.parse_args()
    
    print("🔍 MNIST AONN 实时监控")
    print("=" * 80)
    print(f"监控文件: {args.json}")
    print(f"更新间隔: {args.interval} 秒")
    print("按 Ctrl+C 退出")
    print("=" * 80)
    
    last_step = -1
    
    try:
        while True:
            # 清屏（可选）
            # os.system('clear' if os.name != 'nt' else 'cls')
            
            # 加载最新快照
            snapshot = load_latest_snapshot(args.json)
            
            if snapshot:
                current_step = snapshot.get('step', 0)
                if current_step != last_step:
                    print(f"\n⏰ {time.strftime('%H:%M:%S')}")
                    print(format_snapshot_info(snapshot))
                    last_step = current_step
                    
                    # 检查是否完成
                    if 'final_structure' in snapshot:
                        print("✅ 实验已完成！")
                        break
            
            # 显示日志（如果启用）
            if args.show_log:
                log_lines = monitor_log_file(args.log, last_lines=5)
                if log_lines:
                    print("\n📋 最新日志:")
                    for line in log_lines[-3:]:
                        print(f"   {line.strip()}")
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")


if __name__ == "__main__":
    main()

