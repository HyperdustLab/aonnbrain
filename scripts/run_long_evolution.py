#!/usr/bin/env python3
"""
长时间演化实验启动脚本

默认配置：
- 步数: 15000（约 8 小时）
- Aspect 上限: 10000
- 保存间隔: 100 步
- 检查点: 每 1000 步
- LLM: Ollama cogito:32b（默认）
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="启动 AONN 长时间演化实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置（15000 步，约 8 小时）
  python scripts/run_long_evolution.py

  # 自定义步数
  python scripts/run_long_evolution.py --steps 20000

  # 使用 Ollama LLM
  python scripts/run_long_evolution.py --use-ollama --model cogito:32b

  # 后台运行并保存日志
  nohup python scripts/run_long_evolution.py > evolution.log 2>&1 &
        """
    )
    
    parser.add_argument("--steps", type=int, default=15000,
                       help="演化步数（默认: 15000，约 8 小时）")
    parser.add_argument("--save-interval", type=int, default=100,
                       help="保存快照的间隔（默认: 100 步）")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="检查点目录（默认: data/checkpoints_YYYYMMDD_HHMMSS）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件（默认: data/long_evolution_YYYYMMDD_HHMMSS.json）")
    parser.add_argument("--use-ollama", action="store_true",
                       help="使用 Ollama LLM（已默认启用，此选项已废弃）")
    parser.add_argument("--ollama-model", type=str, default="cogito:32b",
                       help="Ollama 模型名称（默认: cogito:32b）")
    parser.add_argument("--disable-ollama", action="store_true",
                       help="禁用 Ollama LLM（使用 MockLLMClient）")
    parser.add_argument("--use-openai", action="store_true",
                       help="使用 OpenAI LLM")
    parser.add_argument("--disable-llm", action="store_true",
                       help="禁用 LLM（使用 MockLLMClient）")
    parser.add_argument("--verbose", action="store_true",
                       help="输出详细信息")
    parser.add_argument("--device", type=str, default="cpu",
                       help="设备（默认: cpu）")
    
    args = parser.parse_args()
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置默认输出目录和文件
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"data/checkpoints_{timestamp}"
    if args.output is None:
        args.output = f"data/long_evolution_{timestamp}.json"
    
    # 创建输出目录
    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建命令
    cmd = [
        sys.executable,
        "scripts/run_general_ai_experiment.py",
        "--steps", str(args.steps),
        "--save-interval", str(args.save_interval),
        "--checkpoint-dir", args.checkpoint_dir,
        "--output", args.output,
        "--device", args.device,
    ]
    
    # 默认使用 Ollama cogito:32b，除非明确禁用
    if args.disable_ollama or args.disable_llm:
        if args.use_openai:
            cmd.extend(["--use-openai-llm"])
        else:
            # 如果禁用 Ollama，使用 Mock
            cmd.append("--disable-llm")
    elif args.use_openai:
        cmd.extend(["--use-openai-llm"])
    else:
        # 默认使用 Ollama（即使没有 --use-ollama 标志）
        cmd.extend(["--use-ollama-llm", "--ollama-model", args.ollama_model])
    
    if args.verbose:
        cmd.append("--verbose")
    
    # 打印配置信息
    print("=" * 80)
    print("AONN 长时间演化实验")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"演化步数: {args.steps:,}")
    print(f"保存间隔: {args.save_interval} 步")
    print(f"检查点目录: {args.checkpoint_dir}")
    print(f"输出文件: {args.output}")
    print(f"设备: {args.device}")
    if args.disable_ollama or args.disable_llm:
        if args.use_openai:
            print("LLM: OpenAI")
        else:
            print("LLM: 禁用（使用 Mock）")
    elif args.use_openai:
        print("LLM: OpenAI")
    else:
        print(f"LLM: Ollama ({args.ollama_model}) [默认]")
    print("=" * 80)
    print()
    
    # 运行实验
    try:
        subprocess.run(cmd, check=True)
        print()
        print("=" * 80)
        print("实验完成！")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"结果保存在: {args.output}")
        print(f"检查点保存在: {args.checkpoint_dir}")
        print("=" * 80)
    except subprocess.CalledProcessError as e:
        print(f"\n实验失败，退出代码: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
        print(f"部分结果可能已保存在: {args.checkpoint_dir}")
        sys.exit(130)


if __name__ == "__main__":
    main()
