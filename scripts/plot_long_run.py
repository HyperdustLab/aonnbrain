#!/usr/bin/env python3
"""
根据 data/long_run_results.json 绘制自由能曲线
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def plot_free_energy(results, output_path: Path):
    plt.figure(figsize=(8, 5))

    for summary in results:
        steps = [snap["step"] for snap in summary.get("snapshots", [])]
        energies = [snap["free_energy"] for snap in summary.get("snapshots", [])]
        if steps and energies:
            plt.plot(steps, energies, marker="o", label=f"{summary['num_steps']} steps")

    plt.xlabel("Step")
    plt.ylabel("Free Energy")
    plt.title("AONN 长周期演化自由能曲线")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"自由能曲线已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="绘制长周期演化自由能曲线")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/long_run_results.json"),
        help="长周期实验结果 JSON 文件",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_run_free_energy.png"),
        help="输出图像路径",
    )
    args = parser.parse_args()

    results = load_results(args.input)
    if not results:
        raise ValueError("结果文件为空或格式不正确")

    plot_free_energy(results, args.output)


if __name__ == "__main__":
    main()
