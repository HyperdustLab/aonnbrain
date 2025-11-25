#!/usr/bin/env python3
"""
Analyze Office AI experiment outputs.

Given a JSON result file (e.g., data/office_ai_results_steps500.json), this script
prints basic statistics about the free-energy trajectory, structure growth, and
object state norms. Optionally, it renders a matplotlib figure that visualizes
the dynamics across snapshots.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import os

MPL_CACHE_DIR = Path(".cache/matplotlib")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_time_series(data: Dict) -> Tuple[List[int], List[float], List[Dict]]:
    snapshots = data.get("snapshots", [])
    if not snapshots:
        raise ValueError("results file does not contain any snapshots")

    steps = [snap["step"] for snap in snapshots]
    free_energy = [float(snap["free_energy"]) for snap in snapshots]
    structures = [snap["structure"] for snap in snapshots]

    final_step = data.get("num_steps")
    final_fe = data.get("final_free_energy")
    final_structure = data.get("final_structure")

    if (
        final_step is not None
        and final_structure is not None
        and (not steps or steps[-1] != final_step)
    ):
        steps.append(int(final_step))
        free_energy.append(float(final_fe) if final_fe is not None else free_energy[-1])
        structures.append(copy.deepcopy(final_structure))

    return steps, free_energy, structures


def compute_structure_stats(structures: List[Dict]) -> Dict:
    num_objects = [s.get("num_objects", 0) for s in structures]
    num_aspects = [s.get("num_aspects", 0) for s in structures]
    num_pipeline_aspects = [s.get("num_pipeline_aspects", 0) for s in structures]
    llm_active_steps = sum(1 for s in structures if s.get("has_llm_aspect"))

    return {
        "num_objects": num_objects,
        "num_aspects": num_aspects,
        "num_pipeline_aspects": num_pipeline_aspects,
        "llm_active_ratio": llm_active_steps / len(structures),
    }


def collect_object_norms(
    steps: List[int], structures: List[Dict], tracked_objects: Tuple[str, ...]
) -> Dict[str, List[Tuple[int, float]]]:
    norms = defaultdict(list)
    for step, structure in zip(steps, structures):
        for name in tracked_objects:
            obj_state = structure.get("objects", {}).get(name)
            if obj_state and obj_state.get("state_norm") is not None:
                norms[name].append((step, float(obj_state["state_norm"])))
    return norms


def print_summary(
    steps: List[int],
    free_energy: List[float],
    struct_stats: Dict,
    final_structure: Dict,
) -> None:
    start_fe = free_energy[0]
    final_fe = free_energy[-1]
    delta_fe = final_fe - start_fe

    print("=" * 80)
    print("Office AI Evolution Summary")
    print("=" * 80)
    print(f"Steps recorded : {len(steps)} snapshots (0 → {steps[-1]})")
    print(f"Free energy    : {start_fe:.3f} → {final_fe:.3f} (Δ {delta_fe:+.3f})")
    print(f"Free energy µ  : {mean(free_energy):.3f}")
    print("-" * 80)
    print(
        f"Objects        : {struct_stats['num_objects'][0]} → "
        f"{struct_stats['num_objects'][-1]}"
    )
    print(
        f"Aspects        : {struct_stats['num_aspects'][0]} → "
        f"{struct_stats['num_aspects'][-1]}"
    )
    print(
        f"Pipeline aspects: {struct_stats['num_pipeline_aspects'][0]} → "
        f"{struct_stats['num_pipeline_aspects'][-1]}"
    )
    print(
        f"LLM Aspect active in {struct_stats['llm_active_ratio'] * 100:.1f}% of snapshots"
    )
    print("-" * 80)
    if final_structure:
        pipeline_info = final_structure.get("pipelines", [])
        print(f"Final pipelines: {len(pipeline_info)} entries")
        for pipe in pipeline_info:
            spec = pipe.get("spec", {})
            print(
                f"  - depth {pipe.get('depth')} × {pipe.get('num_aspects')} "
                f"({pipe.get('total_aspects_in_pipeline')} aspects) "
                f"{spec.get('input')}→{spec.get('output')} "
                f"[stage={spec.get('metadata', {}).get('stage')}]"
            )
    print("=" * 80)


def plot_dynamics(
    steps: List[int],
    free_energy: List[float],
    struct_stats: Dict,
    object_norms: Dict[str, List[Tuple[int, float]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(steps, free_energy, marker="o", label="Free Energy")
    axes[0].set_ylabel("Free Energy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(steps, struct_stats["num_objects"], marker="s", label="#Objects")
    axes[1].plot(steps, struct_stats["num_aspects"], marker="^", label="#Aspects")
    axes[1].plot(
        steps,
        struct_stats["num_pipeline_aspects"],
        marker="d",
        label="#Pipeline Aspects",
    )
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].set_ylabel("State Norm")
    for name, series in object_norms.items():
        if not series:
            continue
        obj_steps, values = zip(*series)
        axes[2].plot(obj_steps, values, marker=".", label=name)
    axes[2].set_xlabel("Step")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.suptitle("Office AI Evolution", fontsize=14, y=1.02)
    fig.subplots_adjust(top=0.95)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"[+] Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize statistics from Office AI experiment results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/office_ai_results_steps500.json"),
        help="Path to the JSON results file.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("data/office_ai_analysis.png"),
        help="Path to save the generated plot image.",
    )
    parser.add_argument(
        "--track",
        nargs="*",
        default=("internal", "document", "table", "calendar", "action"),
        help="Object names to track for state-norm visualization.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib plot generation.",
    )
    args = parser.parse_args()

    data = load_results(args.input)
    steps, free_energy, structures = build_time_series(data)
    struct_stats = compute_structure_stats(structures)
    norms = collect_object_norms(steps, structures, tuple(args.track))

    print_summary(steps, free_energy, struct_stats, data.get("final_structure"))

    if not args.no_plot:
        plot_dynamics(steps, free_energy, struct_stats, norms, args.plot)


if __name__ == "__main__":
    main()

