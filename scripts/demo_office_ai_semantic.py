#!/usr/bin/env python3
"""
Generate a multi-panel visualization for an Office AI experiment run, similar to
the MNIST recognition demo figures. The script summarizes free energy, structure
growth, key object state norms, pipeline composition, sampled LLM semantic
descriptions, and (optionally) overlays synthetic office event scenarios.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import textwrap

import numpy as np

MPL_CACHE_DIR = Path(".cache/matplotlib")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402


def load_results(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_event_fixture(path: Optional[Path], event_name: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Event fixture file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "event_name" in data:
        event = data
    elif isinstance(data, dict):
        if not event_name:
            raise ValueError("Fixture bundle requires --event-name to select an entry.")
        if event_name not in data:
            raise KeyError(f"Event '{event_name}' not found in fixture bundle.")
        event = data[event_name]
    else:
        raise ValueError("Unsupported fixture format for event fixture.")
    event.setdefault("expectations", [])
    event.setdefault("keywords", [])
    return event


def extract_series(
    snapshots: Sequence[Dict],
    tracked_objects: Sequence[str],
) -> Tuple[List[int], List[float], Dict[str, List[float]], Dict[str, List[int]]]:
    steps = [snap["step"] for snap in snapshots]
    free_energy = [float(snap["free_energy"]) for snap in snapshots]

    object_norms: Dict[str, List[float]] = {name: [] for name in tracked_objects}
    structure_counts = {
        "num_objects": [],
        "num_aspects": [],
        "num_pipeline_aspects": [],
    }
    for snap in snapshots:
        structure = snap.get("structure", {})
        for key in structure_counts:
            structure_counts[key].append(structure.get(key, 0))
        objects = structure.get("objects", {})
        for name in tracked_objects:
            obj_state = objects.get(name)
            object_norms[name].append(
                float(obj_state.get("state_norm"))
                if obj_state and obj_state.get("state_norm") is not None
                else np.nan
            )

    return steps, free_energy, object_norms, structure_counts


def gather_llm_descriptions(snapshots: Sequence[Dict], max_entries: int) -> List[Tuple[int, str]]:
    entries = []
    for snap in snapshots:
        text = snap.get("llm_description")
        if text:
            entries.append((snap["step"], text.strip()))
    entries.sort(key=lambda x: x[0])
    if max_entries and len(entries) > max_entries:
        # pick evenly spaced entries
        idxs = np.linspace(0, len(entries) - 1, max_entries, dtype=int)
        entries = [entries[i] for i in idxs]
    return entries


def plot_office_ai_demo(
    *,
    steps: List[int],
    free_energy: List[float],
    object_norms: Dict[str, List[float]],
    structure_counts: Dict[str, List[int]],
    final_structure: Dict,
    llm_entries: List[Tuple[int, str]],
    output: Path,
    title: str,
    event_info: Optional[Dict],
) -> None:
    fig = plt.figure(figsize=(16, 13))
    gs = GridSpec(4, 2, height_ratios=[1.1, 1.0, 0.9, 0.9], hspace=0.35, wspace=0.28)

    # Panel 1: Free energy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, free_energy, color="#2E86AB", linewidth=2.5, marker="o")
    ax1.set_title("Free Energy Evolution", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Free Energy")
    ax1.grid(True, alpha=0.3)
    ax1.text(
        0.02,
        0.95,
        f"Initial: {free_energy[0]:.2f}\nFinal: {free_energy[-1]:.2f}",
        transform=ax1.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    # Panel 2: Structure growth
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, structure_counts["num_objects"], label="#Objects", marker="s")
    ax2.plot(steps, structure_counts["num_aspects"], label="#Aspects", marker="^")
    ax2.plot(
        steps,
        structure_counts["num_pipeline_aspects"],
        label="#Pipeline Aspects",
        marker="d",
    )
    ax2.set_title("Structure Growth", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Object state norms
    ax3 = fig.add_subplot(gs[1, 0])
    for name, series in object_norms.items():
        ax3.plot(steps, series, label=name)
    ax3.set_title("Key Object State Norms", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("State Norm")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", ncol=2)

    # Panel 4: Pipeline summary
    ax4 = fig.add_subplot(gs[1, 1])
    pipelines = final_structure.get("pipelines", [])
    if pipelines:
        labels = [p.get("spec", {}).get("metadata", {}).get("stage", f"depth {p.get('depth')}") for p in pipelines]
        counts = [p.get("total_aspects_in_pipeline", 0) for p in pipelines]
        positions = np.arange(len(labels))
        bars = ax4.bar(positions, counts, color="#F6A01A", alpha=0.85)
        ax4.bar_label(bars, fmt="%.0f", rotation=0, padding=3)
        ax4.set_xticks(positions)
        ax4.set_xticklabels(labels, rotation=20, ha="right")
    else:
        ax4.text(0.5, 0.5, "No pipelines", ha="center", va="center", fontsize=12, alpha=0.7)
    ax4.set_title("Pipeline Aspect Budget (Final)", fontsize=14, fontweight="bold")
    ax4.set_ylabel("#Aspects")
    ax4.grid(True, axis="y", alpha=0.3)

    # Panel 5: LLM descriptions
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis("off")
    if llm_entries:
        lines = []
        for step, text in llm_entries:
            prefix = f"Step {step:>3}: "
            wrapped = textwrap.fill(
                text,
                width=70,
                initial_indent=prefix,
                subsequent_indent=" " * len(prefix),
            )
            lines.append(wrapped)
        llm_text = "\n".join(lines)
    else:
        llm_text = "No llm_description entries found in snapshots."
    ax5.text(
        0.0,
        1.0,
        llm_text,
        ha="left",
        va="top",
        fontsize=11,
        wrap=True,
        family="monospace",
    )
    ax5.set_title("Sampled LLM Semantic Descriptions", fontsize=14, fontweight="bold", loc="left")

    # Panel 6: Final summary
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    bullets = [
        f"Steps: {steps[0]} → {steps[-1]}",
        f"Final Free Energy: {free_energy[-1]:.3f}",
        f"Objects / Aspects: {final_structure.get('num_objects')} / {final_structure.get('num_aspects')}",
        f"Pipelines: {final_structure.get('num_pipelines')} (LLM: {'ON' if final_structure.get('has_llm_aspect') else 'OFF'})",
    ]
    if llm_entries:
        bullets.append(f"Latest LLM: “{llm_entries[-1][1]}”")
    bullet_text = "\n".join(f"• {line}" for line in bullets)
    ax6.text(0.0, 1.0, bullet_text, ha="left", va="top", fontsize=13, wrap=True)
    ax6.set_title("Run Summary", fontsize=14, fontweight="bold", loc="left")

    # Panel 7: Event overlay (full width)
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis("off")
    if event_info:
        event_lines = [
            f"Scenario: {event_info.get('event_name', 'N/A')}",
            event_info.get("description", ""),
        ]
        expectations = event_info.get("expectations") or []
        if expectations:
            event_lines.append("Expectations:")
            for exp in expectations:
                event_lines.append(f"  - {exp}")
        keywords = event_info.get("keywords") or []
        if keywords:
            event_lines.append(f"Keywords: {', '.join(keywords)}")
        ax7.text(
            0.0,
            1.0,
            "\n".join(event_lines),
            ha="left",
            va="top",
            fontsize=12,
            wrap=True,
        )
    else:
        ax7.text(0.5, 0.5, "No event overlay provided", ha="center", va="center", alpha=0.6)
    ax7.set_title("Office Event Overlay", fontsize=14, fontweight="bold", loc="left")

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Office AI demo figure saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Office AI semantic demo visualization")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/office_ai_results_ollama_steps50.json"),
        help="Path to Office AI experiment results JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/office_ai_demos/office_ai_demo.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--track-objects",
        nargs="*",
        default=("internal", "document", "table", "calendar", "action", "semantic_context"),
        help="Object names to plot state norms for",
    )
    parser.add_argument(
        "--max-llm-entries",
        type=int,
        default=8,
        help="Maximum number of LLM description samples to display",
    )
    parser.add_argument(
        "--event-fixture",
        type=Path,
        default=None,
        help="Path to a single event JSON or bundle generated by generate_office_event_tests.py",
    )
    parser.add_argument(
        "--event-name",
        type=str,
        default=None,
        help="Event name to select when using a fixture bundle.",
    )

    args = parser.parse_args()
    data = load_results(args.input)
    snapshots = data.get("snapshots", [])
    if not snapshots:
        raise ValueError("Input JSON has no snapshots to visualize.")

    steps, free_energy, object_norms, structure_counts = extract_series(
        snapshots,
        args.track_objects,
    )
    llm_entries = gather_llm_descriptions(snapshots, args.max_llm_entries)
    event_info = load_event_fixture(args.event_fixture, args.event_name)

    title = f"Office AI Semantic Evolution Demo • steps {steps[0]}–{steps[-1]}"
    plot_office_ai_demo(
        steps=steps,
        free_energy=free_energy,
        object_norms=object_norms,
        structure_counts=structure_counts,
        final_structure=data.get("final_structure", {}),
        llm_entries=llm_entries,
        output=args.output,
        title=title,
        event_info=event_info,
    )


if __name__ == "__main__":
    main()

