#!/usr/bin/env python3
"""
Generate synthetic office scenarios for regression-style testing.

Each scenario encodes a common office event (e.g., meeting reschedule, urgent
email) by biasing the internal state of the OfficeAIWorldModel in a specific
direction. The script saves JSON test cases that include:
  - scenario description
  - sampled observations (document/table/calendar vectors)
  - state norm summary
  - suggested actions / expectations

This is intended to provide quick fixtures similar to the MNIST recognition
demos, but for semantic office workflows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aonn.models.office_ai_world_model import OfficeAIWorldModel

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


@dataclass
class OfficeEvent:
    name: str
    description: str
    expectations: List[str]
    keywords: List[str]
    seed: int
    document_scale: float
    task_scale: float
    schedule_scale: float
    context_scale: float


EVENT_LIBRARY: List[OfficeEvent] = [
    OfficeEvent(
        name="urgent_board_meeting",
        description="The board demands a Q3 financial summary within two hours, requiring simultaneous document updates and meeting coordination.",
        expectations=[
            "Rapidly curate an existing report template into a concise financial brief",
            "Schedule a 30‑minute sync with the CFO and sales lead",
            "LLMAspect should highlight finance-oriented semantic cues (revenue, margin, forecast)",
        ],
        keywords=["board", "Q3", "finance", "summary", "meeting"],
        seed=11,
        document_scale=0.45,
        task_scale=0.65,
        schedule_scale=0.80,
        context_scale=0.40,
    ),
    OfficeEvent(
        name="compliance_audit_ping",
        description="Compliance sends a fresh audit checklist that must be annotated item-by-item while updating the approval calendar.",
        expectations=[
            "Document signals emphasize checklists, annotations and reviewer notes",
            "Task state shows many pending items transitioning to review",
            "Schedule state adds multiple audit-review blocks",
        ],
        keywords=["audit", "compliance", "checklist", "review"],
        seed=22,
        document_scale=0.35,
        task_scale=0.75,
        schedule_scale=0.55,
        context_scale=0.30,
    ),
    OfficeEvent(
        name="product_launch_briefing",
        description="During launch week the team prepares a cross-functional briefing combining marketing copy, sales sheets, and meeting logistics.",
        expectations=[
            "Document and table observations spike together with marketing / sales semantics",
            "LLM output should mention launch timeline, go-to-market, enablement materials",
            "Recommended actions focus on meeting orchestration and collateral distribution",
        ],
        keywords=["launch", "marketing", "sales", "briefing", "timeline"],
        seed=33,
        document_scale=0.55,
        task_scale=0.40,
        schedule_scale=0.60,
        context_scale=0.50,
    ),
    OfficeEvent(
        name="finance_budget_adjustment",
        description="A quarterly budget adjustment request from the CFO requires spreadsheet edits and updated approval meetings.",
        expectations=[
            "Table modality dominates with cost / budget / variance semantics",
            "Actions lean toward editing spreadsheets and sending approval notices",
            "Schedule state reflects several finance review calls",
        ],
        keywords=["budget", "finance", "cost", "variance", "approval"],
        seed=44,
        document_scale=0.30,
        task_scale=0.55,
        schedule_scale=0.70,
        context_scale=0.35,
    ),
    OfficeEvent(
        name="security_incident_notification",
        description="Security operations issues a potential intrusion alert that must be triaged immediately and broadcast to stakeholders.",
        expectations=[
            "Task and context states surge to reflect incident response urgency",
            "LLM semantics should emphasize alert, containment, investigation wording",
            "Recommended actions include broadcasting notices and creating high-priority tasks",
        ],
        keywords=["security", "incident", "alert", "response"],
        seed=55,
        document_scale=0.25,
        task_scale=0.85,
        schedule_scale=0.40,
        context_scale=0.65,
    ),
]


def generate_state(world: OfficeAIWorldModel, event: OfficeEvent, context_override: Optional[torch.Tensor] = None) -> None:
    torch.manual_seed(event.seed)

    world.document_state = torch.randn(world.document_dim, device=world.device) * event.document_scale
    world.task_state = torch.randn(world.task_dim, device=world.device) * event.task_scale
    world.schedule_state = torch.randn(world.schedule_dim, device=world.device) * event.schedule_scale
    if context_override is not None:
        world.context_state = context_override
    else:
        world.context_state = torch.randn(world.context_dim, device=world.device) * event.context_scale
    world.set_context_metadata(
        {
            "text": event.description,
            "keywords": event.keywords,
            "expectations": event.expectations,
        }
    )


def summarize_tensor(name: str, tensor: torch.Tensor) -> Dict:
    return {
        "dim": tensor.numel(),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "norm": tensor.norm().item(),
    }


def build_semantic_context(
    event: OfficeEvent,
    embedder: Optional["SentenceTransformer"],
    semantic_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if embedder is None:
        return None
    text = event.description
    if event.expectations:
        text += " Expectations: " + " ".join(event.expectations)
    embedding = embedder.encode(text, normalize_embeddings=True)
    tensor = torch.tensor(embedding, dtype=torch.float32, device=device)
    if tensor.numel() >= semantic_dim:
        tensor = tensor[:semantic_dim]
    else:
        pad = torch.zeros(semantic_dim - tensor.numel(), device=device)
        tensor = torch.cat([tensor, pad])
    return tensor


def simulate_event(
    world: OfficeAIWorldModel,
    event: OfficeEvent,
    embedder: Optional["SentenceTransformer"],
    semantic_dim: int,
) -> Dict:
    context_vec = build_semantic_context(event, embedder, semantic_dim, world.device)
    generate_state(world, event, context_override=context_vec)
    observation = world.get_observation()
    dummy_action = torch.zeros(world.action_dim, device=world.device)
    reward = world.get_reward(dummy_action).item()

    summary = {
        "event_name": event.name,
        "description": event.description,
        "expectations": event.expectations,
        "keywords": event.keywords,
        "context_metadata": world.get_context_metadata(),
        "state_summary": {
            "document": summarize_tensor("document", world.document_state),
            "task": summarize_tensor("task", world.task_state),
            "schedule": summarize_tensor("schedule", world.schedule_state),
            "context": summarize_tensor("context", world.context_state),
        },
        "observation_summary": {
            sense: summarize_tensor(sense, obs)
            for sense, obs in observation.items()
        },
        "reward_if_idle": reward,
    }

    # Store raw tensors for downstream tests (converted to lists for JSON)
    summary["state_vectors"] = {
        "document": world.document_state.detach().cpu().tolist(),
        "task": world.task_state.detach().cpu().tolist(),
        "schedule": world.schedule_state.detach().cpu().tolist(),
        "context": world.context_state.detach().cpu().tolist(),
    }
    summary["observation_vectors"] = {
        sense: obs.detach().cpu().tolist() for sense, obs in observation.items()
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Office AI event regression fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/office_event_tests"),
        help="Directory to store generated JSON files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for Office AI world model",
    )
    parser.add_argument(
        "--embed-text",
        action="store_true",
        help="Use a sentence-transformers model to encode description+expectations into semantic_context.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (used when --embed-text is set).",
    )
    parser.add_argument(
        "--semantic-dim",
        type=int,
        default=128,
        help="Semantic context dimension to target when embedding text.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    world = OfficeAIWorldModel(device=device, context_dim=args.semantic_dim)
    fixtures: Dict[str, Dict] = {}

    embedder = None
    if args.embed_text:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed. Run `pip install sentence-transformers`.")
        embedder = SentenceTransformer(args.embed_model, device=str(device))

    for event in EVENT_LIBRARY:
        summary = simulate_event(world, event, embedder, args.semantic_dim)
        fixtures[event.name] = summary
        event_path = output_dir / f"{event.name}.json"
        with event_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved event fixture: {event_path}")

    bundle_path = output_dir / "office_event_fixtures.json"
    with bundle_path.open("w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2, ensure_ascii=False)
    print(f"✓ Bundle summary saved: {bundle_path}")


if __name__ == "__main__":
    main()
