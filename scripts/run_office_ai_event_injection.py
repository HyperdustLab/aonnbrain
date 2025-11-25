#!/usr/bin/env python3
"""
Office AI Experiment with Runtime Event Injection
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    pass

import json
import math
import time
from typing import Dict, Optional, List
from dataclasses import dataclass

import torch
from tqdm import tqdm

from aonn.models.office_ai_world_model import OfficeAIWorldModel, OfficeAIWorldInterface
from aonn.models.aonn_brain_v3 import AONNBrainV3
from aonn.core.active_inference_loop import ActiveInferenceLoop
from aonn.aspects.mock_llm_client import MockLLMClient

try:
    from aonn.aspects.openai_llm_client import OpenAILLMClient
except Exception:
    OpenAILLMClient = None

try:
    from aonn.aspects.ollama_llm_client import OllamaLLMClient
except Exception:
    OllamaLLMClient = None

# --- Event Definitions (Copied from generate_office_event_tests.py) ---
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
    # ... other events can be added here if needed
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

def run_experiment(
    num_steps: int,
    config: Dict,
    device: torch.device,
    event_name: Optional[str] = None,
    injection_step: int = 10,
    *,
    verbose: bool = False,
    use_openai_llm: bool = False,
    use_ollama_llm: bool = False,
    openai_api_key: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
    save_interval: int = 100,
    checkpoint_dir: str = "data/checkpoints",
):
    """Run Office AI experiment with event injection"""
    
    # Create World Model
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
        state_noise_std=config.get("world_model", {}).get("state_noise_std", 0.01),
        observation_noise_std=config.get("world_model", {}).get("observation_noise_std", 0.01),
    )
    world_interface = OfficeAIWorldInterface(world_model)
    
    # Initialize LLM Client
    llm_client = None
    sem_dim = config.get("sem_dim", 128)
    
    if use_ollama_llm and OllamaLLMClient is not None:
        llm_cfg = config.get("llm", {})
        print(f"Initializing Ollama client with model: {ollama_model or llm_cfg.get('model', 'llama3')}")
        llm_client = OllamaLLMClient(
            input_dim=sem_dim,
            output_dim=sem_dim,
            base_url=ollama_base_url or llm_cfg.get("base_url", "http://localhost:11434"),
            model=ollama_model or llm_cfg.get("model", "llama3"),
            embedding_model=llm_cfg.get("embedding_model"),
            summary_size=llm_cfg.get("summary_size", 8),
            max_tokens=llm_cfg.get("max_tokens", 120),
            temperature=llm_cfg.get("temperature", 0.7),
            timeout=llm_cfg.get("timeout", 120.0),
            verbose=verbose,
            device=device,
            system_prompt="You convert latent state summaries into brief semantic descriptions. Keep the response concise (<= 30 tokens). Always answer in English.",
        )
    elif use_openai_llm and OpenAILLMClient is not None:
        llm_cfg = config.get("llm", {})
        llm_client = OpenAILLMClient(
            input_dim=sem_dim,
            output_dim=sem_dim,
            api_key=openai_api_key or llm_cfg.get("api_key"),
            model=llm_cfg.get("model", "gpt-4o-mini"),
            embedding_model=llm_cfg.get("embedding_model", "text-embedding-3-small"),
            summary_size=llm_cfg.get("summary_size", 8),
            max_tokens=llm_cfg.get("max_tokens", 120),
            temperature=llm_cfg.get("temperature", 0.7),
            verbose=verbose,
            device=device,
        )
    else:
        llm_client = MockLLMClient(
            input_dim=sem_dim,
            output_dim=sem_dim,
            hidden_dims=config.get("llm", {}).get("hidden_dims", [256, 512, 256]),
            device=device,
        )
    
    # Create AONN Brain
    brain = AONNBrainV3(
        config=config,
        llm_client=llm_client,
        device=device,
        enable_evolution=config.get("enable_evolution", True),
    )
    
    # Initialize Environment
    obs = world_interface.reset()
    for sense, value in obs.items():
        if sense in brain.objects:
            brain.objects[sense].set_state(value)
    
    prev_obs = None
    prev_action = None
    snapshots = []
    
    action = torch.randn(config["act_dim"], device=device) * 0.1
    
    progress = tqdm(range(num_steps), desc=f"OfficeAI Injection {num_steps}")
    
    try:
        for step in progress:
            step_start_time = time.perf_counter()
            
            # --- Event Injection Logic ---
            if event_name and step == injection_step:
                if verbose:
                    print(f"\n[Step {step}] Injecting event: {event_name}")
                
                # Find the event
                event = next((e for e in EVENT_LIBRARY if e.name == event_name), None)
                if event:
                    # Inject state into world model
                    generate_state(world_model, event)
                    
                    # Force observation update immediately
                    obs = world_interface.get_observation()
                    
                    # Log injection
                    if verbose:
                        print(f"  Event '{event.name}' injected. Description: {event.description}")
                else:
                    print(f"  Warning: Event '{event_name}' not found in library.")
            
            elif step > 0:
                obs, reward = world_interface.step(action)
            
            # Set observations to brain
            for sense, value in obs.items():
                if sense in brain.objects:
                    brain.objects[sense].set_state(value)
            
            # Network Evolution
            full_state = world_model.get_true_state()
            if full_state.shape[-1] >= config["state_dim"]:
                target_state = full_state[:config["state_dim"]]
            else:
                padding = torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)
                target_state = torch.cat([full_state, padding], dim=-1)
            
            # Sync semantic context
            if "semantic_context" in brain.objects and hasattr(world_model, "context_state"):
                sem_dim = config.get("sem_dim", 128)
                world_context = world_model.context_state[:sem_dim].detach().clone()
                metadata = None
                if hasattr(world_model, "get_context_metadata"):
                    metadata = world_model.get_context_metadata()
                brain.update_semantic_context(world_semantic_state=world_context, metadata=metadata)
            
            try:
                brain.evolve_network(obs, target=target_state)
            except Exception:
                pass
            
            # Active Inference
            if len(brain.aspects) > 0:
                try:
                    # Detach states
                    for obj_name, obj in brain.objects.items():
                        state = obj.state
                        if state.requires_grad and state.is_leaf and state.grad is not None:
                            obj.set_state(state.detach())
                        elif not state.is_leaf:
                            obj.set_state(state.detach())
                    
                    loop = ActiveInferenceLoop(
                        brain.objects,
                        brain.aspects,
                        infer_lr=config.get("infer_lr", 0.02),
                        max_grad_norm=config.get("max_grad_norm", None),
                        device=device,
                    )
                    num_iters = config.get("num_infer_iters", 2)
                    loop.infer_states(target_objects=("internal",), num_iters=num_iters, sanitize_callback=brain.sanitize_states)
                    brain.sanitize_states()
                    
                except Exception:
                    pass
            
            # Action Generation
            if "action" in brain.objects and len(brain.aspect_pipelines) > 0:
                action = brain.objects["internal"].state
                for pipeline in brain.aspect_pipelines:
                    action = pipeline(action)
                brain.objects["action"].set_state(action)
            else:
                action = torch.randn(config["act_dim"], device=device) * 0.1
                if "action" in brain.objects:
                    brain.objects["action"].set_state(action)
            
            # World Model Learning
            if prev_obs is not None and prev_action is not None:
                full_state = world_model.get_true_state()
                if full_state.shape[-1] >= config["state_dim"]:
                    target_state = full_state[:config["state_dim"]]
                else:
                    padding = torch.zeros(config["state_dim"] - full_state.shape[-1], device=device)
                    target_state = torch.cat([full_state, padding], dim=-1)
                try:
                    brain.learn_world_model(
                        observation=prev_obs,
                        action=prev_action,
                        next_observation=obs,
                        target_state=target_state,
                        learning_rate=config.get("learning_rate", 0.0015),
                    )
                    brain.sanitize_states()
                except Exception:
                    pass
            
            prev_obs = {sense: value.clone() for sense, value in obs.items()}
            prev_action = action.clone()
            
            if step % 10 == 0:
                brain.sanitize_states()
            
            # Record Snapshot
            F = brain.compute_free_energy().item()
            if not math.isfinite(F):
                brain.sanitize_states()
                F = 1e-6
            self_model_snapshot = brain.observe_self_model()
            structure = self_model_snapshot.get("structure", {})
            
            if step % save_interval == 0 or step == num_steps - 1 or step == injection_step or step == injection_step + 1:
                snapshot = {
                    "step": step,
                    "free_energy": F,
                    "structure": structure,
                    "llm_description": llm_client._last_generated_text if llm_client and hasattr(llm_client, '_last_generated_text') else None,
                }
                snapshots.append(snapshot)
            
            progress.set_postfix(
                F=f"{F:.2f}",
                Obj=structure.get('num_objects', 0),
                Asp=structure.get('num_aspects', 0),
                LLM='LLM✓' if structure.get('has_llm_aspect', False) else 'LLM✗'
            )
            
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Result
    final_snapshot = brain.observe_self_model()
    final_F = brain.compute_free_energy().item()
    
    result = {
        "num_steps": num_steps,
        "final_free_energy": final_F,
        "final_structure": final_snapshot.get("structure", {}),
        "snapshots": snapshots,
        "evolution_summary": brain.evolution.get_evolution_summary() if brain.evolution else {},
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Office AI Event Injection Experiment")
    parser.add_argument("--steps", type=int, default=60, help="Number of steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--output", type=Path, default=Path("data/office_ai_injection_results.json"), help="Output file")
    parser.add_argument("--event-name", type=str, default=None, help="Name of event to inject")
    parser.add_argument("--injection-step", type=int, default=10, help="Step to inject event")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--use-openai-llm", action="store_true", help="Use OpenAI LLM")
    parser.add_argument("--use-ollama-llm", action="store_true", help="Use Ollama LLM")
    parser.add_argument("--openai-api-key", type=str, default=None, help="OpenAI API Key")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434", help="Ollama Base URL")
    parser.add_argument("--ollama-model", type=str, default="llama3", help="Ollama Model")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    config = {
        "state_dim": 576,
        "act_dim": 128,
        "obs_dim": 448,
        "auto_classification_aspect": False,
        "sem_dim": 128,
        "sense_dims": {
            "document": 256,
            "table": 128,
            "calendar": 64,
        },
        "enable_evolution": False,
        "use_world_model_pipelines": True,
        "world_model_pipeline_map": {
            "document": "document_dim",
            "table": "task_dim",
            "calendar": "schedule_dim",
        },
        "sensory_pipeline": {
            "depth": 3,
            "width": 32,
            "use_gate": False,
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
                "min_per_sense": 6,
                "error_threshold": 0.07,
                "error_multiplier": 0.7,
            },
        },
        "pipeline_growth": {
            "enable": False,
            "initial_depth": 3,
            "initial_width": 32,
            "depth_increment": 1,
            "width_increment": 12,
            "max_stages": 6,
            "min_interval": 80,
            "free_energy_trigger": None,
            "max_depth": 10,
        },
        "state_clip_value": 5.0,
        "infer_lr": 0.02,
        "learning_rate": 0.0015,
        "num_infer_iters": 5,
        "max_grad_norm": 100.0,
    }
    
    print("=" * 80)
    print(f"Office AI Event Injection: {args.event_name} at step {args.injection_step}")
    print(f"LLM: {'Ollama (' + args.ollama_model + ')' if args.use_ollama_llm else 'Mock'}")
    print("=" * 80)
    
    result = run_experiment(
        num_steps=args.steps,
        config=config,
        device=device,
        event_name=args.event_name,
        injection_step=args.injection_step,
        verbose=args.verbose,
        use_openai_llm=args.use_openai_llm,
        use_ollama_llm=args.use_ollama_llm,
        openai_api_key=args.openai_api_key,
        ollama_base_url=args.ollama_base_url,
        ollama_model=args.ollama_model,
    )
    
    output_path = Path(__file__).parent.parent / args.output if not args.output.is_absolute() else args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
