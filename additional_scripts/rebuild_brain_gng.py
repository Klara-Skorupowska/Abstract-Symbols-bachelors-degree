"""
rebuild_brain_gng.py

Re-trains/re-calculates all GNG predicates and the transition graph from scratch 
using historical transition data (.npz) across session ancestor chains, 
applying optimized hyperparameters.
"""

import os
import sys
import json
import argparse
import numpy as np

# Adjust module path to import local modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agent.brain_network import BrainNetwork
from agent.predicate import Predicate


# -------------------------------------------------------------------------
# Session History Tracing & Data Aggregation
# -------------------------------------------------------------------------
def get_session_dir(session_arg: str) -> str:
    logs_root = "logs"
    if session_arg.lower() in ("latest", "last", "continue"):
        session_dirs = [
            os.path.join(logs_root, d) for d in os.listdir(logs_root)
            if os.path.isdir(os.path.join(logs_root, d)) and not d.startswith("opt_")
        ]
        if not session_dirs:
            raise FileNotFoundError(f"No session directories found in '{logs_root}'.")
        return max(session_dirs, key=os.path.basename)

    if session_arg.startswith(logs_root):
        target_dir = session_arg
    else:
        target_dir = os.path.join(logs_root, session_arg)

    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Session directory '{target_dir}' does not exist.")
    return target_dir


def find_predecessor_session(session_dir: str):
    stats_file = os.path.join(session_dir, "execution_statistics.txt")
    if not os.path.exists(stats_file):
        return None

    prefix = "CONTINUATION FROM SESSION:"
    with open(stats_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(prefix):
                raw_path = line[len(prefix):].strip().rstrip(";").strip()
                normalized_path = os.path.normpath(raw_path)
                if os.path.isdir(normalized_path):
                    return normalized_path
                alt_path = os.path.join("logs", os.path.basename(normalized_path))
                if os.path.isdir(alt_path):
                    return alt_path
    return None


def collect_transitions(session_arg: str):
    """Traces session history backward and aggregates all transition arrays in chronological order."""
    current_dir = get_session_dir(session_arg)
    root_session_dir = current_dir

    session_chain = []
    visited = set()
    while current_dir and current_dir not in visited:
        visited.add(current_dir)
        session_chain.append(current_dir)
        current_dir = find_predecessor_session(current_dir)

    session_chain.reverse()
    print(f"[Dataset] Session lineage chain ({len(session_chain)} sessions):")
    for s in session_chain:
        print(f"  -> {s}")

    all_prev_states = []
    all_skill_names = []
    all_post_states = []
    all_succeeds = []

    for s_dir in session_chain:
        # Check both the concatenated file and the raw transitions file
        npz_candidate = os.path.join(s_dir, "all_offline_transitions.npz")
        if not os.path.exists(npz_candidate):
            npz_candidate = os.path.join(s_dir, "offline_transitions.npz")

        if os.path.exists(npz_candidate):
            try:
                data = np.load(npz_candidate, allow_pickle=True)
                all_prev_states.append(data['prev_states'])
                all_skill_names.append(data['skill_names'])
                all_post_states.append(data['post_states'])
                all_succeeds.append(data['succeeds'])
                print(f"[Dataset] Loaded {len(data['prev_states'])} transitions from {npz_candidate}")
            except Exception as e:
                print(f"[Dataset] Warning: Failed to load {npz_candidate}: {e}")

    if not all_prev_states:
        raise FileNotFoundError(f"No offline transitions found in lineage of '{root_session_dir}'.")

    prev_states = np.concatenate(all_prev_states, axis=0)
    skill_names = np.concatenate(all_skill_names, axis=0)
    post_states = np.concatenate(all_post_states, axis=0)
    succeeds = np.concatenate(all_succeeds, axis=0)

    print(f"[Dataset] Total combined transitions: {len(prev_states)}")
    return prev_states, skill_names, post_states, succeeds, root_session_dir


# -------------------------------------------------------------------------
# Hyperparameter Loading & Setup
# -------------------------------------------------------------------------
def load_hyperparameters(config_source: str) -> dict:
    """Loads parameter dictionary from a JSON tuning result file or raw JSON string."""
    default_params = {
        "max_points": 50,
        "base_radius": 0.1,
        "max_radius": 1.0,
        "learning_rate_b": 0.2,
        "learning_rate_n": 0.02,
        "max_edge_age": 30,
        "lambda_step": 10,
        "alpha": 0.5,
        "d": 0.99,
        "min_overlap_ratio": 0.55
    }

    if not config_source:
        print("[Config] Using default parameters.")
        return default_params

    # 1. Check if argument points to a JSON file (e.g., hyperparam_tuning_*.json)
    if os.path.isfile(config_source):
        with open(config_source, "r") as f:
            data = json.load(f)
        if "best_params" in data and data["best_params"]:
            print(f"[Config] Loaded 'best_params' from {config_source}")
            params = data["best_params"]
        else:
            params = data
    else:
        # 2. Treat as inline JSON string
        try:
            params = json.loads(config_source)
        except Exception:
            raise ValueError(f"Could not parse parameter source: {config_source}")

    # Merge with default fallback
    merged = default_params.copy()
    merged.update(params)
    return merged


def apply_params_to_brain(brain: BrainNetwork, params: dict):
    brain.max_points = int(params["max_points"])
    brain.base_radius = float(params["base_radius"])
    brain.max_radius = float(params["max_radius"])
    brain.learning_rate_b = float(params["learning_rate_b"])
    brain.learning_rate_n = float(params["learning_rate_n"])
    brain.max_edge_age = int(params["max_edge_age"])
    brain.lambda_step = int(params["lambda_step"])
    brain.alpha = float(params["alpha"])
    brain.d = float(params["d"])

    min_overlap = float(params.get("min_overlap_ratio", 0.55))
    orig_update_transitions = brain.update_transitions

    def wrapped_update_transitions(skill_name: str, min_overlap_ratio: float = min_overlap):
        return orig_update_transitions(skill_name, min_overlap_ratio=min_overlap)

    brain.update_transitions = wrapped_update_transitions


# -------------------------------------------------------------------------
# Rebuild Pipeline
# -------------------------------------------------------------------------
def rebuild_gng(session_arg: str, config_source: str, out_dir: str = None, batch_size: int = 10):
    params = load_hyperparameters(config_source)
    print("\n[Hyperparameters to apply]:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    prev_states, skill_names, post_states, succeeds, root_session_dir = collect_transitions(session_arg)

    # Determine destination folder
    if out_dir is None:
        target_log_dir = root_session_dir
    else:
        target_log_dir = out_dir
        os.makedirs(target_log_dir, exist_ok=True)

    print(f"\n[Rebuild] Initializing fresh BrainNetwork targeting: {target_log_dir}")
    brain = BrainNetwork(log_dir=target_log_dir)
    apply_params_to_brain(brain, params)

    # Feed transitions in batches
    total_transitions = len(prev_states)
    batch = []
    print(f"[Rebuild] Replaying {total_transitions} transitions (batch size: {batch_size})...")

    for i in range(total_transitions):
        transition_tuple = (
            prev_states[i],
            str(skill_names[i]),
            post_states[i],
            bool(succeeds[i])
        )
        batch.append(transition_tuple)

        if len(batch) >= batch_size:
            brain.update(batch)
            batch = []

    if batch:
        brain.update(batch)

    # Finalize abstractions and save predicates & graph
    print("[Rebuild] Abstracting symbols and exporting GNG JSON files...")
    brain.abstract_symbols()
    brain._logger()

    # Save used parameters metadata alongside outputs
    params_path = os.path.join(brain.gas_dir, "applied_hyperparameters.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    print("=" * 80)
    print("[Rebuild Completed Successfully]")
    print(f"  - Nodes trained: {brain.get_nodes()}")
    print(f"  - GNG files saved to: {brain.gas_dir}")
    print(f"  - Graph file saved to: {os.path.join(brain.graphs_dir, 'trans_graph.json')}")
    print(f"  - Parameters metadata: {params_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalculate GNG predicates using historical transitions and tuned parameters.")
    parser.add_argument("--session", type=str, default="latest",
                        help="Session directory in logs/ (e.g., 'latest' or '2026-09-18_12-52').")
    parser.add_argument("--config", type=str, default="",
                        help="Path to tuning JSON file (e.g. logs/hyperparam_tuning_*.json) or raw JSON string.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Optional output directory. Defaults to overwriting the session folder.")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Buffer batch size for brain.update().")

    args = parser.parse_args()
    rebuild_gng(
        session_arg="2026-09-21_01-07",
        config_source=".\logs\optim\hyperparam_tuning_20261006_152106.json",
        out_dir= ".\logs\offline",
        batch_size=1000
    )