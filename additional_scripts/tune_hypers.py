"""
tune_hyperparameters.py

Hyperparameter optimization for BrainNetwork & Predicate components:
  - max_points
  - base_radius
  - max_radius
  - learning_rate_b
  - learning_rate_n
  - max_edge_age
  - lambda_step
  - alpha
  - d
  - min_overlap_ratio
"""

import os
import sys
import json
import time
import shutil
import random
import argparse
import numpy as np
from datetime import datetime

# Adjust module path to import agent modules if placed in the parent directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from communicator import Communicator
    from agent.the_agent import TheAgent
    from agent.brain_network import BrainNetwork
    from agent.predicate import Predicate
except ImportError as e:
    print(f"[Warning] Failed to import one or more modules: {e}")

# Try importing Optuna, fallback gracefully if not present
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAVE_OPTUNA = True
except ImportError:
    HAVE_OPTUNA = False

# -------------------------------------------------------------------------
# Session History Tracing & Transition Concatenation
# -------------------------------------------------------------------------
def get_session_dir(session_arg: str) -> str:
    """Resolves session argument to its folder directory under logs/."""
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
    """
    Parses execution_statistics.txt in the session folder to detect the
    'CONTINUATION FROM SESSION: <path>' header line.
    """
    stats_file = os.path.join(session_dir, "execution_statistics.txt")
    if not os.path.exists(stats_file):
        return None

    prefix = "CONTINUATION FROM SESSION:"
    with open(stats_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(prefix):
                raw_path = line[len(prefix):].strip().rstrip(";").strip()
                # Normalize OS separators (e.g., logs\2026-09-18_12-52)
                normalized_path = os.path.normpath(raw_path)
                if os.path.isdir(normalized_path):
                    return normalized_path
                # In case only folder name was stored
                alt_path = os.path.join("logs", os.path.basename(normalized_path))
                if os.path.isdir(alt_path):
                    return alt_path
    return None


def collect_and_concatenate_transitions(session_arg: str, output_filename: str = "all_offline_transitions.npz") -> str:
    """
    Traces back through all ancestor sessions using execution_statistics.txt,
    collects all offline_transitions.npz arrays, concatenates them chronologically,
    and saves the combined dataset.
    """
    current_dir = get_session_dir(session_arg)
    target_session_dir = current_dir

    # 1. Trace the lineage chain backwards
    session_chain = []
    visited = set()
    while current_dir and current_dir not in visited:
        visited.add(current_dir)
        session_chain.append(current_dir)
        current_dir = find_predecessor_session(current_dir)

    # Reverse chain to place oldest sessions first
    session_chain.reverse()
    print(f"[Dataset] Session lineage chain ({len(session_chain)} sessions):")
    for s in session_chain:
        print(f"  -> {s}")

    # 2. Collect .npz arrays across all found sessions
    all_prev_states = []
    all_skill_names = []
    all_post_states = []
    all_succeeds = []
    all_timestamps = []

    for s_dir in session_chain:
        npz_path = os.path.join(s_dir, "offline_transitions.npz")
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path, allow_pickle=True)
                all_prev_states.append(data['prev_states'])
                all_skill_names.append(data['skill_names'])
                all_post_states.append(data['post_states'])
                all_succeeds.append(data['succeeds'])
                if 'timestamps' in data:
                    all_timestamps.append(data['timestamps'])
                print(f"[Dataset] Loaded {len(data['prev_states'])} transitions from {npz_path}")
            except Exception as e:
                print(f"[Dataset] Warning: Failed to read {npz_path}: {e}")

    # Raise error only if no offline data could be found across all sessions
    if not all_prev_states:
        raise FileNotFoundError(
            f"No offline transitions found in session chain starting from '{target_session_dir}'."
        )

    # 3. Concatenate arrays
    combined_prev = np.concatenate(all_prev_states, axis=0)
    combined_skills = np.concatenate(all_skill_names, axis=0)
    combined_post = np.concatenate(all_post_states, axis=0)
    combined_succeeds = np.concatenate(all_succeeds, axis=0)
    combined_timestamps = (
        np.concatenate(all_timestamps, axis=0)
        if all_timestamps
        else np.zeros(len(combined_prev), dtype=np.float64)
    )

    # 4. Save combined NPZ inside the target session folder
    output_path = os.path.join(target_session_dir, output_filename)
    np.savez_compressed(
        output_path,
        prev_states=combined_prev,
        skill_names=combined_skills,
        post_states=combined_post,
        succeeds=combined_succeeds,
        timestamps=combined_timestamps
    )
    print(f"[Dataset] Successfully concatenated {len(combined_prev)} total transitions -> {output_path}")
    return output_path

# -------------------------------------------------------------------------
# Hyperparameter Search Space Definition
# -------------------------------------------------------------------------
PARAM_BOUNDS = {
    "max_points": (20, 100),            # int
    "base_radius": (0.02, 0.25),        # float
    "max_radius": (0.25, 2.0),          # float
    "learning_rate_b": (0.05, 0.5),     # float
    "learning_rate_n": (0.001, 0.05),   # float
    "max_edge_age": (10, 80),           # int
    "lambda_step": (5, 25),             # int
    "alpha": (0.1, 0.9),                # float
    "d": (0.90, 0.999),                 # float
    "min_overlap_ratio": (0.25, 1.0),   # float
}


def sample_hyperparameters():
    """Generates a random configuration within defined search bounds."""
    params = {
        "max_points": random.randint(PARAM_BOUNDS["max_points"][0], PARAM_BOUNDS["max_points"][1]),
        "base_radius": round(random.uniform(PARAM_BOUNDS["base_radius"][0], PARAM_BOUNDS["base_radius"][1]), 4),
        "max_radius": round(random.uniform(PARAM_BOUNDS["max_radius"][0], PARAM_BOUNDS["max_radius"][1]), 4),
        "learning_rate_b": round(random.uniform(PARAM_BOUNDS["learning_rate_b"][0], PARAM_BOUNDS["learning_rate_b"][1]), 4),
        "learning_rate_n": round(random.uniform(PARAM_BOUNDS["learning_rate_n"][0], PARAM_BOUNDS["learning_rate_n"][1]), 4),
        "max_edge_age": random.randint(PARAM_BOUNDS["max_edge_age"][0], PARAM_BOUNDS["max_edge_age"][1]),
        "lambda_step": random.randint(PARAM_BOUNDS["lambda_step"][0], PARAM_BOUNDS["lambda_step"][1]),
        "alpha": round(random.uniform(PARAM_BOUNDS["alpha"][0], PARAM_BOUNDS["alpha"][1]), 4),
        "d": round(random.uniform(PARAM_BOUNDS["d"][0], PARAM_BOUNDS["d"][1]), 5),
        "min_overlap_ratio": round(random.uniform(PARAM_BOUNDS["min_overlap_ratio"][0], PARAM_BOUNDS["min_overlap_ratio"][1]), 3),
    }
    if params["base_radius"] >= params["max_radius"]:
        params["max_radius"] = params["base_radius"] + 0.2
    return params


def apply_params_to_brain(brain: BrainNetwork, params: dict):
    """Injects sampled hyperparameters into the BrainNetwork instance."""
    brain.max_points = int(params["max_points"])
    brain.base_radius = float(params["base_radius"])
    brain.max_radius = float(params["max_radius"])
    brain.learning_rate_b = float(params["learning_rate_b"])
    brain.learning_rate_n = float(params["learning_rate_n"])
    brain.max_edge_age = int(params["max_edge_age"])
    brain.lambda_step = int(params["lambda_step"])
    brain.alpha = float(params["alpha"])
    brain.d = float(params["d"])

    # Monkey-patch default min_overlap_ratio for update_transitions
    original_update_transitions = brain.update_transitions
    min_overlap = float(params["min_overlap_ratio"])

    def wrapped_update_transitions(skill_name: str, min_overlap_ratio: float = min_overlap):
        return original_update_transitions(skill_name, min_overlap_ratio=min_overlap)

    brain.update_transitions = wrapped_update_transitions

    # Sync any existing predicate nodes
    for node in brain.trans_graph.nodes.values():
        for pred in (node.precondition, node.effect):
            pred.max_points = brain.max_points
            pred.base_radius = brain.base_radius
            pred.max_radius = brain.max_radius
            pred.eb = brain.learning_rate_b
            pred.en = brain.learning_rate_n
            pred.max_age = brain.max_edge_age
            pred.lambda_step = brain.lambda_step
            pred.alpha = brain.alpha
            pred.d = brain.d


# -------------------------------------------------------------------------
# Evaluation Routine 1: Live Robot/Simulation Test Runs
# -------------------------------------------------------------------------
def evaluate_live_tests(bus: Communicator, params: dict, n_tests: int, source_dir: str = 'continue'):
    """Initializes TheAgent, executes test_run(), and returns composite fitness."""
    agent = TheAgent(bus=bus, how_many_runs=1, how_many_tests=n_tests, source_dir=source_dir)
    apply_params_to_brain(agent.brain, params)

    agent.test_run()

    if not agent.tests:
        return 0.0, {"success_rate": 0.0, "mean_steps": 0.0}

    results = [1.0 if t[5] else 0.0 for t in agent.tests]
    steps_executed = [t[2] for t in agent.tests]
    durations = [t[4] for t in agent.tests]

    success_rate = float(np.mean(results))
    mean_steps = float(np.mean(steps_executed))
    mean_time = float(np.mean(durations))

    score = (success_rate * 100.0) + (mean_steps * 5.0) - min(mean_time * 0.1, 10.0)

    stats = {
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "mean_time_s": mean_time,
        "score": score
    }
    return score, stats


# -------------------------------------------------------------------------
# Evaluation Routine 2: Offline Dataset Replay
# -------------------------------------------------------------------------
def evaluate_offline_dataset(dataset_path: str, params: dict):
    """
    Replays offline transition history from logs/<session>/offline_transitions.npz
    through BrainNetwork with candidate hyperparameters.
    """
    data = np.load(dataset_path, allow_pickle=True)
    prev_states = data['prev_states']
    skill_names = data['skill_names']
    post_states = data['post_states']
    succeeds = data['succeeds']

    split_idx = int(0.7 * len(prev_states))
    train_slice = range(0, split_idx)
    val_slice = range(split_idx, len(prev_states))

    temp_log_dir = os.path.join("optim_logs", f"opt_temp_{int(time.time() * 1000)}")
    os.makedirs(temp_log_dir, exist_ok=True)

    try:
        brain = BrainNetwork(log_dir=temp_log_dir)
        apply_params_to_brain(brain, params)

        # 1. Train on training split
        batch = []
        for i in train_slice:
            batch.append((prev_states[i], str(skill_names[i]), post_states[i], bool(succeeds[i])))
            if len(batch) >= 10:
                brain.update(batch)
                batch = []
        if batch:
            brain.update(batch)

        # 2. Evaluate on validation split
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in val_slice:
            actual_succeed = bool(succeeds[i])
            pred_met = brain.preconditions_met(str(skill_names[i]), prev_states[i])

            if pred_met and actual_succeed:
                tp += 1
            elif not pred_met and not actual_succeed:
                tn += 1
            elif pred_met and not actual_succeed:
                fp += 1
            else:
                fn += 1

        total = max(1, tp + tn + fp + fn)
        accuracy = (tp + tn) / total

        # Check graph connectivity
        num_edges = sum(len(sources) for sources in brain.trans_graph.edges_by_target.values())
        connectivity_bonus = min(num_edges, 10) * 1.5

        score = (accuracy * 100.0) + connectivity_bonus
        metrics = {
            "val_accuracy": accuracy,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "num_edges": num_edges,
            "score": score
        }
        return score, metrics

    finally:
        shutil.rmtree(temp_log_dir, ignore_errors=True)


# -------------------------------------------------------------------------
# Optimizer Runner
# -------------------------------------------------------------------------
def run_optimization(mode: str, num_trials: int, n_tests: int, session: str, source_dir: str):
    dataset_path = None
    if mode == "offline":
        dataset_path = collect_and_concatenate_transitions(session)
        print(f"[Optimizer] Using aggregated dataset: {dataset_path}")

    bus = None
    if mode == "live":
        print("[Optimizer] Connecting to Communicator bus for live testing...")
        bus = Communicator()

    best_score = -float('inf')
    best_params = None
    all_trials_records = []

    out_file = os.path.join("logs", f"hyperparam_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs("logs", exist_ok=True)

    print(f"\n[Optimizer] Starting optimization in '{mode}' mode across {num_trials} trials.")
    print("=" * 80)

    if HAVE_OPTUNA:
        print("[Optimizer] Using Optuna for Bayesian search.")

        def objective(trial):
            params = {
                "max_points": trial.suggest_int("max_points", PARAM_BOUNDS["max_points"][0], PARAM_BOUNDS["max_points"][1]),
                "base_radius": trial.suggest_float("base_radius", PARAM_BOUNDS["base_radius"][0], PARAM_BOUNDS["base_radius"][1]),
                "max_radius": trial.suggest_float("max_radius", PARAM_BOUNDS["max_radius"][0], PARAM_BOUNDS["max_radius"][1]),
                "learning_rate_b": trial.suggest_float("learning_rate_b", PARAM_BOUNDS["learning_rate_b"][0], PARAM_BOUNDS["learning_rate_b"][1]),
                "learning_rate_n": trial.suggest_float("learning_rate_n", PARAM_BOUNDS["learning_rate_n"][0], PARAM_BOUNDS["learning_rate_n"][1]),
                "max_edge_age": trial.suggest_int("max_edge_age", PARAM_BOUNDS["max_edge_age"][0], PARAM_BOUNDS["max_edge_age"][1]),
                "lambda_step": trial.suggest_int("lambda_step", PARAM_BOUNDS["lambda_step"][0], PARAM_BOUNDS["lambda_step"][1]),
                "alpha": trial.suggest_float("alpha", PARAM_BOUNDS["alpha"][0], PARAM_BOUNDS["alpha"][1]),
                "d": trial.suggest_float("d", PARAM_BOUNDS["d"][0], PARAM_BOUNDS["d"][1]),
                "min_overlap_ratio": trial.suggest_float("min_overlap_ratio", PARAM_BOUNDS["min_overlap_ratio"][0], PARAM_BOUNDS["min_overlap_ratio"][1]),
            }
            if params["base_radius"] >= params["max_radius"]:
                params["max_radius"] = params["base_radius"] + 0.1

            if mode == "live":
                score, metrics = evaluate_live_tests(bus, params, n_tests=n_tests, source_dir=source_dir)
            else:
                score, metrics = evaluate_offline_dataset(dataset_path, params)

            trial.set_user_attr("metrics", metrics)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=num_trials)

        best_params = study.best_params
        best_score = study.best_value
        all_trials_records = [
            {"trial": t.number, "params": t.params, "score": t.value, "metrics": t.user_attrs.get("metrics")}
            for t in study.trials
        ]

    else:
        print("[Optimizer] Optuna not found. Defaulting to Random Search.")
        for trial_idx in range(1, num_trials + 1):
            params = sample_hyperparameters()
            print(f"\n[Trial {trial_idx}/{num_trials}] Testing config: {params}")

            if mode == "live":
                score, metrics = evaluate_live_tests(bus, params, n_tests=n_tests, source_dir=source_dir)
            else:
                score, metrics = evaluate_offline_dataset(dataset_path, params)

            print(f"[Trial {trial_idx}] Score: {score:.3f} | Metrics: {metrics}")

            record = {"trial": trial_idx, "params": params, "score": score, "metrics": metrics}
            all_trials_records.append(record)

            if score > best_score:
                best_score = score
                best_params = params
                print(f"*** New best score: {best_score:.3f} ***")

    # Save summary
    results_summary = {
        "mode": mode,
        "dataset_used": dataset_path if mode == "offline" else None,
        "best_score": best_score,
        "best_params": best_params,
        "all_trials": all_trials_records,
    }

    with open(out_file, "w") as f:
        json.dump(results_summary, f, indent=4)

    print("\n" + "=" * 80)
    print("OPTIMIZATION FINISHED")
    print("=" * 80)
    print(f"Best Score: {best_score}")
    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k} = {v}")
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="Tune GNG & BrainNetwork hyperparameters.")
    parser.add_argument("--mode", choices=["live", "offline"], default="offline",
                        help="Choose 'live' for robot test_run() evaluation, or 'offline' to replay .npz transitions.")
    parser.add_argument("--session", type=str, default="latest",
                        help="Session name inside logs/ (e.g. '2026-03-29_14-30', 'latest', or full path).")
    parser.add_argument("--trials", type=int, default=30, help="Number of tuning trials.")
    parser.add_argument("--n_tests", type=int, default=5, help="Number of tests per trial in live mode.")
    parser.add_argument("--source_dir", type=str, default="continue", help="Session directory to load brain from for live tests.")

    args = parser.parse_args()
    '''
    run_optimization(
        mode="offline",
        num_trials=30,
        n_tests=5,
        session="2026-09-21_01-07",
        source_dir="continue"
    )