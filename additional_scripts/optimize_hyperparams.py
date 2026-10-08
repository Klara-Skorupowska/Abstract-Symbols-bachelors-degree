#!/usr/bin/env python3
import os
import sys
import glob
import re
import argparse
from datetime import datetime
from collections import defaultdict
import numpy as np

from agent.brain_network import BrainNetwork
from agent.predicate import Predicate

# Target sequence of skills
REQUIRED_SEQUENCE = [
    "Start",
    "Find_the_Door",
    "Go_to_the_Door",
    "Open_the_Door",
    "Find_the_Goal",
    "Go_to_the_Goal",
    "Finish"
]


def find_target_pddl_gng_dir(logs_root: str = "logs", session_name: str = None) -> str:
    if session_name:
        candidate_dir = os.path.join(logs_root, session_name, "PDDL", "GNG")
        if not os.path.isdir(candidate_dir):
            sys.exit(f"Error: Path does not exist: {candidate_dir}")
        return candidate_dir

    if not os.path.isdir(logs_root):
        sys.exit(f"Error: Logs root '{logs_root}' does not exist.")

    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})$")
    sessions = []
    for entry in os.listdir(logs_root):
        if os.path.isdir(os.path.join(logs_root, entry)) and pattern.match(entry):
            try:
                sessions.append((datetime.strptime(entry, "%Y-%m-%d_%H-%M"), entry))
            except ValueError:
                continue

    if not sessions:
        sys.exit(f"Error: No matching session directories in '{logs_root}'.")

    sessions.sort(key=lambda x: x[0], reverse=True)
    selected = sessions[0][1]
    print(f"[Session Discovery] Using latest session: '{selected}'")
    return os.path.join(logs_root, selected, "PDDL", "GNG")


def load_predicates(gas_dir: str):
    init_files = glob.glob(os.path.join(gas_dir, "*_init.json"))
    nodes, inits, effs = [], {}, {}

    for init_path in init_files:
        skill_name = os.path.basename(init_path)[:-10]
        eff_path = os.path.join(gas_dir, f"{skill_name}_eff.json")
        if not os.path.exists(eff_path):
            continue

        nodes.append(skill_name)
        inits[skill_name] = BrainNetwork.load_predicate(init_path)
        effs[skill_name] = BrainNetwork.load_predicate(eff_path)

    return sorted(nodes), inits, effs


def recalculate_predicate_radii(pred: Predicate, new_rf: float, baseline_rf: float):
    pred.radii_factor = new_rf
    if pred.cached_radii is not None and len(pred.cached_radii) == len(pred.nodes):
        pred.cached_radii = [
            float(np.clip((r / baseline_rf if baseline_rf > 0 else r) * new_rf, pred.base_radius, pred.max_radius))
            for r in pred.cached_radii
        ]
    else:
        pred.cached_radii = None


def compute_pairwise_overlap(eff_pred: Predicate, init_pred: Predicate) -> float:
    if not eff_pred.nodes or not init_pred.nodes:
        return 0.0
    overlapping = [pt for pt in eff_pred.nodes if init_pred.is_active(pt)]
    return len(overlapping) / float(len(eff_pred.nodes))


def evaluate_path_and_edges(nodes, inits, effs, rf, th, baseline_rf, required_path):
    # Recalculate radii for all loaded predicates
    for n in nodes:
        recalculate_predicate_radii(inits[n], rf, baseline_rf)
        recalculate_predicate_radii(effs[n], rf, baseline_rf)

    # 1. Check path validity and find the bottleneck ratio on this path
    path_bottleneck = 1.0
    path_overlaps = []
    path_valid = True

    for u, v in zip(required_path[:-1], required_path[1:]):
        ratio = compute_pairwise_overlap(effs[u], inits[v])
        path_overlaps.append((u, v, ratio))
        if ratio < path_bottleneck:
            path_bottleneck = ratio
        if ratio < th:
            path_valid = False

    # 2. Count total graph edges under current threshold
    total_edges = 0
    all_active_edges = []
    for src in nodes:
        for tgt in nodes:
            if src == tgt:
                continue
            r = compute_pairwise_overlap(effs[src], inits[tgt])
            if r >= th:
                total_edges += 1
                all_active_edges.append((src, tgt, r))

    return path_valid, total_edges, path_bottleneck, path_overlaps, all_active_edges


def main():
    parser = argparse.ArgumentParser(description="Optimize rf and min_overlap to preserve the ground-truth sequence with minimal edges.")
    parser.add_argument("--logs_root", type=str, default="logs")
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--baseline_rf", type=float, default=1.0)

    # Search boundaries
    parser.add_argument("--rf_min", type=float, default=0.5)
    parser.add_argument("--rf_max", type=float, default=3.0)
    parser.add_argument("--rf_steps", type=int, default=26)

    parser.add_argument("--th_min", type=float, default=0.1)
    parser.add_argument("--th_max", type=float, default=0.9)
    parser.add_argument("--th_steps", type=int, default=17)

    args = parser.parse_args()

    gas_dir = find_target_pddl_gng_dir(args.logs_root, args.session)
    nodes, inits, effs = load_predicates(gas_dir)
    print(f"Loaded {len(nodes)} skills from: {gas_dir}")

    # Validate that all required skills exist in the files
    missing = [skill for skill in REQUIRED_SEQUENCE if skill not in nodes]
    if missing:
        sys.exit(f"Error: Missing predicate files for required skills: {missing}")

    rf_grid = np.linspace(args.rf_min, args.rf_max, args.rf_steps)
    th_grid = np.linspace(args.th_min, args.th_max, args.th_steps)

    best_score = None
    best_config = None
    best_info = None

    print("\nTarget sequence:")
    print(" -> ".join(REQUIRED_SEQUENCE))
    print(f"Minimum required path edges: {len(REQUIRED_SEQUENCE) - 1}\n")

    print(f"{'rf':>7s} | {'th':>6s} | {'Path OK':>8s} | {'Edges':>6s} | {'Bottleneck':>10s}")
    print("-" * 48)

    for rf in rf_grid:
        for th in th_grid:
            path_valid, total_edges, bottleneck, path_overlaps, active_edges = evaluate_path_and_edges(
                nodes, inits, effs, rf, th, args.baseline_rf, REQUIRED_SEQUENCE
            )

            # Ranking:
            # 1. path_valid (must be True: 1 vs 0)
            # 2. Minimize total_edges (maximize -total_edges)
            # 3. Maximize min_overlap_threshold (th)
            # 4. Minimize radii_factor (maximize -rf)
            score = (
                1 if path_valid else 0,
                -total_edges,
                th,
                -rf
            )

            if best_score is None or score > best_score:
                best_score = score
                best_config = (rf, th)
                best_info = (total_edges, bottleneck, path_overlaps, active_edges)

            status = "YES" if path_valid else "no"
            print(f"{rf:7.3f} | {th:6.3f} | {status:>8s} | {total_edges:6d} | {bottleneck:10.3f}")

    print("\n" + "=" * 55)
    if best_config and best_score[0] == 1:
        opt_rf, opt_th = best_config
        opt_edges, opt_bottleneck, opt_path_overlaps, opt_all_edges = best_info

        print("Optimal Parameters Found:")
        print(f"  radii_factor:          {opt_rf:.3f}")
        print(f"  min_overlap_threshold: {opt_th:.3f}")
        print(f"  Total Graph Edges:     {opt_edges} (Minimal path has {len(REQUIRED_SEQUENCE) - 1} edges)")
        print(f"  Path Bottleneck Ratio: {opt_bottleneck:.3f}")

        print("\nRequired Path Overlap Ratios:")
        for u, v, w in opt_path_overlaps:
            print(f"  {u} -> {v:<16s} : {w:.3f}")

        spurious_edges = [e for e in opt_all_edges if (e[0], e[1]) not in set(zip(REQUIRED_SEQUENCE[:-1], REQUIRED_SEQUENCE[1:]))]
        if spurious_edges:
            print(f"\nRemaining Extraneous Edges ({len(spurious_edges)}):")
            for u, v, w in spurious_edges:
                print(f"  {u} -> {v:<16s} : {w:.3f}")
        else:
            print("\nZero spurious edges! The graph contains strictly the ground-truth path.")
    else:
        print("No parameter combination kept the complete path connected.")
        print("Suggestion: Expand --rf_max or decrease --th_min.")


if __name__ == "__main__":
    main()