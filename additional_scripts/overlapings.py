import os
import glob
from datetime import datetime
import numpy as np

from agent.brain_network import BrainNetwork
from agent.predicate import Predicate


def get_session(logs_root, session_dir):
    if not os.path.exists(logs_root):
        return None
    if session_dir is not None and os.path.exists(os.path.join(logs_root, session_dir)):
        return os.path.join(logs_root, session_dir)

    def parse_session_timestamp(dir_name):
        try:
            return datetime.strptime(dir_name, "%Y-%m-%d_%H-%M")
        except ValueError:
            return None

    valid_sessions = []
    for entry in os.scandir(logs_root):
        if entry.is_dir() and parse_session_timestamp(entry.name):
            valid_sessions.append(entry.path)

    return max(valid_sessions, key=os.path.basename, default=None)


class OverlapMatrixGenerator:
    """
    Computes and exports an NxN matrix of overlap ratios between all actions:
        Overlap(A -> B) = |Eff(A) ∩ Pre(B)| / |Eff(A)|
    Uses BrainNetwork.load_predicate() and Predicate.is_forbidden() directly.
    """
    def __init__(self, logs_root="logs", session_dir=None, output_filename="overlap_matrix.txt"):
        self.logs_root = logs_root
        self.session_dir = session_dir
        self.output_filename = output_filename

    def _resolve_gng_dir(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session:
            return None, None
        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
        if not os.path.isdir(symbols_dir):
            return latest_session, None
        return latest_session, symbols_dir

    @staticmethod
    def _calculate_overlap(eff_pred: Predicate, pre_pred: Predicate):
        """
        Mirrors the exact overlap logic from BrainNetwork.calculate_overlap:
        An effect prototype point is counted if it is NOT forbidden (i.e. TRUE or UNKNOWN).
        """
        if not eff_pred or not eff_pred.nodes or len(eff_pred.nodes) < 1:
            return None
        if not pre_pred or not pre_pred.nodes or len(pre_pred.nodes) < 1:
            return None

        overlapping_points = [
            node for node in eff_pred.nodes 
            if not pre_pred.is_forbidden(node)
        ]
        return len(overlapping_points) / len(eff_pred.nodes)

    def run(self):
        latest_session, symbols_dir = self._resolve_gng_dir()
        if not symbols_dir:
            print(f"[OverlapMatrix] GNG folder not found in session: {latest_session}")
            return

        # 1. Discover all unique action/skill names from *_init.json and *_eff.json
        action_names = set()
        for f in glob.glob(os.path.join(symbols_dir, "*_init.json")):
            base = os.path.splitext(os.path.basename(f))[0]
            action_names.add(base[:-5])
        for f in glob.glob(os.path.join(symbols_dir, "*_eff.json")):
            base = os.path.splitext(os.path.basename(f))[0]
            action_names.add(base[:-4])

        # Canonical execution order fallback
        desired_order = [
            "Start", "Find_the_Door", "Go_to_the_Door", "Open_the_Door",
            "Find_the_Goal", "Go_to_the_Goal", "Finish"
        ]
        actions = [a for a in desired_order if a in action_names]
        actions.extend(sorted(list(action_names - set(actions))))

        if not actions:
            print("[OverlapMatrix] No actions discovered.")
            return

        # 2. Pre-load Predicate instances via BrainNetwork deserializer
        effects = {}
        preconditions = {}

        for act in actions:
            eff_path = os.path.join(symbols_dir, f"{act}_eff.json")
            pre_path = os.path.join(symbols_dir, f"{act}_init.json")
            
            effects[act] = BrainNetwork.load_predicate(eff_path) if os.path.isfile(eff_path) else None
            preconditions[act] = BrainNetwork.load_predicate(pre_path) if os.path.isfile(pre_path) else None

        # 3. Compute the pairwise overlap matrix
        N = len(actions)
        matrix = np.full((N, N), np.nan, dtype=float)

        for i, src_act in enumerate(actions):
            eff_pred = effects[src_act]

            for j, tgt_act in enumerate(actions):
                if i == j:
                    matrix[i, j] = 0.0  # No self-loops
                    continue

                pre_pred = preconditions[tgt_act]
                ratio = self._calculate_overlap(eff_pred, pre_pred)
                matrix[i, j] = ratio if ratio is not None else np.nan

        # 4. Format and save to text file
        os.makedirs(latest_session, exist_ok=True)
        out_path = os.path.join(latest_session, self.output_filename)

        col_w = max(14, max(len(a) for a in actions) + 2)
        row_header_w = max(len(a) for a in actions) + 2

        lines = []
        lines.append("=" * 110)
        lines.append("                         INTER-ACTION OVERLAP RATIO MATRIX")
        lines.append("  Definition : Overlap(Row -> Col) = |Eff(Row) intersect Pre(Col)| / |Eff(Row)|")
        lines.append(f"  Session    : {latest_session}")
        lines.append("=" * 110)
        lines.append("")

        # Column Header Row
        header_str = f"{'Source / Target':<{row_header_w}}" + "".join(f"{act:>{col_w}}" for act in actions)
        lines.append(header_str)
        lines.append("-" * len(header_str))

        # Matrix Body Rows
        for i, src_act in enumerate(actions):
            row_items = []
            for j in range(N):
                val = matrix[i, j]
                val_str = "N/A" if np.isnan(val) else f"{val:.4f}"
                row_items.append(f"{val_str:>{col_w}}")
            lines.append(f"{src_act:<{row_header_w}}" + "".join(row_items))

        lines.append("-" * len(header_str))
        lines.append("")

        # Filtered Valid Transitions (Overlap >= 0.85)
        lines.append("SIGNIFICANT TRANSITIONS (Overlap >= 0.85):")
        lines.append("-" * 55)
        sig_count = 0
        for i, src_act in enumerate(actions):
            for j, tgt_act in enumerate(actions):
                val = matrix[i, j]
                if not np.isnan(val) and val >= 0.85:
                    lines.append(f"  • {src_act} -> {tgt_act:<20} : {val * 100:.2f}%")
                    sig_count += 1
        if sig_count == 0:
            lines.append("  (None found above threshold)")

        lines.append("=" * 110)

        with open(out_path, "w") as f:
            f.write("\n".join(lines))

        print(f"[OverlapMatrix] Saved overlap matrix to {out_path}")
        return out_path


if __name__ == "__main__":
    matrix_gen = OverlapMatrixGenerator(session_dir="2026-10-12_22-20")
    matrix_gen.run()