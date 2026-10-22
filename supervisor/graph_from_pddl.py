import json
import os
import re

from .picture_generator import get_session

SKILL_ORDER = [
    "start",
    "find_the_door",
    "go_to_the_door",
    "open_the_door",
    "find_the_goal",
    "go_to_the_goal",
    "finish",
]


class GraphFromPDDL:
    def __init__(self, session_dir=None, logs_dir="logs"):
        self.session_dir = session_dir
        self.logs_dir = logs_dir

    def _parse_actions(self, content: str) -> dict:
        """
        Parses all (:action ...) blocks in domain.pddl and extracts
        precondition symbols and add (positive effect) symbols.
        """
        actions = {}
        raw_actions = re.findall(
            r"\(:action\s+([a-zA-Z0-9_\-]+)(.*?)(?=\(:action|\)\s*\Z)",
            content,
            re.DOTALL | re.IGNORECASE,
        )

        reserved = {"and", "not", "or", "imply", "forall", "exists"}

        for act_name, body in raw_actions:
            act_name = act_name.strip().lower()

            pre_match = re.search(r":precondition\s*(.*?)(?=:effect|\Z)", body, re.DOTALL | re.IGNORECASE)
            pre_str = pre_match.group(1) if pre_match else ""

            eff_match = re.search(r":effect\s*(.*?)\Z", body, re.DOTALL | re.IGNORECASE)
            eff_str = eff_match.group(1) if eff_match else ""

            raw_pre = re.findall(r"\(\s*([a-zA-Z0-9_\-]+)", pre_str)
            pre_symbols = {p.lower() for p in raw_pre if p.lower() not in reserved}

            eff_clean = re.sub(r"\(\s*not\s*\([^)]*\)\s*\)", "", eff_str, flags=re.IGNORECASE)
            raw_eff = re.findall(r"\(\s*([a-zA-Z0-9_\-]+)", eff_clean)
            eff_symbols = {e.lower() for e in raw_eff if e.lower() not in reserved}

            actions[act_name] = {
                "preconditions": pre_symbols,
                "effects": eff_symbols,
            }

        return actions

    def domain_to_json_graph(self, domain_file_path: str) -> str:
        if not os.path.exists(domain_file_path):
            return None

        with open(domain_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        actions = self._parse_actions(content)
        if not actions:
            return json.dumps({"nodes": [], "edges": []}, indent=4)

        nodes = sorted(list(actions.keys()))
        ordered_present = [s for s in SKILL_ORDER if s in actions]
        n = len(ordered_present)

        edges = []
        for i in range(n):
            for j in range(n):
                u = ordered_present[i]
                v = ordered_present[j]

                u_eff = actions[u]["effects"]
                v_pre = actions[v]["preconditions"]

                if len(v_pre) == 0:
                    fraction = 1.0
                else:
                    satisfied = len(v_pre.intersection(u_eff))
                    fraction = satisfied / len(v_pre)

                is_consecutive = (j == i + 1)
                is_full = (fraction >= 1.0)

                # Keep edge if on the consecutive main path or if fully satisfied (1.0)
                if is_consecutive or is_full:
                    edges.append({
                        "source": u,
                        "target": v,
                        "weight": round(fraction, 2),
                    })

        return json.dumps({"nodes": nodes, "edges": edges}, indent=4)

    def save_file(self, data, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data)

    def run(self):
        session = get_session(self.logs_dir, self.session_dir)
        if not session:
            print("[GraphFromPDDL] No valid session found.")
            return

        domain_path = os.path.join(session, "PDDL", "task", "domain.pddl")
        json_graph = self.domain_to_json_graph(domain_path)
        if json_graph is None:
            return

        out_path = os.path.join(session, "graphs", "trans_graph_PDDL.json")
        self.save_file(json_graph, out_path)


if __name__ == "__main__":
    sd = "2026-10-17_15-38"
    gen = GraphFromPDDL(session_dir=sd)
    gen.run()