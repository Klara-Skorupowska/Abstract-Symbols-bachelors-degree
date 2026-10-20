import json
import re
import os

from .picture_generator import get_session

class GraphFromPDDL:
    def __init__(self, session_dir=None, logs_dir="logs"):
        self.session_dir = session_dir
        self.logs_dir = logs_dir

    def _parse_actions(self, content: str) -> dict:
        """
        Parses all (:action ...) definitions in domain.pddl and extracts
        precondition symbols and add (positive effect) symbols.
        """
        actions = {}

        # Match blocks: (:action <name> ... :precondition (...) :effect (...))
        pattern = re.compile(
            r"\(:action\s+([a-zA-Z0-9_\-]+)\s+:parameters\s*\([^)]*\)\s+:precondition\s*(.*?)\s+:effect\s*(.*?)\s*\)",
            re.DOTALL | re.IGNORECASE,
        )

        for match in pattern.finditer(content):
            act_name = match.group(1).strip()
            pre_str = match.group(2).strip()
            eff_str = match.group(3).strip()

            # Extract precondition predicate symbols (ignoring 'and')
            pre_symbols = set(re.findall(r"\(([a-zA-Z0-9_\-]+)\)", pre_str))
            pre_symbols.discard("and")

            # Extract only positive added effect symbols (ignoring 'not' and 'and')
            # First remove (not (...)) clauses so deleted symbols aren't treated as adds
            eff_clean = re.sub(r"\(not\s*\([^)]*\)\)", "", eff_str)
            eff_symbols = set(re.findall(r"\(([a-zA-Z0-9_\-]+)\)", eff_clean))
            eff_symbols.discard("and")

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

        action_names = list(actions.keys())
        nodes = sorted(action_names)

        edges = []
        # Draw directed transition from action A to action B if B's preconditions
        # are non-empty and fully satisfied by the effects of A
        for u in nodes:
            u_eff = actions[u]["effects"]
            for v in nodes:
                v_pre = actions[v]["preconditions"]
                # Transition is possible if all preconditions of v are satisfied by u's effects
                if v_pre and v_pre.issubset(u_eff):
                    edges.append({"from": u, "to": v})

        graph_data = {
            "nodes": nodes,
            "edges": edges,
        }
        return json.dumps(graph_data, indent=4)

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
            #print(f"[GraphFromPDDL] domain.pddl not found at: {domain_path}")
            return

        out_path = os.path.join(session, "graphs", "trans_graph_PDDL.json")
        self.save_file(json_graph, out_path)
        #print(f"[GraphFromPDDL] Saved transition graph to: {out_path}")


if __name__ == "__main__":
    sd = "2026-10-17_15-38"
    gen = GraphFromPDDL(session_dir=sd)
    gen.run()