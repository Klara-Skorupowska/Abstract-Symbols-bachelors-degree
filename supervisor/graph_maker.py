import json
import re
import os


from .visualize import get_session

class GraphFromPDDL():

    def __init__(self, session_dir=None, logs_dir="logs"):
        self.session_dir = session_dir
        self.logs_dir = logs_dir

    def domain_to_json_graph(self, domain_file_path: str) -> str:
        if not os.path.exists(domain_file_path):
            return None
        with open(domain_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Match each :action block: (:action <name> :parameters (...) :precondition (...) :effect (...))
        action_pattern = re.compile(
            r"\(:action\s+([a-zA-Z0-9_\-]+)\s*:parameters\s*\([^)]*\)\s*:precondition\s*\((.*?)\)\s*:effect\s*\((.*?)\)\s*\)",
            re.DOTALL | re.IGNORECASE,
        )

        actions = {}

        for match in action_pattern.finditer(content):
            action_name_raw = match.group(1).strip()
            precond_text = match.group(2)
            effect_text = match.group(3)

            # Standardize name formatting (e.g. 'find_the_door' -> 'Find_the_Door')
            canonical_name = "_".join(
                w.capitalize() for w in action_name_raw.split("_")
            )

            # Extract all positive preconditions
            # Removes '(and' or other outer structural words
            preconds = set(re.findall(r"\(([a-zA-Z0-9_\-]+)\)", precond_text))

            # Extract effects: identify positive adds and exclude '(not (...))'
            negated_effects = set(re.findall(r"\(not\s+\(([a-zA-Z0-9_\-]+)\)\)", effect_text))
            all_predicates_in_effects = set(re.findall(r"\(([a-zA-Z0-9_\-]+)\)", effect_text))
            positive_effects = (all_predicates_in_effects - negated_effects) - {"not", "and"}

            actions[canonical_name] = {
                "preconditions": preconds,
                "effects": positive_effects,
            }

        nodes = list(actions.keys())
        edges = []

        # An action A enables action B if A adds a predicate that B requires as a precondition
        for source_name, source_data in actions.items():
            for target_name, target_data in actions.items():
                if source_data["effects"] & target_data["preconditions"]:
                    edges.append({
                        "source": source_name,
                        "target": target_name,
                        "symbol_name": f"sym_{source_name}_enables_{target_name}",
                    })

        graph_dict = {
            "nodes": nodes,
            "edges": edges,
        }

        return json.dumps(graph_dict, indent=4)

    def save_file(self, data, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
                f.write(data)

    def run(self):
        session = get_session(self.logs_dir, self.session_dir)
        json_graph = self.domain_to_json_graph(f"{session}\\PDDL\\task\\domain.pddl")
        if json_graph is None:
            return
        self.save_file(json_graph, f"{session}\\graphs\\trans_graph_PDDL.json")

if __name__ == "__main__":
    # Example usage:
    sd = "2026-10-17_15-38"
    
    gen = GraphFromPDDL(session_dir=sd)
    gen.run()