import os
import re
import json
import time
import numpy as np

from .predicate import Predicate

class BrainNetwork:
    def __init__(self, log_dir=None):
        self.state_dim = None 
        # Flat dictionary mapping skill_name -> {"precondition": Predicate, "effect": Predicate}
        self.skills = {}

        ### HYPERPARAMETERS ###
        self.min_overlap_ratio_HARD = 0.75 # for only actiive overlap
        self.min_overlap_ratio_SOFT = 0.75 # for active and unknown and coflicted

        self.max_points = 10
        self.base_radius = 0.1
        self.max_radius = 8 
        self.learning_rate_b = 0.2
        self.learning_rate_n = 0.01
        self.max_edge_age = 15
        self.lambda_step = 5
        self.alpha = 0.7
        self.d = 0.99
        ### --------------- ###

        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, "PDDL")
            self.gas_dir = os.path.join(self.log_dir, "PDDL", "GNG")
            self.graphs_dir = os.path.join(self.log_dir, "graphs")
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.gas_dir, exist_ok=True)
            os.makedirs(self.graphs_dir, exist_ok=True)

    def add_node(self, skill_name: str, precondition: Predicate, effect: Predicate):
        self.skills[skill_name] = {
            "precondition": precondition,
            "effect": effect
        }
            
    def create_node(self, skill_name: str):
        if skill_name not in self.skills:
            precondition = Predicate(
                self.max_points, self.state_dim, self.base_radius, self.max_radius, 
                self.learning_rate_b, self.learning_rate_n, self.max_edge_age, 
                self.lambda_step, self.alpha, self.d
            )
            effect = Predicate(
                self.max_points, self.state_dim, self.base_radius, self.max_radius, 
                self.learning_rate_b, self.learning_rate_n, self.max_edge_age, 
                self.lambda_step, self.alpha, self.d
            )
            self.add_node(skill_name, precondition, effect)

    def prepare_nodes(self, skillset):
        for skill_name in skillset:
            self.create_node(skill_name)

    def preconditions_status(self, skill_name: str, state_vector) -> str:
        """Returns 'TRUE', 'FALSE', or 'UNKNOWN' for the skill's precondition."""
        if skill_name not in self.skills:
            return "UNKNOWN"
        return self.skills[skill_name]["precondition"].evaluate(state_vector)

    def preconditions_met(self, skill_name: str, state_vector) -> bool:
        """Execution check: Only execute in plans if positive is active and negative is inactive."""
        return self.preconditions_status(skill_name, state_vector) == "TRUE"

    def get_unknown_skills(self, state_vector) -> list:
        """Returns skills where neither positive nor negative GNG is active."""
        unknown_skills = []
        for name, predicates in self.skills.items():
            if predicates["precondition"].is_unknown(state_vector):
                unknown_skills.append(name)
        return unknown_skills

    def get_active_skills(self, vector) -> list:
        """Returns skills with active positive preconditions."""
        if self.state_dim is None:
            self.state_dim = len(vector)

        active_skills = []
        for name, predicates in self.skills.items():
            if predicates["precondition"].is_active(vector):
                active_skills.append(name)
        return active_skills

    def get_nodes(self) -> list:
        return list(self.skills.keys())

    def update(self, buffor):
        print("\t[Brain] Updating Predicates (GNG)")
        for data in buffor:
            pre_vector_state, skill_name, post_vector_state, succeed = data
        
            if self.state_dim is None:
                self.state_dim = len(pre_vector_state)

            if skill_name not in self.skills:
                self.create_node(skill_name)

            self.skills[skill_name]["precondition"].update(pre_vector_state, succeed)
            if succeed:
                self.skills[skill_name]["effect"].update(post_vector_state, True)

        self._logger()

    def calculate_overlap(self, preceeding: str, following: str, mode='HARD'):
        if preceeding == following: 
            return None, None
            
        src_skill = self.skills.get(preceeding)
        tgt_skill = self.skills.get(following)
        if not src_skill or not tgt_skill:
            return None, None

        eff_gng = src_skill["effect"]
        if not eff_gng.nodes or len(eff_gng.nodes) < 1:
            return None, None
            
        init_gng = tgt_skill["precondition"]
        if not init_gng.nodes or len(init_gng.nodes) < 1:
            return None, None

        # Check overlap: soft = not forbidden, hard = active
        if mode == 'SOFT':
            overlapping_points = [
                node for node in eff_gng.nodes 
                if not init_gng.is_forbidden(node)
            ]
        elif mode == 'HARD':
            overlapping_points = [
                node for node in eff_gng.nodes 
                if init_gng.is_acceptable(node)
            ]
        else:
            raise ValueError(f"mode should be SOFT or HARD, not {mode}")

        overlap_ratio = len(overlapping_points) / len(eff_gng.nodes)

        sym_pred = Predicate(
            max_points=max(self.max_points, len(overlapping_points)),
            dim=self.state_dim,
            base_radius=self.base_radius,
            max_radius=self.max_radius,
            learning_rate_b=self.learning_rate_b,
            learning_rate_n=self.learning_rate_n,
            max_edge_age=self.max_edge_age,
            lambda_step=1,
            alpha=self.alpha, 
            d=self.d
        )
        for pt in overlapping_points:
            sym_pred.update(pt, valence=True)

        return overlap_ratio, sym_pred

    def compute_transitions(self, min_overlap_ratio=None, mode='HARD') -> list:
        """
        Dynamically computes valid transitions and transition symbols on the fly.
        Returns a list of dicts: [{'source': str, 'target': str, 'weight': float, 'symbol': Predicate}]
        """
        if min_overlap_ratio is None:
            if mode == 'HARD':
                min_overlap_ratio = self.min_overlap_ratio_HARD
            if mode == 'SOFT':
                min_overlap_ratio = self.min_overlap_ratio_SOFT


        transitions = []
        skills_list = list(self.skills.keys())

        for src in skills_list:
            for tgt in skills_list:
                if src == tgt:
                    continue
                ratio, sym_pred = self.calculate_overlap(src, tgt, mode)
                if ratio is not None and ratio >= min_overlap_ratio:
                    transitions.append({
                        "source": src,
                        "target": tgt,
                        "weight": ratio,
                        "symbol": sym_pred
                    })
        return transitions

    def export_transition_graph_json(self, file_path=None, mode='HARD') -> dict:
        """
        Generates and optionally saves transition graph JSON for visualization.
        """
        transitions = self.compute_transitions(mode=mode)
        edges_list = []

        for edge in transitions:
            symbol_name = f"sym_{edge['source']}_enables_{edge['target']}"
            edges_list.append({
                "source": edge["source"],
                "target": edge["target"],
                "symbol_name": symbol_name,
                "weight": edge["weight"]
            })

        graph_data = {
            "nodes": list(self.skills.keys()),
            "edges": edges_list
        }

        if file_path:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            temp_file = f"{file_path}.tmp"
            with open(temp_file, "w") as f:
                json.dump(graph_data, f, indent=4)
            for _ in range(5):
                try:
                    os.replace(temp_file, file_path)
                    break
                except (PermissionError, OSError):
                    time.sleep(0.05)
            else:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass

        return graph_data

    def generate_domain_pddl(self, domain_name="robot_domain", mode='HARD') -> str:
        """Generates a STRIPS PDDL domain adhering strictly to Pyperplan syntax rules."""
        pddl = [
            f"(define (domain {domain_name})",
            "  (:requirements :strips)",
            "  (:predicates"
        ]

        def format_symbol_name(src: str, tgt: str) -> str:
            return f"sym_{src.lower()}_to_{tgt.lower()}"

        transitions = self.compute_transitions(mode=mode)

        # 1. Collect all declared predicates
        predicates = set()
        for skill in self.skills.keys():
            predicates.add(f"pre_{skill.lower()}")
            predicates.add(f"goal_reached_{skill.lower()}")

        for trans in transitions:
            predicates.add(format_symbol_name(trans["source"], trans["target"]))

        for pred in sorted(predicates):
            pddl.append(f"    ({pred})")
        pddl.append("  )")

        def format_effect(adds, deletes):
            effects = [f"({a})" for a in adds]
            effects.extend([f"(not ({d}))" for d in deletes])
            if len(effects) == 1:
                return effects[0]
            return f"(and {' '.join(effects)})"

        # 2. Build actions
        for skill_name in sorted(self.skills.keys()):
            sk_lower = skill_name.lower()

            incoming = [
                format_symbol_name(trans["source"], skill_name)
                for trans in transitions if trans["target"] == skill_name
            ]

            outgoing = [
                format_symbol_name(skill_name, trans["target"])
                for trans in transitions if trans["source"] == skill_name
            ]

            # Action 1: Starting from initial precondition
            adds_init = [f"goal_reached_{sk_lower}"] + outgoing
            dels_init = [f"pre_{sk_lower}"]
            pddl.extend([
                "",
                f"  (:action {sk_lower}_from_init",
                "    :parameters ()",
                f"    :precondition (and (pre_{sk_lower}))",
                f"    :effect {format_effect(adds_init, dels_init)}",
                "  )"
            ])

            # Action 2: Triggered by learned transition symbols
            for in_sym in incoming:
                adds_via = [f"goal_reached_{sk_lower}"] + outgoing
                dels_via = [in_sym]
                pddl.extend([
                    "",
                    f"  (:action {sk_lower}_via_{in_sym}",
                    "    :parameters ()",
                    f"    :precondition (and ({in_sym}))",
                    f"    :effect {format_effect(adds_via, dels_via)}",
                    "  )"
                ])

        pddl.append(")")
        return "\n".join(pddl)

    def generate_problem_pddl(self, current_vector, goal_skill, start_skill=None, 
                          problem_name="plan_problem", domain_name="robot_domain", mode='HARD') -> str:
        init_predicates = []

        if start_skill:
            # Enforce planning starting strictly from the designated skill
            init_predicates.append(f"(pre_{start_skill.lower()})")
        else:
            # Fallback: direct preconditions in s_0
            for name, predicates in self.skills.items():
                if predicates["precondition"] and self.preconditions_met(name, current_vector):
                    init_predicates.append(f"(pre_{name.lower()})")

        # Only include active transition symbols if not enforcing a strict starting skill
        if not start_skill:
            transitions = self.compute_transitions(mode=mode)
            for trans in transitions:
                if trans["symbol"] is not None and trans["symbol"].is_active(current_vector):
                    init_predicates.append(f"(sym_{trans['source'].lower()}_to_{trans['target'].lower()})")

        pddl = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "  (:init"
        ]
        for pred in sorted(set(init_predicates)):
            pddl.append(f"    {pred}")
        pddl.extend([
            "  )",
            "  (:goal",
            f"    (goal_reached_{goal_skill.lower()})",
            "  )",
            ")"
        ])
        return "\n".join(pddl)

    def _logger(self):
        if not self.log_dir:
            return

        # 1. Save all skill predicates
        for skill_name, predicates in self.skills.items():
            self.save_predicate(skill_name, predicates["precondition"], 'init')
            self.save_predicate(skill_name, predicates["effect"], 'eff')

        # 2. Clean old pass symbols if present
        for f in os.listdir(self.gas_dir):
            if f.endswith("_pass.json"):
                try:
                    os.remove(os.path.join(self.gas_dir, f))
                except OSError:
                    pass

        # 3. Compute transitions on the fly and save active transition pass predicates
        transitions = self.compute_transitions()
        for edge in transitions:
            symbol_name = f"sym_{edge['source']}_enables_{edge['target']}"
            self.save_predicate(symbol_name, edge["symbol"], 'pass')

        # 4. Generate and save trans_graph.json
        for mode in ('HARD', 'SOFT'):
            file_path = os.path.join(self.graphs_dir, f"trans_graph_{mode}.json")
            self.export_transition_graph_json(file_path=file_path, mode=mode)

    @staticmethod
    def _serialize_single_gng(gng_obj) -> dict:
        parameters = {
            "max_points": gng_obj.max_points,
            "base_radius": gng_obj.base_radius,
            "max_radius": gng_obj.max_radius,
            "learning_rate_b": gng_obj.eb,
            "learning_rate_n": gng_obj.en,
            "max_edge_age": gng_obj.max_age,
            "lambda_step": gng_obj.lambda_step,
            "alpha": gng_obj.alpha,
            "d": gng_obj.d,
        }
        edges_str_keys = {f"{u},{v}": age for (u, v), age in gng_obj.edges.items()}
        radii = [float(gng_obj._get_local_radius(i)) for i in range(len(gng_obj.nodes))]

        return {
            "parameters": parameters,
            "nodes": [n.tolist() for n in gng_obj.nodes],
            "edges": edges_str_keys,
            "local_radii": radii
        }

    @staticmethod
    def _deserialize_single_gng(gng_obj, data: dict):
        if not data:
            return

        params = data.get("parameters", {})
        gng_obj.max_points = params.get("max_points", gng_obj.max_points)
        gng_obj.base_radius = params.get("base_radius", gng_obj.base_radius)
        gng_obj.max_radius = params.get("max_radius", gng_obj.max_radius)
        gng_obj.eb = params.get("learning_rate_b", gng_obj.eb)
        gng_obj.en = params.get("learning_rate_n", gng_obj.en)
        gng_obj.max_age = params.get("max_edge_age", gng_obj.max_age)
        gng_obj.lambda_step = params.get("lambda_step", gng_obj.lambda_step)
        gng_obj.alpha = params.get("alpha", gng_obj.alpha)
        gng_obj.d = params.get("d", gng_obj.d)

        if "nodes" in data:
            gng_obj.nodes = [np.array(n, dtype=float) for n in data["nodes"]]
            gng_obj.errors = [0.0] * len(gng_obj.nodes)

        if "local_radii" in data:
            gng_obj.cached_radii = data["local_radii"]

        if "edges" in data:
            parsed_edges = {}
            for k, v in data["edges"].items():
                u, v_node = map(int, k.split(","))
                parsed_edges[(u, v_node)] = v
            gng_obj.edges = parsed_edges

            gng_obj.adj = {}
            for (u, v) in gng_obj.edges.keys():
                gng_obj.adj.setdefault(u, set()).add(v)
                gng_obj.adj.setdefault(v, set()).add(u)

    def save_predicate(self, skill_name: str, predicate: Predicate, pred_type: str):
        if not self.log_dir or predicate is None or pred_type not in ['init', 'eff', 'pass']:
            return

        os.makedirs(self.gas_dir, exist_ok=True)

        data = {
            "version": "3_val_logic",
            "gng_pos": self._serialize_single_gng(predicate.gng_pos),
            "gng_neg": self._serialize_single_gng(predicate.gng_neg)
        }

        file_path = os.path.join(self.gas_dir, f"{skill_name}_{pred_type}.json")
        temp_file = os.path.join(self.gas_dir, f"{skill_name}_{pred_type}.tmp")

        with open(temp_file, "w") as f:
            json.dump(data, f)

        for _ in range(5):
            try:
                os.replace(temp_file, file_path)
                break
            except (PermissionError, OSError):
                time.sleep(0.05)
        else:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    @classmethod
    def load_predicate(cls, file_path) -> Predicate:
        with open(file_path, "r") as f:
            data = json.load(f)

        if data.get("version") == "3_val_logic":
            pos_data = data.get("gng_pos", {})
            neg_data = data.get("gng_neg", {})
            params = pos_data.get("parameters", {})
        else:
            pos_data = data
            neg_data = None
            params = data.get("parameters", {})

        pred = Predicate(
            max_points=params.get("max_points", 85),
            dim=None,
            base_radius=params.get("base_radius", 0.15),
            max_radius=params.get("max_radius", 1.75),
            learning_rate_b=params.get("learning_rate_b", 0.2),
            learning_rate_n=params.get("learning_rate_n", 0.01),
            max_edge_age=params.get("max_edge_age", 24),
            lambda_step=params.get("lambda_step", 5),
            alpha=params.get("alpha", 0.7),
            d=params.get("d", 0.99)
        )

        cls._deserialize_single_gng(pred.gng_pos, pos_data)
        if neg_data:
            cls._deserialize_single_gng(pred.gng_neg, neg_data)

        return pred

    def load_brain(self, source_dir):
        """Loads node predicates directly from disk into the skills dictionary."""
        print("\t[Brain] Loading brain from files.")
        if source_dir is None or not os.path.isdir(source_dir):
            print("\t[Brain] Invalid source directory.")
            return

        src_gas_dir = os.path.join(source_dir, "PDDL", "GNG")
        if not os.path.exists(src_gas_dir):
            print(f"\t[Brain] Could not find GNG directory: {src_gas_dir}")
            return

        # Find all skills by scanning for *_init.json files
        skill_names = set()
        for filename in os.listdir(src_gas_dir):
            match = re.match(r"^(.*)_init\.json$", filename)
            if match:
                skill_names.add(match.group(1))

        for name in skill_names:
            pre_file = os.path.join(src_gas_dir, f"{name}_init.json")
            eff_file = os.path.join(src_gas_dir, f"{name}_eff.json")

            if os.path.exists(pre_file) and os.path.exists(eff_file):
                precondition = self.load_predicate(pre_file)
                effect = self.load_predicate(eff_file)
                self.add_node(name, precondition, effect)
                if self.state_dim is None and hasattr(precondition, "gng_pos") and precondition.gng_pos.nodes:
                    self.state_dim = len(precondition.gng_pos.nodes[0])
            else:
                print(f"\t[Brain] Incomplete predicates for skill '{name}', skipping.")