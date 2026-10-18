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
        self.min_overlap_ratio = 0.75       # how much of the symbol's nodes have to be active to count
        self.allowed_variancy = 0.02        # for finding invariant dimentions

        self.max_points = 20
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
        """Returns 'TRUE', 'FALSE', 'UNKNOWN' or 'CONFLICT' for the skill's precondition."""
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

        self.update_allowed_variance()
        self._logger()

    @staticmethod
    def _otsu_variance_threshold(variances: np.ndarray, fallback: float = 0.1) -> float:
        """Finds the variance threshold minimizing intra-class variance between invariant vs changing dims."""
        # Filter out negligible/numerical zeros
        valid_variances = variances[variances > 1e-7]
        if len(valid_variances) < 2:
            return fallback

        sorted_vars = np.sort(valid_variances)
        n = len(sorted_vars)
        best_thresh = sorted_vars[0]
        best_intra_variance = float("inf")

        for i in range(1, n):
            w0 = i / n
            w1 = 1.0 - w0
            c0, c1 = sorted_vars[:i], sorted_vars[i:]
            
            # Weighted sum of intra-class variances
            intra_variance = w0 * np.var(c0) + w1 * np.var(c1)
            if intra_variance < best_intra_variance:
                best_intra_variance = intra_variance
                best_thresh = sorted_vars[i]

        return float(best_thresh)

    def update_allowed_variance(self):
        """
        Collects dimension variances across all learned skill predicates (precondition & effect)
        and updates self.allowed_variancy dynamically using Otsu's method.
        """
        collected_variances = []

        for skill in self.skills.values():
            for phase in ("precondition", "effect"):
                pred = skill[phase]
                if pred.nodes and len(pred.nodes) >= 2:
                    nodes_mat = np.asarray(pred.nodes, dtype=float)
                    # Variance along each dimension for this predicate
                    dim_variances = np.var(nodes_mat, axis=0)
                    collected_variances.extend(dim_variances.tolist())

        if collected_variances:
            var_array = np.array(collected_variances, dtype=float)
            self.allowed_variancy = self._otsu_variance_threshold(
                var_array, fallback=self.allowed_variancy
            )

    def invariant_mask(self, predicate: Predicate) -> np.ndarray:
        """
        Returns a boolean mask (np.ndarray of bools), where True indicates an invariant dimension.
        Considers only those nodes belonging to the 'true' GNG (gng_pos).
        """
        if not predicate or not predicate.nodes:
            if self.state_dim is not None:
                return np.zeros(self.state_dim, dtype=bool)
            return np.array([], dtype=bool)

        nodes_mat = np.asarray(predicate.nodes, dtype=float)
        if len(nodes_mat) < 2:
            return np.ones(nodes_mat.shape[1], dtype=bool)

        variances = np.var(nodes_mat, axis=0)
        return variances < self.allowed_variancy
    
    def action_mask(self, skill_name: str) -> np.ndarray:
        """
        It identifies 'variants' – a mask of dimensions that have changed between the initial condition (precondition) and the final condition (effect) for dimensions that are otherwise invariants.
        """
        skill = self.skills.get(skill_name)
        if not skill:
            return np.zeros(self.state_dim or 0, dtype=bool)

        pre_pred = skill["precondition"]
        eff_pred = skill["effect"]

        if not pre_pred.nodes or not eff_pred.nodes:
            return np.zeros(self.state_dim or 0, dtype=bool)

        mask_pre = self.invariant_mask(pre_pred)
        mask_eff = self.invariant_mask(eff_pred)
        candidate_dims = mask_pre | mask_eff

        mean_pre = np.mean(pre_pred.nodes, axis=0)
        mean_eff = np.mean(eff_pred.nodes, axis=0)

        # A change in a dimension value that exceeds the variance tolerance threshold
        diff = np.abs(mean_pre - mean_eff)
        threshold = np.sqrt(self.allowed_variancy)

        return candidate_dims & (diff >= threshold)

    def group_symbols(self, symbols: dict) -> dict:
        """
        It verifies the coverage relationships between symbols and reduces them to independent elements.
        Input and output structure: {skill_name: {'precondition': [gngs], 'effect': [gngs]}}
        """
        grouped = {}
        for skill_name, phases in symbols.items():
            grouped[skill_name] = {}
            for phase, pred_list in phases.items():
                if len(pred_list) <= 1:
                    grouped[skill_name][phase] = list(pred_list)
                    continue

                reduced_preds = []
                for i, sym_a in enumerate(pred_list):
                    others = [sym for j, sym in enumerate(pred_list) if i != j]
                    # If the symbol a is entirely covered by the union of the remaining symbols, we replace it
                    if others and sym_a.is_covered_by(others):
                        continue
                    reduced_preds.append(sym_a)

                grouped[skill_name][phase] = reduced_preds if reduced_preds else pred_list
        return grouped

    def abstract_symbols(self)-> dict:
        """
        1) Identifies invariants for `init` and `eff`.
        2) Checks the dimensions of actions that undergo changes.
        3) Groups the data and creates a map: `{skill_name: {'precondition': [gngs], 'effect': [gngs]}}`.
        """
        raw_symbols = {}
        for skill_name in self.skills.keys():
            raw_symbols[skill_name] = {
                "precondition": [self.skills[skill_name]["precondition"]],
                "effect": [self.skills[skill_name]["effect"]]
            }

        return self.group_symbols(raw_symbols)

    def export_dimension_statistics(self, file_path=None):
        """
        Exports dimension invariant and transition analysis to a formatted text file in English.
        Covers invariant dimensions in precondition, invariant dimensions in effect,
        and dimensions that changed during action execution.
        """
        if not self.log_dir and file_path is None:
            return

        if file_path is None:
            file_path = os.path.join(self.log_dir, "dimension_statistics.txt")

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        lines = [
            "=" * 70,
            "SKILL DIMENSION INVARIANT & TRANSITION STATISTICS",
            "=" * 70,
            f"State Dimension: {self.state_dim}",
            f"Allowed Variance Threshold: {self.allowed_variancy}",
            "-" * 70,
            ""
        ]

        for skill_name in sorted(self.skills.keys()):
            skill = self.skills[skill_name]
            pre_pred = skill["precondition"]
            eff_pred = skill["effect"]

            # Compute invariant masks (boolean arrays)
            pre_inv_mask = self.invariant_mask(pre_pred)
            eff_inv_mask = self.invariant_mask(eff_pred)
            act_changed_mask = self.action_mask(skill_name)

            # Convert boolean masks to list of dimension indices
            pre_invariant_dims = np.where(pre_inv_mask)[0].tolist()
            eff_invariant_dims = np.where(eff_inv_mask)[0].tolist()
            changed_dims = np.where(act_changed_mask)[0].tolist()

            lines.append(f"Skill: {skill_name}")
            lines.append(f"  - Precondition Invariant Dimensions : {pre_invariant_dims}")
            lines.append(f"  - Effect Invariant Dimensions       : {eff_invariant_dims}")
            lines.append(f"  - Changed Dimensions (Action Mask)  : {changed_dims}")
            lines.append("-" * 70)

        temp_file = f"{file_path}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

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

    def save_abstract_symbols(self):
        """
        Saves all reduced abstract symbols to JSON files and exports
        dimension statistics to a text report file.
        """
        if not self.log_dir:
            return

        os.makedirs(self.gas_dir, exist_ok=True)

        # 1. Clean previous abstract symbol files
        for fname in os.listdir(self.gas_dir):
            if fname.endswith("_abs.json"):
                try:
                    os.remove(os.path.join(self.gas_dir, fname))
                except OSError:
                    pass

        # 2. Extract and serialize reduced abstract symbols
        abstract_map = self.abstract_symbols()
        for skill_name, phases in abstract_map.items():
            for phase, pred_list in phases.items():
                for idx, predicate in enumerate(pred_list):
                    suffix = f"_{idx}" if len(pred_list) > 1 else ""
                    symbol_identifier = f"{skill_name}_{phase}{suffix}"
                    self.save_predicate(symbol_identifier, predicate, pred_type='abs')

        # 3. Export dimension statistics report
        self.export_dimension_statistics()

    def _get_active_abstract_symbols(self, vector, abstract_map: dict) -> set:
        """Returns a set of active symbols in the format <skill_name>_<precondition/effect> for a given state vector."""
        active_set = set()
        for sk_name, phases in abstract_map.items():
            for phase in ("precondition", "effect"):
                sym_name = f"{sk_name.lower()}_{phase}"
                for pred in phases.get(phase, []):
                    if pred.is_active(vector):
                        active_set.add(sym_name)
                        break
        return active_set

    def _get_skill_abstract_interactions(self, abstract_map: dict, min_ratio=None):
        """
        It specifies which abstract symbols are active in the states before and after the execution of each action.
        """
        if min_ratio is None:
            min_ratio = self.min_overlap_ratio

        action_pre = {}
        action_eff = {}

        for skill_name, skill_data in self.skills.items():
            sk_lower = skill_name.lower()
            action_pre[sk_lower] = set()
            action_eff[sk_lower] = set()

            # Custom action symbols are always active
            action_pre[sk_lower].add(f"{sk_lower}_precondition")
            action_eff[sk_lower].add(f"{sk_lower}_effect")

            pre_pts = skill_data["precondition"].nodes
            eff_pts = skill_data["effect"].nodes

            # Checking for correlation with symbols of other skills
            for other_name, phases in abstract_map.items():
                oth_lower = other_name.lower()
                for phase in ("precondition", "effect"):
                    sym_label = f"{oth_lower}_{phase}"
                    preds = phases.get(phase, [])
                    if not preds:
                        continue

                    # Precondition
                    if pre_pts:
                        active_cnt = sum(
                            1 for pt in pre_pts if any(p.is_acceptable(pt) for p in preds)
                        )
                        if (active_cnt / len(pre_pts)) >= min_ratio:
                            action_pre[sk_lower].add(sym_label)

                    # Effect
                    if eff_pts:
                        active_cnt = sum(
                            1 for pt in eff_pts if any(p.is_acceptable(pt) for p in preds)
                        )
                        if (active_cnt / len(eff_pts)) >= min_ratio:
                            action_eff[sk_lower].add(sym_label)

        return action_pre, action_eff

    def generate_domain_pddl(self, domain_name="robot_domain") -> str:
        """
        Generates a PDDL domain based on abstract symbols: <skill_name>_<precondition/effect>.
        """
        pddl = [
            f"(define (domain {domain_name})",
            "  (:requirements :strips)",
            "  (:predicates"
        ]

        abstract_map = self.abstract_symbols()
        action_pre, action_eff = self._get_skill_abstract_interactions(abstract_map)

        all_predicates = set()
        for preds in action_pre.values():
            all_predicates.update(preds)
        for preds in action_eff.values():
            all_predicates.update(preds)

        for pred in sorted(all_predicates):
            pddl.append(f"    ({pred})")
        pddl.append("  )")

        def format_effect(adds, deletes):
            effects = [f"({a})" for a in sorted(adds)]
            effects.extend([f"(not ({d}))" for d in sorted(deletes)])
            if len(effects) == 1:
                return effects[0]
            return f"(and {' '.join(effects)})"

        for skill_name in sorted(self.skills.keys()):
            sk_lower = skill_name.lower()
            pre_conditions = action_pre.get(sk_lower, set())
            eff_conditions = action_eff.get(sk_lower, set())

            # adds i deletes
            adds = eff_conditions
            dels = {f"{sk_lower}_precondition"} | (pre_conditions - eff_conditions)

            pre_str = " ".join(f"({p})" for p in sorted(pre_conditions))

            pddl.extend([
                "",
                f"  (:action {sk_lower}",
                "    :parameters ()",
                f"    :precondition (and {pre_str})",
                f"    :effect {format_effect(adds, dels)}",
                "  )"
            ])

        pddl.append(")")
        return "\n".join(pddl)

    def generate_problem_pddl(self, current_vector, goal_skill: str, start_skill: str = None, 
                              problem_name="plan_problem", domain_name="robot_domain") -> str:
        """
        Generates a PDDL problem. The goal is a conjunction of symbols that are active before the execution of `goal_skill`.
        """
        abstract_map = self.abstract_symbols()
        action_pre, _ = self._get_skill_abstract_interactions(abstract_map)

        # 1.initial state (:init)
        init_predicates = set()
        if start_skill:
            start_lower = start_skill.lower()
            init_predicates.update(action_pre.get(start_lower, {f"{start_lower}_precondition"}))
        else:
            init_predicates = self._get_active_abstract_symbols(current_vector, abstract_map)

        # 2. (:goal) - conjunction of active symbols before execution of goal_skill
        goal_lower = goal_skill.lower()
        goal_predicates = action_pre.get(goal_lower, {f"{goal_lower}_precondition"})

        pddl = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "  (:init"
        ]
        for pred in sorted(init_predicates):
            pddl.append(f"    ({pred})")
        pddl.append("  )")

        pddl.append("  (:goal")
        if len(goal_predicates) > 1:
            goals_str = " ".join(f"({g})" for g in sorted(goal_predicates))
            pddl.append(f"    (and {goals_str})")
        elif len(goal_predicates) == 1:
            pddl.append(f"    ({next(iter(goal_predicates))})")
        else:
            pddl.append(f"    ({goal_lower}_precondition)")
        pddl.append("  )")
        pddl.append(")")

        return "\n".join(pddl)


    def _logger(self):
        if not self.log_dir:
            return

        # Save all skill predicates
        for skill_name, predicates in self.skills.items():
            self.save_predicate(skill_name, predicates["precondition"], 'init')
            self.save_predicate(skill_name, predicates["effect"], 'eff')

        # Save abstract symbols:
        self.save_abstract_symbols()

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
        if not self.log_dir or predicate is None or pred_type not in ['init', 'eff', 'pass', 'abs']:
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