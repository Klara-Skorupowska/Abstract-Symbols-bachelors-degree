import os
import re
import time

from .predicate import Predicate
from .symbol_generator import SymbolGenerator


class BrainNetwork:
    def __init__(self, log_dir=None):
        self.state_dim = None
        self.skills = {}

        ### HYPERPARAMETERS ###
        self.min_overlap_ratio = 1.0
        self.min_nodes = 20        # changes no. symbols
        self.span_tolerance = 0.1  # changes no. dimentions in symbols

        self.max_points = 40
        self.base_radius = 0.1
        self.max_radius = 1.0
        self.learning_rate_b = 0.2
        self.learning_rate_n = 0.01
        self.max_edge_age = 15
        self.lambda_step = 5
        self.alpha = 0.7
        self.d = 0.99
        ### --------------- ###

        self.symbol_gen = SymbolGenerator(
            min_overlap_ratio=self.min_overlap_ratio,
            min_nodes_in_symbol = self.min_nodes,
            span_tolerance = self.span_tolerance
        )

        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, "PDDL")
            self.gas_dir = os.path.join(self.log_dir, "PDDL", "GNG")
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.gas_dir, exist_ok=True)
            self.symbol_stats_path = os.path.join(self.log_dir, "symbol_statistics.txt")

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

    def preconditions_met(self, skill_name: str, state_vector) -> bool:
        """
        Checks if all symbols in the skill's precondition are currently active in state_vector.
        """
        if skill_name not in self.skills:
            return False

        sk_lower = skill_name.lower()
        symbols_map = self.symbol_gen.symbols or self.abstract_symbols()
        action_pre, _ = self._get_skill_symbol_interactions(symbols_map)

        required_symbols = action_pre.get(sk_lower, set())

        # Filter out fallback predicates to isolate abstract symbols (sym_*)
        abstract_req_symbols = [
            s for s in required_symbols 
            if s in symbols_map
        ]

        #    verify that all of them are active for the current state_vector
        if abstract_req_symbols:
            return all(symbols_map[sym_name].is_active(state_vector) for sym_name in abstract_req_symbols)

        return False

    def preconditions_status(self, skill_name: str, state_vector) -> str:
        """Returns 'TRUE' if preconditions_met is satisfied, otherwise 'FALSE'."""
        return "TRUE" if self.preconditions_met(skill_name, state_vector) else "FALSE"

    def get_active_skills(self, vector) -> list:
        if self.state_dim is None:
            self.state_dim = len(vector)

        return [
            name for name, predicates in self.skills.items()
            if predicates["precondition"].is_active(vector)
        ]

    def get_nodes(self) -> list:
        return list(self.skills.keys())

    def update(self, buffer):
        if not buffer:
            return
        print("\t[Brain] Updating Predicates (GNG)")
        for pre_vector_state, skill_name, post_vector_state in buffer:
            if self.state_dim is None:
                self.state_dim = len(pre_vector_state)

            if skill_name not in self.skills:
                self.create_node(skill_name)

            self.symbol_gen.update_bounds([pre_vector_state, post_vector_state])
            self.skills[skill_name]["precondition"].update(pre_vector_state)
            self.skills[skill_name]["effect"].update(post_vector_state)

        self.abstract_symbols()
        self._logger()

    def abstract_symbols(self) -> dict:
        """Generates symbols for each skill and aggregates them into minimal sym_i predicates."""
        raw_symbols = []
        for skill_data in self.skills.values():
            raw_symbols.extend(self.symbol_gen.generate_symbols(skill_data))

        agregated = self.symbol_gen.agregate_symbols(raw_symbols)
        return self.symbol_gen.reduce_symbols(file_path = self.symbol_stats_path, symbols = agregated)

    def get_active_symbols(self, vector) -> list:
        """
        Returns active symbols (sym_0, sym_1, ...).
        """
        symbols_map = self.symbol_gen.symbols or self.abstract_symbols()
        if symbols_map:
            return [sym_name for sym_name, pred in symbols_map.items() if pred.is_active(vector)]

        return []

    def _get_skill_symbol_interactions(self, symbols_map: dict):
        """
        Specifies active symbols before and after execution of each action.
        """
        action_pre = {}
        action_eff = {}

        for skill_name, skill_data in self.skills.items():
            sk_lower = skill_name.lower()
            action_pre[sk_lower] = set()
            action_eff[sk_lower] = set()

            pre_pred = skill_data["precondition"]
            eff_pred = skill_data["effect"]
            pre_pts = pre_pred.nodes
            eff_pts = eff_pred.nodes

            for sym_name, sym_pred in symbols_map.items():
                if pre_pts:
                    active_cnt = sum(
                        1 for pt in pre_pts
                        if sym_pred.is_active(pt, vector_dims=pre_pred.active_dimensions)
                    )
                    if (active_cnt / len(pre_pts)) >= self.min_overlap_ratio:
                        action_pre[sk_lower].add(sym_name)

                if eff_pts:
                    active_cnt = sum(
                        1 for pt in eff_pts
                        if sym_pred.is_active(pt, vector_dims=eff_pred.active_dimensions)
                    )
                    if (active_cnt / len(eff_pts)) >= self.min_overlap_ratio:
                        action_eff[sk_lower].add(sym_name)

        return action_pre, action_eff

    def generate_domain_pddl(self, domain_name="robot_domain") -> str:
        """
        Generates a PDDL domain file strictly formatted for pyperplan.
        Ensures all possible goal/action predicates are declared in (:predicates ...).
        """
        symbols_map = self.symbol_gen.symbols or self.abstract_symbols()
        action_pre, action_eff = self._get_skill_symbol_interactions(symbols_map)

        all_predicates = set(symbols_map.keys())
        for preds in action_pre.values():
            all_predicates.update(preds)
        for preds in action_eff.values():
            all_predicates.update(preds)

        pddl = [
            f"(define (domain {domain_name})",
            "  (:requirements :strips)",
            "  (:predicates"
        ]
        for pred in sorted(all_predicates):
            pddl.append(f"    ({pred})")
        pddl.append("  )")

        for skill_name in sorted(self.skills.keys()):
            sk_lower = skill_name.lower()
            pre_conditions = action_pre.get(sk_lower, set())
            eff_conditions = action_eff.get(sk_lower, set())

            # Fallback if an action has no symbol interactions mapped yet
            if not pre_conditions:
                continue
            if not eff_conditions:
                continue

            adds = set(eff_conditions)
            dels = set(pre_conditions - eff_conditions)

            # Precondition: CNF compliant
            pre_body = f"(and {' '.join(f'({p})' for p in sorted(pre_conditions))})"

            # Effect: CNF compliant
            eff_tokens = [f"({a})" for a in sorted(adds)]
            eff_tokens.extend([f"(not ({d}))" for d in sorted(dels)])
            eff_body = f"(and {' '.join(eff_tokens)})"

            pddl.extend([
                "",
                f"  (:action {sk_lower}",
                "    :parameters ()",
                f"    :precondition {pre_body}",
                f"    :effect {eff_body}",
                "  )"
            ])

        pddl.append(")")
        return "\n".join(pddl)

    def generate_problem_pddl(self, active_symbols, goal_skill: str,
                              problem_name="plan_problem", domain_name="robot_domain") -> str:
        """
        Generates a PDDL problem file formatted for pyperplan.
        Guarantees that all init and goal predicates exist in the domain.
        """
        symbols_map = self.symbol_gen.symbols or self.abstract_symbols()
        action_pre, _ = self._get_skill_symbol_interactions(symbols_map)

        init_predicates = set(active_symbols)

        goal_lower = goal_skill.lower()
        goal_predicates = set(action_pre.get(goal_lower, set()))

        if not goal_predicates:
            return ""

        pddl = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "  (:init"
        ]
        for pred in sorted(init_predicates):
            pddl.append(f"    ({pred})")
        pddl.append("  )")

        pddl.append("  (:goal")
        goals_str = " ".join(f"({g})" for g in sorted(goal_predicates))
        pddl.append(f"    (and {goals_str})")
        pddl.append("  )")
        pddl.append(")")

        return "\n".join(pddl)

    def export_symbol_statistics(self, file_path=None):
        """Exports active dimensions and node counts of aggregated symbols and skills."""
        if not self.log_dir and file_path is None:
            return

        if file_path is None:
            file_path = os.path.join(self.log_dir, "symbol_statistics.txt")

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        symbols_map = self.symbol_gen.symbols
        action_pre, action_eff = self._get_skill_symbol_interactions(symbols_map)

        lines = [
            "=" * 70,
            "SYMBOL & SKILL ABSTRACTION STATISTICS",
            "=" * 70,
            f"State Dimension: {self.state_dim}",
            f"Total Aggregated Symbols: {len(symbols_map)}",
            "-" * 70,
            "SYMBOLS:"
        ]

        for sym_name in sorted(symbols_map.keys()):
            pred = symbols_map[sym_name]
            lines.append(
                f"  {sym_name:<10} | dims: {pred.active_dimensions} | nodes: {len(pred.nodes)}"
            )

        lines.append("-" * 70)
        lines.append("SKILL INTERACTIONS:")
        for skill_name in sorted(self.skills.keys()):
            sk_lower = skill_name.lower()
            pre_syms = sorted(action_pre.get(sk_lower, set()))
            eff_syms = sorted(action_eff.get(sk_lower, set()))
            lines.append(f"Skill: {skill_name}")
            lines.append(f"  - Precondition Symbols : {pre_syms}")
            lines.append(f"  - Effect Symbols       : {eff_syms}")
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

    def save_abstract_symbols(self):
        """
        Saves all aggregated symbols (sym_0, sym_1, ...) as Predicate JSON files.
        Deletes old symbol files from disk that are no longer part of the current active symbols.
        """
        if not self.log_dir:
            return

        os.makedirs(self.gas_dir, exist_ok=True)

        symbols_map = self.symbol_gen.symbols or self.abstract_symbols()
        active_filenames = {f"{sym_name}.json" for sym_name in symbols_map.keys()}

        # 1. Scan and delete old symbol files not present in the current symbols_map
        for fname in os.listdir(self.gas_dir):
            is_symbol_file = (
                re.match(r"^sym_\d+\.(json|tmp)$", fname) is not None
            )
            if is_symbol_file and fname not in active_filenames:
                try:
                    os.remove(os.path.join(self.gas_dir, fname))
                except OSError:
                    pass

        # 2. Save only the current, active symbols
        for sym_name, predicate in symbols_map.items():
            predicate.save_predicate(self.gas_dir, sym_name)

        # 3. Export symbol report reflecting only active symbols
        self.export_symbol_statistics(self.symbol_stats_path)

    def _logger(self):
        if not self.log_dir:
            return

        for skill_name, predicates in self.skills.items():
            predicates["precondition"].save_predicate(self.gas_dir, f"{skill_name}_init")
            predicates["effect"].save_predicate(self.gas_dir, f"{skill_name}_eff")

        self.save_abstract_symbols()

    def load_brain(self, source_dir):
        """Loads skill predicates and abstract symbols from disk using Predicate.load_predicate."""
        print("\t[Brain] Loading brain from files.")
        if source_dir is None or not os.path.isdir(source_dir):
            print("\t[Brain] Invalid source directory.")
            return

        src_gas_dir = os.path.join(source_dir, "PDDL", "GNG")
        if not os.path.exists(src_gas_dir):
            print(f"\t[Brain] Could not find GNG directory: {src_gas_dir}")
            return

        skill_names = set()
        for filename in os.listdir(src_gas_dir):
            match = re.match(r"^(.*)_init\.json$", filename)
            if match:
                skill_names.add(match.group(1))

        for name in skill_names:
            pre_file = os.path.join(src_gas_dir, f"{name}_init.json")
            eff_file = os.path.join(src_gas_dir, f"{name}_eff.json")

            if os.path.exists(pre_file) and os.path.exists(eff_file):
                precondition = Predicate(self.max_points).load_predicate(pre_file)
                effect = Predicate(self.max_points).load_predicate(eff_file)
                self.add_node(name, precondition, effect)

                if self.state_dim is None and precondition.nodes:
                    self.state_dim = len(precondition.nodes[0])
                if precondition.nodes:
                    self.symbol_gen.update_bounds(precondition.nodes)
                if effect.nodes:
                    self.symbol_gen.update_bounds(effect.nodes)
            else:
                print(f"\t[Brain] Incomplete predicates for skill '{name}', skipping.")

        # Load saved sym_i predicates if present, otherwise regenerate
        loaded_symbols = {}
        for filename in os.listdir(src_gas_dir):
            match = re.match(r"^(sym_\d+)\.json$", filename)
            if match:
                sym_name = match.group(1) 
                sym_file = os.path.join(src_gas_dir, filename)
                loaded_symbols[sym_name] = Predicate(self.max_points).load_predicate(sym_file)


        if self.skills:
            self.abstract_symbols()
        elif self.symbol_gen.symbols is None and loaded_symbols:
            self.symbol_gen.symbols = loaded_symbols

        self._logger()