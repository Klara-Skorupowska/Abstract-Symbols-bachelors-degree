import os
import re
import time
import json
import numpy as np
from collections import Counter

from .predicate import Predicate
from .symbol_generator import SymbolGenerator


class BrainNetwork:
    def __init__(self, log_dir=None):
        self.state_dim = None
        self.skills = {}

        ### HYPERPARAMETERS ###
        self.precision = 0.9           # Acceptance threshold for overlap/equivalence
        self.n_test_points = 100    # number of random points in symbol space used for testing
        self.min_nodes = None       # minimum nr of nodes in a symbol; if None = as many as in predicate 
        self.span_tolerance = 0.1   # acceptable changes in a value (% of min-max span)

        self.max_nodes = 40

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
            precision=self.precision,
            min_nodes_in_symbol=self.min_nodes,
            span_tolerance=self.span_tolerance,
            n_test_points=self.n_test_points,
        )

        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, "PDDL")
            self.gas_dir = os.path.join(self.log_dir, "PDDL", "GNG")
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.gas_dir, exist_ok=True)
            self.symbol_stats_path = os.path.join(self.log_dir, "symbol_statistics.txt")

    def add_node(self, skill_name: str, precondition: Predicate, effect: Predicate):
        skill_name = skill_name.lower().strip()
        self.skills[skill_name] = {
            "precondition": precondition,
            "effect": effect
        }

    def create_node(self, skill_name: str):
        skill_name = skill_name.lower().strip()
        if skill_name not in self.skills:
            precondition = Predicate(
                self.max_nodes, self.state_dim, self.base_radius, self.max_radius,
                self.learning_rate_b, self.learning_rate_n, self.max_edge_age,
                self.lambda_step, self.alpha, self.d
            )
            effect = Predicate(
                self.max_nodes, self.state_dim, self.base_radius, self.max_radius,
                self.learning_rate_b, self.learning_rate_n, self.max_edge_age,
                self.lambda_step, self.alpha, self.d
            )
            self.add_node(skill_name, precondition, effect)

    def prepare_nodes(self, skillset):
        for skill_name in skillset:
            self.create_node(skill_name)

    def effects_met(self, skill_name: str, state_vector) -> bool:
        """
        Checks if all symbols in the skill's effect are currently active in state_vector.
        """
        skill_name = skill_name.lower().strip()
        if skill_name not in self.skills:
            return False

        symbols_map = self.symbol_gen.get_all_symbols()
        if not symbols_map:
            return False

        skill_map = self.symbol_gen.get_skill_map()
        required_symbols = skill_map.get(skill_name, {}).get("effect", set())

        if not required_symbols:
            return False

        return all(
            sym in symbols_map and symbols_map[sym].is_active(state_vector)
            for sym in required_symbols
        )

    def preconditions_met(self, skill_name: str, state_vector) -> bool:
        """
        Checks if all symbols in the skill's precondition are currently active in state_vector.
        """
        skill_name = skill_name.lower().strip()
        if skill_name not in self.skills:
            return False

        symbols_map = self.symbol_gen.get_all_symbols()
        if not symbols_map:
            return False

        skill_map = self.symbol_gen.get_skill_map()
        required_symbols = skill_map.get(skill_name, {}).get("precondition", set())

        if not required_symbols:
            return False

        return all(
            sym in symbols_map and symbols_map[sym].is_active(state_vector)
            for sym in required_symbols
        )

    def predicates_status(self, skill_name: str, state_vector) -> str:
        pre = self.preconditions_met(skill_name, state_vector)
        eff = self.effects_met(skill_name, state_vector)

        if pre and eff:
            return 'BOTH'
        if pre:
            return 'PRECONDITIONS'
        if eff:
            return 'EFFECTS'
        return 'FALSE'

    def get_nodes(self) -> list:
        return list(self.skills.keys())

    def update(self, buffer):
        if not buffer:
            return
        print("\t[Brain] Updating Predicates (GNG)")

        # 1. Vectorized single update of global bounds for all states in the buffer
        pre_states = [item[0] for item in buffer]
        post_states = [item[2] for item in buffer]
        all_states = np.vstack([*pre_states, *post_states])
        if self.state_dim is None:
            self.state_dim = all_states.shape[1]
        self.symbol_gen.update_bounds(all_states)

        # 2. Update GNG predicates
        updated_skills = set()
        for pre_vector_state, skill_name, post_vector_state in buffer:
            skill_name = skill_name.lower().strip()
            if skill_name not in self.skills:
                self.create_node(skill_name)

            self.skills[skill_name]["precondition"].update(pre_vector_state)
            self.skills[skill_name]["effect"].update(post_vector_state)
            updated_skills.add(skill_name)

        self.abstract_symbols(updated_skills=updated_skills)
        self.generate_and_save_domain()
        self._logger()

    def abstract_symbols(self, updated_skills=None) -> None:
        """Abstracts and aggregates symbols per skill, only for specified updated skills."""
        if updated_skills is None:
            updated_skills = set(self.skills.keys())
        
        for skill_name in updated_skills:
            if skill_name in self.skills:
                skill_name = skill_name.lower().strip()
                self.symbol_gen.generate_symbols(skill_name, self.skills[skill_name])
                self.symbol_gen.agregate_symbols(skill_name)

    def get_active_symbols(self, vector) -> list:
        symbols_map = self.symbol_gen.get_all_symbols()
        if not symbols_map:
            self.abstract_symbols()
            symbols_map = self.symbol_gen.get_all_symbols()
        if symbols_map:
            return [sym_name for sym_name, pred in symbols_map.items() if pred.is_active(vector)]
        return []

    def _get_skill_symbol_interactions(self, symbols_map: dict = None):
        skill_map = self.symbol_gen.get_skill_map()
        action_pre = {}
        action_eff = {}

        for skill_name, roles in skill_map.items():
            sk_lower = skill_name.lower()
            pre = set(roles.get("precondition", set()))
            eff = set(roles.get("effect", set()))
            if symbols_map is not None:
                pre = {s for s in pre if s in symbols_map}
                eff = {s for s in eff if s in symbols_map}
            action_pre[sk_lower] = pre
            action_eff[sk_lower] = eff

        return action_pre, action_eff

    def generate_and_save_domain(self, domain_name="robot_domain"):
        """Pre-generates domain.pddl inside the task directory right after brain update."""
        if not self.log_dir:
            return
        task_dir = os.path.join(self.pddl_dir, "task")
        os.makedirs(task_dir, exist_ok=True)
        domain_file = os.path.join(task_dir, "domain.pddl")

        domain_content = self.generate_domain_pddl(domain_name=domain_name)
        with open(domain_file, "w") as f:
            f.write(domain_content)

    def generate_domain_pddl(self, domain_name="robot_domain") -> str:
        """Generates PDDL domain string after merging identical symbols."""
        self.symbol_gen.merge_symbols()

        symbols_map = self.symbol_gen.get_all_symbols()
        skill_map = self.symbol_gen.get_skill_map()

        all_predicates = set(symbols_map.keys())
        for roles in skill_map.values():
            all_predicates.update(roles["precondition"])
            all_predicates.update(roles["effect"])

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
            roles = skill_map.get(skill_name, {"precondition": set(), "effect": set()})
            pre_conditions = set(roles["precondition"])
            eff_conditions = set(roles["effect"])

            adds = set(eff_conditions)
            dels = set(pre_conditions - eff_conditions)

            pre_body = f"(and {' '.join(f'({p})' for p in sorted(pre_conditions))})" if pre_conditions else "(and)"
            eff_tokens = [f"({a})" for a in sorted(adds)] + [f"(not ({d}))" for d in sorted(dels)]
            eff_body = f"(and {' '.join(eff_tokens)})" if eff_tokens else "(and)"

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
        """Generates problem.pddl using active symbols and goal effects from skill_map."""
        goal_skill = goal_skill.lower().strip()
        skill_map = self.symbol_gen.get_skill_map()
        goal_predicates = set(skill_map.get(goal_skill, {}).get("effect", set()))

        if not goal_predicates:
            return ""

        pddl = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "  (:init"
        ]
        for pred in sorted(active_symbols):
            pddl.append(f"    ({pred})")
        pddl.append("  )")

        pddl.append("  (:goal")
        goals_str = " ".join(f"({g})" for g in sorted(goal_predicates))
        pddl.append(f"    (and {goals_str})")
        pddl.append("  )")
        pddl.append(")")

        return "\n".join(pddl)

    def export_symbol_statistics(self, file_path=None):
        """Exports active dimensions, node counts, and skill_map interactions."""
        if not self.log_dir and file_path is None:
            return

        if file_path is None:
            file_path = os.path.join(self.log_dir, "symbol_statistics.txt")

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        symbols_map = self.symbol_gen.get_all_symbols()
        # Read the exact mapping used by domain.pddl
        skill_map = self.symbol_gen.get_skill_map()

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
                f"  {sym_name:<10} | dims: {len(pred.active_dimensions)} | nodes: {len(pred.nodes)}"
            )

        lines.append("-" * 70)
        lines.append("SKILL INTERACTIONS:")
        for skill_name in sorted(self.skills.keys()):
            roles = skill_map.get(skill_name, {"precondition": set(), "effect": set()})
            pre_syms = sorted(roles["precondition"])
            eff_syms = sorted(roles["effect"])
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
        Saves all symbols (sym_0, sym_1, ...) as Predicate JSON files.
        Deletes old symbol files from disk that are no longer part of the current active symbols.
        """
        if not self.log_dir:
            return

        os.makedirs(self.gas_dir, exist_ok=True)

        ## here we need unconnected skills
        symbols_map = self.symbol_gen.get_all_symbols() # returns {sym_0: gng}
        active_filenames = {f"{sym_name}.json" for sym_name in symbols_map.keys()}

        # 1. Scan and delete old symbol files not present in the current symbols_map
        for fname in os.listdir(self.gas_dir):
            if fname.endswith(".json") and re.match(r"^sym_\d+\.json$", fname):
                if fname not in active_filenames:
                    try:
                        os.remove(os.path.join(self.gas_dir, fname))
                    except OSError:
                        pass

        # 2. Save only the current, active symbols
        for sym_name, predicate in symbols_map.items():
            predicate.save_predicate(self.gas_dir, sym_name)

        # 3. Export symbol report reflecting only active symbols
        self.export_symbol_statistics(self.symbol_stats_path)

    def save_skill_map(self):
        if not self.log_dir:
            return
        map_path = os.path.join(self.gas_dir, "skill_map.json")
        serializable_map = {
            "skills": {
                sk: {role: list(syms) for role, syms in roles.items()}
                for sk, roles in self.symbol_gen.skill_map.items()
            },
            "bounds": {
                "dim_min": self.symbol_gen.dim_min.tolist() if self.symbol_gen.dim_min is not None else None,
                "dim_max": self.symbol_gen.dim_max.tolist() if self.symbol_gen.dim_max is not None else None,
                "state_dim": self.state_dim,
            }
        }
        with open(map_path, "w") as f:
            json.dump(serializable_map, f, indent=2)

    def _logger(self):
        if not self.log_dir:
            return

        for skill_name, predicates in self.skills.items():
            predicates["precondition"].save_predicate(self.gas_dir, f"{skill_name}_init")
            predicates["effect"].save_predicate(self.gas_dir, f"{skill_name}_eff")

        self.save_abstract_symbols()
        self.save_skill_map()

    def save_symbol_stats(self, filename="symbol_stats_log.txt") -> str:
        """
        Appends a raw line of summary statistics to a text file.
    
        Metrics included:
        - total_symbols: Total count of active symbols
        - total_skills: Total count of skills
        - min_dim: Minimum active dimension count and occurrence (val (count))
        - dominant_dim: Most frequent dimension count and occurrence (val (count))
        - max_dim: Maximum active dimension count and occurrence (val (count))
        - avg_pre_symbols: Preconditions symbol density (avg per skill)
        - avg_eff_symbols: Effects symbol density (avg per skill)
        - reusability_ratio: Average number of skills referencing each symbol
        """
        if not self.log_dir:
            target_path = filename
        else:
            target_path = os.path.join(self.log_dir, filename)

        symbols_map = self.symbol_gen.get_all_symbols() or {}
        skill_map = self.symbol_gen.get_skill_map() or {}

        total_symbols = len(symbols_map)
        total_skills = len(self.skills)

        # 4. Dimension distribution (Min, Dominant/Mode, Max) with counts
        if total_symbols > 0:
            dim_counts = [len(getattr(pred, "active_dimensions", [])) for pred in symbols_map.values()]
            counter = Counter(dim_counts)
        
            min_val = min(counter.keys())
            min_dim_str = f"{min_val} ({counter[min_val]})"
        
            max_val = max(counter.keys())
            max_dim_str = f"{max_val} ({counter[max_val]})"
        
            # Most common / dominant
            dominant_val, dominant_count = counter.most_common(1)[0]
            dominant_dim_str = f"{dominant_val} ({dominant_count})"
        else:
            min_dim_str = "0 (0)"
            dominant_dim_str = "0 (0)"
            max_dim_str = "0 (0)"

        # 5 & 6. Precondition and Effect Symbol Density (Average per skill)
        if total_skills > 0:
            pre_counts = [len(roles.get("precondition", set())) for roles in skill_map.values()]
            eff_counts = [len(roles.get("effect", set())) for roles in skill_map.values()]
            avg_pre_symbols = np.mean(pre_counts)
            avg_eff_symbols = np.mean(eff_counts)
        else:
            avg_pre_symbols = 0.0
            avg_eff_symbols = 0.0

        # 7. Reusability Ratio (Average number of skills referencing each symbol)
        if total_symbols > 0:
            symbol_usage = Counter()
            for roles in skill_map.values():
                for sym in set(roles.get("precondition", set())) | set(roles.get("effect", set())):
                    symbol_usage[sym] += 1
            reusability_ratio = sum(symbol_usage.values()) / total_symbols
        else:
            reusability_ratio = 0.0

        # Create CSV / TSV style raw string
        header = (
            "timestamp,"
            "total_symbols,"
            "total_skills,"
            "min_dims,"
            "dominant_dims,"
            "max_dims,"
            "avg_pre_symbols,"
            "avg_eff_symbols,"
            "reusability_ratio\n"
        )

        row = (
            f"{time.time():.2f},"
            f"{total_symbols},"
            f"{total_skills},"
            f"\"{min_dim_str}\","
            f"\"{dominant_dim_str}\","
            f"\"{max_dim_str}\","
            f"{avg_pre_symbols:.2f},"
            f"{avg_eff_symbols:.2f},"
            f"{reusability_ratio:.2f}\n"
        )

        # Write header if file does not exist or is empty
        file_exists = os.path.exists(target_path) and os.path.getsize(target_path) > 0
        with open(target_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write(header)
            f.write(row)

        return row.strip()

    def load_brain(self, source_dir, regenerate_symbols: bool = False):
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
                precondition = Predicate(self.max_nodes).load_predicate(pre_file)
                effect = Predicate(self.max_nodes).load_predicate(eff_file)
                self.add_node(name, precondition, effect)

                if self.state_dim is None and len(precondition.nodes) > 0:
                    self.state_dim = precondition.nodes.shape[1]
            else:
                print(f"\t[Brain] Incomplete predicates for skill '{name}', skipping.")

        # 1. Load saved sym_i predicates
        loaded_symbols = {}
        for filename in os.listdir(src_gas_dir):
            match = re.match(r"^(sym_\d+)\.json$", filename)
            if match:
                sym_name = match.group(1)
                sym_file = os.path.join(src_gas_dir, filename)
                loaded_symbols[sym_name] = Predicate(self.max_nodes).load_predicate(sym_file)

        # 2. Load symbol map and bounds
        map_path = os.path.join(src_gas_dir, "skill_map.json")
        if os.path.exists(map_path):
            with open(map_path, "r") as f:
                data = json.load(f)

            # Support both old format (dict of skills) and new format (with bounds)
            loaded_map = data.get("skills", data)
            for sk, roles in loaded_map.items():
                self.symbol_gen.skill_map[sk]["precondition"] = set(roles.get("precondition", []))
                self.symbol_gen.skill_map[sk]["effect"] = set(roles.get("effect", []))

            if "bounds" in data:
                b_min = data["bounds"].get("dim_min")
                b_max = data["bounds"].get("dim_max")
                if b_min is not None and b_max is not None:
                    self.symbol_gen.dim_min = np.asarray(b_min, dtype=np.float64)
                    self.symbol_gen.dim_max = np.asarray(b_max, dtype=np.float64)
                if data["bounds"].get("state_dim"):
                    self.state_dim = data["bounds"]["state_dim"]

        # If bounds were not in metadata, fallback to updating from predicate nodes
        if self.symbol_gen.dim_min is None:
            for skill in self.skills.values():
                if len(skill["precondition"].nodes) > 0:
                    self.symbol_gen.update_bounds(skill["precondition"].nodes)
                if len(skill["effect"].nodes) > 0:
                    self.symbol_gen.update_bounds(skill["effect"].nodes)

        # 3. Restore symbols or re-abstract cleanly
        if loaded_symbols and not regenerate_symbols:
            self.symbol_gen.symbols = loaded_symbols
            print(f"\t[Brain] Successfully restored {len(loaded_symbols)} symbols from disk.")
        else:
            print("\t[Brain] Re-abstracting symbols from scratch.")
            self.symbol_gen.symbols.clear()
            self.abstract_symbols()

        self.generate_and_save_domain()
        self.save_abstract_symbols()
        self._logger()