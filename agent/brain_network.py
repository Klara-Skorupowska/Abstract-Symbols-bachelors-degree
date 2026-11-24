from __future__ import annotations
import os
import re
import json
from pathlib import Path
import numpy as np

from .predicate import Predicate
from .symbol_generator import SymbolGenerator
from supervisor.loggers import SymbolStatsLogger


class BrainNetwork:
    """Neural-symbolic representation engine using Growing Neural Gas (GNG) and PDDL predicate abstraction."""

    ### HYPERPARAMETERS ### 
    DEFAULT_HYPERPARAMETERS: dict[str, object] = {
        "precision": 0.80,         # Acceptance threshold for overlap/equivalence
        "n_test_points": 500,     # Number of random points in symbol space for testing
        "min_nodes": 2,            # Minimum nodes in symbol; None defaults to predicate count
        "max_nodes": 15,
        "base_radius": 0.25,
        "max_radius": 4.0,        
        "learning_rate_b": 0.25,
        "learning_rate_n": 0.01,
        "max_edge_age": 25,
        "lambda_step": 10,
        "alpha": 0.5,
        "d": 0.995,
        "radii_factor": 0.85,
    }

    def __init__(self, log_dir: str | Path | None = None, brain_params: dict | None = None):
        self.state_dim: int | None = None
        self.skills: dict[str, dict[str, Predicate]] = {}

        params = dict(self.DEFAULT_HYPERPARAMETERS)
        if brain_params:
            unknown = set(brain_params.keys()) - set(self.DEFAULT_HYPERPARAMETERS.keys())
            if unknown:
                raise ValueError(f"Unknown BrainNetwork hyperparameters: {sorted(unknown)}")
            params.update(brain_params)

        ### HYPERPARAMETERS ###
        self.precision: float = params["precision"]
        self.n_test_points: int = params["n_test_points"]
        self.min_nodes: int | None = params["min_nodes"]

        self.max_nodes: int = params["max_nodes"]
        self.base_radius: float = params["base_radius"]
        self.max_radius: float = params["max_radius"]
        self.learning_rate_b: float = params["learning_rate_b"]
        self.learning_rate_n: float = params["learning_rate_n"]
        self.max_edge_age: int = params["max_edge_age"]
        self.lambda_step: int = params["lambda_step"]
        self.alpha: float = params["alpha"]
        self.d: float = params["d"]
        self.radii_factor: float = params["radii_factor"]
        ### ---------------- ###

        self.symbol_gen = SymbolGenerator(
            precision=self.precision,
            min_nodes_in_symbol=self.min_nodes,
            n_test_points=self.n_test_points,
        )

        self.log_dir: str | None = None
        self.pddl_dir: str | None = None
        self.task_dir: str | None = None
        self.gas_dir: str | None = None
        self.logger: SymbolStatsLogger | None = None

        if log_dir is not None:
            self.set_output_dir(log_dir)

    def get_hyperparameters(self) -> dict[str, object]:
        """Returns the current brain hyperparameter values."""
        return {key: getattr(self, key) for key in self.DEFAULT_HYPERPARAMETERS}

    def set_output_dir(self, target_dir: str | Path | None):
        """Re-targets all brain outputs to an active epoch directory."""
        if target_dir is None:
            self.log_dir = None
            self.pddl_dir = None
            self.task_dir = None
            self.gas_dir = None
            self.logger = None
            return

        self.log_dir = str(target_dir)
        self.pddl_dir = os.path.join(self.log_dir, "PDDL")
        self.task_dir = os.path.join(self.pddl_dir, "task")
        self.gas_dir = os.path.join(self.pddl_dir, "GNG")
        os.makedirs(self.task_dir, exist_ok=True)
        os.makedirs(self.gas_dir, exist_ok=True)
        self.logger = SymbolStatsLogger(self.log_dir)

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
                self.lambda_step, self.alpha, self.d, self.radii_factor
            )
            effect = Predicate(
                self.max_nodes, self.state_dim, self.base_radius, self.max_radius,
                self.learning_rate_b, self.learning_rate_n, self.max_edge_age,
                self.lambda_step, self.alpha, self.d, self.radii_factor
            )
            self.add_node(skill_name, precondition, effect)

    def prepare_nodes(self, skillset):
        for skill_name in skillset:
            self.create_node(skill_name)

    def effects_met(self, skill_name: str, state_vector: np.ndarray) -> bool:
        skill_name = skill_name.lower().strip()
        if skill_name not in self.skills:
            return False

        symbols_map = self.symbol_gen.get_all_symbols()
        if not symbols_map:
            return False

        skill_map = self.symbol_gen.get_skill_map()
        effect_symbols = skill_map.get(skill_name, {}).get("effect", set())
        if not effect_symbols:
            return False

        # Partitioned symbols are mutually disjoint pieces covering the original region
        return any(
            sym in symbols_map and symbols_map[sym].is_active(state_vector)
            for sym in effect_symbols
        )

    def preconditions_met(self, skill_name: str, state_vector: np.ndarray) -> bool:
        skill_name = skill_name.lower().strip()
        if skill_name not in self.skills:
            return False

        symbols_map = self.symbol_gen.get_all_symbols()
        if not symbols_map:
            return False

        skill_map = self.symbol_gen.get_skill_map()
        precondition_symbols = skill_map.get(skill_name, {}).get("precondition", set())
        if not precondition_symbols:
            return False

        # Partitioned symbols are mutually disjoint pieces covering the original region
        return any(
            sym in symbols_map and symbols_map[sym].is_active(state_vector)
            for sym in precondition_symbols
        )

    def predicates_status(self, skill_name: str, state_vector: np.ndarray) -> str:
        pre = self.preconditions_met(skill_name, state_vector)
        eff = self.effects_met(skill_name, state_vector)

        if pre and eff:
            return 'BOTH'
        if pre:
            return 'PRECONDITIONS'
        if eff:
            return 'EFFECTS'
        return 'FALSE'

    def get_nodes(self) -> list[str]:
        return list(self.skills.keys())

    def update(self, buffer: list):
        if not buffer:
            return
        print("\t[Brain] Updating Predicates (GNG)")

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

    def abstract_symbols(self, updated_skills: set[str] | None = None) -> None:
        if updated_skills is None:
            updated_skills = set(self.skills.keys())

        normalized_skills = [
            sk.lower().strip() for sk in updated_skills if sk.lower().strip() in self.skills
        ]

        # Step 1: Generate Q_init and Q_eff for each updated skill (Eq. 4 & 5)
        for skill_name in normalized_skills:
            self.symbol_gen.generate_symbols(skill_name, self.skills[skill_name])

        # Step 2: Partition all symbols into mutually disjoint Voronoi sets (Eq. 6)
        if normalized_skills:
            self.symbol_gen.partition_symbols()

    def get_active_symbols(self, vector: np.ndarray) -> list[str]:
        symbols_map = self.symbol_gen.get_all_symbols()
        if not symbols_map:
            self.abstract_symbols()
            symbols_map = self.symbol_gen.get_all_symbols()
        if symbols_map:
            return [sym_name for sym_name, pred in symbols_map.items() if pred.is_active(vector)]
        return []

    def generate_and_save_domain(self, domain_name: str = "robot_domain") -> str:
        """Writes domain.pddl inside PDDL/task/ of the epoch directory."""
        if not self.task_dir:
            return ""
        os.makedirs(self.task_dir, exist_ok=True)
        domain_file = os.path.join(self.task_dir, "domain.pddl")
        domain_content = self.generate_domain_pddl(domain_name=domain_name)
        with open(domain_file, "w", encoding="utf-8") as f:
            f.write(domain_content)
        return domain_file

    def generate_domain_pddl(self, domain_name: str = "robot_domain") -> str:

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

    def generate_problem_pddl(
        self, active_symbols: list[str], goal_skill: str,
        problem_name: str = "plan_problem", domain_name: str = "robot_domain"
    ) -> str:
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

    def save_abstract_symbols(self):
        if not self.gas_dir:
            return

        os.makedirs(self.gas_dir, exist_ok=True)
        symbols_map = self.symbol_gen.get_all_symbols()
        active_filenames = {f"{sym_name}.json" for sym_name in symbols_map.keys()}

        for fname in os.listdir(self.gas_dir):
            if fname.endswith(".json") and re.match(r"^sym_\d+\.json$", fname):
                if fname not in active_filenames:
                    try:
                        os.remove(os.path.join(self.gas_dir, fname))
                    except OSError:
                        pass

        for sym_name, predicate in symbols_map.items():
            predicate.save_predicate(self.gas_dir, sym_name)

        if self.logger:
            self.logger.export_statistics(
                self.state_dim,
                self.symbol_gen.get_all_symbols(),
                self.symbol_gen.get_skill_map(),
                self.skills
            )

    def save_skill_map(self):
        if not self.gas_dir:
            return
        map_path = os.path.join(self.gas_dir, "skill_map.json")
        serializable_map = {
            "skills": {
                sk: {role: list(syms) for role, syms in roles.items()}
                for sk, roles in self.symbol_gen.skill_map.items()
            }
        }
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(serializable_map, f, indent=2)

    def _logger(self):
        if not self.log_dir or not self.gas_dir:
            return

        for skill_name, predicates in self.skills.items():
            predicates["precondition"].save_predicate(self.gas_dir, f"{skill_name}_init")
            predicates["effect"].save_predicate(self.gas_dir, f"{skill_name}_eff")

        self.save_abstract_symbols()
        self.save_skill_map()

    def save_symbol_stats(self, filename: str = "symbol_stats_log.txt") -> str:
        if self.logger:
            return self.logger.log_continuous_stats(
                self.symbol_gen.get_all_symbols() or {},
                self.symbol_gen.get_skill_map() or {},
                len(self.skills),
                filename=filename
            )
        return ""

    def load_brain(self, source_epoch_dir: str | Path | None, regenerate_symbols: bool = False):
        """Loads skill predicates and abstract symbols directly from source epoch PDDL/GNG directory."""
        if source_epoch_dir is None:
            return

        source_path = Path(source_epoch_dir)
        print(f"\t[Brain] Loading brain state from epoch: {source_path}")

        src_gas_dir = source_path / "PDDL" / "GNG"
        if not src_gas_dir.exists():
            print(f"\t[Brain] Warning: GNG directory not found at {src_gas_dir}")
            return

        # Fallback defaults in case older JSON files omit some GNG parameters
        default_pred_kwargs = {
            "max_points": getattr(self, "max_nodes", 20),
            "dim": getattr(self, "state_dim", None),
        }

        skill_names = sorted({
            match.group(1)
            for filename in os.listdir(src_gas_dir)
            if (match := re.match(r"^(.*)_init\.json$", filename))
        })

        for name in skill_names:
            pre_file = src_gas_dir / f"{name}_init.json"
            eff_file = src_gas_dir / f"{name}_eff.json"

            if pre_file.exists() and eff_file.exists():
                precondition = Predicate.from_file(pre_file, **default_pred_kwargs)
                effect = Predicate.from_file(eff_file, **default_pred_kwargs)
                self.add_node(name, precondition, effect)

                if self.state_dim is None:
                    inferred_dim = (
                        precondition.dim
                        or effect.dim
                        or (precondition.nodes.shape[1] if len(precondition.nodes) > 0 else None)
                        or (effect.nodes.shape[1] if len(effect.nodes) > 0 else None)
                    )
                    if inferred_dim is not None:
                        self.state_dim = int(inferred_dim)
                        default_pred_kwargs["dim"] = self.state_dim
            else:
                print(f"\t[Brain] Incomplete predicates for skill '{name}', skipping.")

        loaded_symbols = {}
        for filename in sorted(os.listdir(src_gas_dir)):
            match = re.match(r"^(sym_\d+)\.json$", filename)
            if match:
                sym_name = match.group(1)
                sym_file = src_gas_dir / filename
                loaded_symbols[sym_name] = Predicate.from_file(sym_file, **default_pred_kwargs)

        map_path = src_gas_dir / "skill_map.json"
        if map_path.exists():
            with map_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            loaded_map = data.get("skills", data)
            for sk, roles in loaded_map.items():
                self.symbol_gen.skill_map[sk]["precondition"] = set(roles.get("precondition", []))
                self.symbol_gen.skill_map[sk]["effect"] = set(roles.get("effect", []))

        if loaded_symbols and not regenerate_symbols:
            self.symbol_gen.symbols = loaded_symbols
            print(f"\t[Brain] Successfully restored {len(loaded_symbols)} symbols from disk.")
        else:
            print("\t[Brain] Re-abstracting symbols from scratch.")
            self.symbol_gen.symbols.clear()
            self.abstract_symbols()

        # If an active epoch output directory is already configured, write state there
        if self.log_dir:
            self.generate_and_save_domain()
            self.save_abstract_symbols()
            self._logger()
