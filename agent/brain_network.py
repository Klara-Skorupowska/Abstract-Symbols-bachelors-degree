import os
import json
import time
import numpy as np

from .predicate import Predicate

class BrainNetwork:
    def __init__(self, log_dir=None):
        self.state_dim = None 

        class Node:
            def __init__(self, name: str, precondition: Predicate, effect: Predicate):
                self.name = name
                self.precondition = precondition
                self.effect = effect
            
            def __repr__(self):
                return f"Node({self.name})"

        class Edge:
            def __init__(self, source_node: str, target_node: str, weight: float):
                self.source_node = source_node
                self.target_node = target_node
                self.symbol = None
                self.weight = weight

            def __repr__(self):
                return f"Edge({self.source_node} -> {self.target_node}, weight={self.weight})"
            
            def update(self, symbol: Predicate, weight: float):
                self.symbol = symbol
                self.weight = weight

        class Graph:
            def __init__(self):
                self.nodes = {}
                self.edges_by_target = {}

            def add_node(self, name: str, precondition: Predicate, effect: Predicate):
                if name in self.nodes:
                    raise ValueError(f"Node '{name}' already exists.")
                self.nodes[name] = Node(name, precondition, effect)
                if name not in self.edges_by_target:
                    self.edges_by_target[name] = {}

            def add_edge(self, from_name: str, to_name: str, overlap: float):
                if from_name is None or to_name is None:
                    return
                if from_name not in self.nodes or to_name not in self.nodes:
                    raise KeyError(f"Both nodes must exist: {from_name}, {to_name}")

                incoming = self.edges_by_target.setdefault(to_name, {})
                if from_name in incoming:
                    incoming[from_name].weight = overlap
                else:
                    incoming[from_name] = Edge(from_name, to_name, overlap)

            def delete_edge(self, from_name: str, to_name: str):
                incoming = self.edges_by_target.get(to_name)
                if not incoming or from_name not in incoming:
                    return
                del incoming[from_name]

            def get_incoming_edges(self, target_name: str) -> dict:
                return self.edges_by_target.get(target_name, {})

            def get_node(self, name: str):
                return self.nodes.get(name)

            def get_nodes_list(self):
                return list(self.nodes.keys())

            def update_precondition(self, node_name: str, vector, valence: bool):
                if node_name not in self.nodes:
                    raise KeyError(f"Node '{node_name}' does not exist.")
                self.nodes[node_name].precondition.update(vector, valence)

            def update_effect(self, node_name: str, vector, valence: bool):
                if node_name not in self.nodes:
                    raise KeyError(f"Node '{node_name}' does not exist.")
                if valence:
                    self.nodes[node_name].effect.update(vector, True)

        self.trans_graph = Graph()
        self.min_overlap_ratio = 0.7

        self.max_points = 85
        self.base_radius = 0.15
        self.max_radius = 1.75 
        self.learning_rate_b = 0.2
        self.learning_rate_n = 0.01
        self.max_edge_age = 24
        self.lambda_step = 5
        self.alpha = 0.7
        self.d = 0.99

        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, "PDDL")
            self.gas_dir = os.path.join(self.log_dir, "PDDL", "GNG")
            self.graphs_dir = os.path.join(self.log_dir, "graphs")
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.gas_dir, exist_ok=True)
            os.makedirs(self.graphs_dir, exist_ok=True)

    def add_node(self, skill_name: str, precondition: Predicate, effect: Predicate):
        self.trans_graph.add_node(skill_name, precondition, effect)
            
    def create_node(self, skill_name):
        if skill_name not in self.trans_graph.nodes:
            precondition = Predicate(self.max_points, self.state_dim, self.base_radius, self.max_radius, 
                                     self.learning_rate_b, self.learning_rate_n, self.max_edge_age, 
                                     self.lambda_step, self.alpha, self.d)
            effect = Predicate(self.max_points, self.state_dim, self.base_radius, self.max_radius, 
                               self.learning_rate_b, self.learning_rate_n, self.max_edge_age, 
                               self.lambda_step, self.alpha, self.d)
            self.trans_graph.add_node(skill_name, precondition, effect)

    def delete_edge(self, from_name: str, to_name: str):
        self.trans_graph.delete_edge(from_name, to_name)
        if self.log_dir:
            symbol_name = f"sym_{from_name}_enables_{to_name}_pass"
            file_path = os.path.join(self.gas_dir, f"{symbol_name}.json")
            temp_file = os.path.join(self.gas_dir, f"{symbol_name}.tmp")
            for path in (file_path, temp_file):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def preconditions_met(self, skill_name, state_vector) -> bool:
        if self.state_dim is None:
            self.state_dim = len(state_vector)

        if skill_name not in self.trans_graph.nodes:
            return False
            
        node = self.trans_graph.nodes[skill_name]
        return node.precondition.is_active(state_vector)

    def get_active_skills(self, vector):
        if self.state_dim is None:
            self.state_dim = len(vector)

        active_skills = []
        for name, node in self.trans_graph.nodes.items():
            if node.precondition.is_active(vector):
                active_skills.append(name)
        return active_skills

    def get_nodes(self):
        return self.trans_graph.get_nodes_list()

    def update(self, buffor):
        print(f"\t[Brain] Updating Predicates (GNG)")
        for data in buffor:
            pre_vector_state, skill_name, post_vector_state, succeed = data
        
            if self.state_dim is None:
                self.state_dim = len(pre_vector_state)

            if skill_name not in self.trans_graph.nodes:
                self.create_node(skill_name)

            self.trans_graph.update_precondition(skill_name, pre_vector_state, succeed)
            self.trans_graph.update_effect(skill_name, post_vector_state, succeed)

            # 1. Update outgoing transitions from skill_name
            self.update_transitions(skill_name)
            
            # 2. Update incoming transitions to skill_name (preconditions have changed)
            incoming_sources = list(self.trans_graph.get_incoming_edges(skill_name).keys())
            for src_skill in incoming_sources:
                self.update_transitions(src_skill)

        self._logger()
            
    def update_transitions(self, skill_name: str, min_overlap_ratio = None):
        # if we can do A -> B, where A = skill_name, B = all others
        if min_overlap_ratio is None:
            min_overlap_ratio = self.min_overlap_ratio
        
        src_node = self.trans_graph.nodes[skill_name] # skill_name is the A
        eff_gng = src_node.effect
        
        existing_targets = [
            target_name 
            for target_name, sources in self.trans_graph.edges_by_target.items() 
            if skill_name in sources
        ]

        if not eff_gng.nodes or len(eff_gng.nodes) < 1:
            for target_skill in existing_targets:
                self.delete_edge(skill_name, target_skill) # delete all if there is no effect
            return {}

        overlapping_points_map = {}

        for target_skill, target_node in self.trans_graph.nodes.items(): # all others
            if skill_name == target_skill: 
                continue # no self loops

            init_gng = target_node.precondition
            if not init_gng.nodes or len(init_gng.nodes) < 1:
                if target_skill in existing_targets:
                    self.delete_edge(skill_name, target_skill) # delete this one if there is no init
                continue

            # Check activation directly
            overlapping_points = [
                node for node in eff_gng.nodes 
                if init_gng.is_active(node) # add if we can start target skill (init is active)
            ]
            overlap_ratio = len(overlapping_points) / len(eff_gng.nodes)
            
            # Require minimum threshold and count to reject noise
            if overlap_ratio >= min_overlap_ratio and len(overlapping_points) >= 1:
                self.trans_graph.add_edge(skill_name, target_skill, overlap_ratio) # add if the overlap is good enough
                overlapping_points_map[target_skill] = overlapping_points
            else:
                if target_skill in existing_targets:
                    self.delete_edge(skill_name, target_skill) # delete if it is no longer good

        return overlapping_points_map

    def abstract_symbols(self): # I guesse it is unnecessary
        # Compute filtered transitions
        all_overlaps = {
            src_skill: self.update_transitions(src_skill)
            for src_skill in self.trans_graph.nodes.keys()
        }

        # Create transition symbols only for verified overlapping nodes
        for target_skill, sources in list(self.trans_graph.edges_by_target.items()):
            for src_skill, edge in list(sources.items()):
                overlapping_points = all_overlaps.get(src_skill, {}).get(target_skill, [])
                if not overlapping_points:
                    self.delete_edge(src_skill, target_skill)
                    continue

                init_gng = self.trans_graph.nodes[target_skill].precondition
                sym_pred = Predicate(
                    max_points=max(self.max_points, len(overlapping_points)),
                    base_radius=init_gng.base_radius,
                    max_radius=init_gng.max_radius,
                    learning_rate_b=self.learning_rate_b,
                    learning_rate_n=self.learning_rate_n
                )
                for pt in overlapping_points:
                    sym_pred.update(pt, valence=True)

                edge.symbol = sym_pred

    def generate_domain_pddl(self, domain_name="robot_domain") -> str:
        pddl = [
            f"(define (domain {domain_name})",
            "  (:requirements :strips)",
            "  (:predicates"
        ]
        
        for skill in self.trans_graph.nodes.keys():
            pddl.append(f"    (can_run_{skill})")
            pddl.append(f"    (executed_{skill})")
            
        pddl.append("  )")

        # Compute max/average outgoing transition weight for sorting source nodes
        def get_source_priority(src_name):
            weights = []
            for target_name, sources in self.trans_graph.edges_by_target.items():
                if src_name in sources:
                    edge = sources[src_name]
                    w = edge.weight if edge.weight is not None else 0.0
                    weights.append(w)
            return max(weights) if weights else 0.0

        # Sort source skills so highest-overlap branches appear first
        sorted_sources = sorted(
            self.trans_graph.nodes.keys(),
            key=get_source_priority,
            reverse=True
        )

        for src_name in sorted_sources:
            targets = [
                target
                for target, sources in self.trans_graph.edges_by_target.items()
                if src_name in sources
            ]

            # Sort target transitions by weight descending (highest overlap first)
            targets.sort(
                key=lambda t: (
                    self.trans_graph.edges_by_target[t][src_name].weight
                    if self.trans_graph.edges_by_target[t][src_name].weight is not None
                    else 0.0
                ),
                reverse=True
            )

            pddl.extend([
                "",
                f"  (:action {src_name}",
                "    :parameters ()",
                f"    :precondition (can_run_{src_name})",
            ])

            adds = [f"(executed_{src_name})"]
            for target in targets:
                adds.append(f"(can_run_{target})")

            deletes = [f"(can_run_{src_name})"]
            pddl.append(f"    :effect (and {' '.join(adds)} (not {' '.join(deletes)}))")
            pddl.append("  )")

        pddl.append(")")
        return "\n".join(pddl)

    def generate_problem_pddl(self, initial_skill, goal_skill, 
                              problem_name="plan_problem", domain_name="robot_domain") -> str:
        pddl = [
            f"(define (problem {problem_name})",
            f"  (:domain {domain_name})",
            "  (:init",
            f"    (can_run_{initial_skill})",
            "  )",
            "  (:goal",
            f"    (executed_{goal_skill})",
            "  )",
            ")"
        ]
        return "\n".join(pddl)

    def _logger(self):
        if not self.log_dir:
            return

        for skill in self.trans_graph.nodes.keys():
            self.update_transitions(skill)
        self.abstract_symbols()

        for skill, node in self.trans_graph.nodes.items():
            self.save_predicate(skill, node.precondition, 'init')
            self.save_predicate(skill, node.effect, 'eff')

        edges_list = []
        for target, sources in self.trans_graph.edges_by_target.items():
            for source, edge in sources.items():
                symbol_name = ""
                if edge.symbol is not None:
                    symbol_name = f"sym_{source}_enables_{target}"
                    self.save_predicate(symbol_name, edge.symbol, 'pass')
                edges_list.append({
                    "source": edge.source_node,
                    "target": edge.target_node,
                    "symbol_name": symbol_name,
                    "weight": edge.weight, 
                })

        graph_data = {
            "nodes": list(self.trans_graph.nodes.keys()),
            "edges": edges_list
        }
            
        file_path = os.path.join(self.graphs_dir, "trans_graph.json")
        temp_file = os.path.join(self.graphs_dir, "trans_graph.tmp")
        with open(temp_file, "w") as f:
            json.dump(graph_data, f)
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

    def save_predicate(self, skill_name: str, predicate: Predicate, pred_type: str):
        if not self.log_dir or predicate is None or pred_type not in ['init', 'eff', 'pass']:
            return

        os.makedirs(self.gas_dir, exist_ok=True)

        parameters = {
            "max_points": self.max_points,
            "base_radius": self.base_radius,
            "max_radius": self.max_radius,
            "learning_rate_b": self.learning_rate_b,
            "learning_rate_n": self.learning_rate_n,
            "max_edge_age": self.max_edge_age,
            "lambda_step": self.lambda_step,
            "alpha": self.alpha,
            "d": self.d,
        }
            
        edges_str_keys = {f"{u},{v}": age for (u, v), age in predicate.edges.items()}
        radii = [float(predicate._get_local_radius(i)) for i in range(len(predicate.nodes))]
        data = {
            "parameters": parameters,
            "nodes": [n.tolist() for n in predicate.nodes],
            "edges": edges_str_keys,
            "local_radii": radii
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

    @staticmethod
    def load_predicate(file_path) -> Predicate:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        params = data.get("parameters", {})
        pred = Predicate(
            max_points=params.get("max_points", 50),
            base_radius=params.get("base_radius", 0.1),
            max_radius=params.get("max_radius", 1.0),
            learning_rate_b=params.get("learning_rate_b", 0.2),
            learning_rate_n=params.get("learning_rate_n", 0.02),
            max_edge_age=params.get("max_edge_age", 30),
            lambda_step=params.get("lambda_step", 10),
            alpha=params.get("alpha", 0.5),
            d=params.get("d", 0.99)
        )

        if "nodes" in data:
            pred.nodes = [np.array(n, dtype=float) for n in data["nodes"]]
            pred.errors = [0.0] * len(pred.nodes)

        # Restore the saved radii!
        if "local_radii" in data:
            pred.cached_radii = data["local_radii"]
        
        if "edges" in data:
            parsed_edges = {}
            for k, v in data["edges"].items():
                u, v_node = map(int, k.split(","))
                parsed_edges[(u, v_node)] = v
            pred.edges = parsed_edges
        
            # Populate adj
            pred.adj = {}
            for (u, v) in pred.edges.keys():
                pred.adj.setdefault(u, set()).add(v)
                pred.adj.setdefault(v, set()).add(u)
        
        return pred

    def load_brain(self, source_dir):
        print(f"\t[Brain] Loading brain from the files.")
        if source_dir is None:
            print(f"\t[Brain] No source directory.")
            return
        
        if os.path.isdir(source_dir):
            src_graphs_file = os.path.join(source_dir, "graphs", "trans_graph.json")
            src_gas_dir = os.path.join(source_dir, "PDDL", "GNG")
            
            target_graphs_file = os.path.join(self.graphs_dir, "trans_graph.json")
            target_gas_dir = self.gas_dir
            
            if not os.path.exists(src_graphs_file):
                print(f"\t[Brain] Could not find trans_graph.json in {source_dir}")
                return
                
            with open(src_graphs_file, "r") as src_f, open(target_graphs_file, "w") as dst_f:
                dst_f.write(src_f.read())
                
            if os.path.exists(src_gas_dir):
                for filename in os.listdir(src_gas_dir):
                    if filename.endswith(".json"):
                        src_file = os.path.join(src_gas_dir, filename)
                        dst_file = os.path.join(target_gas_dir, filename)
                        with open(src_file, "r") as src_f, open(dst_file, "w") as dst_f:
                            dst_f.write(src_f.read())

            with open(target_graphs_file, "r") as f:
                graph_data = json.load(f)
                
            for idx, node_name in enumerate(graph_data.get("nodes", [])):
                precondition = None
                effect = None

                precondition_file = os.path.join(target_gas_dir, f"{node_name}_init.json")
                if os.path.exists(precondition_file):
                    precondition = self.load_predicate(precondition_file)
                else:
                    print(f"\t[Brain] Warning: Missing precondition file for node {node_name}")
                
                effect_file = os.path.join(target_gas_dir, f"{node_name}_eff.json")
                if os.path.exists(effect_file):
                    effect = self.load_predicate(effect_file)
                else:
                    print(f"\t[Brain] Warning: Missing effect file for node {node_name}")

                if precondition and effect:
                    self.add_node(node_name, precondition, effect)    
            
            for edge in graph_data.get("edges", []):
                source = edge.get("source")
                target = edge.get("target")
                symbol_name = edge.get("symbol_name")
                weight = edge.get("weight")

                self.trans_graph.add_edge(source, target, weight)

                symbol_file = os.path.join(target_gas_dir, f"{symbol_name}_pass.json")
                if os.path.exists(symbol_file):
                    symbol = self.load_predicate(symbol_file)
                    if target in self.trans_graph.edges_by_target and source in self.trans_graph.edges_by_target[target]:
                        self.trans_graph.edges_by_target[target][source].symbol = symbol
                else:
                    print(f"\t[Brain] Warning: Missing symbol file for edge: {source} --> {target}")
            # recalculate
            self.abstract_symbols()