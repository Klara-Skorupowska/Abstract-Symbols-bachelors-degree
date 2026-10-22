import os
import json
import time
import numpy as np


class GNG:
    """Standard, stable Growing Neural Gas network."""
    def __init__(self, max_points=20, dim=None, base_radius=0.1, max_radius=1.0,
                 learning_rate_b=0.2, learning_rate_n=0.006,
                 max_edge_age=50, lambda_step=10, alpha=0.5, d=0.995):
        self.dim = int(dim) if dim is not None else None
        self.max_points = max_points
        self.base_radius = base_radius
        self.max_radius = max_radius
        self.radii_factor = 0.5
        self.eb = learning_rate_b
        self.en = learning_rate_n
        self.max_age = max_edge_age
        self.lambda_step = lambda_step
        self.alpha = alpha
        self.d = d

        self.nodes = []
        self.errors = []
        self.edges = {}
        self.adj = {}
        self.cached_radii = None
        self.update_count = 0

    def _add_edge(self, u, v, age=0):
        if u == v:
            return
        self.edges[tuple(sorted((u, v)))] = age
        self.adj.setdefault(u, set()).add(v)
        self.adj.setdefault(v, set()).add(u)

    def _remove_edge(self, u, v):
        edge = tuple(sorted((u, v)))
        if edge in self.edges:
            del self.edges[edge]
        if u in self.adj and v in self.adj[u]:
            self.adj[u].remove(v)
        if v in self.adj and u in self.adj[v]:
            self.adj[v].remove(u)

    def _get_local_radius(self, node_idx):
        if self.cached_radii is not None and node_idx < len(self.cached_radii):
            return self.cached_radii[node_idx]
        neighbors = self.adj.get(node_idx, set())
        nodes_mat = np.asarray(self.nodes, dtype=float)
        norm_factor = np.sqrt(self.dim) if (self.dim and self.dim > 0) else 1.0

        if not neighbors:
            return self.base_radius

        ref_node = nodes_mat[node_idx]
        dists = [np.linalg.norm(ref_node - nodes_mat[n]) / norm_factor for n in neighbors]
        raw_radius = min(dists) * self.radii_factor
        return float(np.clip(raw_radius, self.base_radius, self.max_radius))

    def is_active(self, vector) -> bool:
        if not self.nodes:
            return False
        vec = np.asarray(vector, dtype=float)
        nodes_mat = np.asarray(self.nodes, dtype=float)
        dim = len(vec)
        if dim == 0 or nodes_mat.shape[1] != dim:
            return False
        norm_factor = np.sqrt(dim)

        dists = np.linalg.norm(nodes_mat - vec, axis=1) / norm_factor
        n1_idx = int(np.argmin(dists))
        return bool(dists[n1_idx] <= self._get_local_radius(n1_idx))

    def _insert_node(self):
        if not self.errors:
            return
        q = int(np.argmax(self.errors))
        neighbors = list(self.adj.get(q, set()))
        if not neighbors:
            return
        f = neighbors[int(np.argmax([self.errors[n] for n in neighbors]))]
        r_pos = 0.5 * (self.nodes[q] + self.nodes[f])
        self.nodes.append(r_pos)
        self.errors.append(0.0)
        r = len(self.nodes) - 1
        self._remove_edge(q, f)
        self._add_edge(q, r, age=0)
        self._add_edge(f, r, age=0)
        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha
        self.errors[r] = self.errors[q]

    def tidy(self):
        for edge in list(self.edges.keys()):
            if self.edges[edge] > self.max_age:
                del self.edges[edge]
        connected = set()
        for u, v in self.edges.keys():
            connected.add(u)
            connected.add(v)
        if len(connected) < len(self.nodes) and len(self.nodes) > 1:
            new_nodes, new_errors, idx_map = [], [], {}
            for i in range(len(self.nodes)):
                if i in connected:
                    idx_map[i] = len(new_nodes)
                    new_nodes.append(self.nodes[i])
                    new_errors.append(self.errors[i])
            self.nodes, self.errors = new_nodes, new_errors
            new_edges = {}
            for (u, v), age in self.edges.items():
                if u in idx_map and v in idx_map:
                    new_edges[tuple(sorted((idx_map[u], idx_map[v])))] = age
            self.edges = new_edges
        self.adj = {}
        for u, v in self.edges.keys():
            self.adj.setdefault(u, set()).add(v)
            self.adj.setdefault(v, set()).add(u)

    def update(self, vector):
        vector = np.asarray(vector, dtype=float)
        if self.dim is None:
            self.dim = len(vector)

        self.cached_radii = None

        if len(self.nodes) < 2:
            self.nodes.append(vector.copy())
            self.errors.append(0.0)
            if len(self.nodes) == 2:
                self._add_edge(0, 1, age=0)
            return

        self.update_count += 1
        norm_factor = np.sqrt(self.dim) if self.dim > 0 else 1.0
        dists = [np.linalg.norm(vector - n) / norm_factor for n in self.nodes]
        sorted_idx = np.argsort(dists)
        s1, s2 = int(sorted_idx[0]), int(sorted_idx[1])

        for (u, v) in list(self.edges.keys()):
            if u == s1 or v == s1:
                self.edges[(u, v)] += 1

        self.errors[s1] += dists[s1] ** 2
        self.nodes[s1] += self.eb * (vector - self.nodes[s1])

        for (u, v) in self.edges.keys():
            if u == s1:
                self.nodes[v] += self.en * (vector - self.nodes[v])
            elif v == s1:
                self.nodes[u] += self.en * (vector - self.nodes[u])

        self._add_edge(s1, s2, age=0)
        self.tidy()

        if self.update_count % self.lambda_step == 0 and len(self.nodes) < self.max_points:
            self._insert_node()

        for i in range(len(self.errors)):
            self.errors[i] *= self.d

    def serialize(self) -> dict:
        parameters = {
            "max_points": self.max_points,
            "base_radius": self.base_radius,
            "max_radius": self.max_radius,
            "learning_rate_b": self.eb,
            "learning_rate_n": self.en,
            "max_edge_age": self.max_age,
            "lambda_step": self.lambda_step,
            "alpha": self.alpha,
            "d": self.d,
            "dim": self.dim,
        }
        edges_str_keys = {f"{u},{v}": age for (u, v), age in self.edges.items()}
        radii = [float(self._get_local_radius(i)) for i in range(len(self.nodes))]

        return {
            "parameters": parameters,
            "nodes": [n.tolist() for n in self.nodes],
            "edges": edges_str_keys,
            "local_radii": radii
        }

    def deserialize(self, data: dict):
        if not data:
            return

        params = data.get("parameters", {})
        self.max_points = params.get("max_points", self.max_points)
        self.base_radius = params.get("base_radius", self.base_radius)
        self.max_radius = params.get("max_radius", self.max_radius)
        self.eb = params.get("learning_rate_b", self.eb)
        self.en = params.get("learning_rate_n", self.en)
        self.max_age = params.get("max_edge_age", self.max_age)
        self.lambda_step = params.get("lambda_step", self.lambda_step)
        self.alpha = params.get("alpha", self.alpha)
        self.d = params.get("d", self.d)
        self.dim = params.get("dim", self.dim)

        if "nodes" in data:
            self.nodes = [np.array(n, dtype=float) for n in data["nodes"]]
            self.errors = [0.0] * len(self.nodes)
            if self.dim is None and self.nodes:
                self.dim = len(self.nodes[0])

        if "local_radii" in data:
            self.cached_radii = data["local_radii"]

        if "edges" in data:
            parsed_edges = {}
            for k, v in data["edges"].items():
                u, v_node = map(int, k.split(","))
                parsed_edges[(u, v_node)] = v
            self.edges = parsed_edges

            self.adj = {}
            for (u, v) in self.edges.keys():
                self.adj.setdefault(u, set()).add(v)
                self.adj.setdefault(v, set()).add(u)


class Predicate:
    """Predicate wrapping a GNG manifold over full or sliced active dimensions."""
    def __init__(self, max_points=20, dim=None, base_radius=0.1, max_radius=1.0,
                 learning_rate_b=0.2, learning_rate_n=0.006,
                 max_edge_age=50, lambda_step=10, alpha=0.5, d=0.995):
        params = dict(
            max_points=max_points, dim=dim, base_radius=base_radius, max_radius=max_radius,
            learning_rate_b=learning_rate_b, learning_rate_n=learning_rate_n,
            max_edge_age=max_edge_age, lambda_step=lambda_step, alpha=alpha, d=d
        )
        self.gng = GNG(**params)
        self.dim = int(dim) if dim is not None else None
        self.active_dimensions = list(range(self.dim)) if self.dim is not None else None

    @property
    def nodes(self):
        return self.gng.nodes

    def set_active_dimensions(self, dimensions=None):
        if dimensions is not None:
            self.active_dimensions = [int(d) for d in dimensions]
            self.dim = len(self.active_dimensions)
            self.gng.dim = self.dim
        elif self.dim is not None:
            self.active_dimensions = list(range(self.dim))
        else:
            print('\t[PREDICATE] Wrong active dimensions')

    def _project_vector(self, vector, vector_dims=None):
        vec = np.asarray(vector, dtype=float)
        if self.active_dimensions is None:
            return vec

        if vector_dims is not None:
            dim_to_idx = {int(d): idx for idx, d in enumerate(vector_dims)}
            if all(d in dim_to_idx for d in self.active_dimensions):
                return np.array([vec[dim_to_idx[d]] for d in self.active_dimensions], dtype=float)
            return None

        if len(vec) == len(self.active_dimensions):
            return vec

        if len(vec) > max(self.active_dimensions, default=-1):
            return vec[self.active_dimensions]

        return None

    def update(self, vector):
        vec = np.asarray(vector, dtype=float)
        if self.dim is None:
            self.dim = len(vec)
            self.gng.dim = self.dim
            if self.active_dimensions is None:
                self.active_dimensions = list(range(self.dim))

        projected = self._project_vector(vec)
        if projected is not None and len(projected) == self.dim:
            self.gng.update(projected)

    def evaluate(self, vector, vector_dims=None) -> str:
        projected = self._project_vector(vector, vector_dims=vector_dims)
        if projected is None:
            return "FALSE"
        return "TRUE" if self.gng.is_active(projected) else "FALSE"

    def is_active(self, vector, vector_dims=None) -> bool:
        return self.evaluate(vector, vector_dims=vector_dims) == "TRUE"

    def clone_subspace(self, target_dims=None, node_indices=None) -> "Predicate":
        """Creates a new Predicate sliced to target_dims and/or filtered to node_indices."""
        if target_dims is None:
            target_dims = list(self.active_dimensions) if self.active_dimensions is not None else []
        else:
            target_dims = [int(d) for d in target_dims]

        new_pred = Predicate(
            max_points=self.gng.max_points,
            dim=len(target_dims),
            base_radius=self.gng.base_radius,
            max_radius=self.gng.max_radius,
            learning_rate_b=self.gng.eb,
            learning_rate_n=self.gng.en,
            max_edge_age=self.gng.max_age,
            lambda_step=self.gng.lambda_step,
            alpha=self.gng.alpha,
            d=self.gng.d
        )
        new_pred.set_active_dimensions(target_dims)

        if not self.nodes or not target_dims:
            return new_pred

        if self.active_dimensions is not None:
            dim_map = {int(d): i for i, d in enumerate(self.active_dimensions)}
            local_cols = [dim_map[d] for d in target_dims if d in dim_map]
        else:
            local_cols = target_dims

        if len(local_cols) != len(target_dims):
            return new_pred

        selected_nodes = list(range(len(self.nodes))) if node_indices is None else [int(i) for i in node_indices]
        if not selected_nodes:
            return new_pred

        old_to_new = {}
        for old_idx in selected_nodes:
            old_to_new[old_idx] = len(new_pred.gng.nodes)
            sliced_vec = np.asarray(self.nodes[old_idx], dtype=float)[local_cols].copy()
            new_pred.gng.nodes.append(sliced_vec)
            err = self.gng.errors[old_idx] if old_idx < len(self.gng.errors) else 0.0
            new_pred.gng.errors.append(err)

        for (u, v), age in self.gng.edges.items():
            if u in old_to_new and v in old_to_new:
                new_pred.gng._add_edge(old_to_new[u], old_to_new[v], age=age)

        # Fallback: connect nearest neighbors if filtering left nodes without edges
        if len(new_pred.gng.nodes) >= 2 and not new_pred.gng.edges:
            nodes_mat = np.asarray(new_pred.gng.nodes, dtype=float)
            for i in range(len(nodes_mat)):
                dists = np.linalg.norm(nodes_mat - nodes_mat[i], axis=1)
                dists[i] = float("inf")
                nearest = int(np.argmin(dists))
                new_pred.gng._add_edge(i, nearest, age=0)

        return new_pred

    def save_predicate(self, log_dir, pred_name: str):
        if not log_dir:
            return

        os.makedirs(log_dir, exist_ok=True)

        data = {
            "gng": self.gng.serialize(),
            "dim": self.dim,
            "active_dimensions": list(self.active_dimensions) if self.active_dimensions is not None else None,
        }

        file_path = os.path.join(log_dir, f"{pred_name}.json")
        temp_file = os.path.join(log_dir, f"{pred_name}.tmp")

        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)

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

    def load_predicate(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        gng_data = data.get("gng", {})
        params = gng_data.get("parameters", {})
        dim = data.get("dim", params.get("dim", None))
        dimensions = data.get("active_dimensions", None)

        self.dim = int(dim) if dim is not None else None
        self.gng = GNG(**params)
        self.gng.deserialize(gng_data)

        if self.dim is None and self.gng.nodes:
            self.dim = len(self.gng.nodes[0])
            self.gng.dim = self.dim

        if dimensions is not None:
            self.set_active_dimensions(dimensions)
        elif self.dim is not None:
            self.set_active_dimensions(list(range(self.dim)))
        return self