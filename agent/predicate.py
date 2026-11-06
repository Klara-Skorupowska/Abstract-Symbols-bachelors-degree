import os
import json
import time
import uuid
import numpy as np
from scipy.spatial.distance import cdist


class GNG:
    """Zoptymalizowana sieć Growing Neural Gas dla wysokich wymiarów."""
    def __init__(self, max_points=20, dim=None, base_radius=0.1, max_radius=1.0,
                 learning_rate_b=0.2, learning_rate_n=0.006,
                 max_edge_age=50, lambda_step=10, alpha=0.5, d=0.995):
        self.dim = int(dim) if dim is not None else None
        self.max_points = max_points
        self.base_radius = float(base_radius)
        self.max_radius = float(max_radius)
        self.radii_factor = 0.5
        self.eb = float(learning_rate_b)
        self.en = float(learning_rate_n)
        self.max_age = int(max_edge_age)
        self.lambda_step = int(lambda_step)
        self.alpha = float(alpha)
        self.d = float(d)

        self.nodes = np.empty((0, self.dim if self.dim is not None else 0), dtype=np.float64)
        self.errors = np.empty((0,), dtype=np.float64)
        self.edges = {}
        self.adj = {}
        self.cached_radii = None
        self.update_count = 0

    @property
    def num_nodes(self):
        return len(self.nodes)

    def _add_edge(self, u, v, age=0):
        if u == v:
            return
        edge = (u, v) if u < v else (v, u)
        self.edges[edge] = age
        self.adj.setdefault(u, set()).add(v)
        self.adj.setdefault(v, set()).add(u)

    def _remove_edge(self, u, v):
        edge = (u, v) if u < v else (v, u)
        if edge in self.edges:
            del self.edges[edge]
        if u in self.adj:
            self.adj[u].discard(v)
        if v in self.adj:
            self.adj[v].discard(u)

    def _recompute_radii(self):
        n = self.num_nodes
        if n == 0:
            self.cached_radii = np.empty((0,), dtype=np.float64)
            return

        radii = np.full(n, self.base_radius, dtype=np.float64)

        for i in range(n):
            nbrs = list(self.adj.get(i, ()))
            if nbrs:
                diff = self.nodes[nbrs] - self.nodes[i]
                min_d = np.min(np.sqrt(np.sum(diff * diff, axis=1)))
                radii[i] = np.clip(min_d * self.radii_factor, self.base_radius, self.max_radius)

        self.cached_radii = radii

    def _get_local_radius(self, node_idx):
        if self.cached_radii is None:
            self._recompute_radii()
        if node_idx < len(self.cached_radii):
            return self.cached_radii[node_idx]
        return self.base_radius

    def is_active(self, vectors) -> np.ndarray:
        if self.num_nodes == 0:
            arr = np.asarray(vectors)
            return False if arr.ndim == 1 else np.zeros(len(arr), dtype=bool)

        arr = np.asarray(vectors, dtype=np.float64)
        is_single = (arr.ndim == 1)
        if is_single:
            arr = arr.reshape(1, -1)

        dim = arr.shape[1]
        if dim == 0 or self.nodes.shape[1] != dim:
            res = np.zeros(len(arr), dtype=bool)
            return bool(res[0]) if is_single else res

        if self.cached_radii is None:
            self._recompute_radii()

        dists = cdist(arr, self.nodes, metric='euclidean')
        nearest_idx = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(len(arr)), nearest_idx]

        active = min_dists <= self.cached_radii[nearest_idx]
        return bool(active[0]) if is_single else active

    def _insert_node(self):
        if len(self.errors) == 0:
            return
        q = int(np.argmax(self.errors))
        neighbors = list(self.adj.get(q, ()))
        if not neighbors:
            return
        f = neighbors[int(np.argmax(self.errors[neighbors]))]

        r_pos = 0.5 * (self.nodes[q] + self.nodes[f])
        self.nodes = np.vstack([self.nodes, r_pos])
        self.errors = np.append(self.errors, 0.0)
        r = self.num_nodes - 1

        self._remove_edge(q, f)
        self._add_edge(q, r, age=0)
        self._add_edge(f, r, age=0)

        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha
        self.errors[r] = self.errors[q]
        self.cached_radii = None

    def tidy(self):
        max_age = self.max_age
        for edge, age in list(self.edges.items()):
            if age > max_age:
                self._remove_edge(edge[0], edge[1])

        connected = set()
        for u, v in self.edges.keys():
            connected.add(u)
            connected.add(v)

        n = self.num_nodes
        if 0 < len(connected) < n:
            conn_mask = np.zeros(n, dtype=bool)
            conn_indices = np.array(sorted(connected), dtype=int)
            conn_mask[conn_indices] = True

            idx_map = {old_i: new_i for new_i, old_i in enumerate(conn_indices)}
            self.nodes = self.nodes[conn_mask]
            self.errors = self.errors[conn_mask]

            new_edges = {}
            for (u, v), age in self.edges.items():
                if u in idx_map and v in idx_map:
                    nu, nv = idx_map[u], idx_map[v]
                    new_edges[(nu, nv) if nu < nv else (nv, nu)] = age
            self.edges = new_edges

            self.adj = {}
            for u, v in self.edges.keys():
                self.adj.setdefault(u, set()).add(v)
                self.adj.setdefault(v, set()).add(u)

            self.cached_radii = None

    def update(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if self.dim is None:
            self.dim = len(vector)
            self.nodes = np.empty((0, self.dim), dtype=np.float64)

        self.cached_radii = None

        if self.num_nodes < 2:
            self.nodes = np.vstack([self.nodes, vector])
            self.errors = np.append(self.errors, 0.0)
            if self.num_nodes == 2:
                self._add_edge(0, 1, age=0)
            return

        self.update_count += 1

        diff = self.nodes - vector
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        sorted_idx = np.argpartition(dists, 1)[:2]
        if dists[sorted_idx[0]] > dists[sorted_idx[1]]:
            s1, s2 = sorted_idx[1], sorted_idx[0]
        else:
            s1, s2 = sorted_idx[0], sorted_idx[1]

        for edge in list(self.edges.keys()):
            if edge[0] == s1 or edge[1] == s1:
                self.edges[edge] += 1

        self.errors[s1] += dists[s1] ** 2
        self.nodes[s1] += self.eb * (vector - self.nodes[s1])

        nbrs = list(self.adj.get(s1, ()))
        if nbrs:
            self.nodes[nbrs] += self.en * (vector - self.nodes[nbrs])

        self._add_edge(s1, s2, age=0)
        self.tidy()

        if self.update_count % self.lambda_step == 0 and self.num_nodes < self.max_points:
            self._insert_node()

        self.errors *= self.d

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
        if self.cached_radii is None:
            self._recompute_radii()

        return {
            "parameters": parameters,
            "nodes": self.nodes.tolist(),
            "edges": edges_str_keys,
            "local_radii": self.cached_radii.tolist()
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
            nodes_data = data["nodes"]
            self.nodes = np.array(nodes_data, dtype=np.float64) if nodes_data else np.empty((0, self.dim or 0))
            self.errors = np.zeros(len(self.nodes), dtype=np.float64)
            if self.dim is None and len(self.nodes) > 0:
                self.dim = self.nodes.shape[1]

        if "local_radii" in data:
            self.cached_radii = np.array(data["local_radii"], dtype=np.float64)

        if "edges" in data:
            self.edges = {}
            for k, v in data["edges"].items():
                u, v_node = map(int, k.split(","))
                edge = (u, v_node) if u < v_node else (v_node, u)
                self.edges[edge] = v
            self.adj = {}
            for u, v in self.edges.keys():
                self.adj.setdefault(u, set()).add(v)
                self.adj.setdefault(v, set()).add(u)


class Predicate:
    """Predykat owijający rozmaitość GNG w pełnej lub zredukowanej podprzestrzeni."""
    def __init__(self, max_points=20, dim=None, base_radius=0.1, max_radius=1.0,
                 learning_rate_b=0.2, learning_rate_n=0.006,
                 max_edge_age=50, lambda_step=10, alpha=0.5, d=0.995):
        self.gng = GNG(
            max_points=max_points, dim=dim, base_radius=base_radius, max_radius=max_radius,
            learning_rate_b=learning_rate_b, learning_rate_n=learning_rate_n,
            max_edge_age=max_edge_age, lambda_step=lambda_step, alpha=alpha, d=d
        )
        self.dim = int(dim) if dim is not None else None
        self.active_dimensions = np.arange(self.dim, dtype=int) if self.dim is not None else None

    @property
    def nodes(self):
        return self.gng.nodes

    def set_active_dimensions(self, dimensions=None):
        if dimensions is not None:
            self.active_dimensions = np.array(dimensions, dtype=int)
            self.dim = len(self.active_dimensions)
            self.gng.dim = self.dim
            if self.gng.nodes.shape[1] != self.dim and self.gng.num_nodes == 0:
                self.gng.nodes = np.empty((0, self.dim), dtype=np.float64)
        elif self.dim is not None:
            self.active_dimensions = np.arange(self.dim, dtype=int)
            self.gng.dim = self.dim
            if self.gng.nodes.shape[1] != self.dim and self.gng.num_nodes == 0:
                self.gng.nodes = np.empty((0, self.dim), dtype=np.float64)
        else:
            print('\t[PREDICATE] Wrong active dimensions')

    def _project_vector(self, vectors, vector_dims=None):
        arr = np.asarray(vectors, dtype=np.float64)
        if self.active_dimensions is None:
            return arr

        is_single = (arr.ndim == 1)
        if is_single:
            arr = arr.reshape(1, -1)

        act_dims = self.active_dimensions

        if vector_dims is not None:
            vector_dims = np.asarray(vector_dims, dtype=int)
            if len(vector_dims) == 0 or len(act_dims) == 0:
                return None

            # 1. Zabezpieczone wyszukiwanie z użyciem sorter
            sorter = np.argsort(vector_dims)
            raw_idx = np.searchsorted(vector_dims, act_dims, sorter=sorter)
            
            # Zapobiegamy wyjściu poza zakres tablicy sorter
            clipped_idx = np.clip(raw_idx, 0, len(vector_dims) - 1)
            matched_indices = sorter[clipped_idx]

            # 2. Sprawdzamy, czy WSZYSTKIE act_dims faktycznie istnieją w vector_dims
            is_valid = (raw_idx < len(vector_dims)) & (vector_dims[matched_indices] == act_dims)
            if np.all(is_valid):
                proj = arr[:, matched_indices]
                return proj[0] if is_single else proj
            return None

        # Wektor pełnowymiarowy: wycinamy kolumny
        if arr.shape[1] > np.max(act_dims, initial=-1):
            proj = arr[:, act_dims]
            return proj[0] if is_single else proj

        # Wektor o dokładnie dopasowanej liczbie kolumn do self.active_dimensions
        if arr.shape[1] == len(act_dims):
            return vectors

        return None

    def update(self, vector, vector_dims=None):
        vec = np.asarray(vector, dtype=np.float64)
        if self.dim is None:
            self.dim = len(vec)
            self.gng.dim = self.dim
            if self.gng.nodes.shape[1] != self.dim and self.gng.num_nodes == 0:
                self.gng.nodes = np.empty((0, self.dim), dtype=np.float64)
            if self.active_dimensions is None:
                self.active_dimensions = np.arange(self.dim, dtype=int)

        projected = self._project_vector(vec, vector_dims=vector_dims)
        if projected is not None and len(projected) == self.dim:
            self.gng.update(projected)

    def is_active(self, vectors, vector_dims=None):
        """Obsługuje wektory 1D oraz 2D wsadowo."""
        projected = self._project_vector(vectors, vector_dims=vector_dims)
        if projected is None:
            arr = np.asarray(vectors)
            return False if arr.ndim == 1 else np.zeros(len(arr), dtype=bool)
        return self.gng.is_active(projected)

    def evaluate(self, vector, vector_dims=None) -> str:
        return "TRUE" if bool(self.is_active(vector, vector_dims=vector_dims)) else "FALSE"

    def clone_subspace(self, target_dims=None, node_indices=None) -> "Predicate":
        if target_dims is None:
            target_dims = self.active_dimensions.copy() if self.active_dimensions is not None else np.array([], dtype=int)
        else:
            target_dims = np.array(target_dims, dtype=int)

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

        if self.gng.num_nodes == 0 or len(target_dims) == 0 or self.active_dimensions is None:
            return new_pred

        curr_dims = self.active_dimensions
        if len(curr_dims) == 0:
            return new_pred

        # 1. Bezpieczne wyszukiwanie z użyciem sorter
        sorter = np.argsort(curr_dims)
        raw_idx = np.searchsorted(curr_dims, target_dims, sorter=sorter)
        
        # Zapobiegamy wyjściu poza rozmiar tablicy (indeks len(curr_dims))
        clipped_idx = np.clip(raw_idx, 0, len(curr_dims) - 1)
        matched = sorter[clipped_idx]

        # 2. Sprawdzamy, czy WSZYSTKIE target_dims faktycznie istnieją w curr_dims
        # (jeśli raw_idx == len(curr_dims) lub wartość się nie zgadza, warunek nie zajdzie)
        is_valid = (raw_idx < len(curr_dims)) & (curr_dims[matched] == target_dims)
        if not np.all(is_valid):
            # Brakuje części wymiarów (tak jak w candidate_joined w reduce_symbols) -> zwracamy pusty predykat
            return new_pred

        selected = np.arange(self.gng.num_nodes) if node_indices is None else np.asarray(node_indices, dtype=int)
        if len(selected) == 0:
            return new_pred

        new_pred.gng.nodes = self.gng.nodes[selected][:, matched].copy()
        new_pred.gng.errors = self.gng.errors[selected].copy()

        old_to_new = {old_i: new_i for new_i, old_i in enumerate(selected)}
        for (u, v), age in self.gng.edges.items():
            if u in old_to_new and v in old_to_new:
                new_pred.gng._add_edge(old_to_new[u], old_to_new[v], age=age)

        if new_pred.gng.num_nodes >= 2 and not new_pred.gng.edges:
            dists = cdist(new_pred.gng.nodes, new_pred.gng.nodes)
            np.fill_diagonal(dists, np.inf)
            nearest = np.argmin(dists, axis=1)
            for i, near_i in enumerate(nearest):
                new_pred.gng._add_edge(i, near_i, age=0)

        return new_pred

    def save_predicate(self, log_dir, pred_name: str):
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        data = {
            "gng": self.gng.serialize(),
            "dim": self.dim,
            "active_dimensions": self.active_dimensions.tolist() if self.active_dimensions is not None else None,
        }
        unique_id = uuid.uuid4().hex
        temp_file = os.path.join(log_dir, f"{pred_name}_{unique_id}.tmp")
        file_path = os.path.join(log_dir, f"{pred_name}.json")

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

        if dimensions is not None:
            self.set_active_dimensions(dimensions)
        elif self.dim is not None:
            self.set_active_dimensions(list(range(self.dim)))
        return self