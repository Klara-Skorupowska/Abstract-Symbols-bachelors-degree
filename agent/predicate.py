import numpy as np

class Predicate:
    def __init__(self, max_points, dim=None, base_radius=0.1, max_radius=1.0, 
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
        nodes_mat = np.asarray(self.nodes)
        norm_factor = np.sqrt(self.dim) if (self.dim and self.dim > 0) else 1.0
        
        if not neighbors:
            # Fall back to base radius if there are no topological neighbors
            return self.base_radius
            
        ref_node = nodes_mat[node_idx]
        dists = [np.linalg.norm(ref_node - nodes_mat[n]) / norm_factor for n in neighbors]
        raw_radius = min(dists) * self.radii_factor
        return float(np.clip(raw_radius, self.base_radius, self.max_radius))

    def is_active(self, vector):
        if not self.nodes:
            return False
        self.dim = len(vector)
        vec = np.asarray(vector, dtype=float)
        nodes_mat = np.asarray(self.nodes, dtype=float)

        if vec.shape[-1] != nodes_mat.shape[-1]:
            raise ValueError(f"Vector dimension {vec.shape[-1]} does not match node dimension {nodes_mat.shape[-1]}.")
        
        dists = np.linalg.norm(nodes_mat - vec, axis=1) / np.sqrt(self.dim)
        
        n1_idx = int(np.argmin(dists))
        n1_dist = dists[n1_idx]

    
        local_radius = self._get_local_radius(n1_idx)
        return bool(n1_dist <= local_radius)

    def _insert_node(self):
        if not self.errors:
            return
            
        q = np.argmax(self.errors)
        neighbors = list(self.adj.get(q, set()))
        if not neighbors:
            return
            
        f = neighbors[np.argmax([self.errors[n] for n in neighbors])]
        
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
            new_nodes, new_errors = [], []
            idx_map = {}
            for i in range(len(self.nodes)):
                if i in connected:
                    idx_map[i] = len(new_nodes)
                    new_nodes.append(self.nodes[i])
                    new_errors.append(self.errors[i])
                    
            self.nodes = new_nodes
            self.errors = new_errors
            
            new_edges = {}
            for (u, v), age in self.edges.items():
                if u in idx_map and v in idx_map:
                    new_edges[tuple(sorted((idx_map[u], idx_map[v])))] = age
            self.edges = new_edges

        self.adj = {}
        for u, v in self.edges.keys():
            self.adj.setdefault(u, set()).add(v)
            self.adj.setdefault(v, set()).add(u)

    def update(self, vector, valence: bool = True):
        vector = np.array(vector, dtype=float)
        if self.dim is None:
            self.dim = len(vector)
        elif len(vector) != self.dim:
            raise ValueError(f"Vector dimension {len(vector)} does not match predicate dim {self.dim}.")
        
        if len(self.nodes) < 2:
            if valence:
                self.nodes.append(vector)
                self.errors.append(0.0)
                if len(self.nodes) == 2:
                    self._add_edge(0, 1, age=0)
            return

        self.update_count += 1

        dists = [np.linalg.norm(vector - n) for n in self.nodes]
        
        sorted_idx = np.argsort(dists)
        s1 = sorted_idx[0]
        s2 = sorted_idx[1]

        if valence:
            for (u, v) in list(self.edges.keys()):
                if u == s1 or v == s1:
                    self.edges[(u, v)] += 1
                    
            self.errors[s1] += dists[s1] ** 2
            self.nodes[s1] += self.eb * (vector - self.nodes[s1])
            
            for (u, v) in self.edges.keys():
                if u == s1: 
                    self.nodes[v] += self.en * (vector - self.nodes[v])
                    self.nodes[v] = np.clip(self.nodes[v], 0.0, 1.0)
                elif v == s1: 
                    self.nodes[u] += self.en * (vector - self.nodes[u])
                    self.nodes[u] = np.clip(self.nodes[u], 0.0, 1.0)
                
            self._add_edge(s1, s2, age=0)
                 
        elif not valence and dists[s1] <= self._get_local_radius(s1): 
            for (u, v) in list(self.edges.keys()):
                if u == s1 or v == s1:
                    self.edges[(u, v)] += 1
                    
            self.errors[s1] = max(0.0, self.errors[s1] - (dists[s1] ** 2))
            
            self.nodes[s1] -= self.eb * (vector - self.nodes[s1])
            for (u, v) in self.edges.keys():
                if u == s1: 
                    self.nodes[v] -= self.en * (vector - self.nodes[v])
                    self.nodes[v] = np.clip(self.nodes[v], 0.0, 1.0)
                elif v == s1: 
                    self.nodes[u] -= self.en * (vector - self.nodes[u])
                    self.nodes[u] = np.clip(self.nodes[u], 0.0, 1.0)
                
            v_pos = self.nodes[s1]
            w_pos = self.nodes[s2]
            l2 = np.sum((w_pos - v_pos) ** 2)
            if l2 == 0:
                dist_to_edge = np.linalg.norm(vector - v_pos)
            else:
                t = max(0.0, min(1.0, np.dot(vector - v_pos, w_pos - v_pos) / l2))
                proj = v_pos + t * (w_pos - v_pos)
                dist_to_edge = np.linalg.norm(vector - proj)
                
            if dist_to_edge <= self.base_radius:
                self._remove_edge(s1, s2)
            
        self.tidy()
            
        if self.update_count % self.lambda_step == 0 and len(self.nodes) < self.max_points:
            self._insert_node()
                
        for i in range(len(self.errors)):
            self.errors[i] *= self.d