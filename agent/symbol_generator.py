import numpy as np
from .predicate import Predicate


class SymbolGenerator:
    def __init__(self, min_overlap_ratio=0.75, min_nodes_in_symbol = 4):
        self.symbols = {}
        self.min_overlap_ratio = min_overlap_ratio
        self.min_nodes = min_nodes_in_symbol
        self.dim_min = None
        self.dim_max = None

    def update_bounds(self, samples):
        """Tracks lowest and highest values per dimension across [0, 1] normalized data."""
        arr = np.asarray(samples, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return

        batch_min = np.min(arr, axis=0)
        batch_max = np.max(arr, axis=0)

        if self.dim_min is None or len(self.dim_min) != arr.shape[1]:
            self.dim_min = batch_min.copy()
            self.dim_max = batch_max.copy()
        else:
            self.dim_min = np.minimum(self.dim_min, batch_min)
            self.dim_max = np.maximum(self.dim_max, batch_max)

    def _calculate_variance(self, predicate: Predicate):
            """
            Calculates the variance dimension-wise and a global threshold cutoff.
            Dimensions where at least min_overlap_ratio of the samples fall within
            mean +/- sqrt(threshold) are characterized as invariant.
            """
            if predicate is None or not predicate.nodes:
                return np.array([]), 0.0

            nodes = np.asarray(predicate.nodes, dtype=float)
            if nodes.ndim == 1:
                nodes = nodes.reshape(1, -1)

            # Slice to active dimensions if defined
            if predicate.active_dimentions is not None:
                active_dims = list(predicate.active_dimentions)
                nodes = nodes[:, active_dims]

            n_samples, n_dims = nodes.shape
            if n_samples < 2:
                return np.zeros(n_dims), 0.0

            # Dimension-wise variance and standard deviation
            variances = np.var(nodes, axis=0)

            # Reference scale per dimension using bounds (or fallback to unit/observed bounds)
            if self.dim_min is not None and self.dim_max is not None:
                dim_span = self.dim_max - self.dim_min
                if predicate.active_dimentions is not None:
                    dim_span = dim_span[list(predicate.active_dimentions)]
                # Guard against zero range
                dim_span = np.where(dim_span <= 1e-12, 1.0, dim_span)
            else:
                dim_span = np.ones(n_dims)

            # Normalized deviations from the mean
            means = np.mean(nodes, axis=0)
            abs_deviations = np.abs(nodes - means) / dim_span

            # For each dimension, find the smallest radius (std equivalent)
            # needed to cover at least min_overlap_ratio of the points
            required_radii = np.quantile(
                abs_deviations,
                q=self.min_overlap_ratio,
                axis=0,
                method="higher"
            )

            # Convert back to unnormalized variance scale and pick the global cutoff
            candidate_cutoffs = (required_radii * dim_span) ** 2
        
            # A global threshold corresponding to candidate dimensions
            threshold = float(np.min(candidate_cutoffs)) if len(candidate_cutoffs) > 0 else 0.0

            return variances, threshold

    def _find_intersection(self, predicate: Predicate) -> Predicate:
        """Finds invariant dimensions by checking variance per dimension and slices node vectors."""
        if predicate is None or not predicate.nodes:
            return None

        nodes_mat = np.asarray(predicate.nodes, dtype=float)
        base_dims = (
            list(predicate.active_dimentions)
            if predicate.active_dimentions is not None
            else list(range(nodes_mat.shape[1]))
        )

        if len(nodes_mat) < 2:
            return predicate.clone_subspace(target_dims=base_dims)

        variances, threshold = self._calculate_variance(predicate)
        invariant_mask = variances <= threshold

        # If all dimensions exceed threshold, pick the most stable ones
        if not np.any(invariant_mask):
            min_var = np.min(variances)
            invariant_mask = np.isclose(variances, min_var, atol=1e-6)

        invariant_dims = [base_dims[i] for i, is_inv in enumerate(invariant_mask) if is_inv]
        if not invariant_dims:
            return None

        return predicate.clone_subspace(target_dims=invariant_dims)

    def _find_difference(self, predicate_A: Predicate, predicate_B: Predicate) -> Predicate:
        """
        Node-level set difference A \\ B: retains nodes in predicate_A that are not
        active in predicate_B.
        """
        if predicate_A is None or not predicate_A.nodes:
            return None
        if predicate_B is None or not predicate_B.nodes:
            return predicate_A.clone_subspace(target_dims=predicate_A.active_dimentions)

        dims_A = list(predicate_A.active_dimentions or range(len(predicate_A.nodes[0])))
        dims_B = list(predicate_B.active_dimentions or range(len(predicate_B.nodes[0])))
        shared_dims = sorted(set(dims_A) & set(dims_B))

        if not shared_dims:
            return predicate_A.clone_subspace(target_dims=dims_A)

        # Project B onto shared dimensions so node evaluation compares matching coordinates
        proj_B = predicate_B.clone_subspace(target_dims=shared_dims)

        kept_node_indices = []
        for idx, node_vec in enumerate(predicate_A.nodes):
            if not proj_B.is_active(node_vec, vector_dims=dims_A):
                kept_node_indices.append(idx)

        if not kept_node_indices:
            return None

        return predicate_A.clone_subspace(target_dims=dims_A, node_indices=kept_node_indices)

    def generate_symbols(self, skill) -> list:
        """
        Generates precondition and effect symbols for a given skill:
        before = pre_inter \\ eff_inter
        after  = eff_inter \\ pre_inter
        """
        pre_pred = skill.get("precondition")
        eff_pred = skill.get("effect")

        if pre_pred and pre_pred.nodes:
            self.update_bounds(pre_pred.nodes)
        if eff_pred and eff_pred.nodes:
            self.update_bounds(eff_pred.nodes)

        pre_inter = self._find_intersection(pre_pred)
        eff_inter = self._find_intersection(eff_pred)

        generated = []
        if pre_inter is not None:
            before = self._find_difference(pre_inter, eff_inter) if eff_inter is not None else pre_inter
            if before is not None and before.nodes and len(before.nodes) >= self.min_nodes:
                generated.append(before)

        if eff_inter is not None:
            after = self._find_difference(eff_inter, pre_inter) if pre_inter is not None else eff_inter
            if after is not None and after.nodes and len(after.nodes) >= self.min_nodes:
                generated.append(after)

        return generated

    def _manifolds_equivalent(self, pred_a: Predicate, pred_b: Predicate) -> bool:
        """Checks if two predicates on the same dimensions cover each other's GNG nodes."""
        if not pred_a.nodes or not pred_b.nodes:
            return False
        dims_a = pred_a.active_dimentions
        dims_b = pred_b.active_dimentions
        if dims_a != dims_b:
            return False

        a_in_b = sum(1 for pt in pred_a.nodes if pred_b.is_active(pt, vector_dims=dims_a)) / len(pred_a.nodes)
        b_in_a = sum(1 for pt in pred_b.nodes if pred_a.is_active(pt, vector_dims=dims_b)) / len(pred_b.nodes)
        return a_in_b >= self.min_overlap_ratio and b_in_a >= self.min_overlap_ratio

    def agregate_symbols(self, symbols: list) -> dict:
        """
        Finds the minimal set of elementary perceptual symbols covering all generated symbols.
        If symbol B's dimensions are a subset of symbol A's dimensions, A is replaced by
        B and A \\ B (dimension-wise). Equivalent symbols over the same subspace are merged.
        """
        # Clear old symbols completely
        self.symbols.clear()

        valid_symbols = [
            s for s in symbols
            if s is not None and s.nodes and s.active_dimentions
        ]
        if not valid_symbols:
            self.symbols = {}
            return self.symbols

        # 1. Dimension-wise atomic decomposition: split A into (A & B) and (A \ B)
        changed = True
        current_list = list(valid_symbols)
        while changed:
            changed = False
            next_list = []
            for i, sym_a in enumerate(current_list):
                dims_a = set(sym_a.active_dimentions)
                split_done = False

                for j, sym_b in enumerate(current_list):
                    if i == j:
                        continue
                    dims_b = set(sym_b.active_dimentions)
                    common = dims_a & dims_b
                    diff = dims_a - dims_b

                    # If B is a proper subset of A (or overlaps partially), split A into common and diff
                    if common and diff:
                        part_common = sym_a.clone_subspace(target_dims=sorted(common))
                        part_diff = sym_a.clone_subspace(target_dims=sorted(diff))
                        if part_common.nodes:
                            next_list.append(part_common)
                        if part_diff.nodes:
                            next_list.append(part_diff)
                        split_done = True
                        changed = True
                        break

                if not split_done:
                    next_list.append(sym_a)
            current_list = next_list

        # 2. Deduplicate symbols that share identical dimension subspaces and overlapping GNG nodes
        unique_symbols = []
        for cand in current_list:
            if not any(self._manifolds_equivalent(cand, existing) for existing in unique_symbols):
                unique_symbols.append(cand)

        self.symbols = {f"sym_{idx}": pred for idx, pred in enumerate(unique_symbols)}
        return self.symbols