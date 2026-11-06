import os
import ast
import re
from collections import defaultdict
from typing import Optional, List, Set, Tuple
import numpy as np

from .predicate import Predicate


class SymbolGenerator:
    def __init__(
        self,
        precision: float = 0.9,
        min_nodes_in_symbol: Optional[int] = None,
        span_tolerance: float = 0.1,
        n_test_points: int = 100,
    ):
        self.symbols = {}
        # Mapping: {skill_name: {"precondition": {sym_name, ...}, "effect": {sym_name, ...}}}
        self.skill_map = defaultdict(lambda: {"precondition": set(), "effect": set()})
        self.precision = float(precision)
        self.min_nodes = int(min_nodes_in_symbol) if min_nodes_in_symbol is not None else None
        self.span_tolerance = float(span_tolerance)
        self.n_test_points = int(n_test_points)
        self.dim_min = None
        self.dim_max = None

    def update_bounds(self, samples):
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return

        b_min = np.min(arr, axis=0)
        b_max = np.max(arr, axis=0)

        if self.dim_min is None or len(self.dim_min) != arr.shape[1]:
            self.dim_min = b_min.copy()
            self.dim_max = b_max.copy()
        else:
            self.dim_min = np.minimum(self.dim_min, b_min)
            self.dim_max = np.maximum(self.dim_max, b_max)

    def _get_dimension_scale(self, dims: np.ndarray) -> np.ndarray:
        if self.dim_min is not None and self.dim_max is not None:
            dims = np.asarray(dims, dtype=int)
            if len(dims) > 0 and np.max(dims) < len(self.dim_min):
                full_span = self.dim_max - self.dim_min
                scale = full_span[dims]
                return np.where(scale <= 1e-9, 1.0, scale)
        return np.ones(len(dims), dtype=np.float64)

    def _sample_subspace_points(self, dims: np.ndarray, n_points: Optional[int] = None) -> np.ndarray:
        """Randomly samples n_test_points within [dim_min, dim_max] for specified subspace dimensions."""
        n = n_points if n_points is not None else self.n_test_points
        dims = np.asarray(dims, dtype=int)
        dim_count = len(dims)

        if dim_count == 0:
            return np.empty((0, 0), dtype=np.float64)

        if self.dim_min is not None and self.dim_max is not None:
            if np.max(dims) < len(self.dim_min):
                low = self.dim_min[dims]
                high = self.dim_max[dims]
                # Guard against zero-width bounds
                high = np.where(high <= low, low + 1.0, high)
                return np.random.uniform(low=low, high=high, size=(n, dim_count))

        return np.random.uniform(low=0.0, high=1.0, size=(n, dim_count))

    def get_skill_map(self) -> dict:
        return {
            sk: {"precondition": set(v["precondition"]), "effect": set(v["effect"])}
            for sk, v in self.skill_map.items()
        }

    def get_all_symbols(self) -> dict:
        return {
            name: pred
            for name, pred in self.symbols.items()
            if not name.startswith("raw_")
        }

    def _find_intersection(self, predicate: Predicate) -> Optional[Predicate]:
        """
        Extracts invariant subspace dimensions using sampled points within bounds.
        Tests sampled variation across stable dimensions against span_tolerance.
        """
        if predicate is None or len(predicate.nodes) == 0:
            return None

        nodes = predicate.nodes
        base_dims = (
            predicate.active_dimensions
            if predicate.active_dimensions is not None
            else np.arange(nodes.shape[1], dtype=int)
        )
        base_dims = np.asarray(base_dims, dtype=int)

        if len(nodes) < 2:
            return predicate.clone_subspace(target_dims=base_dims)

        # Guard: slice only if nodes represent full dimensionality
        if predicate.active_dimensions is not None and nodes.shape[1] > len(base_dims):
            if np.max(base_dims) < nodes.shape[1]:
                nodes = nodes[:, base_dims]

        dim_scale = self._get_dimension_scale(base_dims)
        median = np.median(nodes, axis=0)

        # Draw n_test_points bounded samples to test invariant volume dimensions
        sample_pts = self._sample_subspace_points(base_dims, n_points=self.n_test_points)
        diff_to_median = np.abs(sample_pts - median) / dim_scale
        within_tolerance = diff_to_median <= (self.span_tolerance / 2.0)

        # Invariant dimensions satisfy tolerance in at least `precision` fraction of tests
        invariant_mask = np.mean(within_tolerance, axis=0) >= self.precision
        invariant_dims = base_dims[invariant_mask]

        # Fallback to empirical MAD if bounds sampling yields no dimensions
        if len(invariant_dims) == 0:
            mad = np.median(np.abs(nodes - median), axis=0) / dim_scale
            stable_mask = (mad <= (self.span_tolerance / 2.0))
            invariant_dims = base_dims[stable_mask]

        if len(invariant_dims) == 0:
            return None

        return predicate.clone_subspace(target_dims=invariant_dims)

    def _find_difference(self, predicate_A: Predicate, predicate_B: Predicate) -> Optional[Predicate]:
        if predicate_A is None or len(predicate_A.nodes) == 0:
            return None
        if predicate_B is None or len(predicate_B.nodes) == 0:
            return predicate_A.clone_subspace(target_dims=predicate_A.active_dimensions)

        dims_A = predicate_A.active_dimensions if predicate_A.active_dimensions is not None else np.arange(predicate_A.nodes.shape[1])
        dims_B = predicate_B.active_dimensions if predicate_B.active_dimensions is not None else np.arange(predicate_B.nodes.shape[1])

        common_dims = np.intersect1d(dims_A, dims_B)
        if len(common_dims) == 0:
            return predicate_A.clone_subspace(target_dims=dims_A)

        # predicate_B.is_active handles dimensional projection via vector_dims
        active_in_B = predicate_B.is_active(predicate_A.nodes, vector_dims=dims_A)
        retained_indices = np.nonzero(~active_in_B)[0]

        threshold = self.min_nodes if self.min_nodes is not None else 1
        if len(retained_indices) < threshold:
            return None

        return predicate_A.clone_subspace(target_dims=dims_A, node_indices=retained_indices)

    def _check_sampled_overlap(
        self,
        pred_a: Predicate,
        pred_b: Predicate,
        common_dims: np.ndarray
    ) -> bool:
        """
        Samples n_test_points within min-max bounds on common_dims.
        Tests both predicates: if >= precision of points active in A are also active in B
        (and vice versa for common region), returns True.
        """
        if len(common_dims) == 0:
            return False

        sample_pts = self._sample_subspace_points(common_dims, n_points=self.n_test_points)
        sub_a = pred_a.clone_subspace(target_dims=common_dims)
        sub_b = pred_b.clone_subspace(target_dims=common_dims)

        if sub_a.gng.num_nodes == 0 or sub_b.gng.num_nodes == 0:
            return False

        active_a = sub_a.gng.is_active(sample_pts)
        active_b = sub_b.gng.is_active(sample_pts)

        total_active_union = np.sum(active_a | active_b)
        if total_active_union == 0:
            act_a_nodes = sub_b.gng.is_active(sub_a.nodes)
            act_b_nodes = sub_a.gng.is_active(sub_b.nodes)
            return bool((np.mean(act_a_nodes) >= self.precision) and (np.mean(act_b_nodes) >= self.precision))

        overlap_count = np.sum(active_a & active_b)
        overlap_ratio = overlap_count / float(total_active_union)
        return bool(overlap_ratio >= self.precision)

    def _manifolds_equivalent(self, pred_a: Predicate, pred_b: Predicate) -> bool:
        """Two predicates are equivalent if dimensions match and sampled overlap meets precision."""
        if pred_a is None or pred_b is None:
            return False
        if len(pred_a.nodes) == 0 or len(pred_b.nodes) == 0:
            return False

        dims_a = tuple(sorted(pred_a.active_dimensions)) if pred_a.active_dimensions is not None else tuple()
        dims_b = tuple(sorted(pred_b.active_dimensions)) if pred_b.active_dimensions is not None else tuple()

        if dims_a != dims_b or len(dims_a) == 0:
            return False

        return self._check_sampled_overlap(pred_a, pred_b, np.array(dims_a))

    def _get_next_symbol_name(self) -> str:
        idx = 0
        while f"sym_{idx}" in self.symbols:
            idx += 1
        return f"sym_{idx}"

    def prune_orphan_symbols(self) -> None:
        referenced = set()
        for roles in self.skill_map.values():
            referenced.update(roles["precondition"])
            referenced.update(roles["effect"])

        self.symbols = {
            k: v for k, v in self.symbols.items()
            if k in referenced or k.startswith("raw_")
        }

    def generate_symbols(self, skill_name: str, skill_data: dict):
        pre_pred = skill_data.get("precondition")
        eff_pred = skill_data.get("effect")

        if pre_pred and len(pre_pred.nodes) > 0:
            self.update_bounds(pre_pred.nodes)
        if eff_pred and len(eff_pred.nodes) > 0:
            self.update_bounds(eff_pred.nodes)

        pre_inter = self._find_intersection(pre_pred)
        eff_inter = self._find_intersection(eff_pred)

        self.skill_map[skill_name]["precondition"].clear()
        self.skill_map[skill_name]["effect"].clear()

        if pre_inter is not None:
            before = self._find_difference(pre_inter, eff_inter) if eff_inter is not None else pre_inter
            min_pre = self.min_nodes if self.min_nodes is not None else len(pre_pred.nodes)
            if before is not None and len(before.nodes) >= min_pre:
                temp_id = f"raw_pre_{skill_name}"
                self.symbols[temp_id] = before
                self.skill_map[skill_name]["precondition"].add(temp_id)

        if eff_inter is not None:
            after = self._find_difference(eff_inter, pre_inter) if pre_inter is not None else eff_inter
            min_eff = self.min_nodes if self.min_nodes is not None else len(eff_pred.nodes)
            if after is not None and len(after.nodes) >= min_eff:
                temp_id = f"raw_eff_{skill_name}"
                self.symbols[temp_id] = after
                self.skill_map[skill_name]["effect"].add(temp_id)

    def agregate_symbols(self, skill_name: str) -> None:
        """
        Aggregates raw symbols by testing overlaps on common dimensions with n_test_points.
        If an existing candidate symbol overlaps on common dimensions:
          - If the candidate has extra dimensions, it is split into common + extra parts.
          - In all skills referencing the candidate, the candidate symbol is replaced by both parts.
        If no canonical symbols match, raw_pred is registered as a new symbol.
        """
        if skill_name not in self.skill_map:
            return

        roles = self.skill_map[skill_name]
        raw_ids_to_clean = list(roles["precondition"] | roles["effect"])
        min_req = self.min_nodes if self.min_nodes is not None else 1

        for role in ("precondition", "effect"):
            current_raw_ids = list(roles[role])
            resolved_symbols = set()

            for raw_id in current_raw_ids:
                raw_pred = self.symbols.get(raw_id)
                if raw_pred is None or len(raw_pred.nodes) == 0:
                    continue

                raw_dims = set(raw_pred.active_dimensions)
                covered_dims = set()
                matched_symbols = set()

                canonical_syms = [
                    (s_name, s_pred) for s_name, s_pred in self.symbols.items()
                    if not s_name.startswith("raw_") and len(s_pred.nodes) >= min_req
                ]
                # Prioritize larger dimensional candidates first
                canonical_syms.sort(key=lambda item: len(item[1].active_dimensions), reverse=True)

                for c_name, c_pred in canonical_syms:
                    if c_name not in self.symbols:
                        continue  # In case c_name was already split in a previous iteration

                    c_dims = set(c_pred.active_dimensions)
                    common = np.array(sorted(list(raw_dims & c_dims)))

                    # Test common dimensions if not already covered
                    if len(common) > 0 and not set(common).issubset(covered_dims):
                        is_overlapping = self._check_sampled_overlap(raw_pred, c_pred, common)
                        if is_overlapping:
                            c_extra = sorted(list(c_dims - set(common)))
                            if c_extra:
                                # Candidate has extra dimensions: split candidate into common + extra
                                c_common_pred = c_pred.clone_subspace(target_dims=common)
                                c_extra_pred = c_pred.clone_subspace(target_dims=np.array(c_extra))

                                sym_common = self._get_next_symbol_name()
                                self.symbols[sym_common] = c_common_pred

                                sym_extra = self._get_next_symbol_name()
                                self.symbols[sym_extra] = c_extra_pred

                                # Replace c_name in all skills with both split symbols
                                for sk in self.skill_map:
                                    for r in ("precondition", "effect"):
                                        if c_name in self.skill_map[sk][r]:
                                            self.skill_map[sk][r].remove(c_name)
                                            self.skill_map[sk][r].add(sym_common)
                                            self.skill_map[sk][r].add(sym_extra)

                                self.symbols.pop(c_name, None)
                                matched_symbols.add(sym_common)
                            else:
                                matched_symbols.add(c_name)

                            covered_dims.update(common)

                    if covered_dims == raw_dims:
                        break

                leftover_dims = sorted(list(raw_dims - covered_dims))

                # Point 1: If no matches were found, register the raw predicate directly as a symbol
                if not matched_symbols:
                    new_sym = self._get_next_symbol_name()
                    self.symbols[new_sym] = raw_pred
                    resolved_symbols.add(new_sym)
                else:
                    resolved_symbols.update(matched_symbols)
                    # If partially covered, assign remaining dimensions to a new symbol
                    if leftover_dims:
                        rem_pred = raw_pred.clone_subspace(target_dims=np.array(leftover_dims))
                        if len(rem_pred.nodes) >= min_req:
                            rem_sym = None
                            for c_name, c_pred in [
                                (s, p) for s, p in self.symbols.items() if not s.startswith("raw_")
                            ]:
                                if self._manifolds_equivalent(rem_pred, c_pred):
                                    rem_sym = c_name
                                    break
                            if rem_sym is None:
                                rem_sym = self._get_next_symbol_name()
                                self.symbols[rem_sym] = rem_pred
                            resolved_symbols.add(rem_sym)

            roles[role] = resolved_symbols

        for raw_id in raw_ids_to_clean:
            self.symbols.pop(raw_id, None)

        self.prune_orphan_symbols()

    def merge_symbols(self) -> None:
        """
        Merges identical symbols based on sampled points within min-max bounds.
        """
        canonical_names = [
            k for k, p in self.symbols.items()
            if not k.startswith("raw_") and len(p.nodes) > 0
        ]
        merged_into = {}

        for i, sym_a in enumerate(canonical_names):
            if sym_a in merged_into:
                continue
            pred_a = self.symbols[sym_a]

            for sym_b in canonical_names[i + 1:]:
                if sym_b in merged_into:
                    continue
                pred_b = self.symbols[sym_b]

                if self._manifolds_equivalent(pred_a, pred_b):
                    merged_into[sym_b] = sym_a

        if not merged_into:
            return

        for sk in self.skill_map:
            for role in ("precondition", "effect"):
                current_syms = self.skill_map[sk][role]
                updated_syms = set()
                for s in current_syms:
                    root_s = s
                    while root_s in merged_into:
                        root_s = merged_into[root_s]
                    updated_syms.add(root_s)
                self.skill_map[sk][role] = updated_syms

        for duplicate_name in merged_into.keys():
            self.symbols.pop(duplicate_name, None)

        self.prune_orphan_symbols()