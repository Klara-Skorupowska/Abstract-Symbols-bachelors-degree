import os
import ast
import re
from collections import defaultdict
import numpy as np

from .predicate import Predicate


class SymbolGenerator:
    def __init__(
        self,
        min_overlap_ratio: float,
        min_nodes_in_symbol: int,
        span_tolerance: float,
    ):
        self.symbols = {}
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.min_nodes = int(min_nodes_in_symbol)
        self.span_tolerance = float(span_tolerance)
        self.dim_min = None
        self.dim_max = None

    def update_bounds(self, samples):
        """Aktualizuje granice przestrzeni cech (dim_min, dim_max) z wektorów stanu."""
        arr = np.asarray(samples, dtype=float)
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

    def _get_dimension_scale(self, dims: list) -> np.ndarray:
        """Zwraca zakres skali (max - min) dla zadanych wymiarów z zabezpieczeniem numerycznym."""
        if self.dim_min is not None and self.dim_max is not None:
            full_span = self.dim_max - self.dim_min
            scale = full_span[dims] if len(full_span) > max(dims, default=0) else full_span[:len(dims)]
            return np.where(scale <= 1e-9, 1.0, scale)
        return np.ones(len(dims), dtype=float)

    def _find_intersection(self, predicate: Predicate) -> Predicate:
        """
        Identyfikuje niezmiennicze wymiary rozmaitości.
        Matematyka: Robust Quantile-Span Ratio + Normalised Deviation.
        Wymiar uważa się za stabilny, jeśli efektywny rozstęp kwartylowy (IQR) oraz
        maksymalne odchylenie od centroidu mieszczą się w span_tolerance.
        """
        if predicate is None or not predicate.nodes:
            return None

        nodes = np.asarray(predicate.nodes, dtype=float)
        if nodes.ndim == 1:
            nodes = nodes.reshape(1, -1)

        base_dims = (
            list(predicate.active_dimensions)
            if predicate.active_dimensions is not None
            else list(range(nodes.shape[1]))
        )

        if len(nodes) < 2:
            return predicate.clone_subspace(target_dims=base_dims)

        if predicate.active_dimensions is not None and nodes.shape[1] > len(base_dims):
            nodes = nodes[:, base_dims]

        dim_scale = self._get_dimension_scale(base_dims)

        # 1. Mediana i MAD (Median Absolute Deviation) - odporność na szum GNG
        median = np.median(nodes, axis=0)
        mad = np.median(np.abs(nodes - median), axis=0) / dim_scale

        # 2. Zakres międzynormowy (skrajne odchylenia)
        ptp_norm = (np.max(nodes, axis=0) - np.min(nodes, axis=0)) / dim_scale

        # Warunek stabilności: rozrzut w wymiarze mieści się w tolerancji
        stable_mask = (ptp_norm <= self.span_tolerance) | (mad <= (self.span_tolerance / 2.0))

        # Fallback: jeśli żaden wymiar nie spełnia progu, bierzemy wymiary o minimalnym MAD
        if not np.any(stable_mask):
            min_mad = np.min(mad)
            stable_mask = mad <= (min_mad * 1.1 + 1e-6)

        invariant_dims = [base_dims[i] for i, ok in enumerate(stable_mask) if ok]
        if not invariant_dims:
            return None

        return predicate.clone_subspace(target_dims=invariant_dims)

    def _find_difference(self, predicate_A: Predicate, predicate_B: Predicate) -> Predicate:
        """
        Różnica topologiczna A \\ B.
        Matematyka: Zamiast binarnego strzału, weryfikuje dystans Mahalanobisa / znormalizowany
        dystans euklidesowy węzła A do chmury węzłów B. Węzeł A zostaje zachowany, jeśli leży
        poza strefą wpływu (Voronoi/Local Radius) rzutu B.
        """
        if predicate_A is None or not predicate_A.nodes:
            return None
        if predicate_B is None or not predicate_B.nodes:
            return predicate_A.clone_subspace(target_dims=predicate_A.active_dimensions)

        dims_A = list(predicate_A.active_dimensions or range(len(predicate_A.nodes[0])))
        dims_B = list(predicate_B.active_dimensions or range(len(predicate_B.nodes[0])))
        shared_dims = sorted(set(dims_A) & set(dims_B))

        if not shared_dims or dims_A != dims_B:
            return predicate_A.clone_subspace(target_dims=dims_A)

        proj_B = predicate_B.clone_subspace(target_dims=shared_dims)
        retained_indices = []

        for idx, node_a in enumerate(predicate_A.nodes):
            # Węzeł nie jest pokryty przez B
            if not proj_B.is_active(node_a, vector_dims=dims_A):
                retained_indices.append(idx)

        # BEZWZGLĘDNA KONTROLA MIN_NODES:
        if len(retained_indices) < self.min_nodes:
            return None

        return predicate_A.clone_subspace(target_dims=dims_A, node_indices=retained_indices)

    def generate_symbols(self, skill: dict) -> list:
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
        """Symetryczny test stopnia pokrycia chmur punktów na tych samych wymiarach."""
        if not pred_a.nodes or not pred_b.nodes:
            return False
        if pred_a.active_dimensions != pred_b.active_dimensions:
            return False

        dims = pred_a.active_dimensions
        cnt_a_in_b = sum(1 for p in pred_a.nodes if pred_b.is_active(p, vector_dims=dims))
        cnt_b_in_a = sum(1 for p in pred_b.nodes if pred_a.is_active(p, vector_dims=dims))

        ratio_a = cnt_a_in_b / len(pred_a.nodes)
        ratio_b = cnt_b_in_a / len(pred_b.nodes)

        return (ratio_a >= self.min_overlap_ratio) and (ratio_b >= self.min_overlap_ratio)

    def agregate_symbols(self, symbols: list) -> dict:
        """
        Agreguje symbole bez ich rozbijania na małe fragmenty.
        Usuwa symbole redundantne, tj. takie, których zbiór wymiarów
        jest w pełni pokryty (reprezentowany) przez sumę innych niezależnych symboli,
        bądź które stanowią nadmiarową kompozycję prostszych symboli.
        """
        self.symbols.clear()

        # 1. Filtrowanie wstępne wg min_nodes
        valid_candidates = [
            s for s in symbols
            if s is not None and s.nodes and len(s.nodes) >= self.min_nodes and s.active_dimensions
        ]
        if not valid_candidates:
            return self.symbols

        # 2. Połączenie/usunięcie duplikatów i rozmaitości równoważnych (min_overlap_ratio)
        unique_symbols = []
        for cand in valid_candidates:
            if not any(self._manifolds_equivalent(cand, existing) for existing in unique_symbols):
                unique_symbols.append(cand)

        # 3. Eliminacja symboli złożonych (Redundancy Elimination):
        # Sortujemy od symboli o największej liczbie wymiarów do najmniejszej,
        # aby sprawdzić, czy duży symbol można zastąpić sumą mniejszych/bazowych.
        unique_symbols.sort(key=lambda s: len(s.active_dimensions), reverse=True)

        retained_symbols = []
        for i, sym_target in enumerate(unique_symbols):
            target_dims = set(sym_target.active_dimensions)

            # Pozostałe symbole (szczególnie prostsze/o mniejszych lub innych wymiarach)
            other_symbols = [
                s for j, s in enumerate(unique_symbols)
                if i != j and len(s.nodes) >= self.min_nodes
            ]

            # Sprawdzamy, które symbole stanowią ścisłe podzbiory wymiarów sym_target
            subset_symbols = [
                s for s in other_symbols
                if set(s.active_dimensions).issubset(target_dims) and set(s.active_dimensions) != target_dims
            ]

            # Zbieramy sumę wymiarów tych pod-symboli
            covered_dims = set()
            for s in subset_symbols:
                covered_dims.update(s.active_dimensions)

            # Jeżeli suma wymiarów mniejszych symboli w pełni pokrywa target_dims,
            # to sym_target jest kompozycją (sym_1 = sym_2 + sym_3) i jest NIEPOTRZEBNY -> wyrzucamy go.
            if target_dims.issubset(covered_dims):
                # Opcjonalnie: upewniamy się, że pod-symbole są aktywne w obszarze sym_target
                continue

            retained_symbols.append(sym_target)

        # 4. Końcowe przypisanie sym_0, sym_1, ...
        # Sortujemy dla powtarzalności (np. po pierwszym wymiarze)
        retained_symbols.sort(key=lambda s: (min(s.active_dimensions), len(s.active_dimensions)))

        self.symbols = {
            f"sym_{idx}": pred
            for idx, pred in enumerate(retained_symbols)
            if len(pred.nodes) >= self.min_nodes
        }
        return self.symbols

    def reduce_symbols(self, file_path: str, symbols: dict) -> dict:
        """
        Konsoliduje symbole występujące zawsze wspólnie (co-occurrence footprint).
        Zwraca spójny słownik mapujący sym_0, sym_1...
        """
        if not symbols:
            return self.symbols

        input_map = symbols if isinstance(symbols, dict) else {
            f"sym_{i}": p for i, p in enumerate(symbols)
        }

        if not file_path or not os.path.exists(file_path):
            self.symbols = input_map
            return self.symbols

        # Odczyt kontekstów
        symbol_contexts = []
        pattern = re.compile(r"-\s*(?:Precondition|Effect)\s*Symbols\s*:\s*(\[[^\]]*\])")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    try:
                        symbol_contexts.append(set(ast.literal_eval(match.group(1))))
                    except Exception:
                        pass

        if not symbol_contexts:
            self.symbols = input_map
            return self.symbols

        # Wyznaczenie grup współwystępowania
        fingerprints = defaultdict(list)
        all_syms = {s for ctx in symbol_contexts for s in ctx}
        for s_name in all_syms:
            fp = tuple(s_name in ctx for ctx in symbol_contexts)
            fingerprints[fp].append(s_name)

        co_groups = [g for g in fingerprints.values() if len(g) > 1]
        sym_to_group = {s: g for g in co_groups for s in g}

        reduced_dict = {}
        processed_groups = set()
        out_idx = 0

        for name, pred in input_map.items():
            if name in sym_to_group:
                group = tuple(sorted(sym_to_group[name]))
                if group in processed_groups:
                    continue
                processed_groups.add(group)

                group_preds = [input_map[m] for m in group if m in input_map]
                if not group_preds:
                    continue

                merged_dims = set()
                for p in group_preds:
                    if p.active_dimensions:
                        merged_dims.update(p.active_dimensions)
                merged_dims = sorted(merged_dims)

                # Wybór bazowego predykatu o największej liczbie próbek/węzłów
                best_base = max(group_preds, key=lambda x: len(x.nodes))
                
                # Walidacja: czy możemy zrzutować na połączone wymiary bez utraty min_nodes
                candidate_joined = best_base.clone_subspace(target_dims=merged_dims)

                if candidate_joined.nodes and len(candidate_joined.nodes) >= self.min_nodes:
                    reduced_dict[f"sym_{out_idx}"] = candidate_joined
                    out_idx += 1
                else:
                    for gp in group_preds:
                        if len(gp.nodes) >= self.min_nodes:
                            reduced_dict[f"sym_{out_idx}"] = gp
                            out_idx += 1
            else:
                if len(pred.nodes) >= self.min_nodes:
                    reduced_dict[f"sym_{out_idx}"] = pred
                    out_idx += 1

        self.symbols = reduced_dict
        return self.symbols