from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .predicate import Predicate


class SymbolGenerator:
    def __init__(
        self,
        precision: float = 0.9,
        min_nodes_in_symbol: Optional[int] = None,
        n_test_points: int = 256,
        seed: int = 42,
    ):
        self.symbols: Dict[str, Predicate] = {}
        # Mapping: {skill_name: {"precondition": {sym_name, ...}, "effect": {sym_name, ...}}}
        self.skill_map: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: {"precondition": set(), "effect": set()}
        )
        self.precision = float(precision)
        self.min_nodes = int(min_nodes_in_symbol) if min_nodes_in_symbol is not None else 1
        self.n_test_points = int(n_test_points)
        self._rng = np.random.default_rng(seed)

    def _get_bounding_box(self, pred: Predicate) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Zwraca (box_min, box_max) uwzględniając promienie hipersfer wokół węzłów."""
        if pred is None or len(pred.nodes) == 0:
            return None
        nodes = np.asarray(pred.nodes, dtype=np.float64)
        if nodes.ndim == 1:
            nodes = nodes.reshape(1, -1)
        radii = np.asarray(pred.radii, dtype=np.float64).reshape(-1, 1)
        return np.min(nodes - radii, axis=0), np.max(nodes + radii, axis=0)

    def _get_symbol_cardinality(self, pred: Predicate) -> Tuple[int, float]:
        """
        Zwraca (liczba_węzłów, suma_promieni) jako bezpieczną w 248D miarę rozmiaru symbolu
        (unika underflow do 0.0 przy mnożeniu 248 wymiarów).
        """
        if pred is None or len(pred.nodes) == 0:
            return (0, 0.0)
        return (len(pred.nodes), float(np.sum(pred.radii)))

    def _same_cardinality(self, card_a: Tuple[int, float], card_b: Tuple[int, float]) -> bool:
        return card_a[0] == card_b[0] and bool(
            np.isclose(card_a[1], card_b[1], rtol=1e-5, atol=1e-9)
        )

    def _strictly_smaller(self, small_card: Tuple[int, float], large_card: Tuple[int, float]) -> bool:
        if self._same_cardinality(small_card, large_card):
            return False
        if small_card[0] != large_card[0]:
            return small_card[0] < large_card[0]
        return small_card[1] < large_card[1]

    def _boxes_intersect(
        self,
        bbox_a: Tuple[np.ndarray, np.ndarray],
        bbox_b: Tuple[np.ndarray, np.ndarray],
        tol: float = 0.0,
    ) -> bool:
        """Checks whether two axis-aligned bounding boxes overlap."""
        min_a, max_a = bbox_a
        min_b, max_b = bbox_b
        return bool(np.all((max_a + tol) >= min_b) and np.all((max_b + tol) >= min_a))

    def _sample_spheres_points(self, pred: Predicate, n_points: int) -> np.ndarray:
        """
        Losuje punkty jednostajnie wewnątrz unii hipersfer węzłów predykatu w D-wymiarach.
        """
        nodes = np.asarray(pred.nodes, dtype=np.float64)
        radii = np.asarray(pred.radii, dtype=np.float64)
        n_nodes, dim = nodes.shape

        # Wybieramy losowe węzły (z wagą proporcjonalną do r, by unikać dominacji małych kul)
        weights = radii / np.sum(radii)
        chosen_idx = self._rng.choice(n_nodes, size=n_points, p=weights)

        # Losowy kierunek na sferze jednostkowej w D wymiarach
        directions = self._rng.normal(size=(n_points, dim))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        unit_dirs = directions / norms

        # Losowy promień wewnątrz kuli D-wymiarowej: r * U^(1/D)
        u = self._rng.uniform(0.0, 1.0, size=(n_points, 1))
        scales = radii[chosen_idx, None] * (u ** (1.0 / dim))

        return nodes[chosen_idx] + unit_dirs * scales

    def get_skill_map(self) -> dict:
        return {
            sk: {"precondition": set(v["precondition"]), "effect": set(v["effect"])}
            for sk, v in self.skill_map.items()
        }

    def get_all_symbols(self) -> dict:
        return dict(self.symbols)

    def _find_difference(self, predicate_A: Predicate, predicate_B: Predicate) -> Optional[Predicate]:
        """
        Computes set difference A \\ B over GNG Voronoi cells:
        retains nodes in A that do not fall inside active Voronoi cells of B.
        """
        if predicate_A is None or len(predicate_A.nodes) == 0:
            return None
        if predicate_B is None or len(predicate_B.nodes) == 0:
            return predicate_A.clone_nodes()

        bbox_a = self._get_bounding_box(predicate_A)
        bbox_b = self._get_bounding_box(predicate_B)
        if bbox_a is not None and bbox_b is not None and not self._boxes_intersect(bbox_a, bbox_b):
            return predicate_A.clone_nodes()

        active_in_B = np.atleast_1d(predicate_B.is_active(predicate_A.nodes))
        retained_indices = np.nonzero(~active_in_B)[0]

        if len(retained_indices) < self.min_nodes:
            return None

        return predicate_A.clone_nodes(node_indices=retained_indices)

    def _find_intersection(self, predicate_A: Predicate, predicate_B: Predicate) -> Optional[Predicate]:
        """
        Approximates A ∩ B by retaining nodes of A active in B together with
        nodes of B active in A.
        """
        if predicate_A is None or predicate_B is None:
            return None
        if len(predicate_A.nodes) == 0 or len(predicate_B.nodes) == 0:
            return None

        bbox_a = self._get_bounding_box(predicate_A)
        bbox_b = self._get_bounding_box(predicate_B)
        if bbox_a is None or bbox_b is None or not self._boxes_intersect(bbox_a, bbox_b):
            return None

        idx_a = np.nonzero(np.atleast_1d(predicate_B.is_active(predicate_A.nodes)))[0]
        idx_b = np.nonzero(np.atleast_1d(predicate_A.is_active(predicate_B.nodes)))[0]

        part_a = predicate_A.clone_nodes(node_indices=idx_a) if len(idx_a) > 0 else None
        part_b = predicate_B.clone_nodes(node_indices=idx_b) if len(idx_b) > 0 else None

        if part_a is not None and part_b is not None:
            inter = part_a.merge_with(part_b)
        else:
            inter = part_a or part_b

        if inter is None or len(inter.nodes) < self.min_nodes:
            return None
        return inter

    def _predicates_overlap(self, pred_a: Predicate, pred_b: Predicate) -> bool:
        if pred_a is None or pred_b is None or len(pred_a.nodes) == 0 or len(pred_b.nodes) == 0:
            return False
        bbox_a = self._get_bounding_box(pred_a)
        bbox_b = self._get_bounding_box(pred_b)
        if bbox_a is None or bbox_b is None or not self._boxes_intersect(bbox_a, bbox_b, tol=0.0):
            return False
        # Consistent with _find_difference / _find_intersection node-level resolution
        return bool(np.any(pred_b.is_active(pred_a.nodes)) or np.any(pred_a.is_active(pred_b.nodes)))

    def _manifolds_equivalent(self, pred_a: Predicate, pred_b: Predicate) -> bool:
        """
        Sprawdza równoważność dwóch predykatów miarą Jaccarda |A ∩ B| / |A ∪ B|
        poprzez próbkowanie bezpośrednio wewnątrz kul A i B (działa stabilnie w 248D).
        """
        if pred_a is None or pred_b is None:
            return False
        if len(pred_a.nodes) == 0 or len(pred_b.nodes) == 0:
            return False

        if not self._predicates_overlap(pred_a, pred_b):
            return False

        half_n = max(64, self.n_test_points // 2)
        pts_a = self._sample_spheres_points(pred_a, half_n)
        pts_b = self._sample_spheres_points(pred_b, half_n)
        sample_pts = np.vstack([pts_a, pts_b])

        active_a = np.atleast_1d(pred_a.is_active(sample_pts))
        active_b = np.atleast_1d(pred_b.is_active(sample_pts))

        union_count = int(np.sum(active_a | active_b))
        if union_count == 0:
            return False

        overlap_ratio = float(np.sum(active_a & active_b)) / float(union_count)
        return bool(overlap_ratio >= self.precision)

    def _get_next_symbol_name(self) -> str:
        idx = 0
        while f"sym_{idx}" in self.symbols:
            idx += 1
        return f"sym_{idx}"

    def _register_or_match_symbol(self, pred: Predicate) -> str:
        """Returns an existing equivalent symbol ID or registers a new sym_<i>."""
        for existing_name, existing_pred in self.symbols.items():
            if self._manifolds_equivalent(pred, existing_pred):
                return existing_name
        sym_name = self._get_next_symbol_name()
        self.symbols[sym_name] = pred
        return sym_name

    def _replace_symbol_in_skill_map(self, old_sym: str, replacement_syms: Set[str]) -> None:
        """Replaces `old_sym` with `replacement_syms` across all skills' preconditions and effects."""
        for roles in self.skill_map.values():
            for role in ("precondition", "effect"):
                if old_sym in roles[role]:
                    roles[role].remove(old_sym)
                    roles[role].update(replacement_syms)

    def prune_orphan_symbols(self) -> None:
        referenced = set()
        for roles in self.skill_map.values():
            referenced.update(roles["precondition"])
            referenced.update(roles["effect"])

        self.symbols = {k: v for k, v in self.symbols.items() if k in referenced}

    def generate_symbols(self, skill_name: str, skill_data: dict) -> None:
        """
        Forms abstract precondition and effect symbols from already-intersected
        GNG predicates P_init and P_eff:
            Q_init = P_init \\ P_eff
            Q_eff  = P_eff  \\ P_init
        """
        p_init: Optional[Predicate] = skill_data.get("precondition")
        p_eff: Optional[Predicate] = skill_data.get("effect")

        new_pre: Set[str] = set()
        new_eff: Set[str] = set()

        if p_init is not None and len(p_init.nodes) > 0:
            q_init = (
                self._find_difference(p_init, p_eff)
                if (p_eff is not None and len(p_eff.nodes) > 0)
                else p_init.clone_nodes()
            )
            if q_init is not None and len(q_init.nodes) >= self.min_nodes:
                new_pre.add(self._register_or_match_symbol(q_init))

        if p_eff is not None and len(p_eff.nodes) > 0:
            q_eff = (
                self._find_difference(p_eff, p_init)
                if (p_init is not None and len(p_init.nodes) > 0)
                else p_eff.clone_nodes()
            )
            if q_eff is not None and len(q_eff.nodes) >= self.min_nodes:
                new_eff.add(self._register_or_match_symbol(q_eff))

        if new_pre or new_eff:
            self.skill_map[skill_name] = {"precondition": new_pre, "effect": new_eff}
        else:
            self.skill_map.pop(skill_name, None)

        self.prune_orphan_symbols()

    def partition_symbols(self) -> None:
        """
        Partitions existing symbols into mutually disjoint regions:
        1. Equivalent symbols of matching cardinality are merged via connected components.
        2. Overlapping pairs (A, B) with |B| < |A| are split without expanding support:
           - If B ⊂ A: A -> {B, A \\ B}
           - If B partially overlaps A: A -> {A \\ B, A ∩ B} and B -> {B \\ A, A ∩ B}
        """
        if len(self.symbols) < 2:
            self.prune_orphan_symbols()
            return

        # Step 1: Merge equivalent same-cardinality symbols via frozen snapshot & connected components
        sym_names = list(self.symbols.keys())
        frozen_cards = {
            name: self._get_symbol_cardinality(pred) for name, pred in self.symbols.items()
        }
        adj: Dict[str, List[str]] = {name: [] for name in sym_names}

        for i in range(len(sym_names)):
            name_a = sym_names[i]
            for j in range(i + 1, len(sym_names)):
                name_b = sym_names[j]
                if self._same_cardinality(frozen_cards[name_a], frozen_cards[name_b]) and (
                    self._manifolds_equivalent(self.symbols[name_a], self.symbols[name_b])
                ):
                    adj[name_a].append(name_b)
                    adj[name_b].append(name_a)

        visited: Set[str] = set()
        for name in sym_names:
            if name in visited or not adj[name]:
                continue
            component: List[str] = []
            stack = [name]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                component.append(curr)
                stack.extend(adj[curr])

            if len(component) > 1:
                component.sort()
                canonical = component[0]
                merged_pred = self.symbols[canonical]
                for other in component[1:]:
                    merged_pred = merged_pred.merge_with(self.symbols[other])
                    self._replace_symbol_in_skill_map(other, {canonical})
                    del self.symbols[other]
                self.symbols[canonical] = merged_pred

        # Step 2: Order symbols from largest to smallest and split overlaps cleanly
        ordered_syms = sorted(
            self.symbols.keys(),
            key=lambda name: self._get_symbol_cardinality(self.symbols[name]),
            reverse=True,
        )

        for idx, large_name in enumerate(ordered_syms):
            if large_name not in self.symbols:
                continue

            remainder_pred: Optional[Predicate] = self.symbols[large_name]
            large_replacement: Set[str] = set()
            was_split = False

            # Iterate from next-largest down to smallest
            for small_name in ordered_syms[idx + 1 :]:
                if small_name not in self.symbols or remainder_pred is None:
                    continue

                rem_card = self._get_symbol_cardinality(remainder_pred)
                small_pred = self.symbols[small_name]
                small_card = self._get_symbol_cardinality(small_pred)

                if not self._strictly_smaller(small_card, rem_card):
                    continue

                if not self._predicates_overlap(remainder_pred, small_pred):
                    continue

                was_split = True
                small_minus_rem = self._find_difference(small_pred, remainder_pred)
                rem_minus_small = self._find_difference(remainder_pred, small_pred)

                if small_minus_rem is None:
                    # small_pred is fully contained within remainder_pred
                    large_replacement.add(small_name)
                    remainder_pred = rem_minus_small
                else:
                    # Partial overlap: split into (rem \ small), (small \ rem), and (rem ∩ small)
                    inter_pred = self._find_intersection(remainder_pred, small_pred)
                    del self.symbols[small_name]

                    small_replacement: Set[str] = set()
                    small_diff_sym = self._register_or_match_symbol(small_minus_rem)
                    small_replacement.add(small_diff_sym)

                    if inter_pred is not None:
                        inter_sym = self._register_or_match_symbol(inter_pred)
                        small_replacement.add(inter_sym)
                        large_replacement.add(inter_sym)

                    self._replace_symbol_in_skill_map(small_name, small_replacement)
                    remainder_pred = rem_minus_small

            if was_split:
                # Remove large_name before registering remainder so it cannot match itself
                del self.symbols[large_name]
                if remainder_pred is not None and len(remainder_pred.nodes) >= self.min_nodes:
                    rem_sym = self._register_or_match_symbol(remainder_pred)
                    large_replacement.add(rem_sym)

                self._replace_symbol_in_skill_map(large_name, large_replacement)

        self.prune_orphan_symbols()