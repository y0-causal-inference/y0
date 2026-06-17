"""Tests for the σ-separation single-door criterion (admissibility primitive).

Covers:
- Acyclic graphs: σ-single-door admissibility agrees with classical back-door
  admissibility (empty set is admissible when there is no confounding).
- 2-cycle x⇄y: no admissible Z exists (correctly non-identified).
- Direct feedback loop (Y→X): no admissible Z exists because the Y→X edge
  creates an unblockable 2-node path [X,Y] in the reduced graph.
- σ-admissibility ⊇ d-admissibility on acyclic graphs (every d-admissible set
  is also σ-admissible).
- find_sigma_single_door_sets yields only admissible sets.
- max_conditions cap is respected.
- Edge-not-present guard returns False.
- Descendant-in-Z guard returns False.
"""

import unittest

from y0.algorithm.separation.sigma_single_door import (
    find_sigma_single_door_sets,
    is_sigma_single_door_admissible,
)
from y0.dsl import Variable
from y0.graph import NxMixedGraph

# ---------------------------------------------------------------------------
# Shared variables
# ---------------------------------------------------------------------------
X, Y, Z1, Z2, W = map(Variable, ["X", "Y", "Z1", "Z2", "W"])


class TestIsSigmaSingleDoorAdmissible(unittest.TestCase):
    """Unit tests for is_sigma_single_door_admissible."""

    # ------------------------------------------------------------------
    # Acyclic graphs
    # ------------------------------------------------------------------

    def test_acyclic_chain_empty_set_admissible(self) -> None:
        """On a simple chain X→Y with no confounders, empty Z is admissible."""
        # X → Y  (no back-door paths)
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        self.assertTrue(is_sigma_single_door_admissible(graph, X, Y, set()))

    def test_acyclic_chain_with_confounder_empty_not_admissible(self) -> None:
        """With a latent confounder X↔Y, empty Z is not admissible."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[(X, Y)])
        self.assertFalse(is_sigma_single_door_admissible(graph, X, Y, set()))

    def test_acyclic_fork_parent_blocks_backdoor(self) -> None:
        """Z1 → X → Y, Z1 → Y: conditioning on Z1 blocks the back-door path."""
        graph = NxMixedGraph.from_edges(directed=[(Z1, X), (X, Y), (Z1, Y)])
        # Empty set: back-door path Z1→Y is open → not admissible
        self.assertFalse(is_sigma_single_door_admissible(graph, X, Y, set()))
        # Conditioning on Z1: blocks the back-door path → admissible
        self.assertTrue(is_sigma_single_door_admissible(graph, X, Y, {Z1}))

    def test_acyclic_descendant_in_z_not_admissible(self) -> None:
        """Z that contains a descendant of Y violates condition (i)."""
        # X → Y → W
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, W)])
        # W is a descendant of Y — must be rejected
        self.assertFalse(is_sigma_single_door_admissible(graph, X, Y, {W}))

    def test_acyclic_y_itself_in_z_not_admissible(self) -> None:
        """Y is a descendant of itself (descendants_inclusive), so Z={Y} is rejected."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        self.assertFalse(is_sigma_single_door_admissible(graph, X, Y, {Y}))

    def test_edge_not_present_returns_false(self) -> None:
        """If the directed edge cause→effect is absent, return False."""
        graph = NxMixedGraph.from_edges(directed=[(X, Z1)])
        self.assertFalse(is_sigma_single_door_admissible(graph, X, Y, set()))

    # ------------------------------------------------------------------
    # Cyclic graphs
    # ------------------------------------------------------------------

    def test_two_cycle_no_admissible_z(self) -> None:
        """2-cycle X⇄Y: no Z can σ-separate X from Y (non-identified)."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
        # Empty set
        self.assertFalse(is_sigma_single_door_admissible(graph, X, Y, set()))
        # Any non-descendant candidate (there are none here, but guard still holds)
        self.assertFalse(is_sigma_single_door_admissible(graph, X, Y, set()))

    def test_feedback_with_instrument_not_admissible(self) -> None:
        """W → X → Y, Y → X: no Z is admissible for X→Y.

        Graph: W → X → Y, Y → X (direct feedback loop).
        After removing X→Y, the reduced graph is W→X, Y→X.
        The disorient of the reduced graph has the edge X-Y (from Y→X).
        This creates a 2-node path [X, Y] with no triples to check.
        Since neither X nor Y is in Z={W}, the path is always Z-σ-open,
        so X and Y are never σ-separated — {W} is NOT admissible.

        Note: any direct Y→X edge in the original graph creates an
        unblockable 2-node path in the reduced graph, making the edge
        X→Y non-identifiable via the σ-separation single-door criterion.
        """
        # W → X → Y, Y → X  (direct feedback loop on X,Y)
        graph = NxMixedGraph.from_edges(directed=[(W, X), (X, Y), (Y, X)])
        # The 2-node path [X,Y] via Y→X is always open; {W} cannot block it.
        self.assertFalse(is_sigma_single_door_admissible(graph, X, Y, {W}))
        # No admissible set exists for X→Y in this graph
        self.assertEqual([], list(find_sigma_single_door_sets(graph, X, Y)))

    def test_cyclic_descendant_guard(self) -> None:
        """In a cycle X→Y→X, X is a descendant of Y; Z={X} must be rejected."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
        self.assertFalse(is_sigma_single_door_admissible(graph, X, Y, {X}))


class TestFindSigmaSingleDoorSets(unittest.TestCase):
    """Unit tests for find_sigma_single_door_sets."""

    def test_simple_chain_yields_empty_set(self) -> None:
        """X→Y with no confounders: empty set is the (only) admissible set."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        sets = list(find_sigma_single_door_sets(graph, X, Y))
        self.assertIn(frozenset(), sets)

    def test_fork_yields_z1(self) -> None:
        """Z1→X→Y, Z1→Y: {Z1} must appear among the admissible sets."""
        graph = NxMixedGraph.from_edges(directed=[(Z1, X), (X, Y), (Z1, Y)])
        sets = list(find_sigma_single_door_sets(graph, X, Y))
        self.assertIn(frozenset({Z1}), sets)

    def test_all_yielded_sets_are_admissible(self) -> None:
        """Every set yielded by find_sigma_single_door_sets must be admissible."""
        graph = NxMixedGraph.from_edges(directed=[(Z1, X), (Z2, X), (X, Y), (Z1, Y)])
        for z in find_sigma_single_door_sets(graph, X, Y):
            with self.subTest(z=z):
                self.assertTrue(is_sigma_single_door_admissible(graph, X, Y, z))

    def test_edge_absent_yields_nothing(self) -> None:
        """If the edge cause→effect is absent, no sets are yielded."""
        graph = NxMixedGraph.from_edges(directed=[(X, Z1)])
        sets = list(find_sigma_single_door_sets(graph, X, Y))
        self.assertEqual([], sets)

    def test_two_cycle_yields_nothing(self) -> None:
        """2-cycle X⇄Y: no admissible sets exist."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
        sets = list(find_sigma_single_door_sets(graph, X, Y))
        self.assertEqual([], sets)

    def test_max_conditions_cap_respected(self) -> None:
        """max_conditions=0 should only yield the empty set (if admissible)."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        sets = list(find_sigma_single_door_sets(graph, X, Y, max_conditions=0))
        # Only the empty set can be yielded
        for z in sets:
            self.assertEqual(frozenset(), z)

    def test_max_conditions_excludes_larger_sets(self) -> None:
        """max_conditions=1 should not yield sets of size > 1."""
        graph = NxMixedGraph.from_edges(
            directed=[(Z1, X), (Z2, X), (X, Y), (Z1, Y), (Z2, Y)]
        )
        for z in find_sigma_single_door_sets(graph, X, Y, max_conditions=1):
            with self.subTest(z=z):
                self.assertLessEqual(len(z), 1)

    def test_sigma_admissibility_subsumes_d_admissibility_on_dag(self) -> None:
        """On a DAG, every d-admissible set (empty set when no confounders) is σ-admissible."""
        # Simple DAG: Z1 → X → Y, no confounders on X→Y after conditioning on Z1
        graph = NxMixedGraph.from_edges(directed=[(Z1, X), (X, Y), (Z1, Y)])
        # {Z1} is d-admissible (blocks the back-door Z1→Y)
        d_admissible = frozenset({Z1})
        sigma_sets = set(find_sigma_single_door_sets(graph, X, Y))
        self.assertIn(d_admissible, sigma_sets)
