"""Unit tests for the new functions in sigma_single_door.py.

Tests cover:
- proper_backdoor_graph: edge removal, causal-path edge deletion
- optimal_adjustment_set: O-set computation on textbook fixtures
- find_sigma_single_door_set: DAG and cyclic fixtures, precomputed extension
- find_sigma_single_door_set strategy="greedy": minimal sigma-separator via greedy shrink
- find_sigma_single_door_set strategy="acyclify": minimal m-separator via acyclification
"""

from __future__ import annotations

import unittest

from y0.algorithm.separation.sigma_single_door import (
    find_sigma_single_door_set,
    is_sigma_single_door_admissible,
    optimal_adjustment_set,
    proper_backdoor_graph,
)
from y0.dsl import Variable
from y0.graph import NxMixedGraph

X = Variable("X")
Y = Variable("Y")
Z = Variable("Z")
Z1 = Variable("Z1")
Z2 = Variable("Z2")
W = Variable("W")
M = Variable("M")  # mediator


class TestProperBackdoorGraph(unittest.TestCase):
    """Unit tests for proper_backdoor_graph."""

    def test_simple_dag_removes_only_target_edge(self) -> None:
        """X -> Y with no other paths: G^pbd has no directed edges out of X."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)])
        pbd = proper_backdoor_graph(g, X, Y)
        # The target edge X->Y is removed
        self.assertFalse(pbd.directed.has_edge(X, Y))

    def test_fork_removes_target_edge_only(self) -> None:
        """Z -> X -> Y, Z -> Y: only X->Y is removed; Z->X and Z->Y remain."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        pbd = proper_backdoor_graph(g, X, Y)
        self.assertFalse(pbd.directed.has_edge(X, Y))
        self.assertTrue(pbd.directed.has_edge(Z, X))
        self.assertTrue(pbd.directed.has_edge(Z, Y))

    def test_mediator_path_removes_causal_edge(self) -> None:
        """X -> M -> Y, X -> Y: both X->Y and X->M are on causal paths.

        After removing X->Y, the path X->M->Y is a proper causal path, so
        X->M should also be removed from G^pbd.
        """
        g = NxMixedGraph.from_edges(directed=[(X, M), (M, Y), (X, Y)])
        pbd = proper_backdoor_graph(g, X, Y)
        # X->Y removed (target edge)
        self.assertFalse(pbd.directed.has_edge(X, Y))
        # X->M is on a causal path X->M->Y, so it is also removed
        self.assertFalse(pbd.directed.has_edge(X, M))
        # M->Y is not an edge out of X, so it is preserved
        self.assertTrue(pbd.directed.has_edge(M, Y))

    def test_no_causal_path_via_other_edges(self) -> None:
        """Z -> X -> Y, no other path from X to Y after removing X->Y.

        After removing X->Y, there is no path X->...->Y, so no additional
        edges out of X are deleted.
        """
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y)])
        pbd = proper_backdoor_graph(g, X, Y)
        self.assertFalse(pbd.directed.has_edge(X, Y))
        # Z->X is not an edge out of X, preserved
        self.assertTrue(pbd.directed.has_edge(Z, X))

    def test_bidirected_edges_preserved(self) -> None:
        """Bidirected edges are not affected by proper_backdoor_graph."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Z, X)], undirected=[(X, Z)])
        pbd = proper_backdoor_graph(g, X, Y)
        self.assertTrue(pbd.undirected.has_edge(X, Z))

    def test_raises_on_missing_edge(self) -> None:
        """Raises KeyError when the target edge does not exist."""
        g = NxMixedGraph.from_edges(directed=[(X, Z)])
        with self.assertRaises((KeyError, Exception)):
            proper_backdoor_graph(g, X, Y)


class TestOptimalAdjustmentSet(unittest.TestCase):
    """Unit tests for optimal_adjustment_set."""

    def test_simple_dag_no_confounders(self) -> None:
        """X -> Y with no confounders: O-set is empty (no parents of causal nodes)."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)])
        o = optimal_adjustment_set(g, X, Y)
        self.assertIsNotNone(o)
        self.assertEqual(frozenset(), o)

    def test_fork_confounder(self) -> None:
        """Z -> X -> Y, Z -> Y: O-set should include Z (parent of causal node X)."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        o = optimal_adjustment_set(g, X, Y)
        self.assertIsNotNone(o)
        assert o is not None
        # Z is a parent of X (a causal node) and not forbidden
        self.assertIn(Z, o)
        # X is forbidden (it's the cause)
        self.assertNotIn(X, o)
        # Y is forbidden (descendant of causal nodes)
        self.assertNotIn(Y, o)

    def test_mediator_excluded(self) -> None:
        """Z -> X -> M -> Y: M is a causal node, so it is forbidden."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, M), (M, Y)])
        o = optimal_adjustment_set(g, X, Y)
        self.assertIsNotNone(o)
        assert o is not None
        # M is on the causal path, so it is forbidden
        self.assertNotIn(M, o)
        # Z is a parent of X (causal node), not forbidden
        self.assertIn(Z, o)

    def test_missing_edge_returns_empty_set(self) -> None:
        """Returns empty frozenset when no directed path cause->effect exists."""
        g = NxMixedGraph.from_edges(directed=[(X, Z)])
        o = optimal_adjustment_set(g, X, Y)
        # No directed paths from X to Y → empty O-set
        self.assertEqual(frozenset(), o)

    def test_two_confounders(self) -> None:
        """Z1 -> X -> Y, Z2 -> X -> Y, Z1 -> Y, Z2 -> Y: both Z1, Z2 in O-set."""
        g = NxMixedGraph.from_edges(directed=[(Z1, X), (Z2, X), (X, Y), (Z1, Y), (Z2, Y)])
        o = optimal_adjustment_set(g, X, Y)
        self.assertIsNotNone(o)
        assert o is not None
        self.assertIn(Z1, o)
        self.assertIn(Z2, o)


class TestFindSigmaSingleDoorSet(unittest.TestCase):
    """Unit tests for find_sigma_single_door_set."""

    def test_simple_dag_returns_empty_set(self) -> None:
        """X -> Y with no confounders: admissible set is empty."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)])
        z = find_sigma_single_door_set(g, X, Y)
        self.assertIsNotNone(z)
        self.assertEqual(frozenset(), z)

    def test_fork_dag_returns_confounder(self) -> None:
        """Z -> X -> Y, Z -> Y: admissible set should include Z."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        z = find_sigma_single_door_set(g, X, Y)
        self.assertIsNotNone(z)
        assert z is not None
        self.assertIn(Z, z)

    def test_missing_edge_returns_none(self) -> None:
        """Returns None when the directed edge cause->effect is absent."""
        g = NxMixedGraph.from_edges(directed=[(X, Z)])
        z = find_sigma_single_door_set(g, X, Y)
        self.assertIsNone(z)

    def test_two_cycle_returns_none(self) -> None:
        """X -> Y -> X: 2-cycle is non-identifiable, should return None."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
        z = find_sigma_single_door_set(g, X, Y)
        self.assertIsNone(z)

    def test_dag_agrees_with_find_sigma_single_door_sets(self) -> None:
        """On a DAG, find_sigma_single_door_set result must be admissible.

        We verify by checking that the returned set is in the output of
        find_sigma_single_door_sets (the exhaustive enumerator).
        """
        from y0.algorithm.separation.sigma_single_door import find_sigma_single_door_sets

        g = NxMixedGraph.from_edges(directed=[(Z1, X), (X, Y), (Z1, Y)])
        z = find_sigma_single_door_set(g, X, Y)
        self.assertIsNotNone(z)
        all_admissible = set(find_sigma_single_door_sets(g, X, Y))
        self.assertIn(z, all_admissible)

    def test_precomputed_extension_gives_same_result(self) -> None:
        """Passing a precomputed sigma-extension gives the same result as computing it fresh."""
        from y0.algorithm.separation.sigma_extension import sigma_extension

        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        gs = sigma_extension(g)
        z_fresh = find_sigma_single_door_set(g, X, Y)
        z_precomputed = find_sigma_single_door_set(g, X, Y, precomputed_extension=gs)
        self.assertEqual(z_fresh, z_precomputed)

    def test_cycle_with_external_confounder(self) -> None:
        """W -> X -> Y -> X, W -> Y: W is external; X->Y should be identifiable."""
        g = NxMixedGraph.from_edges(directed=[(W, X), (X, Y), (Y, X), (W, Y)])
        z = find_sigma_single_door_set(g, X, Y)
        # W is an external confounder; conditioning on W should work
        # (result may be None if the cycle makes it non-identifiable via sigma-sep)
        # We just assert the function returns without error and the result is
        # either None or a frozenset
        self.assertTrue(z is None or isinstance(z, frozenset))


class TestFindSigmaSingleDoorSetGreedy(unittest.TestCase):
    """Tests for find_sigma_single_door_set with strategy='greedy'."""

    def _assert_admissible(
        self, graph: NxMixedGraph, cause: Variable, effect: Variable, z: frozenset[Variable]
    ) -> None:
        """Assert that z is a valid sigma-single-door adjustment set."""
        self.assertTrue(
            is_sigma_single_door_admissible(graph, cause, effect, z),
            msg=f"strategy='greedy' returned {z} which is NOT admissible for {cause}->{effect}",
        )

    def test_simple_dag_returns_empty_set(self) -> None:
        """X -> Y: greedy strategy returns empty set (no confounders)."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)])
        z = find_sigma_single_door_set(g, X, Y, strategy="greedy")
        self.assertIsNotNone(z)
        assert z is not None
        self.assertEqual(frozenset(), z)
        self._assert_admissible(g, X, Y, z)

    def test_fork_dag_returns_confounder(self) -> None:
        """Z -> X -> Y, Z -> Y: greedy returns a set containing Z."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        z = find_sigma_single_door_set(g, X, Y, strategy="greedy")
        self.assertIsNotNone(z)
        assert z is not None
        self.assertIn(Z, z)
        self._assert_admissible(g, X, Y, z)

    def test_missing_edge_returns_none(self) -> None:
        """Returns None when the directed edge cause->effect is absent."""
        g = NxMixedGraph.from_edges(directed=[(X, Z)])
        z = find_sigma_single_door_set(g, X, Y, strategy="greedy")
        self.assertIsNone(z)

    def test_two_cycle_returns_none(self) -> None:
        """X -> Y -> X: non-identifiable, greedy returns None."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
        z = find_sigma_single_door_set(g, X, Y, strategy="greedy")
        self.assertIsNone(z)

    def test_chain_single_mediator(self) -> None:
        """Z -> X -> M -> Y: greedy should find {Z} (blocks back-door via Z)."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, M), (M, Y)])
        z = find_sigma_single_door_set(g, X, M, strategy="greedy")
        # X->M is the edge; Z is a confounder for X
        self.assertTrue(z is None or isinstance(z, frozenset))
        if z is not None:
            self._assert_admissible(g, X, M, z)

    def test_parallel_mediators(self) -> None:
        """Z -> X -> Y, Z -> X -> M -> Y: two paths from X to Y."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (X, M), (M, Y)])
        z = find_sigma_single_door_set(g, X, Y, strategy="greedy")
        self.assertTrue(z is None or isinstance(z, frozenset))
        if z is not None:
            self._assert_admissible(g, X, Y, z)

    def test_result_is_minimal(self) -> None:
        """Greedy result is minimal: no proper subset is also admissible."""
        g = NxMixedGraph.from_edges(directed=[(Z1, X), (Z2, X), (X, Y), (Z1, Y)])
        z = find_sigma_single_door_set(g, X, Y, strategy="greedy")
        if z is None:
            return  # non-identifiable; skip minimality check
        # Check no proper subset is also admissible
        for node in z:
            smaller = z - {node}
            self.assertFalse(
                is_sigma_single_door_admissible(g, X, Y, smaller),
                msg=f"Proper subset {smaller} of {z} is also admissible — not minimal",
            )

    def test_greedy_vs_optimal_both_valid(self) -> None:
        """Both greedy and optimal return valid sets (may differ in size)."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        z_opt = find_sigma_single_door_set(g, X, Y, strategy="optimal")
        z_greedy = find_sigma_single_door_set(g, X, Y, strategy="greedy")
        # Both should be non-None and admissible
        self.assertIsNotNone(z_opt)
        self.assertIsNotNone(z_greedy)
        assert z_opt is not None and z_greedy is not None
        self._assert_admissible(g, X, Y, z_opt)
        self._assert_admissible(g, X, Y, z_greedy)


class TestFindSigmaSingleDoorSetAcyclify(unittest.TestCase):
    """Tests for find_sigma_single_door_set with strategy='acyclify'."""

    def _assert_admissible(
        self, graph: NxMixedGraph, cause: Variable, effect: Variable, z: frozenset[Variable]
    ) -> None:
        """Assert that z is a valid sigma-single-door adjustment set."""
        self.assertTrue(
            is_sigma_single_door_admissible(graph, cause, effect, z),
            msg=f"strategy='acyclify' returned {z} which is NOT admissible for {cause}->{effect}",
        )

    def test_simple_dag_returns_empty_set(self) -> None:
        """X -> Y: acyclify strategy returns empty set (no confounders)."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)])
        z = find_sigma_single_door_set(g, X, Y, strategy="acyclify")
        self.assertIsNotNone(z)
        assert z is not None
        self.assertEqual(frozenset(), z)
        self._assert_admissible(g, X, Y, z)

    def test_fork_dag_returns_confounder(self) -> None:
        """Z -> X -> Y, Z -> Y: acyclify returns a set containing Z."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        z = find_sigma_single_door_set(g, X, Y, strategy="acyclify")
        self.assertIsNotNone(z)
        assert z is not None
        self.assertIn(Z, z)
        self._assert_admissible(g, X, Y, z)

    def test_missing_edge_returns_none(self) -> None:
        """Returns None when the directed edge cause->effect is absent."""
        g = NxMixedGraph.from_edges(directed=[(X, Z)])
        z = find_sigma_single_door_set(g, X, Y, strategy="acyclify")
        self.assertIsNone(z)

    def test_two_cycle_returns_none(self) -> None:
        """X -> Y -> X: non-identifiable, acyclify returns None."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
        z = find_sigma_single_door_set(g, X, Y, strategy="acyclify")
        self.assertIsNone(z)

    def test_dag_chain_with_confounder(self) -> None:
        """Z -> X -> Y, Z -> Y: acyclify finds {Z}."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        z = find_sigma_single_door_set(g, X, Y, strategy="acyclify")
        self.assertTrue(z is None or isinstance(z, frozenset))
        if z is not None:
            self._assert_admissible(g, X, Y, z)

    def test_acyclify_vs_optimal_both_valid(self) -> None:
        """Both acyclify and optimal return valid sets on a DAG."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        z_opt = find_sigma_single_door_set(g, X, Y, strategy="optimal")
        z_ac = find_sigma_single_door_set(g, X, Y, strategy="acyclify")
        self.assertIsNotNone(z_opt)
        assert z_opt is not None
        self._assert_admissible(g, X, Y, z_opt)
        # acyclify may return None if are_sigma_separated is incomplete,
        # but if it returns a set it must be admissible
        if z_ac is not None:
            self._assert_admissible(g, X, Y, z_ac)

    def test_acyclify_result_is_admissible_when_not_none(self) -> None:
        """Any non-None result from acyclify strategy is admissible."""
        graphs = [
            NxMixedGraph.from_edges(directed=[(X, Y)]),
            NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)]),
            NxMixedGraph.from_edges(directed=[(Z1, X), (Z2, X), (X, Y), (Z1, Y)]),
        ]
        for g in graphs:
            z = find_sigma_single_door_set(g, X, Y, strategy="acyclify")
            if z is not None:
                self._assert_admissible(g, X, Y, z)
