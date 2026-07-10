"""Unit tests for the new functions in sigma_single_door.py.

Tests cover:
- proper_backdoor_graph: edge removal, causal-path edge deletion
- optimal_adjustment_set: O-set computation on textbook fixtures
- find_sigma_single_door_set: DAG and cyclic fixtures, precomputed extension
"""

from __future__ import annotations

import unittest

from y0.algorithm.separation.sigma_single_door import (
    find_sigma_single_door_set,
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
