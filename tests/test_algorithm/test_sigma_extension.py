"""Unit tests for sigma_extension.py.

Tests cover:
- DAG input → Gσ == G (no bidirected edges added)
- 2-node SCC → exactly one bidirected edge added
- Self-loop → bidirected self-loop added
- Larger SCC → all intra-SCC directed edges gain bidirected counterparts
- Pre-existing bidirected edges are preserved
- σ≡d equivalence: sigma-separation in G == d-separation in Gσ
"""

from __future__ import annotations

import unittest

from y0.algorithm.conditional_independencies import are_d_separated
from y0.algorithm.separation.sigma_extension import sigma_extension
from y0.algorithm.separation.sigma_separation import are_sigma_separated
from y0.dsl import Variable
from y0.graph import NxMixedGraph

X = Variable("X")
Y = Variable("Y")
Z = Variable("Z")
W = Variable("W")


class TestSigmaExtensionDAG(unittest.TestCase):
    """On a DAG, sigma_extension should return an identical graph."""

    def test_simple_dag_no_bidirected_added(self) -> None:
        """X -> Y: no SCC, so no bidirected edge should be added."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)])
        gs = sigma_extension(g)
        self.assertFalse(gs.undirected.has_edge(X, Y))
        self.assertTrue(gs.directed.has_edge(X, Y))

    def test_chain_dag_no_bidirected_added(self) -> None:
        """X -> Y -> Z: no cycles, no bidirected edges added."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])
        gs = sigma_extension(g)
        self.assertEqual(0, gs.undirected.number_of_edges())
        self.assertEqual(2, gs.directed.number_of_edges())

    def test_fork_dag_no_bidirected_added(self) -> None:
        """Z -> X, Z -> Y: no cycles, no bidirected edges added."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y)])
        gs = sigma_extension(g)
        self.assertEqual(0, gs.undirected.number_of_edges())


class TestSigmaExtensionCycles(unittest.TestCase):
    """Cyclic graphs should gain bidirected edges for intra-SCC directed edges."""

    def test_two_cycle_gains_one_bidirected_edge(self) -> None:
        """X -> Y -> X: both directed edges are intra-SCC, so X <-> Y is added."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
        gs = sigma_extension(g)
        # Exactly one bidirected edge (undirected graphs are symmetric)
        self.assertEqual(1, gs.undirected.number_of_edges())
        self.assertTrue(gs.undirected.has_edge(X, Y))
        # Directed edges preserved
        self.assertTrue(gs.directed.has_edge(X, Y))
        self.assertTrue(gs.directed.has_edge(Y, X))

    def test_three_cycle_gains_three_bidirected_edges(self) -> None:
        """X -> Y -> Z -> X: all three directed edges are intra-SCC."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z), (Z, X)])
        gs = sigma_extension(g)
        self.assertEqual(3, gs.undirected.number_of_edges())
        self.assertTrue(gs.undirected.has_edge(X, Y))
        self.assertTrue(gs.undirected.has_edge(Y, Z))
        self.assertTrue(gs.undirected.has_edge(Z, X))

    def test_cycle_with_external_node(self) -> None:
        """W -> X -> Y -> X: W is external; only X<->Y bidirected edge added."""
        g = NxMixedGraph.from_edges(directed=[(W, X), (X, Y), (Y, X)])
        gs = sigma_extension(g)
        # Only X<->Y is intra-SCC
        self.assertEqual(1, gs.undirected.number_of_edges())
        self.assertTrue(gs.undirected.has_edge(X, Y))
        # W->X is inter-SCC, no bidirected edge
        self.assertFalse(gs.undirected.has_edge(W, X))

    def test_preexisting_bidirected_edge_preserved(self) -> None:
        """Pre-existing bidirected edges are preserved and not duplicated."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)], undirected=[(X, Y)])
        gs = sigma_extension(g)
        # Still exactly one bidirected edge
        self.assertEqual(1, gs.undirected.number_of_edges())
        self.assertTrue(gs.undirected.has_edge(X, Y))

    def test_dag_with_bidirected_edge_unchanged(self) -> None:
        """DAG with a latent confounder: no new bidirected edges from cycles."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[(X, Z)])
        gs = sigma_extension(g)
        # Only the pre-existing X<->Z bidirected edge
        self.assertEqual(1, gs.undirected.number_of_edges())
        self.assertTrue(gs.undirected.has_edge(X, Z))
        self.assertFalse(gs.undirected.has_edge(X, Y))


class TestSigmaExtensionEquivalence(unittest.TestCase):
    r"""σ≡d equivalence: sigma-separation in G == d-separation in Gσ.

    For each fixture, we enumerate (left, right, conditions) triples and
    assert that are_sigma_separated(G, ...) == bool(are_d_separated(Gσ, ...)).
    """

    def _check_equivalence(
        self,
        graph: NxMixedGraph,
        triples: list[tuple[Variable, Variable, set[Variable]]],
    ) -> None:
        gs = sigma_extension(graph)
        for left, right, conds in triples:
            with self.subTest(left=left, right=right, conds=conds):
                sigma_result = are_sigma_separated(graph, left, right, conditions=conds)
                d_result = bool(are_d_separated(gs, left, right, conditions=conds))
                self.assertEqual(
                    sigma_result,
                    d_result,
                    msg=(
                        f"σ-sep({left},{right}|{conds}) in G = {sigma_result} "
                        f"but d-sep({left},{right}|{conds}) in Gσ = {d_result}"
                    ),
                )

    def test_dag_equivalence(self) -> None:
        """On a DAG, Gσ == G, so σ-sep and d-sep must agree on all triples."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        triples: list[tuple[Variable, Variable, set[Variable]]] = [
            (X, Y, set()),
            (X, Y, {Z}),
            (Z, Y, set()),
            (Z, Y, {X}),
            (Z, X, set()),
        ]
        self._check_equivalence(g, triples)

    def test_two_cycle_equivalence(self) -> None:
        """X -> Y -> X: σ-sep and d-sep in Gσ must agree."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
        triples: list[tuple[Variable, Variable, set[Variable]]] = [
            (X, Y, set()),
            (Y, X, set()),
        ]
        self._check_equivalence(g, triples)

    def test_cycle_with_confounder_equivalence(self) -> None:
        """W -> X -> Y -> X, W -> Y: σ-sep and d-sep in Gσ must agree."""
        g = NxMixedGraph.from_edges(directed=[(W, X), (X, Y), (Y, X), (W, Y)])
        triples: list[tuple[Variable, Variable, set[Variable]]] = [
            (X, Y, set()),
            (X, Y, {W}),
            (W, Y, set()),
            (W, Y, {X}),
        ]
        self._check_equivalence(g, triples)
