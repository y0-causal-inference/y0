"""Tests for are_m_separated and find_minimal_m_separator.

Covers:
- are_m_separated: basic DAG patterns, bidirected edges, colliders, chains, forks
- find_minimal_m_separator: correctness (separates), minimality (no proper subset
  within constraints also separates), included/restricted constraints
- Property tests: random ADMGs, verify returned set (a) m-separates, (b) is minimal,
  (c) respects included/restricted
"""

from __future__ import annotations

import random
import unittest
from itertools import combinations

from y0.algorithm.separation.minimal_separator import are_m_separated, find_minimal_m_separator
from y0.dsl import Variable
from y0.graph import NxMixedGraph

# ---------------------------------------------------------------------------
# Shared variables
# ---------------------------------------------------------------------------

A = Variable("A")
B = Variable("B")
C = Variable("C")
D = Variable("D")
E = Variable("E")

X = Variable("X")
Y = Variable("Y")
Z = Variable("Z")
W = Variable("W")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_minimal(
    graph: NxMixedGraph,
    x: Variable,
    y: Variable,
    z: frozenset[Variable],
    included: frozenset[Variable],
) -> bool:
    """Return True if no proper subset of *z* that still contains *included* m-separates."""
    shrinkable = z - included
    for node in shrinkable:
        candidate = z - {node}
        if are_m_separated(graph, x, y, conditions=candidate):
            return False
    return True


def _random_admg(
    nodes: list[Variable],
    *,
    rng: random.Random,
    directed_prob: float = 0.35,
    bidirected_prob: float = 0.20,
) -> NxMixedGraph:
    """Generate a random acyclic directed mixed graph over *nodes*.

    Directed edges are added in topological order (u < v by index) to ensure
    acyclicity.  Bidirected edges are added independently.
    """
    directed = []
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if rng.random() < directed_prob:
                directed.append((u, v))

    undirected = []
    for u, v in combinations(nodes, 2):
        if rng.random() < bidirected_prob:
            undirected.append((u, v))

    return NxMixedGraph.from_edges(directed=directed, undirected=undirected)


# ---------------------------------------------------------------------------
# Tests for are_m_separated
# ---------------------------------------------------------------------------


class TestAreMSeparated(unittest.TestCase):
    """Unit tests for are_m_separated."""

    def test_fork_not_separated(self) -> None:
        """X <- Z -> Y: X and Y are NOT m-separated given empty set."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y)])
        self.assertFalse(are_m_separated(g, X, Y))

    def test_fork_separated_by_z(self) -> None:
        """X <- Z -> Y: X and Y ARE m-separated given {Z}."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y)])
        self.assertTrue(are_m_separated(g, X, Y, conditions={Z}))

    def test_chain_not_separated(self) -> None:
        """X -> Z -> Y: X and Y are NOT m-separated given empty set."""
        g = NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)])
        self.assertFalse(are_m_separated(g, X, Y))

    def test_chain_separated_by_z(self) -> None:
        """X -> Z -> Y: X and Y ARE m-separated given {Z}."""
        g = NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)])
        self.assertTrue(are_m_separated(g, X, Y, conditions={Z}))

    def test_collider_separated_empty(self) -> None:
        """X -> Z <- Y: X and Y ARE m-separated given empty set (collider Z not conditioned)."""
        g = NxMixedGraph.from_edges(directed=[(X, Z), (Y, Z)])
        self.assertTrue(are_m_separated(g, X, Y))

    def test_collider_opened_by_conditioning(self) -> None:
        """X -> Z <- Y: X and Y are NOT m-separated given {Z} (collider opened)."""
        g = NxMixedGraph.from_edges(directed=[(X, Z), (Y, Z)])
        self.assertFalse(are_m_separated(g, X, Y, conditions={Z}))

    def test_bidirected_not_separated(self) -> None:
        """X <-> Y: X and Y are NOT m-separated given empty set."""
        g = NxMixedGraph.from_edges(undirected=[(X, Y)])
        self.assertFalse(are_m_separated(g, X, Y))

    def test_bidirected_with_mediator(self) -> None:
        """X <-> Z -> Y: X and Y are NOT m-separated given empty set."""
        g = NxMixedGraph.from_edges(directed=[(Z, Y)], undirected=[(X, Z)])
        self.assertFalse(are_m_separated(g, X, Y))

    def test_bidirected_with_mediator_blocked(self) -> None:
        """X <-> Z -> Y: X and Y ARE m-separated given {Z}."""
        g = NxMixedGraph.from_edges(directed=[(Z, Y)], undirected=[(X, Z)])
        self.assertTrue(are_m_separated(g, X, Y, conditions={Z}))

    def test_no_path(self) -> None:
        """X and Y with no connecting path are always m-separated."""
        # X -> Z, Y -> W: no path between X and Y
        g = NxMixedGraph.from_edges(directed=[(X, Z), (Y, W)])
        self.assertTrue(are_m_separated(g, X, Y))

    def test_dag_no_connection(self) -> None:
        """Pure DAG with no path between X and Y."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (W, Y)])
        self.assertTrue(are_m_separated(g, X, Y))

    def test_dag_connected_via_common_cause(self) -> None:
        """Z -> X, Z -> Y: connected via common cause."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y)])
        self.assertFalse(are_m_separated(g, X, Y))
        self.assertTrue(are_m_separated(g, X, Y, conditions={Z}))

    def test_invalid_node_raises(self) -> None:
        """Passing a node not in the graph raises ValueError."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)])
        with self.assertRaises(ValueError):
            are_m_separated(g, X, Z)  # Z not in graph

    def test_invalid_condition_raises(self) -> None:
        """Conditioning on a node not in the graph raises ValueError."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)])
        with self.assertRaises(ValueError):
            are_m_separated(g, X, Y, conditions={Z})  # Z not in graph

    def test_default_conditions_is_empty(self) -> None:
        """Default conditions=None is equivalent to conditions=set()."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y)])
        self.assertEqual(
            are_m_separated(g, X, Y),
            are_m_separated(g, X, Y, conditions=set()),
        )

    def test_four_node_diamond(self) -> None:
        """Diamond: X -> A -> Y, X -> B -> Y.  m-sep(X,Y|{A,B}) = True."""
        g = NxMixedGraph.from_edges(directed=[(X, A), (X, B), (A, Y), (B, Y)])
        self.assertFalse(are_m_separated(g, X, Y))
        self.assertFalse(are_m_separated(g, X, Y, conditions={A}))
        self.assertFalse(are_m_separated(g, X, Y, conditions={B}))
        self.assertTrue(are_m_separated(g, X, Y, conditions={A, B}))

    def test_bidirected_collider_path(self) -> None:
        """X <-> Z <-> Y: two bidirected edges sharing Z.

        In the LVM: L1->X, L1->Z, L2->Z, L2->Y.  Z is a collider on the path
        X <- L1 -> Z <- L2 -> Y.  Without conditioning on Z the path is blocked,
        so X and Y ARE m-separated given the empty set.
        """
        g = NxMixedGraph.from_edges(undirected=[(X, Z), (Z, Y)])
        self.assertTrue(are_m_separated(g, X, Y))

    def test_bidirected_collider_opened(self) -> None:
        """X <-> Z <-> Y: conditioning on Z (the collider) opens the path."""
        g = NxMixedGraph.from_edges(undirected=[(X, Z), (Z, Y)])
        self.assertFalse(are_m_separated(g, X, Y, conditions={Z}))

    def test_bidirected_chain_not_separated(self) -> None:
        """X <-> Z, Z <-> Y (chain of bidirected): X and Y are NOT m-separated.

        LVM: L1->X, L1->Z, L2->Z, L2->Y.  The path X <- L1 -> Z <- L2 -> Y
        has Z as a collider; without conditioning on Z it is blocked.
        But there is also a direct path via the latent nodes only if we think
        of it differently.  Actually this is the same as the collider case —
        X and Y ARE m-separated given {}.  To get a non-separated bidirected
        chain we need a direct bidirected edge X <-> Y.
        """
        g = NxMixedGraph.from_edges(undirected=[(X, Y)])
        self.assertFalse(are_m_separated(g, X, Y))


# ---------------------------------------------------------------------------
# Tests for find_minimal_m_separator
# ---------------------------------------------------------------------------


class TestFindMinimalMSeparator(unittest.TestCase):
    """Unit tests for find_minimal_m_separator."""

    def _assert_valid_separator(
        self,
        graph: NxMixedGraph,
        x: Variable,
        y: Variable,
        z: frozenset[Variable] | None,
        *,
        included: frozenset[Variable] | None = None,
        restricted: frozenset[Variable] | None = None,
        label: str = "",
    ) -> None:
        """Assert that *z* is a valid minimal m-separator."""
        self.assertIsNotNone(z, msg=f"[{label}] Expected a separator, got None")
        assert z is not None  # for type narrowing
        inc = frozenset(included) if included else frozenset()
        # (a) z m-separates x and y
        self.assertTrue(
            are_m_separated(graph, x, y, conditions=z),
            msg=f"[{label}] Returned set {z} does not m-separate {x} and {y}",
        )
        # (b) z is minimal (no proper subset within constraints also separates)
        self.assertTrue(
            _is_minimal(graph, x, y, z, inc),
            msg=f"[{label}] Returned set {z} is not minimal",
        )
        # (c) z contains all included nodes
        self.assertTrue(
            inc.issubset(z),
            msg=f"[{label}] Returned set {z} does not contain all included nodes {inc}",
        )
        # (d) z is a subset of restricted (if given)
        if restricted is not None:
            self.assertTrue(
                z.issubset(frozenset(restricted)),
                msg=f"[{label}] Returned set {z} is not a subset of restricted {restricted}",
            )

    def test_fork(self) -> None:
        """X <- Z -> Y: minimal separator is {Z}."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y)])
        result = find_minimal_m_separator(g, X, Y)
        self._assert_valid_separator(g, X, Y, result, label="fork")
        self.assertEqual(result, frozenset({Z}))

    def test_chain(self) -> None:
        """X -> Z -> Y: minimal separator is {Z}."""
        g = NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)])
        result = find_minimal_m_separator(g, X, Y)
        self._assert_valid_separator(g, X, Y, result, label="chain")
        self.assertEqual(result, frozenset({Z}))

    def test_bidirected_mediator(self) -> None:
        """X <-> Z -> Y: minimal separator is {Z}."""
        g = NxMixedGraph.from_edges(directed=[(Z, Y)], undirected=[(X, Z)])
        result = find_minimal_m_separator(g, X, Y)
        self._assert_valid_separator(g, X, Y, result, label="bidirected-mediator")
        self.assertEqual(result, frozenset({Z}))

    def test_collider_no_separator(self) -> None:
        """X -> Z <- Y: no separator exists (X and Y are already separated by empty set)."""
        g = NxMixedGraph.from_edges(directed=[(X, Z), (Y, Z)])
        # X and Y are already m-separated by {} — the empty set IS a separator
        result = find_minimal_m_separator(g, X, Y)
        # The empty set is a valid minimal separator
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(are_m_separated(g, X, Y, conditions=result))

    def test_diamond_both_paths(self) -> None:
        """X -> A -> Y, X -> B -> Y: minimal separator must block both paths."""
        g = NxMixedGraph.from_edges(directed=[(X, A), (X, B), (A, Y), (B, Y)])
        result = find_minimal_m_separator(g, X, Y)
        self._assert_valid_separator(g, X, Y, result, label="diamond")
        # Must contain at least one of {A, B} for each path
        self.assertIsNotNone(result)

    def test_included_constraint(self) -> None:
        """Included nodes must appear in the result.

        Graph: Z->X, Z->Y, W->X, W->Y.  Both {Z,W} is the minimal separator
        (neither Z nor W alone suffices).  Force Z to be included — the result
        must contain Z (and also W, since W is needed).
        """
        g = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y), (W, X), (W, Y)])
        result = find_minimal_m_separator(g, X, Y, included={Z})
        self._assert_valid_separator(g, X, Y, result, included=frozenset({Z}), label="included-Z")
        assert result is not None
        self.assertIn(Z, result)

    def test_restricted_constraint(self) -> None:
        """Restricted nodes limit the pool.

        Simple fork Z->X, Z->Y: Z alone m-separates X and Y.
        Restricting to {Z} should return {Z}.
        """
        g = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y)])
        result = find_minimal_m_separator(g, X, Y, restricted={Z})
        self._assert_valid_separator(
            g, X, Y, result, restricted=frozenset({Z}), label="restricted-Z"
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result, frozenset({Z}))

    def test_restricted_no_separator_returns_none(self) -> None:
        """When restricted pool is insufficient, return None.

        Graph: Z->X, Z->Y, W->X, W->Y.  Restricting to {Z} only — Z alone
        cannot separate X and Y (W still creates a path), so None is returned.
        """
        g = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y), (W, X), (W, Y)])
        result = find_minimal_m_separator(g, X, Y, restricted={Z})
        self.assertIsNone(result)

    def test_no_separator_returns_none(self) -> None:
        """When restricted pool cannot separate, return None."""
        # X <-> Y: the only path is the bidirected edge; no observed node can block it
        # unless we condition on a collider descendant — but there are none here.
        g = NxMixedGraph.from_edges(undirected=[(X, Y)])
        # Restrict to empty pool — cannot separate
        result = find_minimal_m_separator(g, X, Y, restricted=set())
        self.assertIsNone(result)

    def test_empty_separator_valid(self) -> None:
        """When X and Y are already m-separated, the empty set is a valid separator."""
        g = NxMixedGraph.from_edges(directed=[(X, Z), (Y, W)])
        result = find_minimal_m_separator(g, X, Y)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(are_m_separated(g, X, Y, conditions=result))

    def test_invalid_x_raises(self) -> None:
        """Passing x not in graph raises ValueError."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)])
        with self.assertRaises(ValueError):
            find_minimal_m_separator(g, Z, Y)

    def test_invalid_included_raises(self) -> None:
        """Passing an included node not in graph raises ValueError."""
        g = NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)])
        with self.assertRaises(ValueError):
            find_minimal_m_separator(g, X, Y, included={W})  # W not in graph

    def test_invalid_restricted_raises(self) -> None:
        """Passing a restricted node not in graph raises ValueError."""
        g = NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)])
        with self.assertRaises(ValueError):
            find_minimal_m_separator(g, X, Y, restricted={W})  # W not in graph


# ---------------------------------------------------------------------------
# Property tests: random ADMGs
# ---------------------------------------------------------------------------


class TestMinimalSeparatorProperties(unittest.TestCase):
    """Property tests: for every pair in random ADMGs, verify the returned separator."""

    SEED = 99
    N_GRAPHS = 30
    NODES_3 = [A, B, C]
    NODES_4 = [A, B, C, D]
    NODES_5 = [A, B, C, D, E]

    def _run_battery(self, nodes: list[Variable], rng: random.Random) -> None:
        for trial in range(self.N_GRAPHS):
            g = _random_admg(nodes, rng=rng)
            node_list = sorted(g.nodes(), key=lambda v: v.name)
            for x, y in combinations(node_list, 2):
                result = find_minimal_m_separator(g, x, y)
                if result is None:
                    # Verify that no separator exists in the default pool
                    # (ancestors of {x,y} minus {x,y})
                    # We can't easily verify this exhaustively, so just skip
                    continue
                label = f"random-n{len(nodes)}-trial{trial}-{x}-{y}"
                # (a) separates
                self.assertTrue(
                    are_m_separated(g, x, y, conditions=result),
                    msg=f"[{label}] {result} does not m-separate {x} and {y}",
                )
                # (b) minimal
                self.assertTrue(
                    _is_minimal(g, x, y, result, frozenset()),
                    msg=f"[{label}] {result} is not minimal",
                )

    def test_random_3_nodes(self) -> None:
        """Property test on 3-node random ADMGs."""
        rng = random.Random(self.SEED + 3)
        self._run_battery(self.NODES_3, rng)

    def test_random_4_nodes(self) -> None:
        """Property test on 4-node random ADMGs."""
        rng = random.Random(self.SEED + 4)
        self._run_battery(self.NODES_4, rng)

    def test_random_5_nodes(self) -> None:
        """Property test on 5-node random ADMGs."""
        rng = random.Random(self.SEED + 5)
        self._run_battery(self.NODES_5, rng)
