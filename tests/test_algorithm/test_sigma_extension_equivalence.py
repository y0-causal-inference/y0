r"""Property tests for Theorem 5 (Rantanen et al. 2020): sigma-separation equivalence.

Theorem 5 states:
    X and Y are sigma-separated given Z in G
    **if and only if**
    X and Y are d-separated given Z in G^sigma

where G^sigma is the sigma-extension of G (bidirected edges added for all
intra-SCC directed edges).

This module tests that equivalence exhaustively over all (X, Y, Z) triples on
a battery of ~200 small randomly-generated directed mixed graphs (n=3–5 nodes,
directed cycles and bidirected edges allowed), plus a set of hand-crafted
fixtures including the Figure-1 graph from Forré & Mooij (2018).

**Triage results (Phase 1 investigation):**

Two distinct bugs were found in :func:`~y0.algorithm.separation.sigma_separation.are_sigma_separated`:

**Bug A — over-reports separation (False negative for connectivity):**
When a node has *both* a directed edge AND a bidirected edge to a neighbor,
``_only_directed_edge(middle, left)`` returns False even though the directed
edge exists, causing ``is_non_collider_fork`` to incorrectly reject a valid
σ-open fork path.  Example: graph ``A→B, A↔B, A↔C`` — the path ``B-A-C``
is σ-open (A is an unconditioned fork) but ``are_sigma_separated`` returns
True (separated).

**Bug B — over-reports connectivity (False positive for connectivity):**
When a node is simultaneously a collider (both incoming edges present) AND
satisfies the non-collider left-chain condition (due to an intra-SCC directed
edge in the opposite direction), ``is_z_sigma_open`` returns True even when
the path should be blocked.  Example: graph ``X→Y→X, X↔Z``, path ``Y-X-Z``
conditioned on ``{X}`` — X is a collider (Y→X↔Z) opened by conditioning, but
d-separation in G^sigma correctly blocks it because G^sigma adds X↔Y (intra-SCC),
making X a non-collider on Y↔X↔Z that is blocked by conditioning.

**Consequence for the plan:**
The ``sigma_extension`` + ``are_d_separated`` approach is the *correct* oracle
for σ-separation.  ``are_sigma_separated`` is buggy and should not be used as
the ground truth.  The ``are_m_separated`` function in Phase 2 will use
``sigma_extension`` + ``are_d_separated`` internally.

The fixture tests that exercise these bugs are marked ``@unittest.expectedFailure``
to document the known defects without blocking CI.  The random battery tests
are similarly marked.  The hand-crafted fixtures that do NOT exercise the bugs
pass cleanly and serve as regression anchors.
"""

from __future__ import annotations

import random
import unittest
from itertools import combinations

from y0.algorithm.conditional_independencies import are_d_separated
from y0.algorithm.separation.sigma_extension import sigma_extension
from y0.algorithm.separation.sigma_separation import are_sigma_separated
from y0.dsl import Variable
from y0.graph import NxMixedGraph

# ---------------------------------------------------------------------------
# Shared node pool
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

_NODE_POOL_3 = [A, B, C]
_NODE_POOL_4 = [A, B, C, D]
_NODE_POOL_5 = [A, B, C, D, E]


# ---------------------------------------------------------------------------
# Random graph generator
# ---------------------------------------------------------------------------


def _random_dmg(
    nodes: list[Variable],
    *,
    rng: random.Random,
    directed_prob: float = 0.35,
    bidirected_prob: float = 0.20,
) -> NxMixedGraph:
    """Generate a random directed mixed graph over *nodes*.

    Each ordered pair (u, v) with u != v independently gets a directed edge
    u -> v with probability *directed_prob*.  Each unordered pair {u, v}
    independently gets a bidirected edge u <-> v with probability
    *bidirected_prob*.  Directed cycles are allowed.

    :param nodes: The node set.
    :param rng: A seeded :class:`random.Random` instance for reproducibility.
    :param directed_prob: Per-ordered-pair probability of a directed edge.
    :param bidirected_prob: Per-unordered-pair probability of a bidirected edge.
    :returns: A new :class:`~y0.graph.NxMixedGraph`.
    """
    directed = []
    for u in nodes:
        for v in nodes:
            if u != v and rng.random() < directed_prob:
                directed.append((u, v))

    undirected = []
    for u, v in combinations(nodes, 2):
        if rng.random() < bidirected_prob:
            undirected.append((u, v))

    return NxMixedGraph.from_edges(directed=directed, undirected=undirected)


# ---------------------------------------------------------------------------
# Core equivalence checker
# ---------------------------------------------------------------------------


def _check_theorem5(
    graph: NxMixedGraph,
    *,
    test_case: unittest.TestCase,
    label: str = "",
) -> None:
    r"""Assert Theorem 5 for every (X, Y, Z) triple in *graph*.

    For every pair of distinct observed nodes (left, right) and every subset
    Z of the remaining nodes, assert:

        are_sigma_separated(G, left, right, Z)
        ==
        bool(are_d_separated(G^sigma, left, right, Z))

    :param graph: The directed mixed graph to test.
    :param test_case: The :class:`unittest.TestCase` to use for assertions.
    :param label: A human-readable label for error messages.
    """
    g_sigma = sigma_extension(graph)
    nodes = sorted(graph.nodes(), key=lambda v: v.name)

    for left, right in combinations(nodes, 2):
        remaining = [n for n in nodes if n not in (left, right)]
        # Test empty conditioning set plus all singleton and pair subsets
        # (full powerset is 2^(n-2) which is at most 8 for n=5 — tractable).
        for size in range(len(remaining) + 1):
            for cond_tuple in combinations(remaining, size):
                conds = set(cond_tuple)
                sigma_result = are_sigma_separated(graph, left, right, conditions=conds)
                d_result = bool(are_d_separated(g_sigma, left, right, conditions=conds))
                test_case.assertEqual(
                    sigma_result,
                    d_result,
                    msg=(
                        f"[{label}] Theorem 5 violated: "
                        f"sigma-sep({left},{right}|{conds}) in G = {sigma_result} "
                        f"but d-sep({left},{right}|{conds}) in G^sigma = {d_result}\n"
                        f"  directed edges: {list(graph.directed.edges())}\n"
                        f"  bidirected edges: {list(graph.undirected.edges())}"
                    ),
                )


# ---------------------------------------------------------------------------
# Hand-crafted fixture tests
# ---------------------------------------------------------------------------


class TestTheoremFiveFixtures(unittest.TestCase):
    r"""Theorem 5 on hand-crafted graphs including the Forré–Mooij Figure-1 anchor."""

    def test_dag_chain(self) -> None:
        """X -> Y -> Z: pure DAG, G^sigma == G."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])
        _check_theorem5(g, test_case=self, label="chain-DAG")

    def test_dag_fork(self) -> None:
        """Z -> X, Z -> Y: pure DAG fork."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y)])
        _check_theorem5(g, test_case=self, label="fork-DAG")

    def test_dag_collider(self) -> None:
        """X -> Z <- Y: pure DAG collider."""
        g = NxMixedGraph.from_edges(directed=[(X, Z), (Y, Z)])
        _check_theorem5(g, test_case=self, label="collider-DAG")

    def test_dag_with_bidirected(self) -> None:
        """X -> Y, X <-> Z: DAG with latent confounder."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[(X, Z)])
        _check_theorem5(g, test_case=self, label="DAG-with-bidirected")

    def test_two_cycle(self) -> None:
        """X -> Y -> X: minimal 2-cycle."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
        _check_theorem5(g, test_case=self, label="2-cycle")

    def test_three_cycle(self) -> None:
        """X -> Y -> Z -> X: 3-cycle."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z), (Z, X)])
        _check_theorem5(g, test_case=self, label="3-cycle")

    def test_cycle_with_external_node(self) -> None:
        """W -> X -> Y -> X: external node feeding into a 2-cycle."""
        g = NxMixedGraph.from_edges(directed=[(W, X), (X, Y), (Y, X)])
        _check_theorem5(g, test_case=self, label="cycle-with-external")

    @unittest.expectedFailure
    def test_cycle_with_bidirected(self) -> None:
        """X -> Y -> X, X <-> Z: 2-cycle plus a latent confounder.

        Known Bug B in are_sigma_separated: the path Y-X-Z conditioned on {X}
        is incorrectly reported as sigma-open.  X is a collider (Y→X↔Z) that
        is opened by conditioning, but G^sigma adds X↔Y (intra-SCC), making X
        a non-collider on Y↔X↔Z that is blocked by conditioning.
        are_sigma_separated returns False (connected) but d-sep in G^sigma
        returns True (separated) for sigma-sep(Y,Z|{X}).
        """
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)], undirected=[(X, Z)])
        _check_theorem5(g, test_case=self, label="cycle-with-bidirected")

    def test_forre_mooij_figure1(self) -> None:
        r"""Figure-1 anchor from Forré & Mooij (2018).

        The graph has nodes {X1, X2, X3, X4} with:
          directed:   X1 -> X2, X2 -> X3, X3 -> X2, X3 -> X4
          bidirected: X1 <-> X4

        This is the canonical example from the paper demonstrating that
        sigma-separation differs from d-separation on cyclic graphs.
        """
        x1 = Variable("X1")
        x2 = Variable("X2")
        x3 = Variable("X3")
        x4 = Variable("X4")
        g = NxMixedGraph.from_edges(
            directed=[(x1, x2), (x2, x3), (x3, x2), (x3, x4)],
            undirected=[(x1, x4)],
        )
        _check_theorem5(g, test_case=self, label="Forre-Mooij-Fig1")

    def test_two_sccs_with_bridge(self) -> None:
        """Two 2-cycles connected by a bridge: A->B->A, C->D->C, B->C."""
        g = NxMixedGraph.from_edges(
            directed=[(A, B), (B, A), (C, D), (D, C), (B, C)]
        )
        _check_theorem5(g, test_case=self, label="two-SCCs-bridge")

    def test_cycle_with_multiple_bidirected(self) -> None:
        """X -> Y -> X, X <-> Z, Y <-> Z: cycle with two confounders."""
        g = NxMixedGraph.from_edges(
            directed=[(X, Y), (Y, X)],
            undirected=[(X, Z), (Y, Z)],
        )
        _check_theorem5(g, test_case=self, label="cycle-multi-bidirected")

    def test_fully_connected_3_cycle(self) -> None:
        """X -> Y -> Z -> X plus X -> Z, Y -> X, Z -> Y (complete directed graph on 3 nodes)."""
        g = NxMixedGraph.from_edges(
            directed=[(X, Y), (Y, Z), (Z, X), (X, Z), (Y, X), (Z, Y)]
        )
        _check_theorem5(g, test_case=self, label="complete-3-cycle")


# ---------------------------------------------------------------------------
# Random property tests
# ---------------------------------------------------------------------------


class TestTheoremFiveRandom(unittest.TestCase):
    r"""Theorem 5 property test over ~200 randomly-generated directed mixed graphs.

    Uses a fixed seed for reproducibility.  Any failure here indicates either:
    (a) a real bug in sigma_extension or are_sigma_separated, or
    (b) incompleteness in the path-augmentation heuristic inside
        are_sigma_separated (which uses nx.all_simple_paths and may miss
        walks that revisit nodes).

    Failures should be triaged before any production code is changed.
    """

    SEED = 42
    N_GRAPHS_PER_SIZE = 40  # 40 graphs × 5 sizes = 200 total

    def _run_random_battery(self, n_nodes: int, node_pool: list[Variable]) -> None:
        """Run the random battery for graphs of size *n_nodes*."""
        rng = random.Random(self.SEED + n_nodes)  # distinct seed per size
        for trial in range(self.N_GRAPHS_PER_SIZE):
            g = _random_dmg(node_pool[:n_nodes], rng=rng)
            _check_theorem5(
                g,
                test_case=self,
                label=f"random-n{n_nodes}-trial{trial}",
            )

    @unittest.expectedFailure
    def test_random_3_nodes(self) -> None:
        """Random battery: 3-node graphs.

        Expected to fail due to Bug A and Bug B in are_sigma_separated.
        See module docstring for details.
        """
        self._run_random_battery(3, _NODE_POOL_3)

    @unittest.expectedFailure
    def test_random_4_nodes(self) -> None:
        """Random battery: 4-node graphs.

        Expected to fail due to Bug A and Bug B in are_sigma_separated.
        See module docstring for details.
        """
        self._run_random_battery(4, _NODE_POOL_4)

    @unittest.expectedFailure
    def test_random_5_nodes(self) -> None:
        """Random battery: 5-node graphs.

        Expected to fail due to Bug A and Bug B in are_sigma_separated.
        See module docstring for details.
        """
        self._run_random_battery(5, _NODE_POOL_5)

    @unittest.expectedFailure
    def test_random_3_nodes_dense(self) -> None:
        """Random battery: 3-node graphs with higher edge density.

        Expected to fail due to Bug A and Bug B in are_sigma_separated.
        See module docstring for details.
        """
        rng = random.Random(self.SEED + 100)
        for trial in range(self.N_GRAPHS_PER_SIZE):
            g = _random_dmg(
                _NODE_POOL_3,
                rng=rng,
                directed_prob=0.6,
                bidirected_prob=0.4,
            )
            _check_theorem5(
                g,
                test_case=self,
                label=f"random-dense-n3-trial{trial}",
            )

    @unittest.expectedFailure
    def test_random_4_nodes_dense(self) -> None:
        """Random battery: 4-node graphs with higher edge density.

        Expected to fail due to Bug A and Bug B in are_sigma_separated.
        See module docstring for details.
        """
        rng = random.Random(self.SEED + 200)
        for trial in range(self.N_GRAPHS_PER_SIZE):
            g = _random_dmg(
                _NODE_POOL_4,
                rng=rng,
                directed_prob=0.5,
                bidirected_prob=0.35,
            )
            _check_theorem5(
                g,
                test_case=self,
                label=f"random-dense-n4-trial{trial}",
            )
