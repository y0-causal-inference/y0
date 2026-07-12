"""Regression and property-based tests for the reachability-based O-set fix.

These tests verify that:
1. The new reachability-based ``optimal_adjustment_set`` and
   ``proper_backdoor_graph`` produce correct results on hand-crafted graphs,
   including cyclic graphs where the old path-enumeration oracle was wrong.
2. Hypothesis property-based tests confirm agreement on randomly generated
   **acyclic** graphs (where both implementations are equivalent).
3. Hypothesis property-based tests confirm the new implementation satisfies
   the O-set definition directly on random graphs (including cyclic ones).
4. The new implementation is strictly faster on dense-SCC graphs (the
   pathological case that caused timeouts on large E. coli networks).

Background: the old ``all_simple_paths``-based implementation was incorrect
for cyclic graphs because ``nx.all_simple_paths`` cannot traverse cycles, so
it missed causal nodes that are reachable only via cyclic paths.  The new
reachability-based implementation uses ``nx.descendants`` and ``nx.ancestors``
which correctly handle cycles.  The Hypothesis tests below use the old
implementation only as an oracle on **DAGs** (where it is correct), and use
the O-set definition directly for cyclic graphs.
"""

from __future__ import annotations

import time
import unittest
from itertools import combinations

import networkx as nx
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from y0.algorithm.separation.sigma_single_door import (
    find_sigma_single_door_set,
    optimal_adjustment_set,
    proper_backdoor_graph,
)
from y0.dsl import Variable
from y0.graph import NxMixedGraph

# ---------------------------------------------------------------------------
# Reference (old) implementations — path-enumeration-based
# ---------------------------------------------------------------------------


def _old_directed_paths_from(
    graph: NxMixedGraph, source: Variable, target: Variable
) -> list[list[Variable]]:
    if source not in graph.directed or target not in graph.directed:
        return []
    return list(nx.all_simple_paths(graph.directed, source, target))


def _old_remove_directed_edge(graph: NxMixedGraph, u: Variable, v: Variable) -> NxMixedGraph:
    reduced = graph.copy()
    reduced.directed.remove_edge(u, v)
    return reduced


def _old_proper_backdoor_graph(
    graph: NxMixedGraph, cause: Variable, effect: Variable
) -> NxMixedGraph:
    """Compute the reference proper backdoor graph using all_simple_paths."""
    reduced = _old_remove_directed_edge(graph, cause, effect)
    first_hops: set[Variable] = set()
    for path in _old_directed_paths_from(reduced, cause, effect):
        if len(path) >= 2:
            first_hops.add(path[1])
    pbd = reduced.copy()
    for hop in first_hops:
        if pbd.directed.has_edge(cause, hop):
            pbd.directed.remove_edge(cause, hop)
    return pbd


def _old_optimal_adjustment_set(
    graph: NxMixedGraph, cause: Variable, effect: Variable
) -> frozenset[Variable] | None:
    """Compute the reference optimal adjustment set using all_simple_paths."""
    causal_nodes: set[Variable] = set()
    for path in _old_directed_paths_from(graph, cause, effect):
        causal_nodes.update(path)
    if not causal_nodes:
        return frozenset()
    forbidden: set[Variable] = graph.descendants_inclusive(causal_nodes)
    forbidden.add(cause)
    o_set: set[Variable] = set()
    for cn in causal_nodes:
        o_set.update(graph.directed.predecessors(cn))
    o_set -= forbidden
    effect_descendants = graph.descendants_inclusive(effect)
    if o_set & effect_descendants:
        return None
    return frozenset(o_set)


# ---------------------------------------------------------------------------
# Helpers for graph construction
# ---------------------------------------------------------------------------

X = Variable("X")
Y = Variable("Y")
Z = Variable("Z")
Z1 = Variable("Z1")
Z2 = Variable("Z2")
W = Variable("W")
M = Variable("M")


def _make_fan_graph(n_fan: int) -> NxMixedGraph:
    """Dense-SCC fan graph: X → {A0..An} → Y, complete directed graph among Ai."""
    fans = [Variable(f"A{j}") for j in range(n_fan)]
    directed: list[tuple[Variable, Variable]] = [(X, Y)]
    for a in fans:
        directed.append((X, a))
        directed.append((a, Y))
    for i, ai in enumerate(fans):
        for j, aj in enumerate(fans):
            if i != j:
                directed.append((ai, aj))
    return NxMixedGraph.from_edges(directed=directed)


# ---------------------------------------------------------------------------
# Hypothesis strategies for random graphs
# ---------------------------------------------------------------------------

_NODE_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]
_NODES = [Variable(n) for n in _NODE_NAMES]


def _is_dag(g: NxMixedGraph) -> bool:
    """Return True if the directed part of g is acyclic."""
    return nx.is_directed_acyclic_graph(g.directed)


@st.composite
def random_dag_with_xy(draw: st.DrawFn) -> NxMixedGraph:
    """Draw a random DAG (no directed cycles) that always contains X→Y."""
    n_extra = draw(st.integers(min_value=0, max_value=4))
    extra_nodes = _NODES[:n_extra]
    all_nodes = [X, Y, *extra_nodes]

    # Build a random DAG by only allowing edges u→v where u comes before v
    # in a fixed topological order.
    order = draw(st.permutations(all_nodes))
    rank = {v: i for i, v in enumerate(order)}

    possible_directed = [(u, v) for u in all_nodes for v in all_nodes if rank[u] < rank[v]]
    if not possible_directed:
        # Degenerate: only X and Y, force X→Y
        return NxMixedGraph.from_edges(directed=[(X, Y)])

    n_directed = draw(st.integers(min_value=0, max_value=min(len(possible_directed), 8)))
    chosen_directed = draw(
        st.lists(
            st.sampled_from(possible_directed),
            min_size=n_directed,
            max_size=n_directed,
            unique=True,
        )
    )
    # Always include X→Y (may already be present)
    edge_set = list({(X, Y)} | set(chosen_directed))

    # Random bidirected edges
    possible_undirected = list(combinations(all_nodes, 2))
    n_undirected = draw(st.integers(min_value=0, max_value=min(len(possible_undirected), 4)))
    chosen_undirected = draw(
        st.lists(
            st.sampled_from(possible_undirected),
            min_size=n_undirected,
            max_size=n_undirected,
            unique=True,
        )
    )

    return NxMixedGraph.from_edges(directed=edge_set, undirected=chosen_undirected)


@st.composite
def random_mixed_graph_with_xy(draw: st.DrawFn) -> NxMixedGraph:
    """Draw a random NxMixedGraph (possibly cyclic) that always contains X→Y."""
    n_extra = draw(st.integers(min_value=0, max_value=4))
    extra_nodes = _NODES[:n_extra]
    all_nodes = [X, Y, *extra_nodes]

    possible_directed = [(u, v) for u in all_nodes for v in all_nodes if u != v]
    n_directed = draw(st.integers(min_value=1, max_value=min(len(possible_directed), 8)))
    chosen_directed = draw(
        st.lists(
            st.sampled_from(possible_directed),
            min_size=n_directed,
            max_size=n_directed,
            unique=True,
        )
    )
    edge_set = list({(X, Y)} | set(chosen_directed))

    possible_undirected = list(combinations(all_nodes, 2))
    n_undirected = draw(st.integers(min_value=0, max_value=min(len(possible_undirected), 4)))
    chosen_undirected = draw(
        st.lists(
            st.sampled_from(possible_undirected),
            min_size=n_undirected,
            max_size=n_undirected,
            unique=True,
        )
    )

    return NxMixedGraph.from_edges(directed=edge_set, undirected=chosen_undirected)


# ---------------------------------------------------------------------------
# Unit tests: hand-crafted graphs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: compute O-set directly from definition (for verification)
# ---------------------------------------------------------------------------


def _oset_from_definition(
    g: NxMixedGraph, cause: Variable, effect: Variable
) -> frozenset[Variable] | None:
    """Compute O-set directly from the Henckel et al. definition.

    cn   = desc(cause) & anc(effect)   [reachability, handles cycles]
    forb = desc_inclusive(cn) | {cause}
    O    = pa(cn) - forb
    Returns None if O contains a descendant of effect.
    """
    if cause not in g.directed or effect not in g.directed:
        return frozenset()
    desc_cause: set[Variable] = set(nx.descendants(g.directed, cause))
    desc_cause.add(cause)
    anc_effect: set[Variable] = set(nx.ancestors(g.directed, effect))
    anc_effect.add(effect)
    causal_nodes = desc_cause & anc_effect
    if effect not in desc_cause:
        return frozenset()
    forbidden: set[Variable] = g.descendants_inclusive(causal_nodes)
    forbidden.add(cause)
    o_set: set[Variable] = set()
    for cn in causal_nodes:
        o_set.update(g.directed.predecessors(cn))
    o_set -= forbidden
    if o_set & g.descendants_inclusive(effect):
        return None
    return frozenset(o_set)


# ---------------------------------------------------------------------------
# Unit tests: hand-crafted graphs
# ---------------------------------------------------------------------------


class TestOptimalAdjustmentSetReachabilityUnit(unittest.TestCase):
    """Unit tests verifying the reachability O-set on specific graphs."""

    def _assert_agrees_with_old(self, g: NxMixedGraph, cause: Variable, effect: Variable) -> None:
        """On DAGs, new and old must agree."""
        assert _is_dag(g), "Use this helper only for DAGs"
        new = optimal_adjustment_set(g, cause, effect)
        old = _old_optimal_adjustment_set(g, cause, effect)
        self.assertEqual(
            new,
            old,
            f"Disagreement on DAG {list(g.directed.edges())}:\n  new={new}, old={old}",
        )

    def test_simple_dag(self) -> None:
        """Check O-set agreement on a trivial X→Y DAG."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)])
        self._assert_agrees_with_old(g, X, Y)

    def test_fork_confounder(self) -> None:
        """Check O-set agreement on a graph with a single fork confounder Z."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        self._assert_agrees_with_old(g, X, Y)

    def test_mediator(self) -> None:
        """Check O-set agreement on a graph with a mediator M and confounder Z."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, M), (M, Y)])
        self._assert_agrees_with_old(g, X, Y)

    def test_two_confounders(self) -> None:
        """Check O-set agreement on a graph with two independent confounders."""
        g = NxMixedGraph.from_edges(directed=[(Z1, X), (Z2, X), (X, Y), (Z1, Y), (Z2, Y)])
        self._assert_agrees_with_old(g, X, Y)

    def test_no_path(self) -> None:
        """Check that the O-set is empty when there is no path from X to Y."""
        g = NxMixedGraph.from_edges(directed=[(X, Z)])
        # Y not in graph — returns empty
        result = optimal_adjustment_set(g, X, Y)
        self.assertEqual(result, frozenset())

    def test_two_cycle_xy(self) -> None:
        """X→Y→X: both are causal nodes; O-set should be empty (no parents outside forb)."""
        g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
        result = optimal_adjustment_set(g, X, Y)
        # cn = {X,Y}, forb = {X,Y}, pa({X,Y}) = {Y,X}, o_set = {} after removing forb
        self.assertEqual(result, frozenset())

    def test_fan_n3(self) -> None:
        """Fan graph n=3: reachability must match definition (cyclic graph)."""
        g = _make_fan_graph(3)
        result = optimal_adjustment_set(g, X, Y)
        expected = _oset_from_definition(g, X, Y)
        self.assertEqual(result, expected)

    def test_fan_n5(self) -> None:
        """Fan graph n=5: 326 paths — reachability must match definition (cyclic graph)."""
        g = _make_fan_graph(5)
        result = optimal_adjustment_set(g, X, Y)
        expected = _oset_from_definition(g, X, Y)
        self.assertEqual(result, expected)

    def test_external_confounder_with_cycle(self) -> None:
        """W→X→Y→X, W→Y: W is external confounder; new impl correctly includes W."""
        g = NxMixedGraph.from_edges(directed=[(W, X), (X, Y), (Y, X), (W, Y)])
        result = optimal_adjustment_set(g, X, Y)
        # cn = desc(X)&anc(Y) = {X,Y}&{Y,X,W} = {X,Y}
        # forb = desc_inclusive({X,Y}) | {X} = {X,Y} | {X} = {X,Y}
        # pa({X,Y}) = {W,Y} | {X,W} = {W,X,Y}
        # o_set = {W,X,Y} - {X,Y} = {W}
        self.assertEqual(result, frozenset({W}))

    def test_cyclic_oset_hypothesis_counterexample_1(self) -> None:
        """Hypothesis-discovered: B→A, A→X, X→A, X→Y.

        Old path-enum returns frozenset() (misses A as causal node).
        New reachability correctly returns frozenset({B}).
        cn = desc(X)&anc(Y) = {X,A}&{Y,X,A,B} = {X,A,Y}
        forb = desc_inclusive({X,A,Y}) | {X} = {X,A,Y}
        pa({X,A,Y}) = {A,B,X}
        o_set = {A,B,X} - {X,A,Y} = {B}
        """
        A = Variable("A")
        B = Variable("B")
        g = NxMixedGraph.from_edges(directed=[(B, A), (A, X), (X, A), (X, Y)])
        result = optimal_adjustment_set(g, X, Y)
        self.assertEqual(result, frozenset({B}))

    def test_cyclic_oset_hypothesis_counterexample_2_pbd(self) -> None:
        """Hypothesis-discovered PBD: X→B, X→A, X→Y, B→Y, A→X.

        Old path-enum PBD keeps X→A (misses it as causal first-hop).
        New reachability correctly removes both X→A and X→B from PBD.
        After removing X→Y: can_reach_Y = {B,X,A} (B→Y, X→B, A→X→B→Y).
        first_hops = {B, A} → remove X→B and X→A.
        """
        A = Variable("A")
        B = Variable("B")
        g = NxMixedGraph.from_edges(directed=[(X, B), (X, A), (X, Y), (B, Y), (A, X)])
        pbd = proper_backdoor_graph(g, X, Y)
        pbd_edges = set(pbd.directed.edges())
        # X→B and X→A should both be removed; A→X and B→Y remain
        self.assertNotIn((X, B), pbd_edges, "X→B should be removed from PBD")
        self.assertNotIn((X, A), pbd_edges, "X→A should be removed from PBD")
        self.assertIn((A, X), pbd_edges, "A→X should remain in PBD")
        self.assertIn((B, Y), pbd_edges, "B→Y should remain in PBD")


class TestProperBackdoorGraphReachabilityUnit(unittest.TestCase):
    """Unit tests verifying the reachability proper_backdoor_graph on specific graphs."""

    def _assert_pbd_agrees(self, g: NxMixedGraph, cause: Variable, effect: Variable) -> None:
        """On DAGs, new and old PBD must agree."""
        assert _is_dag(g), "Use this helper only for DAGs"
        new_pbd = proper_backdoor_graph(g, cause, effect)
        old_pbd = _old_proper_backdoor_graph(g, cause, effect)
        new_edges = set(new_pbd.directed.edges())
        old_edges = set(old_pbd.directed.edges())
        self.assertEqual(
            new_edges,
            old_edges,
            f"PBD edge disagreement on DAG {list(g.directed.edges())}:\n"
            f"  new={new_edges}, old={old_edges}",
        )

    def test_simple_dag(self) -> None:
        """Check PBD agreement on a trivial X→Y DAG."""
        g = NxMixedGraph.from_edges(directed=[(X, Y)])
        self._assert_pbd_agrees(g, X, Y)

    def test_fork(self) -> None:
        """Check PBD agreement on a graph with a single fork confounder Z."""
        g = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        self._assert_pbd_agrees(g, X, Y)

    def test_mediator(self) -> None:
        """Check PBD agreement on a graph with a mediator M."""
        g = NxMixedGraph.from_edges(directed=[(X, M), (M, Y), (X, Y)])
        self._assert_pbd_agrees(g, X, Y)

    def test_fan_n4(self) -> None:
        """Fan graph n=4 is cyclic; verify PBD removes all first-hop causal edges."""
        g = _make_fan_graph(4)
        # All fan nodes A0..A3 are reachable from X and can reach Y (via cycles too).
        # After removing X→Y, all X→Ai edges should be removed from PBD.
        pbd = proper_backdoor_graph(g, X, Y)
        fans = [Variable(f"A{j}") for j in range(4)]
        for a in fans:
            self.assertNotIn(
                (X, a),
                set(pbd.directed.edges()),
                f"X→{a} should be removed from PBD of fan-4 graph",
            )


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


class TestOptimalAdjustmentSetHypothesis(unittest.TestCase):
    """Property-based tests for the reachability O-set."""

    @given(random_mixed_graph_with_xy())
    @settings(
        max_examples=300,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        deadline=5000,
    )
    def test_reachability_agrees_with_path_enum_on_dags(self, g: NxMixedGraph) -> None:
        """On any random DAG with X→Y, new and old O-set must agree.

        The old path-enumeration oracle is correct on DAGs (no cycles to miss).
        Uses ``assume`` to filter out cyclic graphs.
        """
        assume(_is_dag(g))
        new = optimal_adjustment_set(g, X, Y)
        old = _old_optimal_adjustment_set(g, X, Y)
        self.assertEqual(
            new,
            old,
            f"Disagreement on DAG {list(g.directed.edges())}:\n  new={new}, old={old}",
        )

    @given(random_mixed_graph_with_xy())
    @settings(
        max_examples=300,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        deadline=5000,
    )
    def test_pbd_agrees_with_path_enum_on_dags(self, g: NxMixedGraph) -> None:
        """On any random DAG with X→Y, new and old proper_backdoor_graph must agree.

        Uses ``assume`` to filter out cyclic graphs.
        """
        assume(_is_dag(g))
        new_pbd = proper_backdoor_graph(g, X, Y)
        old_pbd = _old_proper_backdoor_graph(g, X, Y)
        new_edges = set(new_pbd.directed.edges())
        old_edges = set(old_pbd.directed.edges())
        self.assertEqual(
            new_edges,
            old_edges,
            f"PBD disagreement on DAG {list(g.directed.edges())}:\n"
            f"  new={new_edges}, old={old_edges}",
        )

    @given(random_mixed_graph_with_xy())
    @settings(
        max_examples=300,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=5000,
    )
    def test_oset_matches_definition_on_any_graph(self, g: NxMixedGraph) -> None:
        """On any graph (cyclic or not), the O-set must equal the definition.

        The definition is computed independently using nx.descendants/ancestors,
        which is the ground truth for both DAGs and cyclic graphs.
        """
        new = optimal_adjustment_set(g, X, Y)
        expected = _oset_from_definition(g, X, Y)
        self.assertEqual(
            new,
            expected,
            f"O-set does not match definition on {list(g.directed.edges())}:\n"
            f"  new={new}, expected={expected}",
        )

    @given(random_mixed_graph_with_xy())
    @settings(
        max_examples=300,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=5000,
    )
    def test_oset_excludes_forbidden_nodes(self, g: NxMixedGraph) -> None:
        """O-set must never contain X or any descendant of causal nodes."""
        result = optimal_adjustment_set(g, X, Y)
        if result is None:
            return  # non-identifiable — skip
        # X must not be in O-set
        self.assertNotIn(X, result, f"X in O-set for {list(g.directed.edges())}")
        # No descendant of X should be in O-set (they are all forbidden)
        desc_x = set(nx.descendants(g.directed, X)) if X in g.directed else set()
        desc_x.add(X)
        forbidden_in_result = result & desc_x
        self.assertEqual(
            forbidden_in_result,
            set(),
            f"Forbidden nodes {forbidden_in_result} in O-set for {list(g.directed.edges())}",
        )


# ---------------------------------------------------------------------------
# Performance regression test: flat scaling on dense-SCC graphs
# ---------------------------------------------------------------------------


class TestOSetFlatScaling(unittest.TestCase):
    """Verify that the new O-set is fast on dense-SCC graphs (no path blowup)."""

    def test_fan_n8_under_50ms(self) -> None:
        """Fan graph n=8 (109,601 paths) must complete in under 50ms."""
        g = _make_fan_graph(8)
        t0 = time.perf_counter()
        optimal_adjustment_set(g, X, Y)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(
            elapsed_ms,
            50.0,
            f"optimal_adjustment_set took {elapsed_ms:.1f}ms on fan-8 graph "
            f"(expected <50ms with reachability; old path-enum took ~500ms)",
        )

    def test_find_sigma_single_door_set_fan_n8_under_200ms(self) -> None:
        """find_sigma_single_door_set on fan-8 must complete in under 200ms."""
        g = _make_fan_graph(8)
        t0 = time.perf_counter()
        find_sigma_single_door_set(g, X, Y)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(
            elapsed_ms,
            200.0,
            f"find_sigma_single_door_set took {elapsed_ms:.1f}ms on fan-8 graph "
            f"(expected <200ms with reachability; old path-enum took ~1000ms)",
        )
