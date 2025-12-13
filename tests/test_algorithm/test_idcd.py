"""Tests for IDCD algorithm implementation."""

import unittest

from tests.test_algorithm.test_ioscm import simple_cyclic_graph_1
from y0.algorithm.identify.idcd import (
    _get_apt_order_predecessors,
    marginalize_to_ancestors,
    validate_preconditions,
)
from y0.algorithm.ioscm.utils import get_apt_order
from y0.dsl import P, R, Variable, W, X, Y, Z
from y0.graph import NxMixedGraph

# ----------------------------------------------------------------------------


class TestValidatePreconditions(unittest.TestCase):
    """Tests for IDCD precondition validation."""

    def test_empty_targets_raises_error(self) -> None:
        """Empty target set should raise ValueError."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets: set[Variable] = set()
        district = {Y}

        with self.assertRaisesRegex(ValueError, "Target set C cannot be empty"):
            validate_preconditions(graph, targets, district, recursion_level=0)

    def test_empty_district_raises_error(self) -> None:
        """Empty district should raise ValueError."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets = {Y}
        district: set[Variable] = set()

        with self.assertRaisesRegex(ValueError, "District D cannot be empty"):
            validate_preconditions(graph, targets, district, recursion_level=0)

    def test_targets_not_subset_of_district_raises_error(self) -> None:
        """Target must be a subset of district."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])
        targets = {Y, Z}
        district = {Y}  # Z is not in district

        with self.assertRaisesRegex(
            ValueError,
            "Target must be subset of district.",
        ):
            validate_preconditions(graph, targets, district, recursion_level=0)

    def test_district_not_subset_of_nodes_raises_error(self) -> None:
        """District must be a subset of graph nodes."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets = {Y}
        district = {Y, Z}  # Z not in graph

        with self.assertRaisesRegex(
            ValueError,
            "District must be subset of graph nodes.",
        ):
            validate_preconditions(graph, targets, district, recursion_level=0)

    # TODO: this test doesn't pass yet
    def test_district_not_consolidated_raises_error(self) -> None:
        """District must satisfy CD(G_D) = {D} (single consolidated district)."""
        targets = {X}
        district = {X, W}  # the full district is X,W,Z

        with self.assertRaisesRegex(
            ValueError, "D must be a single consolidated district in G[D]."
        ):
            validate_preconditions(simple_cyclic_graph_1, targets, district, recursion_level=0)

    def test_valid_preconditions_pass(self) -> None:
        """Valid inputs should pass without error."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])
        targets = {Z}
        district = {Y, Z}

        # should not raise due to valid inputs
        validate_preconditions(graph, targets, district, recursion_level=0)

    def test_cyclic_graph_valid_preconditions_pass(self) -> None:
        """Valid inputs on cyclic graph should pass without error."""
        # simple cycle - X -> Y -> Z -> X
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z), (Z, X)])

        targets = {Z}
        district = {X, Y, Z}

        # should not raise due to valid inputs
        validate_preconditions(graph, targets, district, recursion_level=0)


# ----------------------------------------------------------------------------


class TestMarginalizationToAncestors(unittest.TestCase):
    """Tests for marginalization to ancestors function."""

    def test_no_marginalization_district_equals_ancestral_closure(self) -> None:
        """If district equals ancestral closure, no marginalization should occur."""
        distribution = P(X, Y)
        district = {X, Y}
        ancestral_closure = {X, Y}

        result = marginalize_to_ancestors(
            distribution, district, ancestral_closure, recursion_level=0
        )

        self.assertEqual(result, distribution)  # should be unchanged

    def test_marginalization_occurs(self) -> None:
        r"""Marginalization should remove district \ ancestral_closure variables."""
        distribution = P(X, Y, Z)
        district = {X, Y, Z}
        ancestral_closure = {X, Y}

        result = marginalize_to_ancestors(
            distribution, district, ancestral_closure, recursion_level=0
        )

        result_str = str(result)
        self.assertIn("Sum", result_str)
        self.assertIn("Z", result_str)

    def test_marginalization_with_single_variable(self) -> None:
        """Test marginalizing out a single variable."""
        distribution = P(X, Y)
        district = {X, Y}
        ancestral_closure = {Y}

        result = marginalize_to_ancestors(
            distribution, district, ancestral_closure, recursion_level=0
        )

        result_str = str(result)
        self.assertIn("Sum", result_str)
        self.assertIn("X", result_str)


# ----------------------------------------------------------------------------


class TestGetAptOrderPredecessors(unittest.TestCase):
    """Tests the _get_apt_order_predecessors function."""

    def test_first_element_has_no_predecessors(self) -> None:
        """First element in apt-order should have no predecessors."""
        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        # first element should not have predecessors
        scc = frozenset([R])
        ancestral_closure = {R, W, X, Z, Y}

        predecessors = _get_apt_order_predecessors(scc, apt_order, ancestral_closure)
        self.assertEqual(predecessors, set())

    def test_predecessors_filtered_by_ancestral_closure(self) -> None:
        """Predecessors should only include those in the ancestral closure."""
        # Apt order = [R, W, X, Z, Y]

        # If ancestral closure = {R, Z}, only R should be predecessor of Z
        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([Z])
        ancestral_closure = {R, Z}

        predecessors = _get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        self.assertEqual(predecessors, {R})

    def test_all_predecessors_included(self) -> None:
        """All variables before SCC in apt order and in ancestral closure should be included."""
        # apt order = [R, W, X, Z, Y], Y is last so all before it should be included

        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([Y])
        ancestral_closure = {R, W, X, Z, Y}

        predecessors = _get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        self.assertEqual(predecessors, {R, W, X, Z})

    def test_multi_node_scc_uses_minimum_position(self) -> None:
        """For multi-node SCC, use the minimum position of its members in apt order."""
        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([W, X, Z])
        ancestral_closure = {R, W, X, Z, Y}

        predecessors = _get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        self.assertEqual(predecessors, {R})

    def test_empty_ancestral_closure(self) -> None:
        """If ancestral closure is empty, there should be no predecessors."""
        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([Z])
        ancestral_closure: set[Variable] = set()

        predecessors = _get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        self.assertEqual(predecessors, set())


# ----------------------------------------------------------------------------
