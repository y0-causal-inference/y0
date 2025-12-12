"""Tests for IDCD algorithm implementation."""

import unittest

from tests.test_algorithm.test_ioscm import simple_cyclic_graph_1
from y0.algorithm.identify.idcd import marginalize_to_ancestors, validate_preconditions
from y0.dsl import P, Q, W, X, Y, Z
from y0.graph import NxMixedGraph


class TestIDCD(unittest.TestCase):
    """Tests for IDCD algorithm implementation."""

    def test_fail(self) -> None:
        """Test failure."""
        self.fail(msg="tests are required before merge")


class TestValidatePreconditions(unittest.TestCase):
    """Tests for IDCD precondition validation."""

    def test_empty_targets_raises_error(self) -> None:
        """Empty target set should raise ValueError."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets = set()
        district = {Y}

        with self.assertRaisesRegex(ValueError, "Target set C cannot be empty"):
            validate_preconditions(graph, targets, district, recursion_level=0)

    def test_empty_district_raises_error(self) -> None:
        """Empty district should raise ValueError."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets = {Y}
        district = set()

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
        district = {Y, Q}  # Q not in graph

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
