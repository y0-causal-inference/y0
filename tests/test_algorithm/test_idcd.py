"""Tests for IDCD algorithm implementation."""

import unittest

from tests.test_algorithm.test_ioscm import simple_cyclic_graph_1
from y0.algorithm.identify import Unidentifiable
from y0.algorithm.identify.idcd import (
    _get_apt_order_predecessors,
    compute_scc_distributions,
    idcd,
    identify_through_scc_decomposition,
    marginalize_to_ancestors,
    validate_preconditions,
)
from y0.algorithm.ioscm.utils import get_apt_order
from y0.dsl import Expression, P, R, Variable, W, X, Y, Z
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
            validate_preconditions(graph, targets, district)

    def test_empty_district_raises_error(self) -> None:
        """Empty district should raise ValueError."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets = {Y}
        district: set[Variable] = set()

        with self.assertRaisesRegex(ValueError, "District D cannot be empty"):
            validate_preconditions(graph, targets, district)

    def test_targets_not_subset_of_district_raises_error(self) -> None:
        """Target must be a subset of district."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])
        targets = {Y, Z}
        district = {Y}  # Z is not in district

        with self.assertRaisesRegex(
            ValueError,
            "Target must be subset of district.",
        ):
            validate_preconditions(graph, targets, district)

    def test_district_not_subset_of_nodes_raises_error(self) -> None:
        """District must be a subset of graph nodes."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets = {Y}
        district = {Y, Z}  # Z not in graph

        with self.assertRaisesRegex(
            ValueError,
            "District must be subset of graph nodes.",
        ):
            validate_preconditions(graph, targets, district)

    # TODO: this test doesn't pass yet
    def test_district_not_consolidated_raises_error(self) -> None:
        """District must satisfy CD(G_D) = {D} (single consolidated district)."""
        targets = {X}
        district = {X, W}

        # When we check CD(G[{X,Y,W,Z}]), we should get TWO districts: {X,Y} and {W,Z}
        # But we passed district={X,Y,W,Z} which claims it's ONE consolidated district
        # This violates the precondition CD(G_D) = {D}

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
        validate_preconditions(graph, targets, district)

    def test_cyclic_graph_valid_preconditions_pass(self) -> None:
        """Valid inputs on cyclic graph should pass without error."""
        # simple cycle - X -> Y -> Z -> X
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z), (Z, X)])

        targets = {Z}
        district = {X, Y, Z}

        # should not raise due to valid inputs
        validate_preconditions(graph, targets, district)


# ----------------------------------------------------------------------------


class TestMarginalizationToAncestors(unittest.TestCase):
    """Tests for marginalization to ancestors function."""

    def test_no_marginalization_district_equals_ancestral_closure(self) -> None:
        """If district equals ancestral closure, no marginalization should occur."""
        distribution = P(X, Y)
        district = {X, Y}
        ancestral_closure = {X, Y}

        result = marginalize_to_ancestors(distribution, district, ancestral_closure)

        self.assertEqual(result, distribution)  # should be unchanged

    def test_marginalization_occurs(self) -> None:
        r"""Marginalization should remove district \ ancestral_closure variables."""
        distribution = P(X, Y, Z)
        district = {X, Y, Z}
        ancestral_closure = {X, Y}

        result = marginalize_to_ancestors(distribution, district, ancestral_closure)

        result_str = str(result)
        self.assertIn("Sum", result_str)
        self.assertIn("Z", result_str)

    def test_marginalization_with_single_variable(self) -> None:
        """Test marginalizing out a single variable."""
        distribution = P(X, Y)
        district = {X, Y}
        ancestral_closure = {Y}

        result = marginalize_to_ancestors(distribution, district, ancestral_closure)

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


class TestIDCDFunction(unittest.TestCase):
    """Tests for the IDCD algorithm implementation."""

    def test_base_case_ancestral_closure_equals_district(self) -> None:
        """Test base case where ancestral closure equals district."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])

        targets = {Z}
        district = {Z}
        distribution = P(Z)

        # should return the distribution as is without recursion
        result = idcd(
            graph=graph,
            targets=targets,
            district=district,
            distribution=distribution,
        )

        self.assertEqual(result, distribution)

    def test_unidentifiable_case_ancestral_closure_equals_district(self) -> None:
        """Line 19-20: When ancestral closure equals district it should be unidentifiable."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z), (Z, X)])

        targets = {X}
        district = {X, Y, Z}
        distribution = P(X, Y, Z)

        # the result should indicate unidentifiability
        with self.assertRaises(Unidentifiable) as context:
            idcd(
                graph=graph,
                targets=targets,
                district=district,
                distribution=distribution,
            )

        self.assertIn("unidentifiable", str(context.exception).lower())

    def test_recursive_case_through_scc_decomposition(self) -> None:
        """Tests the recursive case when SCC decomposition is needed."""
        graph = NxMixedGraph.from_edges(directed=[(R, X), (X, X), (X, Y), (Y, Z), (Z, X)])

        # identify Z from the full graph
        targets = {X}
        district = {X, Y, Z}
        distribution = P(X, Y, Z)

        result = idcd(graph, targets, district, distribution)

        self.assertIsInstance(result, Expression)

    def test_single_scc_in_consolidated_district(self) -> None:
        """Test with a single SCC in the consolidated district."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, Z),
                (Z, X),
            ]
        )

        targets = {Z}
        ancestral_closure = {X, Y, Z}
        with self.assertRaises(Unidentifiable):
            identify_through_scc_decomposition(
                graph=graph,
                targets=targets,
                ancestral_closure=ancestral_closure,
                recursion_level=0,
            )

    def test_multiple_sccs_in_consolidated_district(self) -> None:
        """Test with multiple SCCs in the consolidated district."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, X),
                (Y, W),
                (W, Z),
                (Z, W),
            ]
        )

        targets = {Z}
        ancestral_closure = {X, Y, W, Z}

        with self.assertRaises(Unidentifiable):
            identify_through_scc_decomposition(
                graph=graph,
                targets=targets,
                ancestral_closure=ancestral_closure,
                recursion_level=0,
            )

    def test_no_relevant_sccs_raises_unidentifiable(self) -> None:
        """Test that when no relevant SCCs are found, Unidentifiable is raised."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, X),
                (X, Z),
            ]
        )

        targets = {Z}
        ancestral_closure = {X, Y, Z}

        try:
            estimand = identify_through_scc_decomposition(
                graph=graph, targets=targets, ancestral_closure=ancestral_closure
            )
        except Unidentifiable as e:
            self.assertIn("No SCCs", str(e))
        else:
            self.fail(f"should have been unidentifiable, but got: {estimand}")


# ----------------------------------------------------------------------------


class TestComputeSCCDistributions(unittest.TestCase):
    """Tests for compute_scc_distributions function."""

    def test_single_scc_returns_correct_structure(self) -> None:
        """Test that a single SCC returns the correct distribution structure."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, Z),
                (Z, X),
            ]
        )

        subgraph_a = graph.subgraph({X, Y, Z})

        relevant_sccs = [frozenset({X, Y, Z})]
        ancestral_closure = {X, Y, Z}

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
        )

        # Should return a dictionary
        self.assertIsInstance(result, dict)

        # Should have one entry (one SCC)
        self.assertEqual(len(result), 1)

        # The key should be the SCC
        self.assertIn(frozenset({X, Y, Z}), result)

        # The value should be an Expression
        distribution = result[frozenset({X, Y, Z})]
        self.assertIsInstance(distribution, Expression)

    def test_multiple_sccs_with_cycles(self) -> None:
        """Test multiple SCCs each with cycles in the graph."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, X),
                (Y, X),
                (W, Z),
                (Z, W),
            ]
        )

        # subgraph contains both cycles in the graph
        subgraph_a = graph.subgraph({X, Y, W, Z})

        # two SCCs
        relevant_sccs = [frozenset({X, Y}), frozenset({W, Z})]
        ancestral_closure = {X, Y, W, Z}

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
        )

        # Should return a dictionary
        self.assertIsInstance(result, dict)

        # should have two entries (two SCCs)
        self.assertEqual(len(result), 2)

        # Each key should be an SCC
        self.assertIn(frozenset({X, Y}), result)
        self.assertIn(frozenset({W, Z}), result)

        for scc in relevant_sccs:
            self.assertIsInstance(result[scc], Expression)

    def test_intervention_set_calculation(self) -> None:
        """Test that intervention sets are calculated correctly for SCCs."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (R, X),
                (X, Y),
                (Y, Z),
            ]
        )

        # subgraph contains just Y -> Z
        subgraph_a = graph.subgraph({Y, Z})

        # one SCC
        relevant_sccs = [frozenset({Z})]

        # ancestral closure is {Y, Z}
        # So intervention_set should be nodes - ancestral_closure = {R, X, Y, Z} - {Y, Z} = {R, X}

        ancestral_closure = {Y, Z}

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
        )

        # should successfully compute distribution for the SCC
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)

        self.assertIsInstance(result[frozenset({Z})], Expression)
