"""Tests for the identification of district variables for different branch cases.

This module tests:
1. identify_district_variables_cyclic()
"""

import networkx as nx

from tests.test_algorithm import cases
from y0.algorithm.identify.cyclic_id import (
    cyclic_id,
    identify_district_variables_cyclic,
)
from y0.dsl import Z1, Z2, Fraction, P, Sum, Variable, X, Y
from y0.graph import NxMixedGraph


class TestIdentifyDistrictVariablesCyclic(cases.GraphTestCase):
    """Tests for the generalized identify_district_variables function for the cyclic ID algorithm."""

    def test_base_case_1(self) -> None:
        """When the ancestral set = input variables, return the district probability."""
        graph = NxMixedGraph.from_edges(directed=[], undirected=[])

        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y}),
            input_district=frozenset({Y}),
            district_probability=P(Y),
            graph=graph,
            topo=[Y],
        )

        if result is None:
            self.fail("Expected non-None result from identify_district_variables_cyclic")
        self.assert_expr_equal(P(Y), result)

    def test_base_case_1_marginalization(self) -> None:
        """Marginalize out Z1 when input district = {Y, Z1} but target = {Y}."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[])

        graph.add_node(Z1)  # Add disconnected node Z1

        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y}),
            input_district=frozenset({Y, Z1}),
            district_probability=P(Y, Z1),
            graph=graph,
            topo=[X, Y, Z1],
        )

        expected = Sum.safe(P(Y, Z1), Z1)
        if result is None:
            self.fail("Expected non-None result from identify_district_variables_cyclic")
        self.assert_expr_equal(expected, result)

    def test_base_case_1_multiple_variables(self) -> None:
        """Base Case 1: Multiple Target variables (|C| > 1)."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (X, Z1)], undirected=[])

        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y, Z1}),
            input_district=frozenset({Y, Z1}),
            district_probability=P(Y, Z1),
            graph=graph,
            topo=[X, Y, Z1],
        )

        # No marginalization, ancestral set = input variables = input district
        if result is None:
            self.fail("Expected non-None result from identify_district_variables_cyclic")
        self.assert_expr_equal(P(Y, Z1), result)

    def test_base_case_2_no_confounding_returns_none(self) -> None:
        """Base Case 2: Ancestral set = input district, should return None."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[])

        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y}),
            input_district=frozenset({X, Y}),
            district_probability=P(X, Y),
            graph=graph,
            topo=[X, Y],
        )

        # if ancestral set is equal to the input district, should return a FAIL/None
        self.assertIsNone(result)

    def test_case_3_triggers_recursion(self) -> None:
        """Base Case 3: C ⊂ A ⊂ T triggers recursion with Lemma 4."""
        graph = NxMixedGraph.from_edges(
            directed=[(Z2, Z1), (Z1, X), (X, Y)], undirected=[(Z2, X), (Z2, Y)]
        )

        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y}),
            input_district=frozenset({Z2, X, Y}),
            district_probability=P(Z2, X, Y),
            graph=graph,
            topo=[Z2, Z1, X, Y],
        )

        inner_term = P(X, Y, Z2)
        expected = Fraction(Sum.safe(inner_term, Z2), Sum.safe(Sum.safe(inner_term, Z2), Y))
        if result is None:
            self.fail("Expected non-None result from identify_district_variables_cyclic")
        self.assert_expr_equal(expected, result)

    def test_napkin_full_integration(self) -> None:
        """Integration test that calls the cyclic ID to ensure the right estimand."""
        graph = NxMixedGraph.from_edges(
            directed=[(Z2, Z1), (Z1, X), (X, Y)], undirected=[(Z2, X), (Z2, Y)]
        )

        result = cyclic_id(
            graph=graph,
            outcomes={Y},
            interventions={X},
        )

        inner_term = P(X, Y, Z1, Z2) * P(Z2) / P(Z1, Z2)
        numerator = Sum.safe(inner_term, Z2)
        denominator = Sum.safe(Sum.safe(inner_term, Z2), Y)
        expected = Fraction(numerator, denominator)
        if result is None:
            self.fail("Expected non-None result from identify_district_variables_cyclic")
        self.assert_expr_equal(expected, result)

    def test_incorrect_input_c_not_subset_of_t(self) -> None:
        """Input Validation: Input variables must be a subset of T."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[])

        with self.assertRaises(nx.NetworkXError):
            identify_district_variables_cyclic(
                input_variables=frozenset({Z1}),
                input_district=frozenset({X, Y}),
                district_probability=P(X, Y),
                graph=graph,
                topo=[X, Y, Z1],
            )

    def test_bow_arc_unidentifiable(self) -> None:
        """Regression test: Bow arc structure is unidentifiable."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[(X, Y)])

        input_district = frozenset({X, Y})
        target = frozenset({Y})
        q_t = P(X, Y)

        result = identify_district_variables_cyclic(
            input_variables=target,
            input_district=input_district,
            district_probability=q_t,
            graph=graph,
            topo=[X, Y],
        )

        self.assertIsNone(result)

    def test_tian_pearl_figure_9_deep_recursion(self) -> None:
        """Uses figure 9 of the Tian & Pearl paper to make sure recursion is happening correctly."""
        W1 = Variable("W1")  # noqa: N806
        W2 = Variable("W2")  # noqa: N806
        W3 = Variable("W3")  # noqa: N806
        W4 = Variable("W4")  # noqa: N806
        W5 = Variable("W5")  # noqa: N806

        graph = NxMixedGraph.from_edges(
            directed=[(W1, W2), (W2, X), (W3, W4), (W4, X), (X, Y)],
            undirected=[(W1, W3), (W3, W5), (W4, W5), (W2, W3), (W1, X), (W1, Y)],
        )

        topo = [W3, W5, W4, W1, W2, X, Y]
        input_district = frozenset(graph.nodes())

        q_v = P(W3, W5, W4, W1, W2, X, Y)

        try:
            result_cyclic = identify_district_variables_cyclic(
                input_variables=frozenset({Y}),
                input_district=input_district,
                district_probability=q_v,
                graph=graph,
                topo=topo,
            )
        except Exception as e:
            self.fail(f"Should be Identifiable but raised: {e}")

        self.assertIsInstance(result_cyclic, Fraction, "should produce ratio fraction form")
        if result_cyclic is None:
            self.fail("Expected non-None result from identify_district_variables_cyclic")
        self.assertIn(Y, result_cyclic.get_variables(), "Result should contain outcome variable Y")
