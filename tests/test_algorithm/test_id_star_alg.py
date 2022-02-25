# -*- coding: utf-8 -*-

"""Tests for the ID* algorithm."""

from tests.test_algorithm import cases
from y0.algorithm.identify.id_star import (
    id_star_line_1,
    id_star_line_2,
    id_star_line_3,
    id_star_line_4,
    id_star_line_9,
    idc_star_line_2,
)
from y0.dsl import D, One, P, Sum, Variable, W, D, X, Y, Z, Zero
from y0.examples import figure_9a, figure_9c, figure_9d
d, w, x, y, z = -D, -W, -X, -Y, -Z

class TestIDStar(cases.GraphTestCase):
    # def test_idc_star_line_2(self):
    #     r"""Test line 2 of the IDC* algorithm.
    #
    #     Construct the counterfactual graph Figure 9(c) where the corresponding modified query
    #     is :math:`P(Y_{x} = y|X= x',Z=z,D=d)`
    #     """
    #     input_event= {Y @ -x:  x}
    #     input_conditional = {X: +x, Z: -z, D: -d}
    #     input_graph = figure_9a.graph
    #     actual_graph, actual_query = idc_star_line_2(input_graph, input_event, input_conditional)
    #     expected_graph = figure_9c.graph
    #     expected_query = P(D, X, Y @ ~X, Z)
    #     self.assert_expr_equal(expected=expected_query, actual=actual_query)
    #     self.assert_graph_equal(expected=expected_graph, actual=actual_graph)
    #
    # def test_idc_star_line_4(self):
    #     r"""Test line 4 of the IDC* algorithm.
    #
    #     Check that line 4 or IDC* works correctly moves :math:`Z, D` (with
    #     :math:`D` being redundant due to graph structure) to the
    #     subscript of :math:`Y_\mathbf{x}`, to obtain :math:`P(Y_{X',Z} | X )`,
    #     and calls IDC* with this query recursively.
    #     """
    #     input_query = P(Y @ ~X | X, Z, D)
    #     expected_output_query = P(Y @ (~X, Z) | X)
    #     new_delta = {X, Z, D}
    #     new_event = {Y @ ~X}
    #     graph = figure_9c.graph
    #     for counterfactual in [Z, D]:
    #         # self.assertTrue(are_d_separated(graph.remove_outgoing_edges_from( {counterfactual} ), counterfactual, new_event))
    #         counterfactual_value = Variable(counterfactual.name)
    #         parents = new_delta - {counterfactual}
    #         children = {g.intervene(counterfactual_value) for g in new_event}
    #         # self.assert_expr_equal( P( Y @ {X, counterfactual}  | new_event - {counterfactual}), P(children | parents))

    def test_id_star_line_1(self):
        """Check if event is empty"""
        self.assertEqual(One(), id_star_line_1(graph=figure_9a.graph, event={}))

    def test_id_star_line_2(self):
        """Check to see if the counterfactual event violates the Axiom of Effectiveness."""
        self.assertEqual(Zero(), id_star_line_2(graph=figure_9a.graph, event={X @ x: ~x}))
        self.assertEqual(Zero(), id_star_line_2(graph=figure_9a.graph, event={X @ ~x: x}))
        self.assertEqual(Zero(), id_star_line_2(graph=figure_9a.graph, event={X @ (y, z, x): ~x}))
        self.assertEqual(
            Zero(), id_star_line_2(graph=figure_9a.graph, event={X @ (x, z): x, Y @ (x, -y): +y})
        )
        self.assertEqual(
            Zero(),
            id_star_line_2(
                graph=figure_9a.graph, event={Y @ -x: -y, X @ +x: -x, Y @ x: y, Y @ ~x: y, Y @ ~x: y}
            ),
        )
        self.assertIsNone(id_star_line_2(graph=figure_9a.graph, event={X @ x: x}))
        self.assertIsNone(id_star_line_2(graph=figure_9a.graph, event={X @ ~x: ~x}))
        self.assertIsNone(
            id_star_line_2(
                graph=figure_9a.graph, event={X @ x: x, Y @ x: y }
            )
        )

    def test_id_star_line_3(self):
        """Check to see if the counterfactual event is tautological."""
        self.assertEqual({}, id_star_line_3(graph=figure_9a.graph, event={X @ x: x}))
        self.assertEqual(
            {Y @ x: y}, id_star_line_3(graph=figure_9a.graph, event={Y @ x: y, X @ x: x})
        )
        self.assertEqual({Y @ x: +y}, id_star_line_3(graph=figure_9a.graph, event={Y @ x: +y, X @ ~x: ~x}))

    def test_id_star_line_4(self):
        """Check that the counterfactual graph is correct."""
        new_graph, new_event = id_star_line_4(graph=figure_9a.graph, event={Y @ ~x: ~y, X: x, Z @ d: z, D: d})
        self.assert_graph_equal(figure_9c.graph, new_graph)
        self.assertEqual({Y @ ~x: ~y, X: x, Z: z, D: d}, new_event)

    def test_id_star_line_5(self):
        """Check whether the query is inconsistent with the counterfactual graph."""
        actual_graph3, actual_event3 = id_star_line_4(
            graph=NxMixedGraph.from_edges(directed=[(D, Z), (Z, Y)]),
            event={Z @ -d: -z, Z: +z, D: -d}
        )
        self.assertEqual(Zero(), actual_event3)

    def test_id_star_line_6(self):
        """Check that the input to id_star from each district is properly constructed."""
        graph = figure_9d.graph

        query = P(Y @ (X, Z), X)

    def test_id_star_line_7(self):
        """Check that the graph is entirely one c-component."""

    def test_id_star_line_8(self):
        """Attempt to generate a conflict with an inconsistent value assignment."""

    def test_id_star_line_9(self):
        """Test line 9 of the ID* algorithm.

        Test that estimand returned by taking the effect of all subscripts in
        new_event on variables in new_event is correct
        """
        input_query = P(Y @ (W, Z), X)
        output_query = P[W, Z](Y, X)
        self.assert_expr_equal(output_query, id_star_line_9(input_query))

    def test_id_star(self):
        """Test that the ID* algorithm returns the correct estimand."""
        query = {Y @ (+x, -z): +y, X: -x}
        # actual = id_star( figure_9a.graph, query)
        expected = Sum[W](P(Y @ (Z, W), X @ (Z, W)) * P(W @ X))
        # self.assert_expr_equal(expected, actual)

    def test_idc_star(self):
        """Test that the IDC* algorithm returns the correct estimand."""
        query = P(Y @ ~X | X, Z @ D, D)
        vertices = set(figure_9a.graph.nodes())
        estimand = Sum[W](P(Y @ (Z, W), X @ (Z, W)) * P(W @ X))
        expected = estimand / Sum[vertices - {X, Z @ D, D}](estimand)
        # actual = idc_star( figure_9a.graph, query)
        # self.assert_expr_equal( expected, actual )
