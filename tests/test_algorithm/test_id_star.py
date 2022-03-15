# -*- coding: utf-8 -*-

"""Tests for the ID* algorithm."""

from tests.test_algorithm import cases
from y0.algorithm.identify.id_star import (
    ev,
    id_star,
    id_star_line_1,
    id_star_line_2,
    id_star_line_3,
    id_star_line_4,
    id_star_line_5,
    id_star_line_6,
    id_star_line_8,
    sub,
)
from y0.dsl import D, One, P, Sum, W, X, Y, Z, Zero
from y0.examples import figure_9a, figure_9c, figure_9d
from y0.graph import NxMixedGraph

d, w, x, y, z = -D, -W, -X, -Y, -Z


class TestIDStar(cases.GraphTestCase):
    """Tests for the ``ID*`` algorithm."""

    def test_id_star_line_1(self):
        """Check if event is empty."""
        self.assertTrue(id_star_line_1({}))
        self.assertFalse(id_star_line_1({X @ x: ~x}))

    def test_id_star_line_2(self):
        """Check to see if the counterfactual event violates the Axiom of Effectiveness."""
        # Examples all from figure_9a.graph
        self.assertTrue(id_star_line_2({X @ x: ~x}))
        self.assertTrue(id_star_line_2({X @ ~x: x}))
        self.assertTrue(id_star_line_2({X @ (y, z, x): ~x}))
        self.assertTrue(id_star_line_2({X @ (x, z): x, Y @ (x, -y): +y}))
        self.assertTrue(id_star_line_2({Y @ -x: -y, X @ +x: -x, Y @ x: y, Y @ ~x: y, Y @ ~x: y}))
        self.assertFalse(id_star_line_2({X @ x: x}))
        self.assertFalse(id_star_line_2({X @ ~x: ~x}))
        self.assertFalse(id_star_line_2({X @ x: x, Y @ x: y}))

    def test_id_star_line_3(self):
        """Check to see if the counterfactual event is tautological."""
        self.assertEqual({}, id_star_line_3({X @ x: x}))
        self.assertEqual({Y @ x: y}, id_star_line_3({Y @ x: y, X @ x: x}))
        self.assertEqual({Y @ x: +y}, id_star_line_3({Y @ x: +y, X @ ~x: ~x}))

    def test_id_star_line_4(self):
        """Check that the counterfactual graph is correct."""
        new_graph, new_event = id_star_line_4(
            graph=figure_9a.graph, event={Y @ ~x: ~y, X: x, Z @ d: z, D: d}
        )
        self.assert_graph_equal(figure_9c.graph, new_graph)
        self.assertEqual({Y @ ~x: ~y, X: x, Z: z, D: d}, new_event)

    def test_id_star_line_5(self):
        """Check whether the query is inconsistent with the counterfactual graph."""
        actual_graph3, actual_event3 = id_star_line_4(
            graph=NxMixedGraph.from_edges(directed=[(D, Z), (Z, Y)]),
            event={Z @ -d: -z, Z: +z, D: -d},
        )
        self.assertEqual(Zero(), id_star_line_5(actual_graph3, actual_event3))

    def test_id_star_line_6(self):
        """Check that the input to id_star from each district is properly constructed."""
        counterfactual_graph = NxMixedGraph.from_edges(
            undirected=[(Y @ (~X, Z), X)],
            directed=[
                (W @ (~X, Z), Y @ (~X, Z)),
            ],
        )
        event = {Y @ (+X, -Z): -Y, X: -X}
        expected_summand = {-W}
        expected_interventions_of_districts = {
            frozenset([Y @ (~X, Z), X]): {-W},
            frozenset([W @ (~X, Z)]): {-Y, -X},
        }
        self.assertEqual(
            set(expected_interventions_of_districts),
            set(counterfactual_graph.get_c_components()),
        )
        ## Create a counterfactual graph with at least 2 c-components and return the summand and interventions of each
        #
        actual_summand, actual_iod = id_star_line_6(counterfactual_graph, event)
        self.assertEqual(expected_summand, actual_summand)
        self.assertEqual(expected_interventions_of_districts, actual_iod)

    def test_id_star_line_8(self):
        """Attempt to generate a conflict with an inconsistent value assignment."""
        graph = NxMixedGraph.from_edges(undirected=[(Y @ +X, X), (X, D @ -D)])
        self.assertEqual({-D, +X}, sub(graph))
        query1 = {Y @ +X: +Y, X: -X, D @ -D: -D}
        self.assertEqual({-X, -D, +Y}, ev(query1))
        self.assertTrue(id_star_line_8(graph, query1))
        query2 = {D @ -D: -D}
        self.assertEqual({-D}, ev(query2))
        self.assertFalse(id_star_line_8(graph, query2))

    def test_id_star_line_9(self):
        """Test line 9 of the ID* algorithm.

        Test that estimand returned by taking the effect of all subscripts in
        new_event on variables in new_event is correct
        """

    def test_id_star(self):
        """Test that the ID* algorithm returns the correct estimand."""
        query = {Y @ (+x, -z): +y, X: -x}
        actual = id_star(figure_9a.graph, query)
        expected = Sum[W](P(Y @ (Z, W), X @ (Z, W)) * P(W @ X))
        self.assert_expr_equal(expected, actual)

    def test_idc_star(self):
        """Test that the IDC* algorithm returns the correct estimand."""
        query = P(Y @ ~X | X, Z @ D, D)
        vertices = set(figure_9a.graph.nodes())
        estimand = Sum[W](P(Y @ (Z, W), X @ (Z, W)) * P(W @ X))
        expected = estimand / Sum[vertices - {X, Z @ D, D}](estimand)
        # actual = idc_star( figure_9a.graph, query)
        # self.assert_expr_equal( expected, actual )
