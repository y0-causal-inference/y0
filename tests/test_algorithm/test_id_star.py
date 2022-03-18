# -*- coding: utf-8 -*-

"""Tests for the ID* algorithm."""
from typing import Set

from tests.test_algorithm import cases
from y0.algorithm.identify.id_star import (
    domain_of_counterfactual_values,
    ev,
    id_star,
    id_star_line_1,
    id_star_line_2,
    id_star_line_3,
    id_star_line_4,
    id_star_line_6,
    id_star_line_8,
    merge_interventions,
    sub,
)
from y0.dsl import D, One, P, Sum, Variable, W, X, Y, Z, Zero
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
        event = {Y @ (+x, -z): +y, X: -x}
        self.assertEqual(event, id_star_line_3(event))

    def test_id_star_line_4(self):
        """Check that the counterfactual graph is correct."""
        new_graph, new_event = id_star_line_4(
            graph=figure_9a.graph, event={Y @ ~x: ~y, X: x, Z @ d: z, D: d}
        )
        self.assert_graph_equal(figure_9c.graph, new_graph)
        self.assertEqual({Y @ ~x: ~y, X: x, Z: z, D: d}, new_event)

        actual_graph3, actual_event3 = id_star_line_4(
            graph=NxMixedGraph.from_edges(directed=[(D, Z), (Z, Y)]),
            event={Z @ -d: -z, Z: +z, D: -d},
        )
        self.assertIsNone(actual_event3)

        query4 = {Y @ (+x, -z): +y, X: -x}

        expected_graph = figure_9d.graph
        expected = expected_graph, query4
        self.assertEqual(expected, id_star_line_4(figure_9a.graph, query4))
        self.assertFalse(figure_9d.graph.is_connected())

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

    def test_domain_of_counterfactual_values(self):
        """ "Test that we correctly output the domain of a counterfactual"""
        event = {Y @ (+X, -Z): -Y, X: -X}
        cf_graph = figure_9d.graph
        vertices: Set[Variable] = set(cf_graph.nodes())

        for cf in event:
            self.assertIn(cf, vertices)
        for value in event.values():
            self.assertNotIn(value, vertices)

        self.assertEqual(
            {-cf.parent() for cf in vertices} - set(event.values()),
            domain_of_counterfactual_values(event, vertices - set(event)),
        )
        ## TODO: add more tests

    def test_recursion(self):
        """ "Test the recursive aspect of line 6"""
        cf_graph = figure_9d.graph
        new_event = {Y @ (+X, -Z): -Y, X: -X}
        summand, interventions_of_each_district = id_star_line_6(cf_graph, new_event)
        my_list = [
            {
                merge_interventions(element, interventions): Intervention(element.name, star=False)
                for element in district
            }
            for district, interventions in interventions_of_each_district.items()
        ]
        self.assertEqual()
        ## TODO: add more tests

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

    def test_merge_interventions(self):
        """Test that we can safely merge interventions into a counterfactual"""
        element = Y @ (X, Z)
        interventions = (-Z, -W, -D)
        expected = Y @ (D, W, X, Z)
        self.assertEqual(expected, merge_interventions(element, interventions))

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
