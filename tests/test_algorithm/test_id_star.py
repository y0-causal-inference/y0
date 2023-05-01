# -*- coding: utf-8 -*-

"""Tests for the ID* algorithm."""
from typing import Set

from networkx import NetworkXPointlessConcept

from tests.test_algorithm import cases
from y0.algorithm.conditional_independencies import are_d_separated
from y0.algorithm.identify.cg import is_not_self_intervened
from y0.algorithm.identify.id_star import (
    ev,
    get_district_interventions,
    get_free_variables,
    id_star,
    id_star_line_4,
    id_star_line_6,
    id_star_line_8,
    is_event_empty,
    merge_interventions,
    intervene_on_district,
    remove_event_tautologies,
    sub,
    violates_axiom_of_effectiveness,
    is_redundant_counterfactual
)
from y0.dsl import D, Intervention, P, Sum, Variable, W, X, Y, Z
from y0.examples import figure_9a, figure_9c, figure_9d, tikka_figure_5
from y0.graph import NxMixedGraph

d, w, x, y, z = -D, -W, -X, -Y, -Z


class TestIDStar(cases.GraphTestCase):
    """Tests for the ``ID*`` algorithm."""

    def test_id_star_line_1(self):
        """Check if event is empty."""
        self.assertTrue(is_event_empty({}))
        self.assertFalse(is_event_empty({X @ x: ~x}))

    def test_id_star_line_2(self):
        """Check to see if the counterfactual event violates the Axiom of Effectiveness."""
        # Examples all from figure_9a.graph
        self.assertTrue(violates_axiom_of_effectiveness({X @ x: ~x}))
        self.assertTrue(violates_axiom_of_effectiveness({X @ ~x: x}))
        self.assertTrue(violates_axiom_of_effectiveness({X @ (y, z, x): ~x}))
        self.assertTrue(violates_axiom_of_effectiveness({X @ (x, z): x, Y @ (x, -y): +y}))
        self.assertTrue(
            violates_axiom_of_effectiveness(
                {Y @ -x: -y, X @ +x: -x, Y @ x: y, Y @ ~x: y, Y @ ~x: y}
            )
        )
        self.assertFalse(violates_axiom_of_effectiveness({X @ x: x}))
        self.assertFalse(violates_axiom_of_effectiveness({X @ ~x: ~x}))
        self.assertFalse(violates_axiom_of_effectiveness({X @ x: x, Y @ x: y}))

    def test_id_star_line_3(self):
        """Check to see if the counterfactual event is tautological."""
        self.assertEqual({}, remove_event_tautologies({X @ x: x}))
        self.assertEqual({Y @ x: y}, remove_event_tautologies({Y @ x: y, X @ x: x}))
        self.assertEqual({Y @ x: +y}, remove_event_tautologies({Y @ x: +y, X @ ~x: ~x}))
        self.assertEqual({Y @ x: +y}, remove_event_tautologies({Y @ x: +y, X @ (~x, w): ~x}))
        self.assertEqual({Y @ x: +y}, remove_event_tautologies({Y @ x: +y, X @ (w, ~x): ~x}))
        event = {Y @ (+x, -z): +y, X: -x}
        self.assertEqual(event, remove_event_tautologies(event))

    def test_id_star_line_4(self):
        """Check that the counterfactual graph is correct."""
        new_graph, new_event = id_star_line_4(
            graph=figure_9a.graph, event={Y @ ~x: ~y, X: -x, Z @ -d: -z, D: -d}
        )
        self.assert_graph_equal(figure_9c.graph, new_graph)
        self.assertEqual({Y @ ~x: ~y, X: -x, Z: -z, D: -d}, new_event)

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
        input_graph = tikka_figure_5.graph
        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d}
        expected_summand = {W}
        expected_district_interventions = {
            frozenset({X, Y @ -x}): {Y @ (-x, -z, -w, -d): -y, X @ (-z, -w, -d): +x},
            frozenset({Z}): {Z @ (-y, -x, -w, -d): -z},
            frozenset({W @ -x}): {W @ (-y, -x, -z, -d): -w},
            frozenset({D}): {D @ (-y, -x, -z, -w): -d}}
        ## Create a counterfactual graph with at least 2 c-components and return the summand and interventions of each
        #
        actual_summand, actual_district_interventions = id_star_line_6(input_graph, input_event)
        self.assertEqual(expected_summand, actual_summand)
        self.assertEqual(expected_district_interventions, actual_district_interventions)

    def test_get_free_variables(self):
        """Test that each variable not in the event or self-intervened is marginalized out."""
        input_graph = tikka_figure_5.graph
        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d}
        expected_summand = {W}
        self.assertEqual(expected_summand, get_free_variables(input_graph, input_event))

    # Def test_domain_of_counterfactual_values(self):
    #     """ "Test that we correctly output the domain of a counterfactual"""
    #     event = {Y @ (+X, -Z): -Y, X: -X}
    #     cf_graph = figure_9d.graph
    #     vertices: Set[Variable] = set(
    #         node for node in cf_graph.nodes() if is_not_self_intervened(node)
    #     )

    #     for cf in event:
    #         self.assertIn(cf, vertices)
    #     for value in event.values():
    #         self.assertNotIn(value, vertices)

    #     self.assertEqual(
    #         {v.get_base() for v in vertices} - set(e.get_base() for e in event),
    #         domain_of_counterfactual_values(event, vertices - set(event)),
    #     )
    #     ## TODO: add more tests

    # def test_get_district_events(self):
    #     """Ensure that each variable in the district is intervened on and there are no bad interventions"""
    #     interventions_of_districts = {
    #         frozenset([Y @ (+x, -z), X]): (W,),
    #         frozenset([W @ (+x, -z)]): (Y, X),
    #     }
    #     # self.assertEqual()

    def test_intervene_on_district(self):
        """Test that we correctly intervene on a district"""
        district1 = {X, Y @ -x}
        district2 = {Z}
        district3 = {W @ -x}
        district4 = {D}
        districts = district1 | district2 | district3 | district4
        intervention1 = districts - district1
        intervention2 = districts - district2
        intervention3 = districts - district3
        intervention4 = districts - district4

        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d} # from gamma' (Tikka 2022)
        expected_event1 = {Y @ (-x, -z, -w, -d): -y, X @ (-z,-w,-d): +x}
        expected_event2 = {Z @ (-y, -x, -w, -d): -z}
        expected_event3 = {W @ (-y, -x, -z, -d): -w}
        expected_event4 = {D @ (-y, -x, -z, -w): -d}
        self.assertEqual(expected_event1, intervene_on_district(district1, intervention1, input_event))
        self.assertEqual(expected_event2, intervene_on_district(district2, intervention2, input_event))
        self.assertEqual(expected_event3, intervene_on_district(district3, intervention3, input_event))
        self.assertEqual(expected_event4, intervene_on_district(district4, intervention4, input_event))


    def test_get_district_interventions(self):
        """Ensure that for each district, we intervene on the domain of each variable not in the district.
        Confirm that the domain of variables in the event query are restricted to their event value
        """
        # counterfactual_graph = NxMixedGraph.from_edges(
        #     undirected=[(Y @ (~X, Z), X)],
        #     directed=[
        #         (W @ (~X, Z), Y @ (~X, Z)),
        #     ],
        # )

        #event = {Y @ (+X, -Z): -Y, X: -X}
        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d} # from gamma' (Tikka 2022)
        input_graph = tikka_figure_5.graph

        expected_districts = {frozenset({X, Y @ -x}), frozenset({Z}), frozenset({W @ -x}), frozenset({D})}
        expected_summand = {W}
        expected_district_interventions = {
            frozenset({X, Y @ -x}): {Y @ (-x, -z, -w, -d): -y, X @ (-z, -w, -d): +x},
            frozenset({Z}): {Z @ (-y, -x, -w, -d): -z},
            frozenset({W @ -x}): {W @ (-y, -x, -z, -d): -w},
            frozenset({D}): {D @ (-y, -x, -z, -w): -d}}
        actual =  get_district_interventions(input_graph, input_event)
        expected_event1= expected_summand, expected_district_interventions
        self.assertEqual(expected_district_interventions, actual)

    # def test_recursion(self):
    #     """Test the recursive aspect of line 6"""
    #     cf_graph = figure_9d.graph
    #     new_event = {Y @ (+X, -Z): -Y, X: -X}
    #     summand, interventions_of_each_district = id_star_line_6(cf_graph, new_event)
    #     my_list = [
    #         {
    #             merge_interventions(element, interventions): Intervention(element.name, star=False)
    #             for element in district
    #         }
    #         for district, interventions in interventions_of_each_district.items()
    #     ]
    #     expected = [
    #         {X @ (-W, -X, -Z): -X, Y @ (-W, -X, +X, -Z): -Y},
    #         {X @ (-W, -X, +X, -Y, -Z): -X},
    #         {Z @ (-W, -X, +X, -Y, -Z): -Z},
    #         {W @ (-X, +X, -Y, -Z): -W},
    #     ]
    #     self.assertEqual(expected, my_list)
    #     ## TODO: add more tests

    # def test_merge_interventions(self):
    #     """Test that we can merge new interventions into a (potentially) counterfactual variable with existing interventions"""
    #     counterfactual = Y @ (+x, -z)
    #     interventions = (-w,)
    #     expected = Y @ (-w, +x, -z)
    #     self.assertEqual(expected, merge_interventions(counterfactual, interventions))
    #     interventions2 = (W, -d)
    #     expected2 = Y @ (-d, W, +x, -z)
    #     self.assertEqual(expected2, merge_interventions(counterfactual, interventions2))

    #     element = Y @ (X, Z)
    #     interventions = (-Z, -W, -D)
    #     expected = Y @ (D, W, X, Z)
    #     self.assertEqual(expected, merge_interventions(element, interventions))

    # def test_id_star_line_8(self):
    #     """Attempt to generate a conflict with an inconsistent value assignment."""
    #     graph = NxMixedGraph.from_edges(undirected=[(Y @ +X, X), (X, D @ -D)])
    #     self.assertEqual({-D, +X}, sub(graph))
    #     query1 = {Y @ +X: +Y, X: -X, D @ -D: -D}
    #     self.assertEqual({-X, -D, +Y}, ev(query1))
    #     self.assertTrue(id_star_line_8(graph, query1))
    #     query2 = {D @ -D: -D}
    #     self.assertEqual({-D}, ev(query2))
    #     self.assertFalse(id_star_line_8(graph, query2))
    #     graph3 = NxMixedGraph.from_edges(undirected=[(X, Y @ (-W, +X, -Z))])
    #     event3 = {Y @ (-W, +X, -Z): Y, X: X}
    #     self.assertFalse(id_star_line_8(graph3, event3))

    # def test_id_star_line_9(self):
    #     """Test line 9 of the ID* algorithm.

    #     Test that estimand returned by taking the effect of all subscripts in
    #     new_event on variables in new_event is correct
    #     """

    # def test_rule_3_applies(self):
    #     r"""Test whether rule 3 of the do calculus applies

    #     Rule 3 of the do calculus states
    #     .. math::
    #         P_{x,\color{green}{z}}(y|w) = P_x(y|w)\text{ if }(Y\ci Z|X,W)_{G_{\bar{X}\bar{Z(W)}}}

    #     So we will test whether Y is independent of X given Z (and W if it exists)
    #     with this example:

    #     Check the two C-components :math:`\{Y_{x,z}, X\}, \{W_{x,z}\}`:math:
    #     :math:`P(y_{x,z}, w_{x,z}, x' ) = P(y_{x,z,w}, x'_w )P(w_x, z )`:math,
    #     which can be simplified by removing redundant subscripts to :math:`P(y_{z,w}, x' )P(w_x )`:math:.
    #     """
    #     C1 = (Y @ (-x, +z), +X)
    #     C2 = (W @ (-x, z),)
    #     input_C1_prob = P(Y @ (-x, -z, -w), +X @ -w)
    #     input_C2_prob = P(W @ (-x, -z))
    #     expected_C1_prob = P(Y @ (-z, -w), +X)
    #     expected_C2_prob = P(W @ -x)
    #     graph = figure_9d.graph

    def test_is_redundant_counterfactual(self):
        """Test that we can detect if counterfactual variable is redundant."""
        self.assertTrue(is_redundant_counterfactual(Y @ (+x, -y), -y))
        self.assertFalse(is_redundant_counterfactual(Y @ (+x, -y), +y))
        self.assertFalse(is_redundant_counterfactual(Y @ (+x, -z), -y))
        self.assertFalse(is_redundant_counterfactual(Y @ (+x, -z), +y))


    # def test_is_self_intervened(self):
    #     """Test that we can detect when a counterfactual variable intervenes on itself"""
    #     self.assertTrue(is_self_intervened(Y @ (+x, -y)))
    #     self.assertFalse(is_self_intervened(Y @ (+x, -z)))
    #     self.assertTrue(is_self_intervened(Y @ (+x, +y)))

    # def test_id_star(self):
    #     """Test that the ID* algorithm returns the correct estimand."""
    #     query = {Y @ (+x, -z): +y, X: -x}
    #     counterfactual_graph = NxMixedGraph.from_edges(
    #         undirected=[(Y @ (~X, Z), X)],
    #         directed=[
    #             (W @ (~X, Z), Y @ (~X, Z)),
    #         ],
    #     )
    #     actual = id_star(figure_9a.graph, query)
    #     expected = Sum[W](P(Y @ (-z, W), X @ (-z, W)) * P(W @ -x))
    #     self.assert_expr_equal(expected, actual)

    # def test_idc_star(self):
    #     """Test that the IDC* algorithm returns the correct estimand."""
    #     query = P(Y @ ~X | X, Z @ D, D)
    #     vertices = set(figure_9a.graph.nodes())
    #     estimand = Sum[W](P(Y @ (-z, W), X @ (-z, W)) * P(W @ -x))
    #     expected = estimand / Sum[vertices - {X, Z @ D, D}](estimand)
    #     # actual = idc_star( figure_9a.graph, query)
    #     # self.assert_expr_equal( expected, actual )
