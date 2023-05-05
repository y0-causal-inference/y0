# -*- coding: utf-8 -*-

"""Tests for the ID* algorithm."""

from tests.test_algorithm import cases
from y0.algorithm.identify.id_star import (  # rule_3_applies,
    ev,
    get_district_interventions,
    get_events_of_district,
    get_events_of_each_district,
    get_free_variables,
    id_star,
    id_star_line_4,
    id_star_line_6,
    id_star_line_8,
    id_star_line_9,    
    intervene_on_district,
    is_event_empty,
    is_redundant_counterfactual,
    merge_interventions,
    remove_event_tautologies,
    simplify_counterfactual,
    sub,
    violates_axiom_of_effectiveness,
)
from y0.dsl import D, W, X, Y, Z, P, Sum
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
        district1 = frozenset({X, Y @ -x})
        district2 = frozenset({Z})
        district3 = frozenset({W @ -x})
        district4 = frozenset({D})
        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d}
        expected_event1 = {Y @ (-z, -w): -y, X @ (-z, -w): +x}
        expected_event2 = {Z @ -d: -z}
        expected_event3 = {W @ -x: -w}
        expected_event4 = {D: -d}
        expected_events = {
            district1: expected_event1,
            district2: expected_event2,
            district3: expected_event3,
            district4: expected_event4,
        }

        # expected_district_interventions = {
        #     frozenset({X, Y @ -x}): {Y @ (-x, -z, -w, -d): -y, X @ (-z, -w, -d): +x},
        #     frozenset({Z}): {Z @ (-y, -x, -w, -d): -z},
        #     frozenset({W @ -x}): {W @ (-y, -x, -z, -d): -w},
        #     frozenset({D}): {D @ (-y, -x, -z, -w): -d},
        # }
        ## Create a counterfactual graph with at least 2 c-components and return the summand and interventions of each
        #
        actual_summand, actual_district_events = id_star_line_6(input_graph, input_event)
        self.assertEqual(expected_summand, actual_summand)
        self.assertEqual(expected_events, actual_district_events)

    def test_get_free_variables(self):
        """Test that each variable not in the event or self-intervened is marginalized out."""
        input_graph = tikka_figure_5.graph
        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d}
        expected_summand = {W}
        self.assertEqual(expected_summand, get_free_variables(input_graph, input_event))
        input_event2 = {Y @ -x: -y, X: +x, Z @ -d: -z, D: -d}
        expected_summand2 = {W, Z}
        self.assertEqual(expected_summand2, get_free_variables(input_graph, input_event2))
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

    def test_get_events_of_district(self):
        """Ensure that each variable in the district is intervened on the Markov pillow."""
        input_graph = tikka_figure_5.graph
        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d}
        district1 = {X, Y @ -x}
        district2 = {Z}
        district3 = {W @ -x}
        district4 = {D}
        districts = district1 | district2 | district3 | district4
        expected_event1 = {Y @ (-z, -w): -y, X @ (-z, -w): +x}
        expected_event2 = {Z @ -d: -z}
        expected_event3 = {W @ -x: -w}
        expected_event4 = {D: -d}
        self.assertEqual(
            expected_event1, get_events_of_district(input_graph, district1, input_event)
        )
        self.assertEqual(
            expected_event2, get_events_of_district(input_graph, district2, input_event)
        )
        self.assertEqual(
            expected_event3, get_events_of_district(input_graph, district3, input_event)
        )
        self.assertEqual(
            expected_event4, get_events_of_district(input_graph, district4, input_event)
        )

    def test_events_of_each_district(self):
        """Ensure that each variable in the district is intervened on the Markov pillow."""
        input_graph = tikka_figure_5.graph
        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d}
        district1 = frozenset({X, Y @ -x})
        district2 = frozenset({Z})
        district3 = frozenset({W @ -x})
        district4 = frozenset({D})
        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d}
        expected_event1 = {Y @ (-z, -w): -y, X @ (-z, -w): +x}
        expected_event2 = {Z @ -d: -z}
        expected_event3 = {W @ -x: -w}
        expected_event4 = {D: -d}
        expected_events = {
            district1: expected_event1,
            district2: expected_event2,
            district3: expected_event3,
            district4: expected_event4,
        }
        self.assertEqual(expected_events, get_events_of_each_district(input_graph, input_event))

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

        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d}  # from gamma' (Tikka 2022)
        expected_event1 = {Y @ (-x, -z, -w, -d): -y, X @ (-z, -w, -d): +x}
        expected_event2 = {Z @ (-y, -x, -w, -d): -z}
        expected_event3 = {W @ (-y, -x, -z, -d): -w}
        expected_event4 = {D @ (-y, -x, -z, -w): -d}
        self.assertEqual(
            expected_event1, intervene_on_district(district1, intervention1, input_event)
        )
        self.assertEqual(
            expected_event2, intervene_on_district(district2, intervention2, input_event)
        )
        self.assertEqual(
            expected_event3, intervene_on_district(district3, intervention3, input_event)
        )
        self.assertEqual(
            expected_event4, intervene_on_district(district4, intervention4, input_event)
        )

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

        # event = {Y @ (+X, -Z): -Y, X: -X}
        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d}  # from gamma' (Tikka 2022)
        input_graph = tikka_figure_5.graph

        expected_districts = {
            frozenset({X, Y @ -x}),
            frozenset({Z}),
            frozenset({W @ -x}),
            frozenset({D}),
        }
        expected_summand = {W}
        expected_district_interventions = {
            frozenset({X, Y @ -x}): {Y @ (-x, -z, -w, -d): -y, X @ (-z, -w, -d): +x},
            frozenset({Z}): {Z @ (-y, -x, -w, -d): -z},
            frozenset({W @ -x}): {W @ (-y, -x, -z, -d): -w},
            frozenset({D}): {D @ (-y, -x, -z, -w): -d},
        }
        actual = get_district_interventions(input_graph, input_event)
        expected_event1 = expected_summand, expected_district_interventions
        self.assertEqual(expected_district_interventions, actual)

    # def test_recursion(self):
    #     """Test the recursive aspect of line 6"""
    #     cf_graph = figure_9d.graph
    #     new_event = {Y @ (+x, -z): -y, X: -x}
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

    def test_id_star_line_8(self):
        """Attempt to generate a conflict with an inconsistent value assignment."""
        input_graph = NxMixedGraph.from_edges(undirected=[(Y @ +x, X), (X, Z @ -d)])
        self.assertEqual({-D, +X}, sub(input_graph))
        query1 = {Y @ +x: +y, X: -x, D @ -d: -d}
        self.assertEqual({-x, -d, +y}, ev(query1))
        self.assertTrue(id_star_line_8(input_graph, query1))
        query2 = {Z @ -d: -z, X: +x}
        self.assertEqual({-z, +x}, ev(query2))
        self.assertFalse(id_star_line_8(input_graph, query2))
        graph3 = NxMixedGraph.from_edges(undirected=[(X, Y @ (-w, +x, -z))])
        event3 = {Y @ (-w, +x, -z): -y, X: -x}
        self.assertTrue(id_star_line_8(graph3, event3))
        graph4 = NxMixedGraph.from_edges(directed=[(X, Y @ -x)], undirected=[(X, Y @ -x)])
        event4 = {Y @ -x: -y, X: +x}
        self.assertTrue(id_star_line_8(graph4, event4))
        
    def test_id_star_line_9(self):
        """Test line 9 of the ID* algorithm.

        Test that estimand returned by taking the effect of all subscripts in
        new_event on variables in new_event is correct
        """
        input_graph1 = NxMixedGraph.from_edges(undirected=[(X, Y @ (-w, +x, -z))])
        expected1 = P[-w, +x, -z](X, Y)
        input_graph2 = NxMixedGraph.from_edges(undirected=[(X, Y)])
        expected2 = P(X, Y)
        self.assertEqual(expected1, id_star_line_9(input_graph1))
        self.assertEqual(expected2, id_star_line_9(input_graph2))

    #    def test_rule_3_applies(self):
    #        r"""Test whether rule 3 of the do calculus applies
    #
    #        Rule 3 of the do calculus states
    #        .. math::
    #            P_{x,\color{green}{z}}(y|w) = P_x(y|w)\text{ if }(Y\ci Z|X,W)_{G_{\bar{X}\bar{Z(W)}}}
    #
    #        So we will test whether Y is independent of X given Z (and W if it exists)
    #        with this example:
    #
    #        Check the two C-components :math:`\{Y_{x,z}, X\}, \{W_{x,z}\}`:math:
    #        :math:`P(y_{x,z}, w_{x,z}, x' ) = P(y_{x,z,w}, x'_w )P(w_x, z )`:math,
    #        which can be simplified by removing redundant subscripts to :math:`P(y_{z,w}, x' )P(w_x )`:math:.
    #        """
    #
    #        district_event_map = {
    #            frozenset({Z}): {(Z @ (-y, -x, -w, -d), -z): {Z @ -d: -z}},
    #            frozenset({W @ -x}): {(W @ (-y, -x, -z, -d), -w): {W @ -x: -w}},
    #            frozenset({D}): {(D @ (-y, -x, -z, -w), -d): {D: -d}},
    #            frozenset({X, Y @ -x}): {
    #                (Y @ (-x, -z, -w, -d), -y): {Y @ (-w, -z): -y},
    #                (X @ (-z, -w, -d), +x): {X @ (-w, -z): +x},
    #            },
    #        }
    #
    #        input_graph = tikka_figure_5.graph
    #        expected_counterfactual_map1 = {(Z @ (-y, -x, -w, -d), -z): {Z @ -d: -z}}
    #        expected_counterfactual_map2 = {(W @ (-y, -x, -z, -d), -w): {W @ -x: -w}}
    #        expected_counterfactual_map3 = {(D @ (-y, -x, -z, -w), -d): {D: -d}}
    #        expected_counterfactual_map4 = {
    #            (Y @ (-x, -z, -w, -d), -y): {Y @ (-w, -z): -y},
    #            (X @ (-z, -w, -d), +x): {X @ (-w, -z): +x},
    #        }
    #
    #        input_district_expected_counterfactual_map = {
    #            frozenset({Z}): expected_counterfactual_map1,
    #            frozenset({W @ -x}): expected_counterfactual_map2,
    #            frozenset({D}): expected_counterfactual_map3,
    #            frozenset({X, Y @ -x}): expected_counterfactual_map4,
    #        }
    #
    #        for (
    #            input_district,
    #            expected_counterfactual_map,
    #        ) in input_district_expected_counterfactual_map.items():
    #            self.assertTrue(
    #                expected_counterfactual_map, rule_3_applies(input_graph, input_district)
    #            )

    #    def test_simplify_counterfactual(self):
    #        """Test that we can simplify counterfactuals."""
    #
    #        input_graph = tikka_figure_5.graph
    #
    #        district_counterfactual_map = {
    #            frozenset({Z}): {Z @ (-y, -x, -w, -d): Z @ -d},
    #            frozenset({W @ -x}): {W @ (-y, -x, -z, -d): W @ -x},
    #            frozenset({D}): {D @ (-y, -x, -z, -w): D},
    #            frozenset({X, Y @ -x}): {
    #                Y @ (-x, -z, -w, -d): Y @ (-w, -z),
    #                X @ (-z, -w, -d): X @ (-w, -z),
    #            },
    #        }
    #        for input_district, counterfactual_map in district_counterfactual_map.items():
    #            for input_node in input_district:
    #                for input_counterfactual, expected_counterfactual in counterfactual_map.items():
    #                    self.assertEqual(
    #                        expected_counterfactual,
    #                        simplify_counterfactual(
    #                            input_graph, input_district, input_node, input_counterfactual
    #                        ),
    #                    )

    def test_is_redundant_counterfactual(self):
        """Test that we can detect if counterfactual variable is redundant."""
        self.assertTrue(is_redundant_counterfactual(Y @ (+x, -y), -y))
        self.assertFalse(is_redundant_counterfactual(Y @ (+x, -y), +y))
        self.assertFalse(is_redundant_counterfactual(Y @ (+x, -z), -y))
        self.assertFalse(is_redundant_counterfactual(Y @ (+x, -z), +y))


    def test_id_star(self):
        """Test that the ID* algorithm returns the correct estimand."""
        query = {Y @ (+x, -z): +y, X: -x}
        counterfactual_graph = NxMixedGraph.from_edges(
            undirected=[(Y @ (~X, Z), X)],
            directed=[
                (W @ (~X, Z), Y @ (~X, Z)),
            ],
        )
        actual = id_star(figure_9a.graph, query)
        expected = Sum[W](P(Y @ (-z, -w), X @ (-z, -w)) * P(W @ -x))
        self.assert_expr_equal(expected, actual)

        input_graph2 = figure_9a.graph
        input_event2 = {Y @ -x: -y, X: +x, Z @ -d: -z, D: -d}
        expected2 = Sum[W](P(Y @ (-z, -w), X @ (-z, -w)) * P(Z @ -d) * P(W @ -x) * P(D))
        self.assert_expr_equal(expected2, id_star(input_graph2, input_event2))
        
    # def test_idc_star(self):
    #     """Test that the IDC* algorithm returns the correct estimand."""
    #     query = P(Y @ ~X | X, Z @ D, D)
    #     vertices = set(figure_9a.graph.nodes())
    #     estimand = Sum[W](P(Y @ (-z, W), X @ (-z, W)) * P(W @ -x))
    #     expected = estimand / Sum[vertices - {X, Z @ D, D}](estimand)
    #     # actual = idc_star( figure_9a.graph, query)
    #     # self.assert_expr_equal( expected, actual )
