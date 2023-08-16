# -*- coding: utf-8 -*-

"""Tests for the ID* algorithm."""

from tests.test_algorithm import cases
from y0.algorithm.identify._extras import (
    get_district_interventions,
    intervene_on_district,
)
from y0.algorithm.identify.cg import make_counterfactual_graph
from y0.algorithm.identify.id_star import (
    ConflictUnidentifiable,
    get_cf_interventions,
    get_conflicts,
    get_events_of_district,
    get_events_of_each_district,
    get_evidence,
    get_free_variables,
    id_star,
    id_star_line_6,
    id_star_line_9,
    is_redundant_counterfactual,
    remove_event_tautologies,
    violates_axiom_of_effectiveness,
)
from y0.algorithm.identify.idc_star import (
    cf_rule_2_of_do_calculus_applies,
    get_new_outcomes_and_conditions,
    get_remaining_and_missing_events,
    idc_star,
)
from y0.dsl import D, One, P, Sum, W, X, Y, Z, Zero
from y0.examples import (
    figure_9a,
    figure_9c,
    figure_9d,
    tikka_figure_2,
    tikka_figure_5,
    tikka_figure_6a,
    tikka_figure_6b,
    tikka_unidentifiable_cfgraph,
)
from y0.graph import NxMixedGraph

d, w, x, y, z = -D, -W, -X, -Y, -Z


class TestIDCStar(cases.GraphTestCase):
    """Tests for the  ``IDC*`` algorithm."""

    def test_rule_2_of_do_calculus_applies(self):
        """Test that rule 2 of do calculus applies."""
        input_graph1 = NxMixedGraph.from_edges(directed=[(X, Y), (Z, X), (Z, Y)])
        outcomes = {Y}
        condition = X
        self.assertFalse(cf_rule_2_of_do_calculus_applies(input_graph1, outcomes, condition))
        input_graph2 = NxMixedGraph.from_edges(directed=[(X, Y)])
        self.assertTrue(cf_rule_2_of_do_calculus_applies(input_graph2, outcomes, condition))
        self.assertTrue(cf_rule_2_of_do_calculus_applies(input_graph1, outcomes, Z))
        self.assertFalse(
            cf_rule_2_of_do_calculus_applies(tikka_figure_2.graph, outcomes, condition)
        )
        self.assertTrue(cf_rule_2_of_do_calculus_applies(tikka_figure_5.graph, {Y @ -x}, D))
        self.assertFalse(cf_rule_2_of_do_calculus_applies(tikka_figure_6a.graph, {Y}, Z))
        self.assertFalse(cf_rule_2_of_do_calculus_applies(tikka_figure_6a.graph, {Y}, X))
        self.assertFalse(cf_rule_2_of_do_calculus_applies(tikka_figure_6a.graph, {Y @ -x}, X))
        self.assertFalse(cf_rule_2_of_do_calculus_applies(tikka_figure_6b.graph, {Y}, Z))
        self.assertFalse(cf_rule_2_of_do_calculus_applies(tikka_figure_6b.graph, {Y}, X))
        self.assertTrue(cf_rule_2_of_do_calculus_applies(tikka_figure_6a.graph, {Y @ -x}, Z @ -x))


class TestIDStar(cases.GraphTestCase):
    """Tests for the ``ID*`` algorithm."""

    def test_id_star_line_1(self):
        """Check that one is returned when running ID* on an empty event."""
        self.assertEqual(One(), id_star(figure_9a.graph, {}))

    def test_id_star_line_2(self):
        """Check to see if the counterfactual event violates the Axiom of Effectiveness."""
        # Examples all from figure_9a.graph
        self.assertTrue(violates_axiom_of_effectiveness({X @ -x: +x}))
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
        self.assertFalse(violates_axiom_of_effectiveness({Z @ -x: -z, X: +x}))
        self.assertEqual(Zero(), id_star(figure_9a.graph, {X @ x: ~x}))

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
        new_graph, new_event = make_counterfactual_graph(
            graph=figure_9a.graph, event={Y @ -x: -y, X: +x, Z @ -d: -z, D: -d}
        )
        self.assert_graph_equal(figure_9c.graph, new_graph)
        self.assertEqual({Y @ x: y, X: +x, Z: -z, D: -d}, new_event)

        actual_graph3, actual_event3 = make_counterfactual_graph(
            graph=NxMixedGraph.from_edges(directed=[(D, Z), (Z, Y)]),
            event={Z @ -d: -z, Z: +z, D: -d},
        )

        # created a test case where id_star_line_4 returns None
        self.assertIsNone(actual_event3)

        event_4 = {Y @ (-x, -z): +y, X: +x}

        expected_graph = figure_9d.graph
        expected = expected_graph, event_4
        self.assertEqual(expected, make_counterfactual_graph(figure_9a.graph, event_4))
        self.assertFalse(figure_9d.graph.is_connected())

    def test_id_star_line_6(self):
        """For each district, intervene only on the Markov pillow of each district.

        This is the key difference between the original ID* and our ID*.
        Original ID* intervenes on all variables not in the district, but this creates problems with identifiability.
        The identifiability issues are resolved by intervening only on the Markov pillow of each district.
        """
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
        expected_markov_pillow_events = {
            district1: expected_event1,
            district2: expected_event2,
            district3: expected_event3,
            district4: expected_event4,
        }

        expected_district_interventions = {
            frozenset({X, Y @ -x}): {Y @ (-x, -z, -w, -d): -y, X @ (-z, -w, -d): +x},
            frozenset({Z}): {Z @ (-y, -x, -w, -d): -z},
            frozenset({W @ -x}): {W @ (-y, -x, -z, -d): -w},
            frozenset({D}): {D @ (-y, -x, -z, -w): -d},
        }
        # Create a counterfactual graph with at least 2 c-components and return the summand and interventions of each

        actual_summand, actual_district_events = id_star_line_6(input_graph, input_event)
        self.assertEqual(expected_summand, actual_summand)

        # Original ID* intervenes on all variables not in the district, but this creates problems with identifiability
        self.assertNotEqual(expected_district_interventions, actual_district_events)

        # The identifiability issues are resolved by intervening only on the Markov pillow.
        self.assertEqual(expected_markov_pillow_events, actual_district_events)

        # input_graph2 = figure_9c.graph
        # input_event = {Y @ ~x: ~y, X: -x, Z @ -d: -z, D: -d}

    def test_get_free_variables(self):
        """Test that each variable not in the event or self-intervened is marginalized out."""
        input_graph = tikka_figure_5.graph
        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d}
        expected_summand = {W}
        self.assertEqual(expected_summand, get_free_variables(input_graph, input_event))
        input_event2 = {Y @ -x: -y, X: +x, Z @ -d: -z, D: -d}
        expected_summand2 = {W}
        self.assertEqual(expected_summand2, get_free_variables(input_graph, input_event2))

    def test_get_conflicts(self):
        """Test conflicts between interventions in the graph and value assignments in the event."""
        input_graph = NxMixedGraph.from_edges(
            directed=[(X @ +x, Y @ +x), (X @ -x, Y @ -x)], undirected=[(Y @ +x, Y @ -x)]
        )
        event = {Y @ -x: -y, Y @ +x: +y}
        self.assertEqual([(-x, +x), (+x, -x)], sorted(get_conflicts(input_graph, event)))

    def test_get_evidence(self):
        """Test that the interventions and values of a event."""
        self.assertEqual({-x, -w, -d, -y}, get_evidence({Y @ -x: -y, W @ -d: -w}))
        self.assertEqual({-x, -w, -d, -y}, get_evidence({Y @ -x: -y, X: -x, W @ -d: -w}))

    def test_get_cf_interventions(self):
        """Test we extract all interventions from the CF variables and only the counterfactual variables."""
        self.assertEqual(set(), get_cf_interventions(NxMixedGraph.from_edges(directed=[(X, Y)])))
        self.assertEqual(
            {-x, -d, -w},
            get_cf_interventions(NxMixedGraph.from_edges(directed=[(X, Y @ (-x, -d, -w))])),
        )
        self.assertEqual(
            {-x, -d, -w},
            get_cf_interventions(
                NxMixedGraph.from_edges(directed=[(X @ (-d, -w), Y @ (-x, -d, -w))])
            ),
        )
        self.assertEqual(
            {-x, -d, -w},
            get_cf_interventions(NxMixedGraph.from_edges(directed=[(X @ (-d, -w), Y @ -x)])),
        )

    def test_get_events_of_district(self):
        """Ensure that each variable in the district is intervened on the Markov pillow."""
        input_graph = tikka_figure_5.graph
        input_event = {Y @ -x: -y, X: +x, Z: -z, D: -d}
        district1 = {X, Y @ -x}
        district2 = {Z}
        district3 = {W @ -x}
        district4 = {D}
        # districts = district1 | district2 | district3 | district4
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
        input_event5 = {Y @ (-D, -X): -Y, X: +X, Z: -Z}
        graph5 = NxMixedGraph.from_edges(
            nodes=[Y @ (-D, -X), X, Z @ (-D, -X), Z, W @ (-D, -X), D @ (-D, -X), D, X @ (-D, -X)],
            directed=[
                (Z @ (-D, -X), Y @ (-D, -X)),
                (W @ (-D, -X), Y @ (-D, -X)),
                (D @ (-D, -X), Z @ (-D, -X)),
                (D, Z),
                (X @ (-D, -X), W @ (-D, -X)),
            ],
            undirected=[(Y @ (-D, -X), X), (Z @ (-D, -X), Z)],
        )
        district5 = {Z, Z @ (-D, -X)}
        expected_event5 = {Z @ -D: -z}
        self.assertEqual(expected_event5, get_events_of_district(graph5, district5, input_event5))
        input_graph_unidentifiable = tikka_unidentifiable_cfgraph.graph
        self.assertEqual(
            expected_event1,
            get_events_of_district(input_graph_unidentifiable, district1, input_event),
        )

    def test_get_markov_pillow(self):
        """Test that we correctly compute the Markov pillow of the district."""
        district = {Z, Z @ (-d, -x)}
        graph = NxMixedGraph.from_edges(
            nodes=[Y @ (-D, -X), X, Z @ (-D, -X), Z, W @ (-D, -X), D @ (-D, -X), D, X @ (-D, -X)],
            directed=[
                (Z @ (-D, -X), Y @ (-D, -X)),
                (W @ (-D, -X), Y @ (-D, -X)),
                (D @ (-D, -X), Z @ (-D, -X)),
                (D, Z),
                (X @ (-D, -X), W @ (-D, -X)),
            ],
            undirected=[(Y @ (-D, -X), X), (Z @ (-D, -X), Z)],
        )
        expected_pillow = {D, D @ (-d, -x)}
        self.assertEqual(expected_pillow, graph.get_markov_pillow(district))
        graph2 = tikka_unidentifiable_cfgraph.graph
        district2 = {X, Y @ (-x)}
        expected_pillow2 = {W @ -x, Z}
        self.assertEqual(expected_pillow2, graph2.get_markov_pillow(district2))

    def test_events_of_each_district(self):
        """Ensure that each variable in the district is intervened on the Markov pillow."""
        input_graph = tikka_figure_5.graph
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
        """Test that we correctly intervene on a district."""
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

        # expected_summand = {W}
        expected_district_interventions = {
            frozenset({X, Y @ -x}): {Y @ (-x, -z, -w, -d): -y, X @ (-z, -w, -d): +x},
            frozenset({Z}): {Z @ (-y, -x, -w, -d): -z},
            frozenset({W @ -x}): {W @ (-y, -x, -z, -d): -w},
            frozenset({D}): {D @ (-y, -x, -z, -w): -d},
        }
        actual = get_district_interventions(input_graph, input_event)
        self.assertEqual(expected_district_interventions, actual)

    def test_id_star_line_8(self):
        """Attempt to generate a conflict with an inconsistent value assignment."""
        input_graph = NxMixedGraph.from_edges(
            directed=[(X @ +x, Y @ +x), (X @ -x, Y @ -x)], undirected=[(Y @ +x, Y @ -x)]
        )
        event = {Y @ -x: -y, Y @ +x: +y}
        self.assertTrue(get_conflicts(input_graph, event))

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

    def test_is_redundant_counterfactual(self):
        """Test that we can detect if counterfactual variable is redundant."""
        self.assertTrue(is_redundant_counterfactual(Y @ (+x, -y), -y))
        self.assertFalse(is_redundant_counterfactual(Y @ (+x, -y), +y))
        self.assertFalse(is_redundant_counterfactual(Y @ (+x, -z), -y))
        self.assertFalse(is_redundant_counterfactual(Y @ (+x, -z), +y))

    def test_get_remaining_and_missing_events(self):
        """Test that we can extract the missing and remaining events from a new event."""
        old_event = {Y @ -X: -Y, X: +X, Z @ -D: -Z, D: -D}
        new_event = {Y @ -X: -Y, X: +X, D: -D, Z: -Z}
        remaining_events, missing_events = get_remaining_and_missing_events(new_event, old_event)
        expected_remaining = {Y @ -x: -y, X: +x, D: -d}
        expected_missing = {Z @ -d: -z}
        self.assertEqual(expected_remaining, remaining_events)
        self.assertEqual(expected_missing, missing_events)

    def test_get_new_outcomes_and_conditions(self):
        """Test that we can recover all cases of outcomes and conditions from a new event."""
        old_outcomes = {Y @ -x: -y}
        old_conditions = {X: +x, Z @ -d: -z, D: -d}
        new_event = {Y @ -X: -Y, X: +X, D: -D, Z: -Z}
        new_outcomes, new_conditions = get_new_outcomes_and_conditions(
            new_event, old_outcomes, old_conditions
        )
        expected_new_outcomes = {Y @ -x: -y}
        expected_new_conditions = {X: +x, D: -d, Z: -z}
        self.assertEqual(expected_new_outcomes, new_outcomes)
        self.assertEqual(expected_new_conditions, new_conditions)
        new_event2 = {Y: -y, X: +x, D: -d, Z: -z}
        new_outcomes2, new_conditions2 = get_new_outcomes_and_conditions(
            new_event2, old_outcomes, old_conditions
        )
        expected_new_outcomes2 = {Y: -y}
        expected_new_conditions2 = {X: +x, D: -d, Z: -z}
        self.assertEqual(expected_new_outcomes2, new_outcomes2)
        self.assertEqual(expected_new_conditions2, new_conditions2)
        new_event3 = {Y: -y, X: +x, D: -d, Z @ -d: -z}
        new_outcomes3, new_conditions3 = get_new_outcomes_and_conditions(
            new_event3, old_outcomes, old_conditions
        )
        expected_new_outcomes3 = {Y: -y}
        expected_new_conditions3 = {X: +x, D: -d, Z @ -d: -z}
        self.assertEqual(expected_new_outcomes3, new_outcomes3)
        self.assertEqual(expected_new_conditions3, new_conditions3)
        new_event4 = {Y @ -x: -y, X: +x, D: -d, Z @ -d: -z}
        new_outcomes4, new_conditions4 = get_new_outcomes_and_conditions(
            new_event4, old_outcomes, old_conditions
        )
        expected_new_outcomes4 = {Y @ -x: -y}
        expected_new_conditions4 = {X: +x, D: -d, Z @ -d: -z}
        self.assertEqual(expected_new_outcomes4, new_outcomes4)
        self.assertEqual(expected_new_conditions4, new_conditions4)

        expected_new_outcomes5 = {Y: -y}
        expected_new_conditions5 = {Y: -y, Y @ -z: +y}
        new_event5 = {Y: -y, Y @ -z: +y}
        new_outcomes5, new_conditions5 = get_new_outcomes_and_conditions(
            new_event5, outcomes={Y @ -w: -y}, conditions={Y @ -x: -y, Y @ -z: +y}
        )
        self.assertEqual(expected_new_outcomes5, new_outcomes5)
        self.assertEqual(expected_new_conditions5, new_conditions5)

    def test_id_star(self):
        """Test that the ID* algorithm returns the correct estimand."""
        query = {Y @ (+x, -z): +y, X: -x}
        actual = id_star(figure_9a.graph, query)
        expected = Sum[W](P(Y @ (-z, -w), X @ (-z, -w)) * P(W @ -x))
        self.assert_expr_equal(expected, actual)

        input_graph2 = figure_9a.graph
        input_event2 = {Y @ -x: -y, X: +x, Z @ -d: -z, D: -d}
        expected2 = Sum[W](P(Y @ (-z, -w), X @ (-z, -w)) * P(Z @ -d) * P(W @ -x) * P(D))
        self.assert_expr_equal(expected2, id_star(input_graph2, input_event2))

        input_graph_line3 = input_graph2
        input_event_line3 = {D @ +d: +d}
        self.assert_expr_equal(One(), id_star(input_graph_line3, input_event_line3))

        input_graph_line5 = NxMixedGraph.from_edges(directed=[(D, Z), (Z, Y)])
        input_event_line5 = {Z @ -d: -z, Z: +z, D: -d}
        self.assert_expr_equal(Zero(), id_star(input_graph_line5, input_event_line5))

        input_graph_line8 = NxMixedGraph.from_edges(directed=[(X, Y)])
        with self.assertRaises(ConflictUnidentifiable):
            id_star(input_graph_line8, {Y @ -x: -y, Y @ +x: +y})

    def test_idc_star(self):
        """Test that the IDC* algorithm returns the correct estimand."""
        input_graph_line_1 = NxMixedGraph.from_edges(directed=[(D, Z), (Z, Y)])
        input_condition_line_1 = {Z @ -d: -z, Z: +z, D: -d}
        input_outcome_line_1 = {Y @ -d: -y}
        with self.assertRaises(ValueError):
            idc_star(input_graph_line_1, input_outcome_line_1, input_condition_line_1)

        input_graph_line3 = NxMixedGraph.from_edges(directed=[(D, Z), (Z, Y)])
        input_outcome_line3 = {Z @ -d: -z}
        input_condition_line3 = {Z: +z, D: -d}
        self.assert_expr_equal(
            Zero(), idc_star(input_graph_line3, input_outcome_line3, input_condition_line3)
        )

        input_graph_figure2 = tikka_figure_2.graph
        input_outcome_figure2 = {Y @ -x: -y}
        input_conditional_figure2 = {Z @ -x: -z, X: +x}
        expected_estimand_figure2 = P(Y @ (-x, -z))
        self.assert_expr_equal(
            expected_estimand_figure2,
            idc_star(input_graph_figure2, input_outcome_figure2, input_conditional_figure2),
        )

        input_graph_figure9a = figure_9a.graph
        input_outcome_figure9a = {Y @ -x: -y}
        input_conditional_figure9a = {X: +x, Z @ -d: -z, D: -d}
        id_star_estimand = Sum[W](P[-z, -w](-y, +x) * P[-x](-w))
        idc_star_estimand = id_star_estimand / Sum[W, Y, Z](P[-z, -w](-y, +x) * P[-x](-w))
        self.assert_expr_equal(
            idc_star_estimand, id_star_estimand / Sum[W, Y, Z](P[-z, -w](-y, +x) * P[-x](-w))
        )
        expected_estimand_figure_9a = Sum[D, W](
            P(Z @ -D) * P(X @ (-W, -Z), Y @ (-W, -Z)) * P(W @ -X) * P(D)
        ) / Sum[D, W, Y](Sum[D, W](P(Z @ -D) * P(X @ (-W, -Z), Y @ (-W, -Z)) * P(W @ -X) * P(D)))

        self.assert_expr_equal(
            expected_estimand_figure_9a,
            idc_star(input_graph_figure9a, input_outcome_figure9a, input_conditional_figure9a),
        )
