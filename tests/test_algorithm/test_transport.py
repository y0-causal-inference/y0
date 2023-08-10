"""Unit tests for transport."""

import unittest

from tests.test_algorithm import cases
from y0.algorithm.transport import (
    TARGET_DOMAIN,
    TRANSPORT_PREFIX,
    TransportQuery,
    TRSOQuery,
    get_nodes_to_transport,
    surrogate_to_transport,
    transport,
    trso,
    trso_line1,
    trso_line2,
    trso_line3,
    trso_line4,
    trso_line6,
)
from y0.dsl import PP, Y1, Y2, Pi1, Pi2, Sum, Transport, Variable, W, X, Y, Z
from y0.graph import NxMixedGraph

X1, X2 = Variable("X1"), Variable("X2")

# Figure 8 from https://arxiv.org/abs/1806.07172
tikka_trso_figure_8 = NxMixedGraph.from_edges(
    undirected=[(X1, Y1), (Z, W), (Z, X2)],
    directed=[
        (X1, Y1),
        (X1, Y2),
        (W, Y1),
        (W, Y2),
        (Z, Y1),
        (Z, X2),
        (X2, Y2),
        (Z, Y2),
    ],
)

tikka_trso_figure_8_transport = {
    X1: [Pi1],
    Y2: [Pi1],
    X2: [Pi2],
}

transportability_diagram1 = NxMixedGraph.from_edges(
    undirected=[(X1, Y1), (Z, W), (Z, X2)],
    directed=[
        (X1, Y1),
        (X1, Y2),
        (W, Y1),
        (W, Y2),
        (Z, Y1),
        (Z, X2),
        (X2, Y2),
        (Z, Y2),
        (Variable(TRANSPORT_PREFIX + X1.name), X1),
        (Variable(TRANSPORT_PREFIX + Y2.name), Y2),
    ],
)
transportability_diagram2 = NxMixedGraph.from_edges(
    undirected=[(X1, Y1), (Z, W), (Z, X2)],
    directed=[
        (X1, Y1),
        (X1, Y2),
        (W, Y1),
        (W, Y2),
        (Z, Y1),
        (Z, X2),
        (X2, Y2),
        (Z, Y2),
        (Variable(TRANSPORT_PREFIX + X2.name), X2),
    ],
)


class TestTransport(cases.GraphTestCase):
    """Test surrogate outcomes and transportability."""

    # def test_transport_figure_8(self):
    #     """Test transportability from figure 8."""
    #     actual = transport(
    #         tikka_trso_figure_8,
    #         tikka_trso_figure_8_transport,
    #         treatments=[X1, X2],
    #         outcomes=[Y1, Y2],
    #     )
    #     # Query is also P[X1, X2](Y1, Y2)
    #     expected = ...

    #     # TODO probably need to canonicalize both of these
    #     self.assertEqual(expected, actual)

    def test_get_nodes_to_transport(self):
        expected = {X1, Y2}
        actual = get_nodes_to_transport(
            surrogate_interventions=X1, surrogate_outcomes=Y1, graph=tikka_trso_figure_8
        )
        self.assertEqual(actual, expected)
        expected = {X2}
        actual = get_nodes_to_transport(
            surrogate_interventions={X2}, surrogate_outcomes={Y2}, graph=tikka_trso_figure_8
        )
        self.assertEqual(actual, expected)

        # Test for multiple vertices in interventions and surrogate_outcomes
        expected = {X1, X2, Y1}
        actual = get_nodes_to_transport(
            surrogate_interventions={X2, X1}, surrogate_outcomes={Y2, W}, graph=tikka_trso_figure_8
        )
        self.assertEqual(actual, expected)

    def test_surrogate_to_transport(self):
        target_outcomes = {Y1, Y2}
        target_interventions = {X1, X2}
        surrogate_outcomes = {Pi1: {Y1}, Pi2: {Y2}}
        surrogate_interventions = {Pi1: {X1}, Pi2: {X2}}

        available_experiments = [({X1}, {Y1}), ({X2}, {Y2})]
        actual = surrogate_to_transport(
            target_outcomes=target_outcomes,
            target_interventions=target_interventions,
            graph=tikka_trso_figure_8,
            surrogate_outcomes=surrogate_outcomes,
            surrogate_interventions=surrogate_interventions,
        )

        transportability_diagrams = {
            TARGET_DOMAIN: tikka_trso_figure_8,
            Pi1: transportability_diagram1,
            Pi2: transportability_diagram2,
        }

        expected = TransportQuery(
            target_interventions=target_interventions,
            target_outcomes=target_outcomes,
            transportability_diagrams={
                TARGET_DOMAIN: tikka_trso_figure_8,
                Pi1: transportability_diagram1,
                Pi2: transportability_diagram2,
            },
            domains={Pi1, Pi2},
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
            target_experiments=set(),
        )
        self.assertEqual(actual, expected)

    def test_trso_line1(self):
        # triggers line 1
        outcomes = {Y1, Y2}
        interventions = {}
        active_interventions = {}
        domain_graph = tikka_trso_figure_8
        prob = PP[TARGET_DOMAIN](*list(domain_graph.nodes()))
        available_interventions = [{X2}, {X1}]

        expected = Sum.safe(prob, {W, X1, X2, Z})
        actual = trso_line1(
            outcomes,
            prob,
            domain_graph,
        )
        self.assert_expr_equal(expected, actual)

    def test_trso_line2(self):
        # triggers line 2 and then 1
        query = TRSOQuery(
            target_interventions={X1, X2, Y1, Y2},
            target_outcomes={W, Z},
            expression=PP[TARGET_DOMAIN](*list(tikka_trso_figure_8.nodes())),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            transportability_diagrams={
                TARGET_DOMAIN: tikka_trso_figure_8,
                Pi1: transportability_diagram1,
                Pi2: transportability_diagram2,
            },
            surrogate_interventions={Pi1: {X2}, Pi2: {X1}},
        )

        outcomes_anc = {W, Z}
        new_query = TRSOQuery(
            target_interventions=set(),
            target_outcomes={W, Z},
            expression=Sum.safe(query.expression, {X1, Y1, Y2, X2}),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            transportability_diagrams={
                TARGET_DOMAIN: tikka_trso_figure_8.subgraph(outcomes_anc),
                Pi1: transportability_diagram1,
                Pi2: transportability_diagram2,
            },
            surrogate_interventions={Pi1: {X2}, Pi2: {X1}},
        )

        expected = new_query
        # Sum({X1, X2, Y1, Y2}, Sum({X1, X2, Y1, Y2}, Prob))

        actual = trso_line2(
            query,
            outcomes_anc,
        )
        print(actual.expression == expected.expression)

        self.assertEqual(actual, expected)

    def test_trso_line3(self):
        
        transportability_diagram_line3 = NxMixedGraph.from_edges(
            directed=[
                (W, X),
                (X, Y),
                (X, Z),
                (Z, Y),
            ],
        )
        
        
        query = TRSOQuery(
            target_interventions={X},
            target_outcomes={Y},
            expression=PP[TARGET_DOMAIN](*list(transportability_diagram_line3.nodes())),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            transportability_diagrams={
                TARGET_DOMAIN: transportability_diagram_line3,
            },
            surrogate_interventions={},
        )
        
        
        target_interventions_overbar = transportability_diagram_line3.remove_in_edges(
            query.target_interventions
        )
        additional_interventions = (
            transportability_diagram_line3.nodes()
            - query.target_interventions
            - target_interventions_overbar.ancestors_inclusive(query.target_outcomes)
        )
        additional_interventions = {W}
        
        
        new_query = TRSOQuery(
            target_interventions={X,W},
            target_outcomes={Y},
            expression=PP[TARGET_DOMAIN](*list(transportability_diagram_line3.nodes())),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            transportability_diagrams={
                TARGET_DOMAIN: transportability_diagram_line3,
            },
            surrogate_interventions={},
        )
        expected = new_query

        # Sum({X1, X2, Y1, Y2}, Sum({X1, X2, Y1, Y2}, Prob))
        actual = trso_line3(query = query,additional_interventions=additional_interventions)
        self.assertEqual(actual, expected)

    # def test_trso_line4(self):
    #     target_outcomes = {Y1, Y2}
    #     target_interventions = {X1, X2}
    #     transportability_diagram = tikka_trso_figure_8
    #     prob = PP[TARGET_DOMAIN](*list(transportability_diagram.nodes()))
    #     active_interventions = {}
    #     available_interventions = {X1, X2}
    #     districts_without_interventions = transportability_diagram.subgraph(
    #         transportability_diagram.nodes() - target_interventions
    #     ).get_c_components()

    #     expected = {
    #         frozenset([Y2]): dict(
    #             target_outcomes={Y2},
    #             target_interventions={X1, X2, Z, W, Y1},
    #             probability=prob,
    #             active_interventions=active_interventions,
    #             domain=TARGET_DOMAIN,
    #             transportability_diagram=transportability_diagram,
    #             available_interventions=available_interventions,
    #         ),
    #         frozenset([Y1]): dict(
    #             target_outcomes={Y1},
    #             target_interventions={X1, X2, Z, W, Y2},
    #             probability=prob,
    #             active_interventions=active_interventions,
    #             domain=domain,
    #             transportability_diagram=transportability_diagram,
    #             available_interventions=available_interventions,
    #         ),
    #         frozenset([W, Z]): dict(
    #             target_outcomes={W, Z},
    #             target_interventions={X1, X2, Y2, Y1},
    #             probability=prob,
    #             active_interventions=active_interventions,
    #             domain=domain,
    #             transportability_diagram=transportability_diagram,
    #             available_interventions=available_interventions,
    #         ),
    #     }

    #     actual = trso_line4(
    #         target_outcomes,
    #         target_interventions,
    #         prob,
    #         active_interventions,
    #         TARGET_DOMAIN,
    #         transportability_diagram,
    #         available_interventions,
    #         districts_without_interventions,
    #     )
    #     self.assertEqual(expected, actual)
