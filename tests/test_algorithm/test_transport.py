"""Unit tests for transport."""

import unittest

from tests.test_algorithm import cases
from y0.algorithm.transport import (
    add_transportability_nodes,
    find_transport_vertices,
    transport,
    trso,
    trso_line1,
    trso_line2,
    trso_line3,
    trso_line4,
    trso_line6,
)
from y0.dsl import PP, Y1, Y2, Pi1, Pi2, Sum, Variable, W, X, Y, Z
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

    def test_find_transport_vertices(self):
        expected = {X1, Y2}
        actual = find_transport_vertices(X1, Y1, tikka_trso_figure_8)
        self.assertEqual(actual, expected)
        expected = {X2}
        actual = find_transport_vertices({X2}, {Y2}, tikka_trso_figure_8)
        self.assertEqual(actual, expected)

        # Test for multiple vertices in interventions and surrogate_outcomes
        expected = {X1, X2, Y1}
        actual = find_transport_vertices({X2, X1}, {Y2, W}, tikka_trso_figure_8)
        self.assertEqual(actual, expected)

    # def test_add_transportability_nodes(self):
    #     add_transportability_nodes([X1], [Y1], tikka_trso_figure_8)

    def test_trso_line1(self):
        # triggers line 1
        outcomes = {Y1, Y2}
        interventions = {}
        active_interventions = {}
        domain = Variable("pi*")
        domain_graph = tikka_trso_figure_8
        prob = PP[domain](*list(domain_graph.nodes()))
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

        active_interventions = {}
        domain = Variable("pi*")
        domain_graph = tikka_trso_figure_8
        prob = PP[domain](*list(domain_graph.nodes()))
        available_interventions = [{X2}, {X1}]

        outcomes = {W, Z}
        outcomes_anc = {W, Z}
        interventions = {X1, X2, Y1, Y2}
        expected = dict(
            target_outcomes=outcomes,
            target_interventions=interventions.intersection(outcomes_anc),
            probability=Sum.safe(prob, domain_graph.nodes() - outcomes_anc),
            active_interventions=active_interventions,
            domain=domain,
            transportability_diagram=domain_graph.subgraph(outcomes_anc),
            available_interventions=available_interventions,
        )

        # Sum({X1, X2, Y1, Y2}, Sum({X1, X2, Y1, Y2}, Prob))
        actual = trso_line2(
            outcomes,
            interventions,
            prob,
            active_interventions,
            domain,
            domain_graph,
            available_interventions,
            outcomes_anc,
        )
        self.assertEqual(actual, expected)

    def test_trso_line3(self):
        # triggers line 2 and then 1

        transportability_diagram = NxMixedGraph.from_edges(
            directed=[
                (W, X),
                (X, Y),
                (X, Z),
                (Z, Y),
            ],
        )
        target_interventions = {X}
        target_outcomes = {Y}
        active_interventions = {}
        available_interventions = {X}
        domain = Variable("pi*")
        prob = PP[domain](*list(transportability_diagram.nodes()))
        target_interventions_overbar = transportability_diagram.remove_in_edges(
            target_interventions
        )
        additional_interventions = (
            transportability_diagram.nodes()
            - target_interventions
            - target_interventions_overbar.ancestors_inclusive(target_outcomes)
        )

        expected = dict(
            target_outcomes=target_outcomes,
            target_interventions=target_interventions.union(additional_interventions),
            probability=prob,
            active_interventions=active_interventions,
            domain=domain,
            transportability_diagram=transportability_diagram,
            available_interventions=available_interventions,
        )

        # Sum({X1, X2, Y1, Y2}, Sum({X1, X2, Y1, Y2}, Prob))
        actual = trso_line3(
            target_outcomes,
            target_interventions,
            prob,
            active_interventions,
            domain,
            transportability_diagram,
            available_interventions,
            additional_interventions,
        )
        self.assertEqual(actual, expected)

        # def test_trso_line4(self):

        # # triggers line 4 and then Raises(NotImplementedError)
        # outcomes = {Y2}
        # interventions = {X1, X2, W, Z, Y1}
        # with self.assertRaises(NotImplementedError):
        #     trso(
        #         outcomes,
        #         interventions,
        #         Prob,
        #         active_interventions,
        #         domain,
        #         domain_graph,
        #         available_interventions,
        #     )
