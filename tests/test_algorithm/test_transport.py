"""Unit tests for transport."""

import unittest

from y0.algorithm.transport import (
    add_transportability_nodes,
    find_transport_vertices,
    transport,
    trso,
)
from y0.dsl import PP, Y1, Y2, Pi1, Pi2, Sum, Variable, W, Z
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


class TestTransport(unittest.TestCase):
    """Test surrogate outcomes and transportability."""

    def test_transport_figure_8(self):
        """Test transportability from figure 8."""
        actual = transport(
            tikka_trso_figure_8,
            tikka_trso_figure_8_transport,
            treatments=[X1, X2],
            outcomes=[Y1, Y2],
        )
        # Query is also P[X1, X2](Y1, Y2)
        expected = ...

        # TODO probably need to canonicalize both of these
        self.assertEqual(expected, actual)

    def test_find_transport_vertices(self):
        expected = {X1, Y2}
        actual = find_transport_vertices([X1], [Y1], tikka_trso_figure_8)
        self.assertEqual(actual, expected)
        expected = {X2}
        actual = find_transport_vertices([X2], [Y2], tikka_trso_figure_8)
        self.assertEqual(actual, expected)

        # Test for multiple vertices in interventions and surrogate_outcomes
        expected = {X1, X2, Y1}
        actual = find_transport_vertices([X2, X1], [Y2, W], tikka_trso_figure_8)
        self.assertEqual(actual, expected)

    def test_add_transportability_nodes(self):
        add_transportability_nodes([X1], [Y1], tikka_trso_figure_8)


    def test_trso(self):
        #triggers line 1
        outcomes = {Y1,Y2}
        interventions = {}
        Prob = Variable("Prob")
        active_experiments = {}
        domain = Variable("pi*")
        domain_graph = tikka_trso_figure_8
        available_experiment_interventions = [{X2},{X1}]

        expected = Sum({W,X1,X2,Z},Prob)
        actual = trso(outcomes,interventions,Prob,active_experiments,domain,domain_graph,available_experiment_interventions)
        self.assertEqual(actual, expected)

        #triggers line 2 and then 1
        outcomes = {W,Z}
        interventions = {X1,X2,Y1,Y2}
        expected = Sum({X1,X2,Y1,Y2},Sum({X1,X2,Y1,Y2},Prob))
        actual = trso(outcomes,interventions,Prob,active_experiments,domain,domain_graph,available_experiment_interventions)
        self.assertEqual(actual, expected)
        
        #triggers line 4 and then Raises(NotImplementedError)
        outcomes = {Y2}
        interventions = {X1,X2,W,Z,Y1}
        with self.assertRaises(NotImplementedError):
            trso(outcomes,interventions,Prob,active_experiments,domain,domain_graph,available_experiment_interventions)

        

