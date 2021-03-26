# -*- coding: utf-8 -*-


"""Tests for the identify algorithm."""

import unittest



from y0.dsl import Expression, P, Sum, X, Y, Z
from y0.graph import NxMixedGraph
#MixedGraph, 
from y0.algorithm.identify import identify #does not exist yet

P_XY = P(X, Y, Z)
P_XYZ = P(X, Y, Z)


class TestIdentify(unittest.TestCase):
    """Test cases from https://github.com/COVID-19-Causal-Reasoning/Y0/blob/master/ID_whittemore.ipynb."""

    def assert_identify(self, expression: Expression, graph: NxMixedGraph, query: Expression):
        """Assert that the graph returns the same."""
        self.assertEqual(expression, identify(graph, query))

    def test_figure_2a(self):
        """Test Figure 2A."""
        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Y')

        self.assert_identify(P_XY / Sum[Y](P_XY), graph, Y@X)

    def test_figure_2b(self):
        """Test Figure 2B."""
        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Y')
        graph.add_directed_edge('X', 'Z')
        graph.add_directed_edge('Z', 'Y')
        graph.add_undirected_edge('Y', 'Z')

        self.assertEqual(
            Sum[Z](Sum[Y](P_XY) / (Sum[Z](Sum[Y](P_XY))) * (P_XY / Sum[Y](P_XY))),
            identify(graph),
        )

    def test_figure_2c(self):
        """Test Figure 2C."""
        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Y')
        graph.add_directed_edge('Z', 'X')
        graph.add_directed_edge('Z', 'Y')
        graph.add_undirected_edge('Y', 'Z')

        self.assertEqual(
            Sum[Z](Sum[X, Y](P_XYZ) / (Sum[Z](Sum[X, Y](P_XYZ))) * (P_XYZ / Sum[Y](P_XYZ))),
            identify(graph),
        )

    def test_figure_2d(self):
        """Test Figure 2D."""
        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Y')
        graph.add_directed_edge('Z', 'X')
        graph.add_directed_edge('Z', 'Y')
        graph.add_undirected_edge('X', 'Z')

        self.assertEqual(
            Sum[Z](Sum[X, Y](P_XYZ) * P_XYZ / Sum[Y](P_XYZ)),
            identify(graph),
        )

    def test_figure_2e(self):
        """Test Figure 2E."""
        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Z')
        graph.add_directed_edge('Z', 'Y')
        graph.add_undirected_edge('X', 'Y')

        self.assertEqual(
            (
                Sum[Z](Sum[Y](P_XYZ) / Sum[Z](Sum[Y](P_XYZ)))
                * Sum[X](P_XYZ * Sum[Y, Z](P_XYZ) / Sum[Y](P_XYZ) / Sum[X](Sum[Y, Z](P_XYZ)))
            ),
            identify(graph),
        )
