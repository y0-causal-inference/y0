# -*- coding: utf-8 -*-

"""Tests for the identify algorithm."""

import unittest
from typing import Union

from y0.algorithm.identify import Identification, Unidentifiable, identify
from y0.dsl import Distribution, P, Probability, X, Y
from y0.graph import NxMixedGraph


class TestNotIdentifiable(unittest.TestCase):
    """Tests for lack of identifiability.

    These tests are based on the examples from the Figure 1 series on
    https://github.com/COVID-19-Causal-Reasoning/Y0/blob/master/ID_whittemore.ipynb.
    """

    def assert_not_identifiable(
        self, graph: NxMixedGraph, query: Union[Probability, Distribution]
    ) -> None:
        """Asset the graph is not identifiable under the given query."""
        with self.assertRaises(Unidentifiable):
            identify(Identification.from_expression(graph=graph, query=query))

    def test_figure_1a(self):
        """Test Figure 1A."""
        graph_1a = NxMixedGraph()
        graph_1a.add_directed_edge("X", "Y")
        graph_1a.add_undirected_edge("X", "Y")
        self.assert_not_identifiable(graph_1a, P(Y @ ~X))

    def test_figure_1b(self):
        """Test Figure 1B."""
        graph_1b = NxMixedGraph()
        graph_1b.add_directed_edge("X", "Z")
        graph_1b.add_directed_edge("Z", "Y")
        graph_1b.add_undirected_edge("X", "Z")
        self.assert_not_identifiable(graph_1b, P(Y @ ~X))

    def test_figure_1c(self):
        """Test Figure 1c."""
        graph_1c = NxMixedGraph()
        graph_1c.add_directed_edge("X", "Z")
        graph_1c.add_directed_edge("Z", "Y")
        graph_1c.add_directed_edge("X", "Y")
        graph_1c.add_undirected_edge("X", "Z")
        self.assert_not_identifiable(graph_1c, P(Y @ ~X))

    def test_figure_1d(self):
        """Test Figure 1d."""
        graph_1d = NxMixedGraph()
        graph_1d.add_directed_edge("X", "Y")
        graph_1d.add_directed_edge("Z", "Y")
        graph_1d.add_undirected_edge("X", "Z")
        graph_1d.add_undirected_edge("Z", "Y")
        self.assert_not_identifiable(graph_1d, P(Y @ ~X))

    def test_figure_1e(self):
        """Test Figure 1e."""
        graph_1e = NxMixedGraph()
        graph_1e.add_directed_edge("Z", "X")
        graph_1e.add_directed_edge("X", "Y")
        graph_1e.add_undirected_edge("X", "Z")
        graph_1e.add_undirected_edge("Z", "Y")
        self.assert_not_identifiable(graph_1e, P(Y @ ~X))

    def test_figure_1f(self):
        """Test Figure 1f."""
        graph_1f = NxMixedGraph()
        graph_1f.add_directed_edge("X", "Z")
        graph_1f.add_directed_edge("Z", "Y")
        graph_1f.add_undirected_edge("X", "Y")
        graph_1f.add_undirected_edge("Z", "Y")
        self.assert_not_identifiable(graph_1f, P(Y @ ~X))

    def test_figure_1g(self):
        """Test Figure 1g."""
        graph_1g = NxMixedGraph()
        graph_1g.add_directed_edge("X", "Z1")
        graph_1g.add_directed_edge("Z1", "Y")
        graph_1g.add_directed_edge("Z2", "Y")
        graph_1g.add_undirected_edge("X", "Z2")
        graph_1g.add_undirected_edge("Z1", "Z2")
        self.assert_not_identifiable(graph_1g, P(Y @ ~X))

    def test_figure_1h(self):
        """Test Figure 1h."""
        graph_1h = NxMixedGraph()
        graph_1h.add_directed_edge("Z", "X")
        graph_1h.add_directed_edge("X", "W")
        graph_1h.add_directed_edge("W", "Y")
        graph_1h.add_undirected_edge("X", "Z")
        graph_1h.add_undirected_edge("X", "Y")
        graph_1h.add_undirected_edge("W", "Z")
        graph_1h.add_undirected_edge("Y", "Z")
        self.assert_not_identifiable(graph_1h, P(Y @ ~X))


class TestIdentifiable(unittest.TestCase):
    """Tests for lack of identifiability.

    These tests are based on the examples from the Figure 2 series on
    https://github.com/COVID-19-Causal-Reasoning/Y0/blob/master/ID_whittemore.ipynb.
    """

    def assert_identifiable(
        self, graph: NxMixedGraph, query: Union[Probability, Distribution]
    ) -> None:
        """Assert the graph is identifiable under the given query."""
        estimand = identify(Identification.from_expression(graph=graph, query=query))
        self.assertIsNotNone(estimand)

    def test_figure_2a(self):
        """Test Figure 2a."""
        graph_2a = NxMixedGraph()
        graph_2a.add_directed_edge("X", "Y")
        self.assert_identifiable(graph_2a, P(Y @ ~X))

    def test_figure_2b(self):
        """Test Figure 2B."""
        graph_2b = NxMixedGraph()
        graph_2b.add_directed_edge("X", "Y")
        graph_2b.add_directed_edge("X", "Z")
        graph_2b.add_directed_edge("Z", "Y")
        graph_2b.add_undirected_edge("Y", "Z")
        self.assert_identifiable(graph_2b, P(Y @ ~X))

    def test_figure_2c(self):
        """Test Figure 2C."""
        graph_2c = NxMixedGraph()
        graph_2c.add_directed_edge("X", "Y")
        graph_2c.add_directed_edge("Z", "X")
        graph_2c.add_directed_edge("Z", "Y")
        graph_2c.add_undirected_edge("Y", "Z")
        self.assert_identifiable(graph_2c, P(Y @ ~X))

    def test_figure_2d(self):
        """Test Figure 2D."""
        graph_2d = NxMixedGraph()
        graph_2d.add_directed_edge("X", "Y")
        graph_2d.add_directed_edge("Z", "X")
        graph_2d.add_directed_edge("Z", "Y")
        graph_2d.add_undirected_edge("X", "Z")
        self.assert_identifiable(graph_2d, P(Y @ ~X))

    def test_figure_2e(self):
        """Test Figure 2E."""
        graph_2e = NxMixedGraph()
        graph_2e.add_directed_edge("X", "Z")
        graph_2e.add_directed_edge("Z", "Y")
        graph_2e.add_undirected_edge("X", "Y")
        self.assert_identifiable(graph_2e, P(Y @ ~X))

    def test_figure_2f(self):
        """Test Figure 2f."""
        graph_2f = NxMixedGraph()
        graph_2f.add_directed_edge("X", "Y")
        graph_2f.add_directed_edge("X", "Z1")
        graph_2f.add_directed_edge("Z1", "Y")
        graph_2f.add_directed_edge("Z1", "Z2")
        graph_2f.add_directed_edge("Z2", "Y")
        graph_2f.add_undirected_edge("X", "Z2")
        graph_2f.add_undirected_edge("Y", "Z1")
        self.assert_identifiable(graph_2f, P(Y @ ~X))

    def test_figure_2g(self):
        """Test Figure 2g."""
        graph_2g = NxMixedGraph()
        graph_2g.add_directed_edge("Z2", "Z1")
        graph_2g.add_directed_edge("Z2", "X")
        graph_2g.add_directed_edge("Z2", "Z3")
        graph_2g.add_directed_edge("X", "Z1")
        graph_2g.add_directed_edge("Z1", "Y")
        graph_2g.add_directed_edge("Z3", "Y")
        graph_2g.add_undirected_edge("Z2", "X")
        graph_2g.add_undirected_edge("Z2", "Y")
        graph_2g.add_undirected_edge("X", "Z3")
        graph_2g.add_undirected_edge("X", "Y")
        self.assert_identifiable(graph_2g, P(Y @ ~X))
