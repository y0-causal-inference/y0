# -*- coding: utf-8 -*-

"""Test graph construction and conversion."""

import unittest
from textwrap import dedent

from y0.graph import napkin_graph


class TestGraph(unittest.TestCase):
    """Test graph construction and conversion."""

    def test_causaleffect_str_napkin(self):
        """Test generating R code for the napkin graph for causaleffect."""
        expected = dedent('''
        g <- graph.formula(W -+ R, W -+ V1, R -+ X, X -+ Y, V1 -+ Y, W -+ X, X -+ W, W -+ Y, Y -+ W, simplify = FALSE)
        g <- set.edge.attribute(graph = g, name = "description", index = c(6, 7), value = "U")
        g <- set.edge.attribute(graph = g, name = "description", index = c(8, 9), value = "U")
        ''').strip()
        self.assertEqual(expected, napkin_graph.to_causaleffect_str())
