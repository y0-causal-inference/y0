# -*- coding: utf-8 -*-

"""Test graph construction and conversion."""

import unittest
from textwrap import dedent

from y0.examples import verma_1


class TestGraph(unittest.TestCase):
    """Test graph construction and conversion."""

    def test_causaleffect_str_verma_1(self):
        """Test generating R code for the figure 1A graph for causaleffect."""
        expected = dedent('''
        g <- graph.formula(V1 -+ V2, V2 -+ V3, V3 -+ V4, V2 -+ V4, V4 -+ V2, simplify = FALSE)
        g <- set.edge.attribute(graph = g, name = "description", index = c(4, 5), value = "U")
        ''').strip()
        self.assertEqual(expected, verma_1.to_causaleffect_str())
