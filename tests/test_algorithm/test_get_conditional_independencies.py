# -*- coding: utf-8 -*-

"""Test getting conditional independencies."""

import unittest
from typing import Set

from y0.algorithm.conditional_independencies import ConditionalIndependency, get_conditional_independencies
from y0.graph import NxMixedGraph, napkin_graph


class TestGetConditionalIndependencies(unittest.TestCase):
    """Test getting conditional independencies."""

    def assert_conditional_indepencencies(self, graph: NxMixedGraph, expected: Set[ConditionalIndependency]):
        """Assert that the graph has the correct conditional independencies."""
        conditional_independencies = get_conditional_independencies(graph.to_admg())
        self.assertTrue(
            all(
                conditional_independency.is_canonical
                for conditional_independency in conditional_independencies
            ),
            msg='one or more of the returned ConditionalIndependency instances are not canonical',
        )
        self.assertEqual(expected, conditional_independencies)

    def test_napkin(self):
        """Test getting the conditional independencies from the napkin graph."""
        d1 = ConditionalIndependency('R', 'V1', ('W',))
        d2 = ConditionalIndependency('X', 'V1', ('W',))
        self.assert_conditional_indepencencies(napkin_graph, {d1, d2})
