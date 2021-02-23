# -*- coding: utf-8 -*-

"""Test getting conditional independencies."""

import unittest
from typing import Set
from itertools import chain

from y0.algorithm import falsification

from y0.algorithm.conditional_independencies import ConditionalIndependency, get_conditional_independencies

from y0.graph import NxMixedGraph, ADMG #, napkin_graph

class TestDSeparation(unittest.TestCase):
    "Test the d-separation utility."
    #TODO: Migrate to the ADMG representation, not just vanilla networkx
    def test_mit_example(self):
        # Test graph and cases from http://web.mit.edu/jmn/www/6.034/d-separation.pdf
        edges = [("AA","C"), ("B","C"), ("C","D"), ("C","E"), ("D","F"), ("F","G")]
        #layout = {"AA": (0,-1), "B": (2,-1), "C": (1,-2), "D": (0, -3), 
        #          "E": (2, -3), "F":(1,-4), "G": (0,-5)}
        G = ADMG()
        for v in chain(*edges): G.add_vertex(v)
        for edge in edges: G.add_diedge(*edge)
                                               
        self.assertFalse(falsification.are_d_separated(G, "AA", "B", given=["D", "F"]))
        self.assertTrue(falsification.are_d_separated(G, "AA", "B"))
        self.assertTrue(falsification.are_d_separated(G, "D", "E", given=["C"]))
        self.assertFalse(falsification.are_d_separated(G, "AA", "B", given=["C"]))
        self.assertFalse(falsification.are_d_separated(G, "D", "E"))
        self.assertFalse(falsification.are_d_separated(G, "D", "E", given=["AA", "B"]))
        self.assertFalse(falsification.are_d_separated(G, "G", "G", given=["C"]))
        
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
