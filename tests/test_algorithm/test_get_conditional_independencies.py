# -*- coding: utf-8 -*-

"""Test getting conditional independencies."""

import unittest
from typing import Set

import y0.examples
from y0.algorithm.conditional_independencies import get_conditional_independencies, are_d_separated
from y0.examples import examples
from y0.graph import NxMixedGraph
from y0.struct import ConditionalIndependency


class TestDSeparation(unittest.TestCase):
    "Test the d-separation utility."

    def test_mit_example(self):
        graph = y0.examples.d_separation_example.graph.to_admg()

        self.assertFalse(are_d_separated(graph, "AA", "B", given=["D", "F"]))
        self.assertTrue(are_d_separated(graph, "AA", "B"))
        self.assertTrue(are_d_separated(graph, "D", "E", given=["C"]))
        self.assertFalse(are_d_separated(graph, "AA", "B", given=["C"]))
        self.assertFalse(are_d_separated(graph, "D", "E"))
        self.assertFalse(are_d_separated(graph, "D", "E", given=["AA", "B"]))
        self.assertFalse(are_d_separated(graph, "G", "G", given=["C"]))


class TestGetConditionalIndependencies(unittest.TestCase):
    """Test getting conditional independencies."""

    def assert_conditional_independencies(self, graph: NxMixedGraph, expected: Set[ConditionalIndependency]):
        """Assert that the graph has the correct conditional independencies."""
        conditional_independencies = get_conditional_independencies(graph.to_admg())
        self.assertTrue(
            all(
                conditional_independency.is_canonical
                for conditional_independency in conditional_independencies
            ),
            msg='one or more of the returned ConditionalIndependency instances are not canonical',
        )
        self.assertEqual(set(expected), set(conditional_independencies))

    def test_examples(self):
        """Test getting the conditional independencies from the example graphs."""
        for example in examples:
            with self.subTest(name=example.name):
                self.assert_conditional_independencies(
                    graph=example.graph,
                    expected=example.conditional_independencies,
                )
