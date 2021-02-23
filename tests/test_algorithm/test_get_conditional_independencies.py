# -*- coding: utf-8 -*-

"""Test getting conditional independencies."""

import unittest
from typing import Set

import y0.examples
from y0.algorithm import falsification
from y0.algorithm.conditional_independencies import ConditionalIndependency, get_conditional_independencies
from y0.examples import examples
from y0.graph import NxMixedGraph


class TestDSeparation(unittest.TestCase):
    "Test the d-separation utility."

    def test_mit_example(self):
        G = y0.examples.d_separation_example.graph.to_admg()

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
        self.assertEqual(set(expected), set(conditional_independencies))

    def test_examples(self):
        """Test getting the conditional independencies from the example graphs."""
        for example in examples:
            with self.subTest(name=example.name):
                self.assert_conditional_indepencencies(
                    graph=example.graph,
                    expected=example.conditional_independencies,
                )
