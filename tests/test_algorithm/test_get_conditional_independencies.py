# -*- coding: utf-8 -*-

"""Test getting conditional independencies."""

import unittest
from typing import Set

import y0.examples
from y0.algorithm.conditional_independencies import are_d_separated, get_conditional_independencies
from y0.examples import examples
from y0.graph import NxMixedGraph
from y0.struct import DSeparationJudgement


class TestDSeparation(unittest.TestCase):
    "Test the d-separation utility."

    def test_mit_example(self):
        graph = y0.examples.d_separation_example.graph.to_admg()

        self.assertFalse(are_d_separated(graph, "AA", "B", conditions=["D", "F"]))
        self.assertTrue(are_d_separated(graph, "AA", "B"))
        self.assertTrue(are_d_separated(graph, "D", "E", conditions=["C"]))
        self.assertFalse(are_d_separated(graph, "AA", "B", conditions=["C"]))
        self.assertFalse(are_d_separated(graph, "D", "E"))
        self.assertFalse(are_d_separated(graph, "D", "E", conditions=["AA", "B"]))
        self.assertFalse(are_d_separated(graph, "G", "G", conditions=["C"]))


class TestGetConditionalIndependencies(unittest.TestCase):
    """Test getting conditional independencies."""

    def assert_valid_ci_set(self, graph: NxMixedGraph, subject: Set[DSeparationJudgement]):
        graph = graph.to_admg()
        for ci in subject:
            self.assertTrue(are_d_separated(graph, ci.left, ci.right,
                                            conditions=ci.conditions),
                            f"Conditional independency is not a d-separation in {graph}")

        gist = [(ci.left, ci.right) for ci in subject]
        self.assertEqual(len(gist), len(set(gist)), "Duplicate left/right pair observed")

    def assert_conditional_independencies(
        self,
        graph: NxMixedGraph,
        expected: Set[DSeparationJudgement],
    ) -> None:
        """Assert that the graph has the correct conditional independencies.

        :param graph: the graph to test
        :param expected: the set of expexted conditional independencies
        """
        observed = get_conditional_independencies(graph.to_admg())

        self.assertIsNotNone(expected, "Expected independencies is empty.")
        self.assertIsNotNone(observed, "Observed independencies is empty.")
        self.assertTrue(
            all(
                conditional_independency.is_canonical
                for conditional_independency in observed
            ),
            msg='one or more of the returned DSeparationJudgement instances are not canonical',
        )

        self.assert_valid_ci_set(graph, expected)
        self.assert_valid_ci_set(graph, observed)

        expected_gist = {(ind.left, ind.right) for ind in expected}
        observed_gist = {(ind.left, ind.right) for ind in observed}

        # Test that the set of left & right pairs matches
        self.assertEqual(set(expected_gist), set(observed_gist),
                         "Essential independencies do not match")

    def test_examples(self):
        """Test getting the conditional independencies from the example graphs."""

        testable = [e for e in examples
                    if e.conditional_independencies is not None]

        for example in testable:
            with self.subTest(name=example.name):
                self.assert_conditional_independencies(
                    graph=example.graph,
                    expected=example.conditional_independencies,
                )
