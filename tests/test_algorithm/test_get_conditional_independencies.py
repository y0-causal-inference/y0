# -*- coding: utf-8 -*-

"""Test getting conditional independencies."""

import unittest
from typing import Set, Union

from ananke.graphs import SG

from y0.algorithm.conditional_independencies import are_d_separated, get_conditional_independencies
from y0.examples import d_separation_example, examples
from y0.graph import NxMixedGraph
from y0.struct import DSeparationJudgement


class TestDSeparation(unittest.TestCase):
    """Test the d-separation utility."""

    def test_mit_example(self):
        """Test checking D-separation on the MIT example."""
        graph = d_separation_example.graph

        self.assertFalse(are_d_separated(graph, "AA", "B", conditions=["D", "F"]))
        self.assertTrue(are_d_separated(graph, "AA", "B"))
        self.assertTrue(are_d_separated(graph, "D", "E", conditions=["C"]))
        self.assertFalse(are_d_separated(graph, "AA", "B", conditions=["C"]))
        self.assertFalse(are_d_separated(graph, "D", "E"))
        self.assertFalse(are_d_separated(graph, "D", "E", conditions=["AA", "B"]))
        self.assertFalse(are_d_separated(graph, "G", "G", conditions=["C"]))


class TestGetConditionalIndependencies(unittest.TestCase):
    """Test getting conditional independencies."""

    def assert_valid_judgements(self, graph: Union[NxMixedGraph, SG], judgements: Set[DSeparationJudgement]) -> None:
        """Check that a set of judgments are valid with respect to a graph."""
        if isinstance(graph, NxMixedGraph):
            graph = graph.to_admg()
        for judgement in judgements:
            self.assertTrue(
                are_d_separated(
                    graph,
                    judgement.left,
                    judgement.right,
                    conditions=judgement.conditions,
                ),
                msg=f"Conditional independency is not a d-separation in {graph}",
            )

        pairs = [(judgement.left, judgement.right) for judgement in judgements]
        self.assertEqual(len(pairs), len(set(pairs)), "Duplicate left/right pair observed")

    def assert_has_judgements(self, graph: Union[NxMixedGraph, SG], judgements: Set[DSeparationJudgement]) -> None:
        """Assert that the graph has the correct conditional independencies.

        :param graph: the graph to test
        :param judgements: the set of expected conditional independencies
        """
        actual_judgements = get_conditional_independencies(graph)

        self.assertIsNotNone(judgements, "Expected independencies is empty.")
        self.assertIsNotNone(actual_judgements, "Observed independencies is empty.")
        self.assertTrue(
            all(judgement.is_canonical for judgement in actual_judgements),
            msg='one or more of the returned DSeparationJudgement instances are not canonical',
        )

        self.assert_valid_judgements(graph, judgements)
        self.assert_valid_judgements(graph, actual_judgements)

        expected_pairs = {(judgement.left, judgement.right) for judgement in judgements}
        observed_pairs = {(judgement.left, judgement.right) for judgement in actual_judgements}

        # Test that the set of left & right pairs matches
        self.assertEqual(
            set(expected_pairs),
            set(observed_pairs),
            msg='Essential independencies do not match',
        )

    def test_examples(self):
        """Test getting the conditional independencies from the example graphs."""
        testable = (
            example
            for example in examples
            if example.conditional_independencies is not None
        )

        for example in testable:
            with self.subTest(name=example.name):
                self.assert_has_judgements(
                    graph=example.graph,
                    judgements=example.conditional_independencies,
                )
