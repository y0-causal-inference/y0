# -*- coding: utf-8 -*-

"""Test getting conditional independencies."""

import unittest
from typing import Iterable, Set, Union

from ananke.graphs import SG

from y0.algorithm.conditional_independencies import are_d_separated, get_conditional_independencies
from y0.examples import Example, d_separation_example, examples
from y0.graph import NxMixedGraph
from y0.struct import DSeparationJudgement


class TestDSeparation(unittest.TestCase):
    """Test the d-separation utility."""

    def test_mit_example(self):
        """Test checking D-separation on the MIT example."""
        graph = d_separation_example.graph.to_admg()

        self.assertFalse(are_d_separated(graph, "AA", "B", conditions=["D", "F"]))
        self.assertTrue(are_d_separated(graph, "AA", "B"))
        self.assertTrue(are_d_separated(graph, "D", "E", conditions=["C"]))
        self.assertFalse(are_d_separated(graph, "AA", "B", conditions=["C"]))
        self.assertFalse(are_d_separated(graph, "D", "E"))
        self.assertFalse(are_d_separated(graph, "D", "E", conditions=["AA", "B"]))
        self.assertFalse(are_d_separated(graph, "G", "G", conditions=["C"]))


class TestGetConditionalIndependencies(unittest.TestCase):
    """Test getting conditional independencies."""

    def assert_example_has_judgements(self, example: Example) -> None:
        """Assert that the example is consistent w.r.t. D-separations."""
        self.assertIsNotNone(example.conditional_independencies)
        self.assert_has_judgements(
            graph=example.graph,
            judgements=example.conditional_independencies,
        )

    def assert_has_judgements(self, graph: Union[NxMixedGraph, SG], judgements: Iterable[DSeparationJudgement]) -> None:
        """Assert that the graph has the correct conditional independencies.

        :param graph: the graph to test
        :param judgements: the set of expected conditional independencies
        """
        asserted_judgements = set(judgements)
        observed_judgements = get_conditional_independencies(graph)

        self.assertIsNotNone(asserted_judgements, "Expected independencies is empty.")
        self.assertIsNotNone(observed_judgements, "Observed independencies is empty.")
        self.assertTrue(
            all(judgement.is_canonical for judgement in observed_judgements),
            msg='one or more of the returned DSeparationJudgement instances are not canonical',
        )

        self.assert_valid_judgements(graph, asserted_judgements)
        self.assert_valid_judgements(graph, observed_judgements)

        expected_pairs = {(judgement.left, judgement.right) for judgement in asserted_judgements}
        observed_pairs = {(judgement.left, judgement.right) for judgement in observed_judgements}
        self.assertEqual(expected_pairs, observed_pairs, "Judgements do not find same separable pairs")

        def _get_match(ref, options):
            """Finds judgement in options that has the same left/right pair as the reference judgement"""
            for alt in options:
                if ref.left == alt.left and ref.right == alt.right:
                    return alt
            return None

        for judgement in asserted_judgements:
            with self.subTest(name=judgement):
                matching = _get_match(judgement, observed_judgements)
                self.assertIsNotNone(matching, "No matching judgement found")
                self.assertGreaterEqual(len(judgement.conditions), len(matching.conditions),
                                        "Observed conditional independence more complicated than reference.")

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

    def test_examples(self):
        """Test getting the conditional independencies from the example graphs."""
        testable = (
            example
            for example in examples
            if example.conditional_independencies is not None
        )

        for example in testable:
            with self.subTest(name=example.name):
                self.maxDiff = None
                self.assert_example_has_judgements(example)
