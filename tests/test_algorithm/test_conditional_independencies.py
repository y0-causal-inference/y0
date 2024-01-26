# -*- coding: utf-8 -*-

"""Test getting conditional independencies (and related)."""

import typing
import unittest
from typing import Iterable, Set

from pgmpy.estimators import CITests

from y0.algorithm.conditional_independencies import (
    are_d_separated,
    get_conditional_independencies,
)
from y0.dsl import AA, B, C, D, E, F, G, Variable, X, Y
from y0.examples import (
    Example,
    d_separation_example,
    examples,
    frontdoor_backdoor_example,
    frontdoor_example,
)
from y0.graph import NxMixedGraph, iter_moral_links
from y0.struct import CITestTuple, DSeparationJudgement


class TestDSeparation(unittest.TestCase):
    """Test the d-separation utility."""

    def test_mit_example(self):
        """Test checking D-separation on the MIT example."""
        graph = d_separation_example.graph

        self.assertFalse(are_d_separated(graph, AA, B, conditions=[D, F]))
        self.assertTrue(are_d_separated(graph, AA, B))
        self.assertTrue(are_d_separated(graph, D, E, conditions=[C]))
        self.assertFalse(are_d_separated(graph, AA, B, conditions=[C]))
        self.assertFalse(are_d_separated(graph, D, E))
        self.assertFalse(are_d_separated(graph, D, E, conditions=[AA, B]))
        self.assertFalse(are_d_separated(graph, G, G, conditions=[C]))

    def test_examples(self):
        """Check that example conditional independencies are d-separations and conditions (if present) are required.

        This test is using convenient examples to ensure that the d-separation algorithm
        isn't just always returning true or false.
        """
        testable = (
            example for example in examples if example.conditional_independencies is not None
        )

        for example in testable:
            with self.subTest(name=example.name):
                for ci in example.conditional_independencies:
                    self.assertIn(ci.left, example.graph)
                    self.assertIn(ci.right, example.graph)
                    judgement = are_d_separated(
                        example.graph, ci.left, ci.right, conditions=ci.conditions
                    )
                    self.assertTrue(
                        judgement,
                        msg=f"Expected d-separation not found in {example.name}",
                    )
                    if ci.conditions:
                        self.assertFalse(
                            are_d_separated(example.graph, ci.left, ci.right),
                            msg="Unexpected d-separation",
                        )

    def test_moral_links(self):
        """Test adding 'moral links' (part of the d-separation algorithm).

        This test covers several cases around moral links to ensure that they are added when needed.
        """
        graph = NxMixedGraph.from_str_edges(
            nodes=("a", "b", "c"),
            directed=[("a", "b"), ("b", "c")],
        )
        links = list(iter_moral_links(graph))
        self.assertEqual([], links, msg="Unexpected moral links added.")

        graph = NxMixedGraph.from_str_edges(
            nodes=("a", "b", "c"),
            directed=[("a", "c"), ("b", "c")],
        )
        links = set(tuple(sorted(e)) for e in iter_moral_links(graph))
        self.assertEqual(
            {(Variable("a"), Variable("b"))},
            links,
            msg="Moral links not as expected in single-link case.",
        )

        graph = NxMixedGraph.from_str_edges(
            nodes=("a", "b", "aa", "bb", "c"),
            directed=[("a", "c"), ("b", "c"), ("aa", "c"), ("bb", "c")],
        )
        links = set(tuple(sorted(e)) for e in iter_moral_links(graph))
        self.assertEqual(
            {
                (Variable("a"), Variable("b")),
                (Variable("a"), Variable("aa")),
                (Variable("a"), Variable("bb")),
                (Variable("aa"), Variable("b")),
                (Variable("aa"), Variable("bb")),
                (Variable("b"), Variable("bb")),
            },
            links,
            msg="Moral links not as expected in multi-link case.",
        )

        graph = NxMixedGraph.from_str_edges(
            nodes=("a", "b", "c", "d", "e"),
            directed=[("a", "c"), ("b", "c"), ("c", "e"), ("d", "e")],
        )
        links = set(tuple(sorted(e)) for e in iter_moral_links(graph))
        self.assertEqual(
            {(Variable("a"), Variable("b")), (Variable("c"), Variable("d"))},
            links,
            msg="Moral links not as expected in multi-site case.",
        )


class TestGetConditionalIndependencies(unittest.TestCase):
    """Test getting conditional independencies."""

    def assert_example_has_judgements(self, example: Example) -> None:
        """Assert that the example is consistent w.r.t. D-separations."""
        self.assertIsNotNone(example.conditional_independencies)
        self.assert_has_judgements(
            graph=example.graph,
            judgements=example.conditional_independencies,
        )

    def assert_judgement_types(self, judgements: Iterable[DSeparationJudgement]):
        """Assert all judgmenets have the right types."""
        self.assertTrue(
            all(
                (
                    isinstance(judgement.left, Variable)
                    and isinstance(judgement.right, Variable)
                    and all(isinstance(c, Variable) for c in judgement.conditions)
                )
                for judgement in judgements
            )
        )

    def assert_has_judgements(self, graph, judgements: Iterable[DSeparationJudgement]) -> None:
        """Assert that the graph has the correct conditional independencies.

        :param graph: the graph to test
        :type graph: NxMixedGraph or ananke.graphs.SG
        :param judgements: the set of expected conditional independencies
        """
        self.assertTrue(all(isinstance(node, Variable) for node in graph))
        self.assert_judgement_types(judgements)

        asserted_judgements = set(judgements)
        self.assertIsNotNone(asserted_judgements, "Expected independencies is empty.")

        observed_judgements = get_conditional_independencies(graph)
        self.assertIsNotNone(observed_judgements, "Observed independencies is empty.")
        self.assert_judgement_types(asserted_judgements)

        self.assertTrue(
            all(judgement.is_canonical for judgement in observed_judgements),
            msg="one or more of the returned DSeparationJudgement instances are not canonical",
        )

        self.assert_valid_judgements(graph, asserted_judgements)
        self.assert_valid_judgements(graph, observed_judgements)

        expected_pairs = {(judgement.left, judgement.right) for judgement in asserted_judgements}
        observed_pairs = {(judgement.left, judgement.right) for judgement in observed_judgements}
        self.assertEqual(
            expected_pairs,
            observed_pairs,
            "Judgements do not find same separable pairs",
        )

        def _get_match(ref, options):
            """Find a judgement that has the same left/right pair as the reference judgement."""
            for alt in options:
                if ref.left == alt.left and ref.right == alt.right:
                    return alt
            return None

        for judgement in asserted_judgements:
            with self.subTest(name=judgement):
                matching = _get_match(judgement, observed_judgements)
                self.assertIsNotNone(matching, "No matching judgement found.")
                self.assertGreaterEqual(
                    len(judgement.conditions),
                    len(matching.conditions),
                    msg="Observed conditional independence more complicated than reference.",
                )

    def assert_valid_judgements(
        self, graph: NxMixedGraph, judgements: Set[DSeparationJudgement]
    ) -> None:
        """Check that a set of judgments are valid with respect to a graph."""
        self.assertIsInstance(graph, NxMixedGraph)

        for judgement in judgements:
            self.assertTrue(
                all(isinstance(condition, Variable) for condition in judgement.conditions)
            )
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
            example for example in examples if example.conditional_independencies is not None
        )

        for example in testable:
            with self.subTest(name=example.name):
                self.maxDiff = None
                self.assert_example_has_judgements(example)

    def test_ci_test_continuous(self):
        """Test conditional independency test on continuous data."""
        data = frontdoor_example.generate_data(500)  # continuous
        judgement = DSeparationJudgement(
            left=X,
            right=Y,
            separated=...,
            conditions=(),
        )
        test_result_bool = judgement.test(data, method="pearson", boolean=True)
        self.assertIsInstance(test_result_bool, bool)

        test_result_tuple = judgement.test(data, method="pearson", boolean=False)
        self.assertIsInstance(test_result_tuple, CITestTuple)
        self.assertIsNone(test_result_tuple.dof)

        # Test that an error is thrown if using a discrete test on continuous data
        with self.assertRaises(ValueError):
            judgement.test(data, method="chi-square", boolean=True)

    def test_ci_test_discrete(self):
        """Test conditional independency test on discrete data."""
        data = frontdoor_backdoor_example.generate_data(500)  # discrete
        judgement = DSeparationJudgement(
            left=X,
            right=Y,
            separated=...,
            conditions=(),
        )
        for method in typing.get_args(CITests):
            test_result_bool = judgement.test(data, method=method, boolean=True)
            self.assertIsInstance(test_result_bool, bool)

            test_result_tuple = judgement.test(data, method=method, boolean=False)
            self.assertIsInstance(test_result_tuple, CITestTuple)
            self.assertIsNotNone(test_result_tuple.dof)

        # Test that an error is thrown if using a continous test on discrete data
        with self.assertRaises(ValueError):
            judgement.test(data, method="pearson", boolean=True)
