# -*- coding: utf-8 -*-

"""Test falsification of testable implications given a graph."""

import unittest

from y0.algorithm.conditional_independencies import get_conditional_independencies
from y0.algorithm.falsification import get_falsifications, get_graph_falsifications
from y0.examples import asia_example, frontdoor_backdoor_example
from y0.struct import DSeparationJudgement, get_conditional_independence_tests


class TestFalsification(unittest.TestCase):
    """Test the falsifiable implications."""

    def test_discrete_graph_falsifications(self):
        """Test the asia graph against data generated from it."""
        for method in [None, *get_conditional_independence_tests()]:
            if method == "pearson":
                continue
            with self.subTest(method=method):
                issues = get_graph_falsifications(
                    asia_example.graph, asia_example.data, method=method
                )
                self.assertEqual(0, len(issues.failures))
                self.assertGreater(len(issues.evidence), 0)

    def test_continuous_graph_falsifications(self):
        """Test the frontdoor graph against continuous data generated for it."""
        data = frontdoor_backdoor_example.generate_data(1_000)
        # judgements = [DSeparationJudgement(left=)]
        get_graph_falsifications(frontdoor_backdoor_example.graph, df=data, method="pearson")
        # TODO get a graph where we know what the outcome should be, this
        #  should be available in https://github.com/y0-causal-inference/eliater/pull/1

    def test_falsifications(self):
        """Test the asia graph against data generated from it, passing in the implications to test."""
        implications = get_conditional_independencies(asia_example.graph)
        issues = get_falsifications(implications, asia_example.data)
        self.assertEqual(0, len(issues.failures))
        self.assertEqual(len(issues.evidence), len(implications))
