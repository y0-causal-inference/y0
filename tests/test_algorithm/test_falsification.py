# -*- coding: utf-8 -*-

"""Test falsification of testable implications given a graph."""

import unittest

import numpy as np
import pandas as pd

from y0.algorithm.conditional_independencies import get_conditional_independencies
from y0.algorithm.falsification import get_falsifications, get_graph_falsifications
from y0.examples import asia_example, frontdoor_example
from y0.struct import get_conditional_independence_tests


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

    def test_method_mismatch(self):
        """Test when the wrong test is given."""
        data_continuous = pd.DataFrame(
            {v.name: np.random.normal(size=20) for v in asia_example.graph.nodes()}
        )
        with self.assertRaises(ValueError):
            get_graph_falsifications(asia_example.graph, data_continuous, method="cressie_read")

        data_binary = pd.DataFrame(
            {v.name: np.random.binomial(1, 0.5, size=20) for v in asia_example.graph.nodes()}
        )
        with self.assertRaises(ValueError):
            get_graph_falsifications(asia_example.graph, data_binary, method="pearson")

        # This tests what happens when there are two continuous values
        data_sort_of_continuous = data_binary.copy()
        for c in data_sort_of_continuous.columns:
            data_sort_of_continuous[c] = data_sort_of_continuous[c].map(lambda x: x + 32)
        with self.assertRaises(ValueError):
            get_graph_falsifications(
                asia_example.graph, data_sort_of_continuous, method="cressie_read"
            )
        get_graph_falsifications(asia_example.graph, data_sort_of_continuous, method="pearson")

    def test_continuous_graph_falsifications(self):
        """Test the frontdoor graph against continuous data generated for it."""
        data = frontdoor_example.generate_data(1_000)
        get_graph_falsifications(frontdoor_example.graph, df=data, method="pearson")
        # TODO get a graph where we know what the outcome should be, this
        #  should be available in https://github.com/y0-causal-inference/eliater/pull/1

    def test_falsifications(self):
        """Test the asia graph against data generated from it, passing in the implications to test."""
        implications = get_conditional_independencies(asia_example.graph)
        issues = get_falsifications(implications, asia_example.data)
        self.assertEqual(0, len(issues.failures))
        self.assertEqual(len(issues.evidence), len(implications))
