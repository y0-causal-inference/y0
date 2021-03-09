# -*- coding: utf-8 -*-

"""Test falsification of testable implications given a graph."""

import unittest

from y0.algorithm.falsification import falsifications
from y0.algorithm.conditional_independencies import get_conditional_independencies

from y0.examples import asia_example


class TestFalsification(unittest.TestCase):
    "Are falsifiable implications testable?"

    def test_asia_from_graph(self):
        "Test the asia graph against data generated from it."
        G = asia_example.graph.to_admg()
        df = asia_example.data
        issues = falsifications(G, df)
        self.assertEqual(0, len(issues))
        self.assertGreater(len(issues.evidence), 0)

    def test_asia_from_list(self):
        "Test the asia graph against data generated from it, passing in the implications to test."
        G = asia_example.graph.to_admg()
        df = asia_example.data
        implications = get_conditional_independencies(G)
        issues = falsifications(implications, df)
        self.assertEqual(0, len(issues))
        self.assertEqual(len(issues.evidence), len(implications))
