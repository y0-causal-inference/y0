# -*- coding: utf-8 -*-

"""Test falsification of testable implications given a graph."""

import unittest

from y0.algorithm.conditional_independencies import get_conditional_independencies
from y0.algorithm.falsification import get_falsifications, get_graph_falsifications
from y0.examples import asia_example


class TestFalsification(unittest.TestCase):
    """Test the falsifiable implications."""

    def test_graph_falsifications(self):
        """Test the asia graph against data generated from it."""
        issues = get_graph_falsifications(
            asia_example.graph, asia_example.data, method="cressie_read"
        )
        self.assertEqual(0, len(issues.failures))
        self.assertGreater(len(issues.evidence), 0)

    def test_falsifications(self):
        """Test the asia graph against data generated from it, passing in the implications to test."""
        implications = get_conditional_independencies(asia_example.graph)
        issues = get_falsifications(implications, asia_example.data)
        self.assertEqual(0, len(issues.failures))
        self.assertEqual(len(issues.evidence), len(implications))
