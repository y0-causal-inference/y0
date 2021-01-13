# -*- coding: utf-8 -*-

"""Test common mutation functions."""

import unittest

from y0.dsl import P, W, X, Y, Z
from y0.mutate import check_partial_bayes, expand_simple_bayes, reduce_simple_bayes


class TestMutate(unittest.TestCase):
    """Test common mutations."""

    def test_expand_bayes(self):
        """Test the expansion of a simple conditional to a Bayes-like expression."""
        self.assertEqual(P(X | Y) * P(Y) / P(X), expand_simple_bayes(P(Y | X)))
        self.assertEqual(P(X & W | Y) * P(Y) / P(X & W), expand_simple_bayes(P(Y | X | W)))

        with self.assertRaises(NotImplementedError):
            expand_simple_bayes(P(X & Y | Z))

    def test_check_partial_bayes(self):
        self.assertTrue(check_partial_bayes(P(X | Y), P(Y)))
        self.assertTrue(check_partial_bayes(P(X | Y & Z), P(Y & Z)))

    def test_contract_bayes(self):
        """Test the reduction of a Bayes-like expression to a simple conditional."""
        expr = P(X | Y) * P(Y) / P(X)
        self.assertTrue(check_partial_bayes(expr))
        self.assertEqual(P(Y | X), reduce_simple_bayes(expr))
