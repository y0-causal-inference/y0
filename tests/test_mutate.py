# -*- coding: utf-8 -*-

"""Test common mutation functions."""

import unittest

from y0.dsl import P, X, Y
from y0.mutate import expand_bayes, reduce_bayes, expand_simple_bayes


class TestMutate(unittest.TestCase):
    """Test common mutations."""

    def test_expand_bayes(self):
        """Test the expansion of a simple conditional to a Bayes-like expression."""
        self.assertEqual(P(X | Y) * P(Y) / P(X), expand_simple_bayes(P(Y | X)))

    # def test_contract_bayes(self):
    #     """Test the reduction of a Bayes-like expression to a simple conditional."""
    #     self.assertEqual(P(Y | X), reduce_bayes(P(X | Y) * P(Y) / P(X)))
