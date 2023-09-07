"""Tests for contraction functions."""

import unittest

from y0.dsl import A, B, C, D, P, Sum
from y0.mutate.contract import contract


class TestContract(unittest.TestCase):
    """Test case for contraction functions."""

    def test_contract(self):
        """Test the simple contract function."""
        self.assertEqual(P(A | B), contract(P(A, B) / P(B)))
        self.assertEqual(P(A | B), contract(P(A, B) / P(B)))
        self.assertEqual(P(A | B, C), contract(P(A, B, C) / P(B, C)))
        self.assertEqual(P(B, C | A), contract(P(A, B, C) / P(A)))
        self.assertEqual(P(A, B, C) / P(D), contract(P(A, B, C) / P(D)))

        expr = Sum[A](P(B))
        self.assertEqual(expr, contract(expr))
