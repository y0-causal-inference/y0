# -*- coding: utf-8 -*-

"""Tests for expression complexity.

Note: tests in this module should **NOT** use exact
numbers, but should rather encode our intuitive ideas
of what's more complex by comparing expressions.
"""

import unittest

from y0.complexity import complexity
from y0.dsl import A, B, C, D, Expression, Fraction, One, P, X, Y, Z
from y0.mutate import bayes_expand, chain_expand, fraction_expand


class TestComplexity(unittest.TestCase):
    """Test case for complexity tests."""

    def assert_complexity_le(self, left: Expression, right: Expression) -> None:
        """Assert that the complexity of the first expression is less than the second."""
        self.assertIsInstance(left, Expression)
        self.assertIsInstance(right, Expression)
        self.assertLessEqual(complexity(left), complexity(right))

    def assert_complexity_equal(self, left: Expression, right: Expression) -> None:
        """Assert that the complexity of the first expression is less than the second."""
        self.assertIsInstance(left, Expression)
        self.assertIsInstance(right, Expression)
        self.assertEqual(complexity(left), complexity(right))

    def test_complexity_equal(self):
        """Test complexity equivalence."""
        examples = [
            (P(A), P(B), "probability 1-variable isomorphism"),
            (P(A, B), P(B, A), "probability 2-variable isomorphism, same variables"),
            (P(A, B), P(B, C), "probability 2-variable isomorphism, one different"),
            (P(A, B), P(C, D), "probability 2-variable isomorphism, two different"),
        ]
        for left, right, label in examples:
            with self.subTest(label=label):
                self.assert_complexity_equal(left, right)

    def test_expand(self):
        """Test various expansions always make at least more complicated expressions."""
        examples = [
            P(A | X),
            P(A | X, Y),
            P(A | X, Y, Z),
            P(A, B | X),
            P(A, B | X, Y),
            P(A, B | X, Y, Z),
        ]
        for example in examples:
            with self.subTest(expr=example.to_y0(), type="fraction"):
                self.assert_complexity_le(example, fraction_expand(example))
            with self.subTest(expr=example.to_y0(), type="bayes"):
                self.assert_complexity_le(example, bayes_expand(example))
            with self.subTest(expr=example.to_y0(), type="chain"):
                self.assert_complexity_le(example, chain_expand(example))

    def test_fraction_simplify(self):
        """Test simplifying a fraction always results in at least a less complicated expression."""
        examples = [
            Fraction(P(A), One()),
            Fraction(P(A) * P(B), P(B)),
        ]
        for example in examples:
            with self.subTest(expr=example.to_y0()):
                self.assertIsInstance(example, Fraction)
                self.assert_complexity_le(example.simplify(), example)
