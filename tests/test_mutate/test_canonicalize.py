# -*- coding: utf-8 -*-

"""Tests for the canonicalization algorithm."""

import itertools as itt
import unittest
from typing import Sequence

from y0.dsl import A, B, C, D, Expression, P, Sum, Variable, X, Y, Z
from y0.mutate.canonicalize import canonicalize


class TestCanonicalize(unittest.TestCase):
    """Tests for the canonicalization of a simplified algorithm."""

    def test_canonicalize_raises(self):
        """Test a value error is raised for non markov-conditioning expressions."""
        with self.assertRaises(ValueError):
            canonicalize(P(A, B, C), [A, B, C])

    def assert_canonicalize(self, expected: Expression, expression: Expression, ordering: Sequence[Variable]) -> None:
        """Check that the expression is canonicalized properly given an ordering."""
        with self.subTest(expr=str(expression), ordering=', '.join(variable.name for variable in ordering)):
            actual = canonicalize(expression, ordering)
            self.assertEqual(
                expected, actual,
                msg=f'\nExpected: {str(expression)}\nActual:   {str(actual)}',
            )

    def test_atomic(self):
        """Test canonicalization of atomic expressions."""
        for expected, expression, ordering in [
            (P(A), P(A), [A]),
            (P(A | B), P(A | B), [A, B]),
            (P(A | (B, C)), P(A | (B, C)), [A, B, C]),
            (P(A | (B, C)), P(A | (C, B)), [A, B, C]),
        ]:
            self.assert_canonicalize(expected, expression, ordering)

        expected = P(A | (B, C, D))
        for b, c, d in itt.permutations((B, C, D)):
            expression = P(A | (b, c, d))
            self.assert_canonicalize(expected, expression, [A, B, C, D])

    def test_derived_atomic(self):
        """Test canonicalizing."""
        # Sum
        expected = expression = Sum(P(A))
        self.assert_canonicalize(expected, expression, [A])

        # Simple product (only atomic)
        expected = P(A) * P(B) * P(C)
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            expression = a * b * c
            self.assert_canonicalize(expected, expression, [A, B, C])

        # Sum with simple product (only atomic)
        expected = Sum(P(A) * P(B) * P(C))
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            expression = Sum(a * b * c)
            self.assert_canonicalize(expected, expression, [A, B, C])

        # Fraction
        expected = expression = P(A) / P(B)
        self.assert_canonicalize(expected, expression, [A, B])

        # Fraction with simple products (only atomic)
        expected = (P(A) * P(B) * P(C)) / (P(X) * P(Y) * P(Z))
        for (a, b, c), (x, y, z) in itt.product(
            itt.permutations((P(A), P(B), P(C))),
            itt.permutations((P(X), P(Y), P(Z))),
        ):
            expression = (a * b * c) / (x * y * z)
            self.assert_canonicalize(expected, expression, [A, B, C, X, Y, Z])

    def test_mixed(self):
        """Test mixed expressions."""
        expected = expression = P(A) * Sum(P(B))
        self.assert_canonicalize(expected, expression, [A, B])

        expected = P(A) * Sum(P(B)) * Sum(P(C))
        for a, b, c in itt.permutations((P(A), Sum(P(B)), Sum(P(C)))):
            expression = a * b * c
            self.assert_canonicalize(expected, expression, [A, B, C])

        expected = P(D) * Sum(P(A) * P(B) * P(C))
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            sum_expr = Sum(a * b * c)
            for left, right in itt.permutations((P(D), sum_expr)):
                self.assert_canonicalize(expected, left * right, [A, B, C, D])

        expected = P(X) * Sum(P(A) * P(B)) * Sum(P(C) * P(D))
        for (a, b), (c, d) in itt.product(
            itt.permutations((P(A), P(B))),
            itt.permutations((P(C), P(D))),
        ):
            sexpr = Sum(a * b) * Sum(c * d)
            self.assert_canonicalize(expected, sexpr * P(X), [A, B, C, D])
            self.assert_canonicalize(expected, P(X) * sexpr, [A, B, C, D])

        expected = expression = Sum(P(A) / P(B))
        self.assert_canonicalize(expected, expression, [A, B])

        expected = expression = Sum(P(A) / Sum(P(B))) * Sum(P(A) / Sum(P(B) / P(C)))
        self.assert_canonicalize(expected, expression, [A, B, C])
