# -*- coding: utf-8 -*-

"""Tests for the canonicalization algorithm."""

import itertools as itt
import unittest

from y0.canonicalize import canonicalize
from y0.dsl import A, B, C, D, P, Sum, X, Y, Z


class TestCanonicalize(unittest.TestCase):
    """Tests for the canonicalization of a simplified algorithm."""

    def test_canonicalize_raises(self):
        """Test a value error is raised for non markov-conditioning expressions."""
        with self.assertRaises(ValueError):
            canonicalize(P(A, B, C), [A, B, C])

    def test_atomic(self):
        """Test canonicalization of atomic expressions."""
        for expected, expression, ordering in [
            (P(A), P(A), [A]),
            (P(A | B), P(A | B), [A, B]),
            (P(A | (B, C)), P(A | (B, C)), [A, B, C]),
            (P(A | (B, C)), P(A | (C, B)), [A, B, C]),
        ]:
            with self.subTest(e=str(expression), ordering=ordering):
                self.assertEqual(expected, canonicalize(expression, ordering))

        expected = P(A | (B, C, D))
        for b, c, d in itt.permutations((B, C, D)):
            expression = P(A | (b, c, d))
            with self.subTest(e=str(expression)):
                self.assertEqual(expected, canonicalize(expression, [A, B, C, D]))

    def test_derived_atomic(self):
        """Test canonicalizing."""
        # Sum
        expected = expression = Sum(P(A))
        with self.subTest(e=str(expression)):
            self.assertEqual(expected, canonicalize(expression, [A]))

        # Simple product (only atomic)
        expected = P(A) * P(B) * P(C)
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            expression = a * b * c
            with self.subTest(e=str(expression)):
                self.assertEqual(expected, canonicalize(expression, [A, B, C]))

        # Sum with simple product (only atomic)
        expected = Sum(P(A) * P(B) * P(C))
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            expression = Sum(a * b * c)
            with self.subTest(e=str(expression)):
                self.assertEqual(expected, canonicalize(expression, [A, B, C]))

        # Fraction
        expected = expression = P(A) / P(B)
        with self.subTest(e=str(expression)):
            self.assertEqual(expected, canonicalize(expression, [A, B]))

        # Fraction with simple products (only atomic)
        expected = (P(A) * P(B) * P(C)) / (P(X) * P(Y) * P(Z))
        for (a, b, c), (x, y, z) in itt.product(
            itt.permutations((P(A), P(B), P(C))),
            itt.permutations((P(X), P(Y), P(Z))),
        ):
            expression = (a * b * c) / (x * y * z)
            with self.subTest(e=str(expression)):
                self.assertEqual(expected, canonicalize(expression, [A, B, C, X, Y, Z]))

    def test_mixed(self):
        """Test mixed expressions."""
        expected = Sum(P(A) * P(B) * P(C)) * P(D)
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            expression = P(D) * Sum(a * b * c)
            with self.subTest(e=str(expression)):
                self.assertEqual(expected, canonicalize(expression, [A, B, C]))
