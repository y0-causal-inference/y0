# -*- coding: utf-8 -*-

"""Tests for canceling fractions."""

import unittest

from y0.dsl import A, B, Fraction, One, P, Sum

one = One()


class TestCancel(unittest.TestCase):
    """Test cancelling fractions."""

    def test_simple_identity(self):
        """Test cancelling when the numerator and denominator are the same."""
        for label, frac in [
            ("one", Fraction(one, one)),
            ("prob", P(A) / P(A)),
            ("sum", Sum.safe(P(A), ranges=[B]) / Sum.safe(P(A), ranges=[B])),
            ("product", (P(A) * P(B)) / (P(A) * P(B))),
        ]:
            with self.subTest(type=label):
                self.assertIsInstance(frac, Fraction)
                self.assertEqual(one, frac.simplify(), msg=f"\n\nActual:{frac}")

    def test_fraction_simplify(self):
        """Test cancelling on products."""
        for label, expected, frac in [
            ("leave num.", P(B), (P(A) * P(B)) / P(A)),
            ("leave den.", one / P(B), P(A) / (P(A) * P(B))),
            ("unordered", one, (P(A) * P(B)) / (P(B) * P(A))),
            ("canonical", one / P(A), one / P(A)),
            ("flipper", P(A), Fraction(one, Fraction(one, P(A)))),
            ("prob-redundant-one", P(A), Fraction(P(A), one)),
            ("sum-redundant-one", Sum[B](P(A)), Fraction(Sum[B](P(A)), one)),
            ("frac-redundant-one", P(A) / P(B), Fraction(Fraction(P(A), P(B)), one)),
            ("prod-redundant-one", P(A) * P(B), Fraction(P(A) * P(B), one)),
        ]:
            with self.subTest(type=label):
                self.assertIsInstance(frac, Fraction)
                self.assertEqual(expected, frac.simplify(), msg=f"\n\nActual:{frac}")
