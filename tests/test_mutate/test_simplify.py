# -*- coding: utf-8 -*-

"""Tests for canceling fractions."""

import unittest

from y0.dsl import A, B, One, P, Sum

one = One()


class TestCancel(unittest.TestCase):
    """Test cancelling fractions."""

    def test_simple_identity(self):
        """Test cancelling when the numerator and denominator are the same."""
        for label, frac in [
            ("one", one / one),
            ("prob", P(A) / P(A)),
            ("sum", Sum(P(A)) / Sum(P(A))),
            ("product", (P(A) * P(B)) / (P(A) * P(B))),
        ]:
            with self.subTest(type=label):
                self.assertEqual(one, frac.simplify(), msg=f"\n\nActual:{frac}")

    def test_fraction_simplify(self):
        """Test cancelling on products."""
        for label, expected, frac in [
            ("leave num.", P(B), (P(A) * P(B)) / P(A)),
            ("leave den.", one / P(B), P(A) / (P(A) * P(B))),
            ("unordered", one, (P(A) * P(B)) / (P(B) * P(A))),
        ]:
            with self.subTest(type=label):
                self.assertEqual(expected, frac.simplify(), msg=f"\n\nActual:{frac}")
