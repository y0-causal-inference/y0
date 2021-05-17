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
            ('one', one / one),
            ('prob', P(A) / P(A)),
            ('sum', Sum(P(A)) / Sum(P(A))),
            ('product', (P(A) * P(B)) / (P(A) * P(B))),
        ]:
            with self.subTest(type=label):
                # self.assertIsInstance(frac, Fraction)
                self.assertEqual(one, frac.simplify(), msg=f'\n\nActual:{frac}')
