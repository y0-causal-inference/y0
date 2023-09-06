"""Test the internal y0 DSL parser."""

import unittest

from y0.dsl import A, B, C, Expression, P, Q, Sum
from y0.parser import parse_y0


class TestInternalParser(unittest.TestCase):
    """Test the internal y0 DSL parser."""

    def test_parse(self):
        """Test the parse function."""
        for expected, s in [
            (
                Sum[B](P(A | B) * P(B)) / Sum[C](P(A | B) * P(B)),
                "Sum[B](P(A | B) * P(B)) / Sum[C](P(A | B) * P(B))",
            ),
            (
                Q[A](B) * P(A | B),
                "Q[A](B) * P(A | B)",
            ),
        ]:
            with self.subTest(s=s):
                actual = parse_y0(s)
                self.assertIsInstance(actual, Expression)
                self.assertEqual(
                    expected,
                    actual,
                    msg=f"\nExpected: {expected}\nActual:   {actual}",
                )
