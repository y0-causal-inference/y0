# -*- coding: utf-8 -*-

"""Test the causaleffect probability expression parser."""

import unittest

from y0.dsl import C, D, P, Sum, Variable
from y0.parser import parse_causaleffect
from y0.parser.ce.grammar import sum_pe

U_1 = Variable("U_1")


class TestCausaleffectGrammar(unittest.TestCase):
    """Tests for parsing causaleffect probability expressions."""

    def test_sum(self):
        """Test the sum grammar."""
        for expr, s in [
            (
                Sum[U_1, C](P(D | U_1 | C) * P(C) * P(U_1)),
                r"\sum_{U_{1},C}P(D|U_{1},C)P(C)P(U_{1})",
            ),
        ]:
            with self.subTest(s=s):
                parse_result = sum_pe.parseString(s)
                result_expression = parse_result.asList()[0]
                self.assertIsInstance(result_expression, Sum)
                self.assertEqual(
                    expr,
                    result_expression,
                    msg=f"Mismatch\nExpected: {repr(expr)}\nActual:   {repr(result_expression)}",
                )

    def test_parse(self):
        """Test the high-level parser API."""
        for expr, s in [
            (
                Sum[U_1, C](P(D | U_1 | C) * P(C) * P(U_1)),
                r"\sum_{U_{1},C}P(D|U_{1},C)P(C)P(U_{1})",
            ),
        ]:
            with self.subTest(s=s):
                actual = parse_causaleffect(s)
                self.assertEqual(expr, actual, msg=f"\nExpected: {expr}\nActual:   {actual}")
