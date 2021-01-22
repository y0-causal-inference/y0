# -*- coding: utf-8 -*-

"""Test the causaleffect probability expression parser."""

import unittest

from y0.dsl import C, D, P, Sum, Variable
from y0.parser import parse_causaleffect

U_1 = Variable('U_1')


class TestCausaleffectGrammar(unittest.TestCase):
    """Tests for parsing causaleffect probability expressions."""

    def test_parse(self):
        """Test the high-level parser API."""
        for expr, s in [
            (Sum[U_1, C](P(D | U_1 | C) * P(C) * P(U_1)), "\\sum_{U_{1},C}P(D|u_{1},C)P(C)P(U_{1})"),
        ]:
            with self.subTest(s=s):
                actual = parse_causaleffect(s)
                self.assertEqual(expr, actual, msg=f'\nExpected: {expr}\nActual:   {actual}')
