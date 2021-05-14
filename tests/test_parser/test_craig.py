# -*- coding: utf-8 -*-

"""Test the probability parser."""

import unittest

from pyparsing import ParserElement

from y0.dsl import A, B, C, P, Q, Sum, X, Y, Z
from y0.parser.craig.grammar import fraction_pe, grammar, sum_pe
from y0.parser.craig.utils import probability_pe, qfactor_pe, variable_pe


class TestGrammar(unittest.TestCase):
    """Tests for parsing probability expressions."""

    def assert_many(self, expressions, parser_element: ParserElement, direct: bool = True):
        """Help testing many."""
        for expression in expressions:
            with self.subTest(expr=expression.to_text(), parser_element=parser_element.name):
                self.assert_parse_equal(expression, parser_element, direct=direct)

    def assert_parse_equal(self, expression, parser_element: ParserElement, direct: bool = True):
        """Help test parsing works round trip.

        :param expression: The DSL object to check
        :param parser_element: The relevant parser element for specific checking
        :param direct: If true, uses object equals checks. If false, uses stringification then string equals checks.
            Switch to false when things aren't working during development.
        """
        text = expression.to_text()
        parse_result = parser_element.parseString(text)
        result_expression = parse_result.asList()[0]
        self.assertIsInstance(result_expression, expression.__class__)
        if direct:
            self.assertEqual(
                expression, result_expression,
                msg=f'Mismatch\nExpected: {repr(expression)}\nActual:   {repr(result_expression)}',
            )
        else:
            self.assertEqual(text, result_expression.to_text())

    def test_variable(self):
        """Tests for variables, intervention variables, and counterfactual variables."""
        self.assert_many(
            [
                A,
                A @ B,
                A @ ~B,
                A @ B @ C,
                A @ ~B @ C,
                A @ ~B @ ~C,
                A @ B @ ~C,
            ],
            variable_pe,
        )

    def test_probability(self):
        """Tests for probabilities."""
        for g in (probability_pe, grammar):
            self.assert_many(
                [
                    P(A),
                    P(A, B),
                    P(A | B),
                    P(A @ X | B),
                    P(A @ ~X | B),
                    P(A | B @ Y),
                    P(A | B @ ~Y),
                    P(A @ X | B @ ~Y),
                    P(A @ X | B @ ~Y | C @ Z),
                    P(A | [B, C]),
                    P(A, B, C),
                ],
                g,
            )

    def test_sum(self):
        """Tests for sums."""
        for g in (sum_pe, grammar):
            self.assert_many(
                [
                    Sum(P(A)),
                    Sum(P(A, B)),
                    Sum(P(A | B)),
                    Sum(P(A | B) * P(B)),
                    Sum[B](P(A | B) * P(B)),
                    Sum[B, C](P(A | B) * P(B)),
                ],
                g,
            )

    def test_fraction(self):
        """Tests for fractions."""
        for g in (fraction_pe, grammar):
            self.assert_many(
                [
                    Sum(P(A)) / P(B),
                    Sum(P(A, B)) / P(A),
                    Sum[B](P(A | B) * P(B)) / Sum(P(A | B) * P(B)),
                ],
                g,
            )

    def test_q(self):
        """Tests for Q factors."""
        for g in (qfactor_pe, grammar):
            self.assert_many(
                [
                    Q[A](X),
                    Q[A, B](X),
                    Q[A](X, Y),
                    Q[A, B](X, Y),
                ],
                g,
            )
