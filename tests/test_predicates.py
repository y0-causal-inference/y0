# -*- coding: utf-8 -*-

"""Tests for expression predicates."""

import unittest

from y0.dsl import A, B, C, D, P, Sum, Variable, X
from y0.predicates import has_markov_postcondition


class TestMarkovCondition(unittest.TestCase):
    """Tests for checking the markov condition."""

    def test_markov_raises(self):
        """Test the type error is raised on invalid input."""
        for value in [
            Variable("whatever"),
            A @ B,
            A @ ~B,
            "something else",
        ]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                has_markov_postcondition(value)

    def test_markov_postcondition(self):
        """Test the expressions have the markov postcondition."""
        for expression in [
            P(A),
            P(A | B),
            P(A | (B, C)),
            P(A) * P(B),
            P(A) * P(A | B),
            P(A | B) * P(A | C),
            P(A) / P(B),
            P(A) / P(A | B),
            Sum[X](P(A)),
            Sum[X](P(A | B)),
            Sum[X](P(A | B) * P(B)),
        ]:
            with self.subTest(e=expression):
                self.assertTrue(has_markov_postcondition(expression))

    def test_missing_markov_postcondition(self):
        """Test the expressions do not have the markov postcondition."""
        for expression in [
            P(A, B),
            P(A & B | C),
            P(A & D | (B, C)),
            P(A, C) * P(B),
            P(A, C) * P(A | B),
            P(A) * P(A & C | B),
            P(A & C | B) * P(A | C),
            P(A) / P(B & C),
            P(A & C) / P(B),
            P(A) / P(A & C | B),
            Sum[X](P(A, C)),
            Sum[X](P(A & C | B)),
            Sum[X](P(A & C | B) * P(B)),
            Sum[X](P(A | B) * P(B, C)),
        ]:
            with self.subTest(e=str(expression)):
                self.assertFalse(has_markov_postcondition(expression))
