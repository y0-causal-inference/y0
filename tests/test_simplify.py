# -*- coding: utf-8 -*-

"""Tests of the simplification algorithm."""

import itertools as itt
import unittest

from y0.dsl import A, B, C, D, P, Sum, Variable, X, Y, Z
from y0.graph import NxMixedGraph
from y0.simplify import canonicalize, has_markov_postcondition, simplify

#: Corresponds to Figure 1 in https://www.jmlr.org/papers/volume18/16-166/16-166.pdf
figure_1 = NxMixedGraph()
figure_1.add_directed_edge('X', 'Z1')
figure_1.add_directed_edge('Z1', 'Y')
figure_1.add_directed_edge('Z2', 'X')
figure_1.add_directed_edge('Z2', 'Z1')
figure_1.add_directed_edge('Z2', 'Z3')
figure_1.add_directed_edge('Z3', 'Y')
figure_1.add_undirected_edge('X', 'Y')
figure_1.add_undirected_edge('X', 'Z2')
figure_1.add_undirected_edge('X', 'Z3')
figure_1.add_undirected_edge('Y', 'Z2')

Z1, Z2, Z3 = (Variable(s) for s in 'Z1 Z2 Z3'.split())

# These are the parts of pre-equation 1 and equation 1. They don't all fit on one
#  line so they're broken up into reusable bits
_a = P(Z1 | (Z2, X))
_b = P(Z3 | Z2)
_y = P(Y | (Z2, X, Z3, Z1))
_z3 = P(Z3 | (Z2, X))
_x = P(X | Z2)
_i = _y * _z3 * _x * P(Z2)
_c = Sum[X](_i)
_d = Sum[X, Y](_i)
_e = Sum[X, Z3, Y](_i)
pre_equation_1 = _a * _b * _c * _e / _d
equation_1 = P(Z1 | (Z2, X)) * P(Z2) * P(Y) * Sum[X](_y * _z3 * _x)


class TestMarkovCondition(unittest.TestCase):
    """Tests for checking the markov condition."""

    def test_markov_raises(self):
        """Test the type error is raised on invalid input."""
        for value in [
            Variable('whatever'),
            A @ B,
            A @ ~B,
            'something else',
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


class TestCanonicalize(unittest.TestCase):
    """Tests for the canonicalization of a simplified algorithm."""

    def test_canonicalize_raises(self):
        """Test a value error is raised for non markov-conditioning expressions."""
        with self.assertRaises(ValueError):
            canonicalize(P(A, B, C), [A, B, C])

    def test_canonicalize(self):
        """Test canonicalizing."""
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

        expected = P(A) * P(B) * P(C)
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            expression = a * b * c
            with self.subTest(e=str(expression)):
                self.assertEqual(expected, canonicalize(expression, [A, B, C]))

        expected = Sum(P(A) * P(B) * P(C))
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            expression = Sum(a * b * c)
            with self.subTest(e=str(expression)):
                self.assertEqual(expected, canonicalize(expression, [A, B, C]))

        expected = Sum(P(A) * P(B) * P(C)) * P(D)
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            expression = P(D) * Sum(a * b * c)
            with self.subTest(e=str(expression)):
                self.assertEqual(expected, canonicalize(expression, [A, B, C]))

        expected = (P(A) * P(B) * P(C)) / (P(X) * P(Y) * P(Z))
        for (a, b, c), (x, y, z) in itt.product(
            itt.permutations((P(A), P(B), P(C))),
            itt.permutations((P(X), P(Y), P(Z))),
        ):
            expression = (a * b * c) / (x * y * z)
            with self.subTest(e=str(expression)):
                self.assertEqual(expected, canonicalize(expression, [A, B, C, X, Y, Z]))


class TestSimplify(unittest.TestCase):
    """Tests of the simplification algorithm."""

    def test_simplify_example_1(self):
        """Test the :func:`y0.simplify.simplify` function."""
        self.assertEqual(equation_1, simplify(pre_equation_1, figure_1))
