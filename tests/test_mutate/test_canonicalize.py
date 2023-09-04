# -*- coding: utf-8 -*-

"""Tests for the canonicalization algorithm."""

import itertools as itt
import unittest
from typing import Sequence

from y0.dsl import (
    A,
    B,
    C,
    D,
    Expression,
    Fraction,
    One,
    P,
    Product,
    R,
    Sum,
    Variable,
    W,
    X,
    Y,
    Z,
    Zero,
)
from y0.mutate import canonical_expr_equal, canonicalize
from y0.mutate.canonicalize_expr import Canonicalizer


class TestCanonicalize(unittest.TestCase):
    """Tests for the canonicalization of a simplified algorithm."""

    def assert_canonicalize(
        self, expected: Expression, expression: Expression, ordering: Sequence[Variable]
    ) -> None:
        """Check that the expression is canonicalized properly given an ordering."""
        with self.subTest(
            expr=str(expression),
            ordering=", ".join(variable.name for variable in ordering),
        ):
            actual = canonicalize(expression, ordering)
            self.assertEqual(
                expected,
                actual,
                msg=f"\nExpected: {str(expected)}\nActual:   {str(actual)}",
            )

    def test_invalid_ordering(self):
        """Test raising a value error on duplicates in ordering."""
        with self.assertRaises(ValueError):
            Canonicalizer([A, A, B])

    def test_errors(self):
        """Test errors on types."""
        with self.assertRaises(TypeError):
            canonicalize(5, [A, B])

    def test_atomic(self):
        """Test canonicalization of atomic expressions."""
        for expected, expression, ordering in [
            (One(), One(), []),
            (Zero(), Zero(), []),
            (P(A), P(A), [A]),
            (P(A | B), P(A | B), [A, B]),
            (P(A | (B, C)), P(A | (B, C)), [A, B, C]),
            (P(A | (B, C)), P(A | (C, B)), [A, B, C]),
        ]:
            self.assert_canonicalize(expected, expression, ordering)

        expected = P(A | (B, C, D))
        for b, c, d in itt.permutations((B, C, D)):
            expression = P(A | (b, c, d))
            self.assert_canonicalize(expected, expression, [A, B, C, D])

    def test_atomic_interventions(self):
        """Test canonicalization of atomic expressions containing interventions."""
        for expected, expression, ordering in [
            (P(A @ X), P(A @ X), [A, X]),
            (P(A @ [X, Y]), P(A @ [X, Y]), [A, X, Y]),
            (P(A @ [X, Y]), P(A @ [Y, X]), [A, X, Y]),
        ]:
            self.assert_canonicalize(expected, expression, ordering)

    def test_derived_atomic(self):
        """Test canonicalizing."""
        # self.assert_canonicalize(One(), Sum(One(), ()), ())
        self.assert_canonicalize(One(), Product.safe(One()), ())
        self.assert_canonicalize(One(), Product.safe([One()]), ())
        self.assert_canonicalize(One(), Product((One(), One())), ())
        self.assert_canonicalize(Zero(), Sum.safe(Zero(), (A,)), [A])
        self.assert_canonicalize(Zero(), Product.safe(Zero()), ())
        self.assert_canonicalize(Zero(), Product.safe([Zero()]), ())
        self.assert_canonicalize(Zero(), Product((P(A), Product((P(B), Zero())))), [A, B])
        self.assert_canonicalize(Zero(), Product((Zero(), Zero())), ())
        self.assert_canonicalize(P(A), Product((One(), P(A))), [A])
        self.assert_canonicalize(Zero(), Product((Zero(), One(), P(A))), [A])

        # Sum
        expected = expression = Sum[R](P(A))
        self.assert_canonicalize(expected, expression, [A, R])

        # Single Product
        self.assert_canonicalize(P(A), Product.safe(P(A)), [A])
        self.assert_canonicalize(P(A), Product.safe([P(A)]), [A])

        # Simple product (only atomic)
        expected = P(A) * P(B) * P(C)
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            expression = a * b * c
            self.assert_canonicalize(expected, expression, [A, B, C])

        # Nested product
        expected = P(A) * P(B) * P(C)
        for b, c in itt.permutations((P(B), P(C))):
            expression = Product((P(A), Product((b, c))))
            self.assert_canonicalize(expected, expression, [A, B, C])

            expression = Product((Product((P(A), b)), c))
            self.assert_canonicalize(expected, expression, [A, B, C])

        # Sum with simple product (only atomic)
        expected = Sum[R](P(A) * P(B) * P(C))
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            expression = Sum[R](a * b * c)
            self.assert_canonicalize(expected, expression, [A, B, C, R])

        # Fraction
        expected = expression = P(A) / P(B)
        self.assert_canonicalize(expected, expression, [A, B])

        # Fraction with simple products (only atomic)
        expected = (P(A) * P(B) * P(C)) / (P(X) * P(Y) * P(Z))
        for (a, b, c), (x, y, z) in itt.product(
            itt.permutations((P(A), P(B), P(C))),
            itt.permutations((P(X), P(Y), P(Z))),
        ):
            expression = (a * b * c) / (x * y * z)
            self.assert_canonicalize(expected, expression, [A, B, C, X, Y, Z])

        # Compound fractions
        expr = Fraction(
            numerator=Fraction(numerator=P(A), denominator=P(B)),
            denominator=Fraction(numerator=P(C), denominator=P(D)),
        )
        self.assert_canonicalize(P(A) * P(D) / P(B) / P(C), expr, [A, B, C, D])

        self.assert_canonicalize(P(A), Fraction(P(A), One()), [A])
        self.assert_canonicalize(P(A), Fraction(numerator=P(A), denominator=P(B) / P(B)), [A, B])

    def test_mixed(self):
        """Test mixed expressions."""
        expected = expression = P(A) * Sum[R](P(B))
        self.assert_canonicalize(expected, expression, [A, B, R])

        expected = P(A) * Sum[R](P(B)) * Sum[Y](P(C))
        for a, b, c in itt.permutations((P(A), Sum[R](P(B)), Sum[Y](P(C)))):
            expression = a * b * c
            self.assert_canonicalize(expected, expression, [A, B, C, R, Y])

        expected = P(D) * Sum[R](P(A) * P(B) * P(C))
        for a, b, c in itt.permutations((P(A), P(B), P(C))):
            sum_expr = Sum[R](a * b * c)
            for left, right in itt.permutations((P(D), sum_expr)):
                self.assert_canonicalize(expected, left * right, [A, B, C, D, R])

        expected = P(X) * Sum[Y](P(A) * P(B)) * Sum[Z](P(C) * P(D))
        for (a, b), (c, d) in itt.product(
            itt.permutations((P(A), P(B))),
            itt.permutations((P(C), P(D))),
        ):
            sexpr = Sum[Y](a * b) * Sum[Z](c * d)
            self.assert_canonicalize(expected, sexpr * P(X), [A, B, C, D, X, Y, Z])
            self.assert_canonicalize(expected, P(X) * sexpr, [A, B, C, D, X, Y, Z])

        expected = expression = Sum[R](P(A) / P(B))
        self.assert_canonicalize(expected, expression, [A, B, R])

        expected = expression = Sum[X](P(A) / Sum[W](P(B))) * Sum[Z](P(A) / Sum[Y](P(B) / P(C)))
        self.assert_canonicalize(expected, expression, [A, B, C, W, X, Y, Z])

    def test_non_markov(self):
        """Test non-markov distributions (e.g., with multiple children)."""
        for c1, c2 in itt.permutations([A, B]):
            # No conditions
            self.assert_canonicalize(P(A & B), P(c1 & c2), [A, B])
            # One condition, C
            self.assert_canonicalize(P(A & B | C), P(c1 & c2 | C), [A, B, C])
            # Two conditions, C and D
            for p1, p2 in itt.permutations([C, D]):
                expected = P(A & B | C | D)
                expression = P(c1 & c2 | (p1, p2))
                ordering = [A, B, C, D, R]
                self.assert_canonicalize(expected, expression, ordering)
                self.assert_canonicalize(Sum[R](expected), Sum[R](expression), ordering)

        for c1, c2, c3 in itt.permutations([A, B, C]):
            self.assert_canonicalize(P(A, B, C), P(c1, c2, c3), [A, B, C])
            for p1, p2, p3 in itt.permutations([X, Y, Z]):
                expected = P(A & B & C | (X, Y, Z))
                expression = P(c1 & c2 & c3 | (p1 & p2 & p3))
                ordering = [A, B, C, R, X, Y, Z]
                self.assert_canonicalize(expected, expression, ordering)
                self.assert_canonicalize(Sum[R](expected), Sum[R](expression), ordering)


class TestCanonicalizeEqual(unittest.TestCase):
    """Test the ability of the canonicalize function to check expressions being equal."""

    def test_expr_equal(self):
        """Check that canonicalized expressions are equal."""
        self.assertTrue(canonical_expr_equal(P(X), P(X)))
        self.assertFalse(canonical_expr_equal(P(X), P(Y)))
        self.assertFalse(canonical_expr_equal(P(X @ W), P(X)))
        self.assertFalse(canonical_expr_equal(P(X @ W), P(Y)))

        # Order changes
        self.assertTrue(canonical_expr_equal(P(X & Y), P(Y & X)))
