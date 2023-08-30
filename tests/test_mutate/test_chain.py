# -*- coding: utf-8 -*-

"""Tests for chain mutations."""

import unittest

from y0.dsl import A, B, P, Sum, W, X, Y, Z
from y0.mutate import bayes_expand, chain_expand, fraction_expand


class TestChain(unittest.TestCase):
    """Tests for chain mutations."""

    def test_chain_expand_automatic_reordering(self):
        """Test expanding a joint probability to a product of conditional probabilities."""
        self.assertEqual(P(X | Y) * P(Y), chain_expand(P(X, Y), reorder=True))
        self.assertEqual(P(X | (Y, Z)) * P(Y | Z) * P(Z), chain_expand(P(X, Y, Z), reorder=True))
        self.assertEqual(
            P(W | (X, Y, Z)) * P(X | (Y, Z)) * P(Y | Z) * P(Z),
            chain_expand(P(W, X, Y, Z), reorder=True),
        )

        # Test that conditions come along for the ride
        self.assertEqual(P(X | (Y, A)) * P(Y | A), chain_expand(P(X & Y | A), reorder=True))
        self.assertEqual(
            P(X | (Y, A, B)) * P(Y | (A, B)),
            chain_expand(P(X & Y | (A, B)), reorder=True),
        )

    def test_chain_expand_no_reordering(self):
        """Test expanding a joint probability to a product of conditional probabilities."""
        self.assertEqual(P(X | Y) * P(Y), chain_expand(P(X, Y), reorder=False))
        self.assertEqual(P(X | (Y, Z)) * P(Y | Z) * P(Z), chain_expand(P(X, Y, Z), reorder=False))
        self.assertEqual(
            P(W | (X, Y, Z)) * P(X | (Y, Z)) * P(Y | Z) * P(Z),
            chain_expand(P(W, X, Y, Z), reorder=False),
        )

    def test_fraction_expand(self):
        """Test expanding a conditional probability with Bayes' Theorem."""
        self.assertEqual(P(A, X), fraction_expand(P(A, X)))
        self.assertEqual(P(A, B) / P(B), fraction_expand(P(A | B)))
        self.assertEqual(P(W, X, Y, Z) / P(X, Y, Z), fraction_expand(P(W | (X, Y, Z))))

    def test_bayes_expand(self):
        """Test expanding a conditional using extended Bayes' Theorem."""
        self.assertEqual(P(A, X), bayes_expand(P(A, X)))
        self.assertEqual(P(A, X) / Sum[A](P(A, X)), bayes_expand(P(A | X)))
        self.assertEqual(P(A, X, Y) / Sum[A](P(A, X, Y)), bayes_expand(P(A | (X, Y))))
        self.assertEqual(P(A, B, X, Y) / Sum[A, B](P(A, B, X, Y)), bayes_expand(P(A & B | (X, Y))))
