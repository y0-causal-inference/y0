# -*- coding: utf-8 -*-

"""Tests for chain mutations."""

import unittest

from y0.dsl import A, B, P, W, X, Y, Z
from y0.mutate import chain_expand, markov_kernel_to_fraction


class TestChain(unittest.TestCase):
    """Tests for chain mutations."""

    def test_chain_expand_automatic_reordering(self):
        """Test expanding a joint probability to a product of conditional probabilities."""
        self.assertEqual(P(X | Y) * P(Y), chain_expand(P(X, Y), reorder=True))
        self.assertEqual(P(X | (Y, Z)) * P(Y | Z) * P(Z), chain_expand(P(X, Y, Z), reorder=True))
        self.assertEqual(P(W | (X, Y, Z)) * P(X | (Y, Z)) * P(Y | Z) * P(Z), chain_expand(P(W, X, Y, Z), reorder=True))

    def test_chain_expand_no_reordering(self):
        """Test expanding a joint probability to a product of conditional probabilities."""
        self.assertEqual(P(X | Y) * P(Y), chain_expand(P(X, Y), reorder=False))
        self.assertEqual(P(X | (Y, Z)) * P(Y | Z) * P(Z), chain_expand(P(X, Y, Z), reorder=False))
        self.assertEqual(P(W | (X, Y, Z)) * P(X | (Y, Z)) * P(Y | Z) * P(Z), chain_expand(P(W, X, Y, Z), reorder=False))

    def test_bayes_expand(self):
        """Test expanding a conditional probability with Bayes' Theorem."""
        self.assertEqual(P(A, B) / P(B), markov_kernel_to_fraction(P(A | B)))
        self.assertEqual(P(W, X, Y, Z) / P(X, Y, Z), markov_kernel_to_fraction(P(W | (X, Y, Z))))
