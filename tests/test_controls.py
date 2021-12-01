# -*- coding: utf-8 -*-

"""Tests for good, bad, and neutral controls."""

import unittest

from y0.controls import is_bad_control, is_good_control
from y0.dsl import U1, U2, M, P, Variable, X, Y, Z
from y0.graph import NxMixedGraph

U = Variable("U")
query = P(Y @ X)

model_1 = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y), (X, Y)])
model_2 = NxMixedGraph.from_edges(directed=[(U, Z), (Z, X), (X, Y), (U, Y)])
model_3 = NxMixedGraph.from_edges(directed=[(U, X), (U, Z), (Z, Y), (X, Y)])
model_4 = NxMixedGraph.from_edges(directed=[(Z, X), (Z, M), (X, M), (M, Y)])
model_5 = NxMixedGraph.from_edges(directed=[(U, Z), (Z, X), (U, M), (X, M), (M, Y)])
model_6 = NxMixedGraph.from_edges(directed=[(U, X), (U, Z), (Z, M), (X, M), (M, Y)])

good_test_models = [
    model_1,
    model_2,
    model_3,
    model_4,
    model_5,
    model_6,
]

# M-bias
model_7 = NxMixedGraph.from_edges(directed=[(U1, Z), (U2, Z), (U1, X), (U2, Y), (X, Y)])
# Bias amplification
model_10 = NxMixedGraph.from_edges(directed=[(Z, X), (U, X), (U, Y), (X, Y)])
#
model_11 = NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)])
model_11_variation = NxMixedGraph.from_edges(directed=[(X, Z), (U, Z), (Z, Y), (U, Y)])
model_12 = NxMixedGraph.from_edges(directed=[(X, M), (M, Y), (M, Z)])
# Selection bias
model_16 = NxMixedGraph.from_edges(directed=[(X, Z), (U, Z), (U, Y), (X, Y)])
model_17 = NxMixedGraph.from_edges(directed=[(X, Z), (Y, Z), (X, Y)])
# case-control bias
model_18 = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])

bad_test_models = [
    model_7,
    model_10,
    model_11,
    model_11_variation,
    model_12,
    model_16,
    model_17,
    model_18,
]


class TestControls(unittest.TestCase):
    """Test case for good, bad, and neutral controls."""

    def test_good_controls(self):
        """Test good controls."""
        for model in good_test_models:
            with self.subTest():
                self.assertTrue(is_good_control(model, query, Z))
        for model in bad_test_models:
            with self.subTest():
                self.assertFalse(is_good_control(model, query, Z))

        # TODO need alternative negative examples

    def test_bad_controls(self):
        """Test bad controls."""
        for model in good_test_models:
            with self.subTest():
                self.assertFalse(is_bad_control(model, query, Z))
        for model in bad_test_models:
            with self.subTest():
                self.assertTrue(is_bad_control(model, query, Z))

        # TODO need alternative negative examples
