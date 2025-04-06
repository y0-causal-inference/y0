"""Tests for good, bad, and neutral controls."""

import unittest

from y0.controls import (
    is_bad_control,
    is_good_control,
    is_middle_mediator,
    is_outcome_ancestor,
)
from y0.dsl import U1, U2, M, U, W, X, Y, Z, Variable
from y0.graph import NxMixedGraph

model_1 = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y), (X, Y)])
model_2 = NxMixedGraph.from_edges(directed=[(U, Z), (Z, X), (X, Y), (U, Y)])
model_3 = NxMixedGraph.from_edges(directed=[(U, X), (U, Z), (Z, Y), (X, Y)])
model_4 = NxMixedGraph.from_edges(directed=[(Z, X), (Z, M), (X, M), (M, Y)])
model_5 = NxMixedGraph.from_edges(directed=[(U, Z), (Z, X), (U, M), (X, M), (M, Y)])
model_6 = NxMixedGraph.from_edges(directed=[(U, X), (U, Z), (Z, M), (X, M), (M, Y)])

good_test_models = [
    ("good model 1", model_1),
    ("good model 2", model_2),
    ("good model 3", model_3),
    ("good model 4", model_4),
    ("good model 5", model_5),
    ("good model 6", model_6),
]

# bad control, M-bias
model_7 = NxMixedGraph.from_edges(directed=[(U1, Z), (U2, Z), (U1, X), (U2, Y), (X, Y)])
# bad control, Bias amplification
model_10 = NxMixedGraph.from_edges(directed=[(Z, X), (U, X), (U, Y), (X, Y)])
# bad control
model_11 = NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)])
model_11_variation = NxMixedGraph.from_edges(directed=[(X, Z), (U, Z), (Z, Y), (U, Y)])
model_12 = NxMixedGraph.from_edges(directed=[(X, M), (M, Y), (M, Z)])
# bad control, Selection bias
model_16 = NxMixedGraph.from_edges(directed=[(X, Z), (U, Z), (U, Y), (X, Y)])
model_17 = NxMixedGraph.from_edges(directed=[(X, Z), (Y, Z), (X, Y)])
# bad control, case-control bias
model_18 = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])

bad_test_models = [
    ("bad model 7", model_7),
    ("bad model 10", model_10),
    ("bad model 11", model_11),
    ("bad model 11 (variation)", model_11_variation),
    ("bad model 12", model_12),
    ("bad model 16", model_16),
    ("bad model 17", model_17),
    ("bad model 18", model_18),
]

# neutral control, possibly good for precision
model_8 = NxMixedGraph.from_edges(directed=[(X, Y), (Z, Y)])
# neutral control, possibly bad for precision
model_9 = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y)])
# neutral control, possibly good for precision
model_13 = NxMixedGraph.from_edges(directed=[(X, W), (Z, W), (W, Y)])
# neutral control, possibly helpful in the case of selection bias
model_14 = NxMixedGraph.from_edges(directed=[(X, Y), (X, Z)])
model_15 = NxMixedGraph.from_edges(directed=[(X, Z), (Z, W), (X, Y), (U, W), (U, Y)])

neutral_test_models = [
    ("neutral model 8", model_8),
    ("neutral model 9", model_9),
    ("neutral model 13", model_13),
    ("neutral model 14", model_14),
    ("neutral model 15", model_15),
]


class TestControls(unittest.TestCase):
    """Test case for good, bad, and neutral controls."""

    def test_preconditions(self):
        """Test the preconditions are checked properly for good controls."""
        irrelevant_variable = Variable("nopenopennope")
        for func in is_good_control, is_bad_control:
            with self.subTest(name=func.__name__):
                with self.assertRaises(ValueError):
                    func(model_1, X, Y, irrelevant_variable)

    def test_good_controls(self):
        """Test good controls."""
        for name, model in good_test_models:
            with self.subTest(name=name):
                self.assertTrue(is_good_control(model, X, Y, Z))
        for name, model in bad_test_models + neutral_test_models:
            with self.subTest(name=name):
                self.assertFalse(is_good_control(model, X, Y, Z))

    def test_bad_controls(self):
        """Test bad controls."""
        for name, model in good_test_models + neutral_test_models:
            with self.subTest(name=name):
                self.assertFalse(is_bad_control(model, X, Y, Z))
        for name, model in bad_test_models:
            with self.subTest(name=name):
                self.assertTrue(is_bad_control(model, X, Y, Z))

    def test_outcome_ancestors(self):
        """Test outcome ancestors."""
        self.assertTrue(is_outcome_ancestor(model_8, X, Y, Z))
        for name, model in good_test_models + bad_test_models:
            with self.subTest(name=name):
                self.assertFalse(is_outcome_ancestor(model, X, Y, Z))

    def test_middle_mediators(self):
        """Test middle mediators."""
        self.assertTrue(is_middle_mediator(model_13, X, Y, Z))
        for name, model in good_test_models + bad_test_models:
            with self.subTest(name=name):
                self.assertFalse(is_middle_mediator(model, X, Y, Z))
