"""Test sigma separation."""

import unittest

from y0.algorithm.conditional_independencies import are_d_separated
from y0.algorithm.sigma_separation import (
    are_sigma_separated,
    get_equivalence_classes,
    is_z_sigma_open,
)
from y0.dsl import V1, V2, V3, V4, V5, V6, Variable
from y0.graph import NxMixedGraph

V7, V8 = map(Variable, ["V7", "V8"])

#: Figure 3 from https://arxiv.org/abs/1807.03024
graph = NxMixedGraph.from_edges(
    directed=[
        (V1, V2),
        (V2, V3),
        (V3, V4),
        (V4, V5),
        (V5, V2),
        (V5, V8),
        (V6, V7),
        (V7, V6),
    ],
    undirected=[
        (V1, V2),
        (V4, V6),
        (V4, V7),
        (V6, V7),
    ],
)


class TestSigmaSeparation(unittest.TestCase):
    """Test sigma separation.

    These tests come from Table 1 in https://arxiv.org/abs/1807.03024.
    The sigma equivalence classes in Figure 3 are {v1}, {v2, v3, v4, v5},
    {v6, v7}, and {v8}.
    """

    def test_equivalence_classes(self):
        """Test getting equivalence classes."""
        equivalent_classes = {
            frozenset([V1]),
            frozenset([V2, V3, V4, V5]),
            frozenset([V6, V7]),
            frozenset([V8]),
        }
        expected_equivalent_classes = {n: c for c in equivalent_classes for n in c}
        actual_equivalent_classes = get_equivalence_classes(graph)
        self.assertEqual(expected_equivalent_classes, actual_equivalent_classes)

    def test_z_sigma_open(self):
        """Tests for z-sigma-open paths."""
        equivalent_classes = get_equivalence_classes(graph)

        # this is a weird example since it backtracks
        path = [V1, V2, V3, V4, V5, V4, V6]
        self.assertTrue(is_z_sigma_open(graph, path, conditions={V3, V5}, sigma=equivalent_classes))

    def test_separations_figure_3(self):
        """Test comparisons of d-separation and sigma-separation."""
        self.assertTrue(are_d_separated(graph, V2, V4, conditions=[V3, V5]))
        self.assertFalse(are_sigma_separated(graph, V2, V4, conditions=[V3, V5]))

        self.assertTrue(are_d_separated(graph, V1, V6))
        self.assertTrue(are_sigma_separated(graph, V1, V6))

        self.assertTrue(are_d_separated(graph, V1, V6, conditions=[V3, V5]))
        self.assertFalse(are_sigma_separated(graph, V1, V6, conditions=[V3, V5]))

        self.assertFalse(are_d_separated(graph, V1, V8))
        self.assertFalse(are_sigma_separated(graph, V1, V8))

        self.assertTrue(are_d_separated(graph, V1, V8, conditions=[V3, V5]))
        self.assertFalse(are_sigma_separated(graph, V1, V8, conditions=[V3, V5]))

        self.assertTrue(are_d_separated(graph, V1, V8, conditions=[V4]))
        self.assertTrue(are_sigma_separated(graph, V1, V8, conditions=[V4]))
