"""Test sigma separation."""

import unittest

from y0.algorithm.conditional_independencies import are_d_separated
from y0.algorithm.separation.sigma_separation import (
    are_sigma_separated,
    get_equivalence_classes,
    is_collider,
    is_non_collider_fork,
    is_non_collider_left_chain,
    is_non_collider_right_chain,
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
        (V4, V8),
        (V5, V2),
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

    def setUp(self) -> None:
        """Set up the test case."""
        self.sigma = get_equivalence_classes(graph)

    def test_equivalence_classes(self):
        """Test getting equivalence classes."""
        equivalent_classes = {
            frozenset([V1]),
            frozenset([V2, V3, V4, V5]),
            frozenset([V6, V7]),
            frozenset([V8]),
        }
        expected_equivalent_classes = {n: c for c in equivalent_classes for n in c}
        self.assertEqual(expected_equivalent_classes, self.sigma)

    def test_collider(self):
        """Test checking colliders."""
        self.assertTrue(is_collider(graph, left=V4, middle=V5, right=V4, conditions={V3, V5}))

    def test_left_chain(self):
        """Test checking non-colliders (left chain)."""
        self.assertTrue(
            is_non_collider_left_chain(
                graph, left=V5, middle=V4, right=V6, conditions={V3, V5}, sigma=self.sigma
            )
        )

    def test_right_chain(self):
        """Test checking non-colliders (right chain)."""
        self.assertTrue(
            is_non_collider_right_chain(
                graph, left=V1, middle=V2, right=V3, conditions={V3, V5}, sigma=self.sigma
            )
        )
        self.assertTrue(
            is_non_collider_right_chain(
                graph, left=V2, middle=V3, right=V4, conditions={V3, V5}, sigma=self.sigma
            )
        )
        self.assertTrue(
            is_non_collider_right_chain(
                graph, left=V3, middle=V4, right=V5, conditions={V3, V5}, sigma=self.sigma
            )
        )

    def test_fork(self):
        """Test checking non-colliders (fork)."""
        self.assertTrue(
            is_non_collider_fork(
                graph, left=V5, middle=V4, right=V8, conditions={V3, V5}, sigma=self.sigma
            )
        )

    def test_z_sigma_open(self):
        """Tests for z-sigma-open paths."""
        # this is a weird example since it backtracks
        path = [V1, V2, V3, V4, V5, V4, V6]
        self.assertFalse(is_z_sigma_open(graph, path, sigma=self.sigma))
        self.assertTrue(is_z_sigma_open(graph, path, conditions={V3, V5}, sigma=self.sigma))

    def test_separations_figure_3(self):
        """Test comparisons of d-separation and sigma-separation."""
        for left, right, conditions, d, s in [
            (V2, V4, [V3, V5], True, False),
            (V1, V6, [], True, True),
            (V1, V6, [V3, V5], True, False),
            (V1, V8, [], False, False),
            (V1, V8, [V3, V5], True, False),
            (V1, V8, [V4], True, True),
        ]:
            with self.subTest(left=left, right=right, conditions=conditions):
                self.assertEqual(
                    d, are_d_separated(graph, left, right, conditions=conditions).separated
                )
                self.assertEqual(s, are_sigma_separated(graph, left, right, conditions=conditions))
