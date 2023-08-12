import unittest

from y0.algorithm.estimation import is_markov_blanket_shielded
from y0.graph import NxMixedGraph


class TestEstimation(unittest.TestCase):
    def test_is_mb_shielded(self):
        # Adapted from https://gitlab.com/causal/ananke/-/blob/dev/tests/estimation/test_automated_if.py#L80-92
        graph_unshielded = NxMixedGraph.from_str_edges(
            directed=[("T", "M"), ("M", "L"), ("L", "Y")], undirected=[("M", "Y")]
        )
        # use Ananke method for sanity check
        self.assertFalse(graph_unshielded.to_admg().mb_shielded())
        # test our method
        self.assertFalse(is_markov_blanket_shielded(graph_unshielded))

        # TODO: Come up with true test case
        # graph = ...
        # self.assertTrue(is_markov_blanket_shielded(graph))
