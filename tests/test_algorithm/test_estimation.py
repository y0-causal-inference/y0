import unittest

from y0.algorithm.estimation import is_mb_shielded


class TestEstimation(unittest.TestCase):
    def test_is_mb_shielded(self):
        graph = ...
        self.assertTrue(is_mb_shielded(graph))
        graph_2 = ...
        self.assertFalse(is_mb_shielded(graph_2))
        # TODO: Come up with test cases
