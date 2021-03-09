import unittest

from y0.algorithm.falsification import falsifications
from y0.examples import asia_example


class TestFalsification(unittest.TestCase):
    def test_asia(self):
        G = asia_example.graph.to_admg()
        df = asia_example.data
        issues = falsifications(G, df)
        self.assertEqual(0, len(issues))
        self.assertGreater(len(issues.evidence), 0)
