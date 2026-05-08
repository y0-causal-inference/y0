"""Tests for ConcreteDistribution — discrete PMF for Kolmogorov axiom verification.

These are hand-written unit tests for the ConcreteDistribution class itself.
The auto-generated Kolmogorov tests in test_dafny_correspondence.py will use
ConcreteDistribution as a fixture.
"""

import unittest

from y0.dsl import Variable
from y0.probability import ConcreteDistribution


class TestConcreteDistribution(unittest.TestCase):
    """Test ConcreteDistribution construction and basic queries."""

    def setUp(self):
        self.A = Variable("A")
        self.B = Variable("B")
        self.dist = ConcreteDistribution.from_random(
            [self.A, self.B], n_values=2, seed=42
        )

    def test_is_valid(self):
        """A randomly generated PMF is a valid distribution."""
        self.assertTrue(self.dist.is_valid())

    def test_prob_event_nonneg(self):
        """All event probabilities are non-negative."""
        for a in range(2):
            for b in range(2):
                p = self.dist.prob_event({self.A: a, self.B: b})
                self.assertGreaterEqual(p, 0.0)

    def test_prob_event_sums_to_one(self):
        """Sum of all atomic probabilities equals 1."""
        total = sum(
            self.dist.prob_event({self.A: a, self.B: b})
            for a in range(2)
            for b in range(2)
        )
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_prob_marginal(self):
        """Marginal P(A=a) = sum_b P(A=a, B=b)."""
        for a in range(2):
            marginal = self.dist.prob_marginal({self.A: a})
            joint_sum = sum(
                self.dist.prob_event({self.A: a, self.B: b})
                for b in range(2)
            )
            self.assertAlmostEqual(marginal, joint_sum, places=10)

    def test_prob_cond(self):
        """P(A=a | B=b) = P(A=a, B=b) / P(B=b)."""
        for a in range(2):
            for b in range(2):
                p_b = self.dist.prob_marginal({self.B: b})
                if p_b > 0:
                    cond = self.dist.prob_cond({self.A: a}, given={self.B: b})
                    joint = self.dist.prob_event({self.A: a, self.B: b})
                    self.assertAlmostEqual(cond, joint / p_b, places=10)

    def test_values(self):
        """values() returns the domain of a variable."""
        vals = self.dist.values(self.A)
        self.assertEqual(vals, [0, 1])

    def test_empty_event(self):
        """P(empty event) = 0 — no rows match an impossible assignment."""
        # An assignment with a value outside the domain
        p = self.dist.prob_event({self.A: 999})
        self.assertAlmostEqual(p, 0.0, places=10)

    def test_intervene(self):
        """Intervening on A=0 produces a valid distribution where P(A=1)=0."""
        intervened = self.dist.intervene({self.A: 0})
        self.assertTrue(intervened.is_valid())
        self.assertAlmostEqual(
            intervened.prob_marginal({self.A: 0}), 1.0, places=10
        )
        self.assertAlmostEqual(
            intervened.prob_marginal({self.A: 1}), 0.0, places=10
        )

    def test_three_vars(self):
        """Construction with 3 variables works correctly."""
        C = Variable("C")
        dist3 = ConcreteDistribution.from_random(
            [self.A, self.B, C], n_values=2, seed=99
        )
        self.assertTrue(dist3.is_valid())
        total = sum(
            dist3.prob_event({self.A: a, self.B: b, C: c})
            for a in range(2)
            for b in range(2)
            for c in range(2)
        )
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_from_dag(self):
        """from_dag produces a valid Markov-compatible distribution."""
        C = Variable("C")
        dist = ConcreteDistribution.from_dag(
            directed_edges=[(self.A, self.B), (self.B, C)],
            variables=[self.A, self.B, C],
            n_values=2,
            seed=42,
        )
        self.assertTrue(dist.is_valid())

    def test_do_graph(self):
        """do_graph produces a valid interventional distribution."""
        dist = ConcreteDistribution.from_dag(
            directed_edges=[(self.A, self.B)],
            variables=[self.A, self.B],
            n_values=2,
            seed=42,
        )
        truncated = dist.do_graph({self.A: 0})
        self.assertTrue(truncated.is_valid())
        # All rows have A=0
        self.assertAlmostEqual(
            truncated.prob_marginal({self.A: 0}), 1.0, places=10
        )

    def test_do_graph_empty(self):
        """do_graph({}) returns the original distribution."""
        dist = ConcreteDistribution.from_dag(
            directed_edges=[(self.A, self.B)],
            variables=[self.A, self.B],
            n_values=2,
            seed=42,
        )
        trivial = dist.do_graph({})
        for a in range(2):
            for b in range(2):
                p_orig = dist.prob_event({self.A: a, self.B: b})
                p_trunc = trivial.prob_event({self.A: a, self.B: b})
                self.assertAlmostEqual(p_orig, p_trunc, places=10)


if __name__ == "__main__":
    unittest.main()
