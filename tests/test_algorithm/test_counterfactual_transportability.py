"""Tests for counterfactualy transportabiltiy."""

import unittest

from y0.algorithm.counterfactual_transportability import simplify
from y0.dsl import X, Y


class TestSimplify(unittest.TestCase):
    """Test the simplify algorithm from counterfactual transportability."""

    def test_inconsistent(self):
        """Test simplifying an inconsistent event."""
        event = [
            (Y @ -X, +Y),
            (Y @ -X, -Y),
        ]
        result = simplify(event)
        self.assertIsNone(result)
