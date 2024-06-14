# -*- coding: utf-8 -*-

"""Test getting Verma constraints."""

import unittest
from typing import Set

from y0.algorithm.verma_constraints import VermaConstraint, get_verma_constraints
from y0.dsl import P, Sum, Variable, X, Y
from y0.examples import napkin
from y0.graph import NxMixedGraph

Z1, Z2 = map(Variable, ("Z1", "Z2"))


class TestVermaConstraints(unittest.TestCase):
    """Test getting Verma constraints."""

    def assert_verma_constraints(self, graph: NxMixedGraph, expected: Set[VermaConstraint]):
        """Assert that the graph has the correct conditional independencies."""
        verma_constraints = get_verma_constraints(graph)
        self.assertTrue(
            all(verma_constraint.is_canonical for verma_constraint in verma_constraints),
            msg="one or more of the returned VermaConstraint instances are not canonical",
        )
        self.assertEqual(expected, verma_constraints)

    def test_napkin(self):
        """Test getting Verma constraints on the napkin graph."""
        # TODO how is Q[Y](Y, X) <-> V1 represented as an expression?
        # TODO it also can be represented as r1:
        r1 = Sum[Z2](P(Y | (Z2 | Z1, X)) * P(X | (Z1, Z2)) * P(Z2)) / Sum[Z2](
            P(X | (Z2, Z1) * P(Z2))
        )

        c1 = VermaConstraint(..., ("R",))
        self.assert_verma_constraints(napkin, {c1})
