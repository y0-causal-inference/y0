# -*- coding: utf-8 -*-

"""Tests for the CausalEffect wrapper."""

import unittest

from y0.dsl import A, B, C, D, P, Q, Sum, Variable
from y0.examples import verma_1

try:
    from y0.causaleffect import CAUSALEFFECT, IGRAPH, VermaConstraint, r_get_verma_constraints
except ImportError:  # rpy2 is not installed
    missing_rpy2 = True
else:
    missing_rpy2 = False

u_1 = Variable('u_1')


@unittest.skipIf(missing_rpy2, 'rpy2 is not installed')
class TestCausalEffect(unittest.TestCase):
    """Tests for the causaleffect wrapper."""

    @classmethod
    def setUpClass(cls) -> None:
        """Make imports for the class."""
        from rpy2.robjects.packages import importr
        from rpy2.robjects.packages import PackageNotInstalledError

        try:
            importr(CAUSALEFFECT)
            importr(IGRAPH)
        except PackageNotInstalledError:
            raise unittest.SkipTest('R packages not properly installed.')

    def test_verma_constraint(self):
        """Test getting the single Verma constraint from the Figure 1A graph."""
        actual = r_get_verma_constraints(verma_1)
        self.assertEqual(1, len(actual))
        verma_constraint = actual[0]
        self.assertIsInstance(verma_constraint, VermaConstraint)
        self.assertEqual(Q[D](C, D), verma_constraint.rhs_cfactor)

        expected_rhs_expr = Sum[u_1, C](P(D | u_1 | C) * P(C) * P(u_1))
        self.assertEqual(
            expected_rhs_expr,
            verma_constraint.rhs_expr,
            msg=f'Expected: {expected_rhs_expr}\nActual:  {verma_constraint.rhs_expr}'
        )
        self.assertEqual(Sum[B](Q[B, D](A, B, C, D)), verma_constraint.lhs_cfactor)
        self.assertEqual(Sum[B](P(D | (A, B, C)) * P(B | A)), verma_constraint.lhs_expr)
        self.assertEqual((A,), verma_constraint.variables)
