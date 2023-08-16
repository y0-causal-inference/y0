# -*- coding: utf-8 -*-

"""Tests for the CausalEffect wrapper."""

import unittest

import pyparsing

from y0.dsl import P, Q, Sum, Variable
from y0.examples import examples, verma_1

try:
    from y0.causaleffect import r_get_verma_constraints
    from y0.r_utils import CAUSALEFFECT, IGRAPH
    from y0.struct import VermaConstraint
except ImportError:  # rpy2 is not installed
    missing_rpy2 = True
else:
    missing_rpy2 = False

u_1 = Variable("u_1")
V1 = Variable("V1")
V2 = Variable("V2")
V3 = Variable("V3")
V4 = Variable("V4")


@unittest.skipIf(missing_rpy2, "rpy2 is not installed")
class TestCausalEffect(unittest.TestCase):
    """Tests for the causaleffect wrapper."""

    @classmethod
    def setUpClass(cls) -> None:
        """Make imports for the class."""
        from rpy2.robjects.packages import PackageNotInstalledError, importr

        try:
            importr(CAUSALEFFECT)
            importr(IGRAPH)
        except PackageNotInstalledError:
            raise unittest.SkipTest("R packages not properly installed.")

    def test_verma_constraint(self):
        """Test getting the single Verma constraint from the Figure 1A graph."""
        for example in examples:
            with self.subTest(name=example.name):
                try:
                    actual = r_get_verma_constraints(example.graph)
                except pyparsing.ParseException:
                    continue
                expected = example.verma_constraints
                self.assertEqual(set(expected or ()), set(actual))

        actual = r_get_verma_constraints(verma_1)
        self.assertEqual(1, len(actual))
        verma_constraint = actual[0]
        self.assertIsInstance(verma_constraint, VermaConstraint)
        self.assertEqual(Q[V4](V3, V4), verma_constraint.rhs_cfactor)

        expected_rhs_expr = Sum[u_1, V3](P(V4 | u_1 | V3) * P(V3) * P(u_1))
        self.assertEqual(
            expected_rhs_expr,
            verma_constraint.rhs_expr,
            msg=f"Expected: {expected_rhs_expr}\nActual:  {verma_constraint.rhs_expr}",
        )
        self.assertEqual(Sum[V2](Q[V2, V4](V1, V2, V3, V4)), verma_constraint.lhs_cfactor)
        self.assertEqual(Sum[V2](P(V4 | (V1, V2, V3)) * P(V2 | V1)), verma_constraint.lhs_expr)
        self.assertEqual((V1,), verma_constraint.variables)
