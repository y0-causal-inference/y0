# -*- coding: utf-8 -*-

"""Tests for the CausalEffect wrapper."""

import unittest

from y0.graph import figure_1

try:
    from y0.causaleffect import CAUSALEFFECT, IGRAPH, VermaConstraint, r_get_verma_constraints
except ImportError:  # rpy2 is not installed
    missing_rpy2 = True
else:
    missing_rpy2 = False


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
        actual = r_get_verma_constraints(figure_1)
        self.assertEqual(1, len(actual))
        self.assertEqual(
            VermaConstraint(
                rhs_cfactor="Q[\\{D\\}](C,D)",
                rhs_expr="\\sum_{u_{1},C}P(D|u_{1},C)P(C)P(u_{1})",
                lhs_cfactor="\\sum_{B}Q[\\{B,D\\}](A,B,C,D)",
                lhs_expr="\\sum_{B}P(D|A,B,C)P(B|A)",
                variables="A",
            ),
            actual[0],
        )
