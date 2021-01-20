# -*- coding: utf-8 -*-

"""Tests of the simplification algorithm."""

import unittest

from y0.dsl import P, Sum, Variable
from y0.graph import NxMixedGraph
from y0.simplify import simplify

#: Corresponds to Figure 1 in https://www.jmlr.org/papers/volume18/16-166/16-166.pdf
figure_1 = NxMixedGraph()
figure_1.add_directed_edge('X', 'Z1')
figure_1.add_directed_edge('Z1', 'Y')
figure_1.add_directed_edge('Z2', 'X')
figure_1.add_directed_edge('Z2', 'Z1')
figure_1.add_directed_edge('Z2', 'Z3')
figure_1.add_directed_edge('Z3', 'Y')
figure_1.add_undirected_edge('X', 'Y')
figure_1.add_undirected_edge('X', 'Z2')
figure_1.add_undirected_edge('X', 'Z3')
figure_1.add_undirected_edge('Y', 'Z2')

X, Y, Z1, Z2, Z3 = (Variable(s) for s in 'X Y Z1 Z2 Z3'.split())

# These are the parts of pre-equation 1 and equation 1. They don't all fit on one
#  line so they're broken up into reusable bits
_a = P(Z1 | (Z2, X))
_b = P(Z3 | Z2)
_y = P(Y | (Z2, X, Z3, Z1))
_z3 = P(Z3 | (Z2, X))
_x = P(X | Z2)
_i = _y * _z3 * _x * P(Z2)
_c = Sum[X](_i)
_d = Sum[X, Y](_i)
_e = Sum[X, Z3, Y](_i)
pre_equation_1 = _a * _b * _c * _e / _d
equation_1 = P(Z1 | (Z2, X)) * P(Z2) * P(Y) * Sum[X](_y * _z3 * _x)


class TestSimplify(unittest.TestCase):
    """Tests of the simplification algorithm."""

    def test_simplify_example_1(self):
        """Test the :func:`y0.simplify.simplify` function."""
        self.assertEqual(equation_1, simplify(pre_equation_1, figure_1))
