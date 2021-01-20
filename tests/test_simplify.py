# -*- coding: utf-8 -*-

"""Tests of the simplification algorithm."""

import unittest

from y0.dsl import Variable
from y0.graph import NxMixedGraph

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


class TestSimplify(unittest.TestCase):
    """Tests of the simplification algorithm."""

    def test_simplify(self):
        """Test the :func:`y0.simplify.simplify` function."""
        raise NotImplementedError
