# -*- coding: utf-8 -*-

"""Tests for the identify algorithm."""

import unittest

from y0.algorithm.identify import identify  # does not exist yet
from y0.dsl import Expression, P, Sum, X, Y, Z
from y0.graph import NxMixedGraph
from y0.mutate import canonicalize
from y0.parser import parse_craig
P_XY = P(X, Y)
P_XYZ = P(X, Y, Z)


class TestIdentify(unittest.TestCase):
    """Test cases from https://github.com/COVID-19-Causal-Reasoning/Y0/blob/master/ID_whittemore.ipynb."""

    def assert_expr_equal(self, expected: Expression, actual: Expression):
        """Assert that two expressions are the same"""
        expected_vars = expected.get_variables()
        self.assertEqual(expected_vars, actual.get_variables())
        ordering = list(expected_vars)
        expected_canonical = canonicalize(expected, ordering)
        actual_canonical = canonicalize(actual, ordering)
        self.assertEqual(
            expected_canonical,
            actual_canonical,
            msg=f"\nExpected: {str(expected_canonical)}\nActual:   {str(actual_canonical)}",
        )

    def assert_identify(
        self, expression: Expression, graph: NxMixedGraph, query: Expression
    ):
        """Assert that the graph returns the same."""
        self.assert_expr_equal(expression, identify(graph, query))

    def test_figure_2a(self):
        """Test Figure 2A."""
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Y")
        print(identify(graph, Y @ X).to_text())
        expr = "[ sum_{} P(Y|X) ]"
        frac_expr = P_XY / Sum[Y](P_XY)
        self.assert_identify(parse_craig(expr), graph, Y @ X)

    def test_figure_2b(self):
        """Test Figure 2B."""
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Y")
        graph.add_directed_edge("X", "Z")
        graph.add_directed_edge("Z", "Y")
        graph.add_undirected_edge("Y", "Z")
        print(identify(graph, Y @ X).to_text())
        expr = "[ sum_{Z} P(Z|X) P(Y|X,Z) ]"
        cond_expr = Sum[Z](P(Z|X)*P(Y|X,Z))
        frac_expr = Sum[Z](Sum[Y](P_XY) / (Sum[Z](Sum[Y](P_XY))) * (P_XY / Sum[Y](P_XY)))
        self.assert_identify(
            parse_craig(expr),
            graph, Y @ X)

    def test_figure_2c(self):
        """Test Figure 2C."""
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Y")
        graph.add_directed_edge("Z", "X")
        graph.add_directed_edge("Z", "Y")
        graph.add_undirected_edge("Y", "Z")
        print(identify(graph, Y @ X).to_text())
        expr = "[ sum_{Z} P(Z) P(Y|X,Z) ]"
        cond_expr = Sum[Z](P(Z)* P(Y|X,Z))
        frac_expr = Sum[Z](
                Sum[X, Y](P_XYZ) / (Sum[Z](Sum[X, Y](P_XYZ))) * (P_XYZ / Sum[Y](P_XYZ))
            )
        self.assert_identify(
            parse_craig(expr),
            graph, Y @ X)
        # self.assert_identify(grammar.parseString(expr)[0], graph, Y@X)

    def test_figure_2d(self):
        """Test Figure 2D.
        frac_expr = Sum[Z](Sum[X, Y](P_XYZ) * P_XYZ / Sum[Y](P_XYZ))
        """
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Y")
        graph.add_directed_edge("Z", "X")
        graph.add_directed_edge("Z", "Y")
        graph.add_undirected_edge("X", "Z")
        print(identify(graph, Y @ X).to_text())

        expr = '[ sum_{Z} [ sum_{} P(Y|X,Z) ] [ sum_{} [ sum_{X,Y} P(X,Y,Z) ] ] ]'
        self.assert_identify(
            parse_craig( expr ),
            graph, Y @ X)
        # self.assert_identify(grammar.parseString(expr)[0], graph, Y@X)

    def test_figure_2e(self):
        """Test Figure 2E."""
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Z")
        graph.add_directed_edge("Z", "Y")
        graph.add_undirected_edge("X", "Y")
        expr = '[ sum_{Z} [ sum_{} P(Z|X) ] [ sum_{} [ sum_{X} P(X) P(Y|X,Z) ] ] ]'
        cond_expr =             Sum[Z](P(Z|X))*Sum[X](P(X)*P(Y|X,Z))
        frac_expr = (
                Sum[Z](Sum[Y](P_XYZ) / Sum[Z](Sum[Y](P_XYZ)))
                * Sum[X](
                    P_XYZ * Sum[Y, Z](P_XYZ) / Sum[Y](P_XYZ) / Sum[X](Sum[Y, Z](P_XYZ))
                )
            )
        self.assert_identify(
            parse_craig(expr),
            graph, Y @ X)
        # self.assert_identify(grammar.parseString(expr)[0], graph, Y@X)


if __name__ == "__main__":
    unittest.main()
