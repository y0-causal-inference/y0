# -*- coding: utf-8 -*-

"""Tests for the identify algorithm."""

import unittest

from y0.algorithm.identify import identify
from y0.algorithm.taheri_design import _get_result, iterate_lvdags
from y0.dsl import Expression, P, Sum, X, Y, Z, W1, W2, Y1, Y2
from y0.graph import DEFAULT_TAG, NxMixedGraph, admg_to_latent_variable_dag
from y0.mutate import canonicalize
from y0.parser import parse_craig
from y0.resources import VIRAL_PATHOGENESIS_PATH

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
        """Test Figure 2A.
        Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy. Journal of Machine Learning Research.
        """
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Y")
        print(identify(graph, Y @ X).to_text())
        expr = "[ sum_{} P(Y|X) ]"
        frac_expr = P_XY / Sum[Y](P_XY)
        self.assert_identify(parse_craig(expr), graph, Y @ X)

    def test_figure_2b(self):
        """Test Figure 2B.
        Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy. Journal of Machine Learning Research.
        """
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Y")
        graph.add_directed_edge("X", "Z")
        graph.add_directed_edge("Z", "Y")
        graph.add_undirected_edge("Y", "Z")
        print(identify(graph, Y @ X).to_text())
        expr = "[ sum_{Z} P(Z|X) P(Y|X,Z) ]"
        cond_expr = Sum[Z](P(Z | X) * P(Y | X, Z))
        frac_expr = Sum[Z](
            Sum[Y](P_XY) / (Sum[Z](Sum[Y](P_XY))) * (P_XY / Sum[Y](P_XY))
        )
        self.assert_identify(parse_craig(expr), graph, Y @ X)

    def test_figure_2c(self):
        """Test Figure 2C.
        Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy. Journal of Machine Learning Research.
        """
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Y")
        graph.add_directed_edge("Z", "X")
        graph.add_directed_edge("Z", "Y")
        graph.add_undirected_edge("Y", "Z")
        print(identify(graph, Y @ X).to_text())
        expr = "[ sum_{Z} P(Z) P(Y|X,Z) ]"
        cond_expr = Sum[Z](P(Z) * P(Y | X, Z))
        frac_expr = Sum[Z](
            Sum[X, Y](P_XYZ) / (Sum[Z](Sum[X, Y](P_XYZ))) * (P_XYZ / Sum[Y](P_XYZ))
        )
        self.assert_identify(parse_craig(expr), graph, Y @ X)
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

        expr = "[ sum_{Z} [ sum_{} P(Y|X,Z) ] [ sum_{} [ sum_{X,Y} P(X,Y,Z) ] ] ]"
        self.assert_identify(parse_craig(expr), graph, Y @ X)
        # self.assert_identify(grammar.parseString(expr)[0], graph, Y@X)

    def test_figure_2e(self):
        """Test Figure 2E.
        Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy. Journal of Machine Learning Research.
        """
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Z")
        graph.add_directed_edge("Z", "Y")
        graph.add_undirected_edge("X", "Y")
        expr = "[ sum_{Z} [ sum_{} P(Z|X) ] [ sum_{} [ sum_{X} P(X) P(Y|X,Z) ] ] ]"
        cond_expr = Sum[Z](P(Z | X)) * Sum[X](P(X) * P(Y | X, Z))
        frac_expr = Sum[Z](Sum[Y](P_XYZ) / Sum[Z](Sum[Y](P_XYZ))) * Sum[X](
            P_XYZ * Sum[Y, Z](P_XYZ) / Sum[Y](P_XYZ) / Sum[X](Sum[Y, Z](P_XYZ))
        )
        self.assert_identify(parse_craig(expr), graph, Y @ X)
        # self.assert_identify(grammar.parseString(expr)[0], graph, Y@X)

    def test_figure_3a(self):
        """Test Figure 3a. A graph hedge-less for P(y1,y2|do(x))
        Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy. Journal of Machine Learning Research.
        """
        graph = NxMixedGraph()
        #W1,W2,Y1,Y2 = Variable('W1'), Variable('W2'), Variable('Y1'), Variable('Y2')
        graph.add_directed_edge("X", "Y1")
        graph.add_directed_edge("W1", "X")
        graph.add_directed_edge("W2", "Y2")
        graph.add_undirected_edge("W1", "W2")
        graph.add_undirected_edge("W1", "Y1")
        graph.add_undirected_edge("W1", "Y2")
        graph.add_undirected_edge("X", "W2")
        cond_expr = Sum[W2](P(Y1,W2))*Sum[W1](P(Y1|(X,W1))*P(W1))
        self.assert_identify(cond_expr, graph, P(Y1 @ X, Y2 @ X))
    def test_taheri(self):
        """Test that all graphs produced by Sara's design algorithm can be run with :func:`identify`."""
        graph = NxMixedGraph.from_causalfusion_path(VIRAL_PATHOGENESIS_PATH)

        cause = 'EGFR'
        effect = 'CytokineStorm'
        stop = 5
        tag = DEFAULT_TAG
        dag = admg_to_latent_variable_dag(graph.to_admg(), tag=tag)
        fixed_latent = {
            node
            for node, data in dag.nodes(data=True)
            if data[tag]
        }
        for latents, observed, lvdag in iterate_lvdags(
            dag,
            fixed_observed={cause, effect},
            fixed_latents=fixed_latent,
            stop=stop,
        ):
            with self.subTest(latents=latents):
                result = _get_result(
                    lvdag=lvdag,
                    latents=latents,
                    observed=observed,
                    cause=cause,
                    effect=effect,
                )
                self.assertIsNotNone(result)  # throwaway test


if __name__ == "__main__":
    unittest.main()
