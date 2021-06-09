# -*- coding: utf-8 -*-

"""Tests for the identify algorithm."""

import unittest

from y0.algorithm.identify import ancestors_and_self, identify, line_1, line_2, line_3, line_4, line_5, line_6, line_7, get_c_components, outcomes_and_treatments_to_query, query_to_outcomes_and_treatments
from y0.dsl import Expression, P, Sum, X, Y, Z
from y0.examples import line_1_example, line_2_example, line_3_example, line_4_example, Identification
from y0.graph import NxMixedGraph
from y0.identify import _get_outcomes, _get_treatments
from y0.mutate import canonicalize, expr_equal

P_XY = P(X, Y)
P_XYZ = P(X, Y, Z)



class TestIdentify(unittest.TestCase):
    """Test cases from https://github.com/COVID-19-Causal-Reasoning/Y0/blob/master/ID_whittemore.ipynb."""

    def assert_expr_equal(self, expected: Expression, actual: Expression):
        """Assert that two expressions are the same"""
        expected_outcomes, expected_treatments = query_to_outcomes_and_treatments( query=expected )
        actual_outcomes, actual_treatments = query_to_outcomes_and_treatments( query = actual )
        self.assertEqual(expected_treatments, actual_treatments)
        self.assertEqual(expected_outcomes, actual_outcomes)
        ordering = list(expected.get_variables())
        expected_canonical = canonicalize(expected, ordering)
        actual_canonical = canonicalize(actual, ordering)
        self.assertEqual(
            str(expected_canonical),
            str(actual_canonical),
            msg=f"\nExpected: {str(expected_canonical)}\nActual:   {str(actual_canonical)}",
        )

    def assert_identify(
        self, expression: Expression, graph: NxMixedGraph, query: Expression
    ):
        """Assert that the graph returns the same."""
        self.assert_expr_equal(expression, identify(graph, query))

    def assert_identification_equal( self, expected: Identification, actual: Identification ):
        """Assert that the recursive call to ID has the correct input parameters"""

        self.assert_expr_equal(expected.query, actual.query)
        self.assert_expr_equal(expected.estimand, actual.estimand)
        self.assertEqual(expected.graph, actual.graph)

    def test_ancestors_and_self(self):
        """Tests whether the ancestors_and_self actually returns the ancestors and itself"""
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Z")
        graph.add_directed_edge("Z", "Y")
        graph.add_undirected_edge("X", "Y")
        self.assertEqual({"X", "Y", "Z"}, ancestors_and_self(graph, {"Y"}))
        self.assertEqual({"X", "Z"}, ancestors_and_self(graph, {"Z"}))
        self.assertEqual({"X"}, ancestors_and_self(graph, {"X"}))

    def test_subgraph(self):
        """Tests whether the subgraph restriction algorithm returns the correct subgraph"""
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Z")
        graph.add_directed_edge("Z", "Y")
        graph.add_undirected_edge("Z", "Y")

    def test_get_c_components(self):
        """Tests that get_c_components works correctly"""
        g1 = NxMixedGraph().from_edges(directed=[('X','Y'), ('Z','X'), ('Z','Y')])
        c1 = [frozenset('X'), frozenset('Y'), frozenset('Z')]
        g2 = NxMixedGraph().from_edges(directed=[('X','Y')], undirected=[('X','Y')])
        c2 = [frozenset(['X', 'Y'])]
        g3 = NxMixedGraph().from_edges(directed=[('X','M'),("M","Y")], undirected=[('X','Y')])
        c3 = [frozenset(['X','Y']), frozenset('M')]
        for g, c in [(g1,c1), (g2,c2), (g3,c3)]:
            self.assertEqual(c, get_c_components(g))



    def test_line_1(self):
        r"""Test that line 1 of ID algorithm works correctly.

        If no action has been taken, the effect on :math:`\mathbf Y` is just the marginal of
        the observational distribution :math:`P(\mathbf v)` on :math:`\mathbf Y`.
        """
        for identification in line_1_example.identifications:
            self.assert_expr_equal(
                expected = identification['id_out'][0].estimand,
                actual   = line_1(
                    outcomes=set(_get_outcomes(identification['id_in'][0].query.get_variables())),
                    treatments=set(_get_treatments(identification['id_in'][0].query.get_variables())),
                    estimand=identification['id_in'][0].estimand,
                    G=identification['id_in'][0].graph,
                )
            )

    def test_line_2(self):
        r"""Test line 2 of the identification algorithm.

        2. If we are interested in the effect on :math:`\mathbf Y`, it is sufficient to restrict our
        attention on the parts of the model ancestral to :math:`\mathbf Y`.
        """
        for identification in line_2_example.identifications:
            self.assert_identification_equal(
                expected = identification['id_out'][0],
                actual  = line_2(
                    outcomes   = set(_get_outcomes(identification['id_in'][0].query.get_variables())),
                    treatments = set(_get_treatments(identification['id_in'][0].query.get_variables())),
                    estimand   = identification['id_in'][0].estimand,
                    G          =  identification['id_in'][0].graph
                ),
            )

    def test_line_3(self):
        r"""Test line 3 of the identification algorithm.

        3. Forces an action on any node where such an action would have no effect on :math:`\mathbf Y`—assuming
        we already acted on :math:`\mathbf X`. Since actions remove incoming arrows, we can view line 3 as
        simplifying the causal graph we consider by removing certain arcs from the graph, without
        affecting the overall answer.
        """
        for identification in line_3_example.identifications:
            outcomes, treatments = query_to_outcomes_and_treatments(query=identification['id_in'][0].query )
            self.assert_identification_equal(
                expected = identification['id_out'][0],
                actual   = line_3(
                    outcomes=outcomes,
                    treatments=treatments,
                    estimand=identification['id_in'][0].estimand,
                    G = identification['id_in'][0].graph))




    def test_line_4(self):
        r"""Test line 4 of the identification algorithm.

        4. The key line of the algorithm, it decomposes the problem into a set of smaller problems
        using the key property of *c-component factorization* of causal models. If the entire graph
        is a single C-component already, further problem decomposition is impossible, and we must
        provide base cases. :math:`\mathbf{ID}` has three base cases.
        """
        for identification in line_4_example.identifications:
            outcomes, treatments = query_to_outcomes_and_treatments(query= identification['id_in'][0].query )
            actuals = line_4(
                outcomes   = outcomes,
                treatments = treatments,
                estimand   = identification['id_in'][0].estimand,
                G          = identification['id_in'][0].graph)
            expecteds = identification['id_out']
            self.assertEqual(len(expecteds), len(equals))
            match = {}
            for expected in expecteds:
                for actual in actuals:
                    if expected == actual:
                        self.assert_identification_equal(
                                  expected, actual)
                        match[expected] = actual
            self.assertEqual(len(expecteds), len(match))



    def test_line_5(self):
        r"""Test line 5 of the identification algorithm.

        5. Fails because it finds two C-components, the graph :math:`G` itself, and a subgraph :math:`S` that
        does not contain any :math:`\mathbf X` nodes. But that is exactly one of the properties of C-forests
        that make up a hedge. In fact, it turns out that it is always possible to recover a hedge
        from these two c-components.
        """
        pass
    def test_line_6(self):
        r"""Test line 6 of the identification algorithm.

        6. Asserts that if there are no bidirected arcs from X to the other nodes in the current
        subproblem under consideration, then we can replace acting on X by conditioning, and thus
        solve the subproblem.
        """
        pass
    def test_line_7(self):
        r"""Test line 2 of the identification algorithm.

        7. The most complex case where :math:`\mathbf X` is partitioned into two sets, :math:`\mathbf W` which
        contain bidirected arcs into other nodes in the subproblem, and :math:`\mathbf Z` which do not.
        In this situation, identifying :math:`P(\mathbf y|do(\mathbf x))` from :math:`P(v)` is equivalent to
        identifying :math:`P(\mathbf y|do(\mathbf w))` from :math:`P(\mathbf V|do(\mathbf z))`, since
        :math:`P(\mathbf y|do(\mathbf x)) = P(\mathbf y|do(\mathbf w), do(\mathbf z))`. But the term
        :math:`P(\mathbf V|do(\mathbf z))` is identifiable using the previous base case, so we can consider
        the subproblem of identifying :math:`P(\mathbf y|do(\mathbf w))`.
        """
        pass
    # def test_figure_2a(self):
    #     """Test Figure 2A.
    #     Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.
    #     Journal of Machine Learning Research.
    #     """
    #     graph = NxMixedGraph()
    #     graph.add_directed_edge("X", "Y")
    #     print(identify(graph, Y @ X).to_text())
    #     expr = "[ sum_{} P(Y|X) ]"
    #     frac_expr = P_XY / Sum[Y](P_XY)
    #     cond_expr = P(Y|X)
    #     self.assert_identify(cond_expr, graph, Y @ X)

    # def test_figure_2b(self):
    #     """Test Figure 2B.
    #     Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.
    #     Journal of Machine Learning Research.
    #     """
    #     graph = NxMixedGraph()
    #     graph.add_directed_edge("X", "Y")
    #     graph.add_directed_edge("X", "Z")
    #     graph.add_directed_edge("Z", "Y")
    #     graph.add_undirected_edge("Y", "Z")
    #     print(identify(graph, Y @ X).to_text())
    #     expr = "[ sum_{Z} P(Z|X) P(Y|X,Z) ]"
    #     cond_expr = Sum[Z](P(Z | X) * P(Y | X, Z))
    #     frac_expr = Sum[Z](
    #         Sum[Y](P_XY) / (Sum[Z](Sum[Y](P_XY))) * (P_XY / Sum[Y](P_XY))
    #     )
    #     self.assert_identify(cond_expr, graph, Y @ X)

    # def test_figure_2c(self):
    #     """Test Figure 2C.
    #     Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.
    #     Journal of Machine Learning Research.
    #     """
    #     graph = NxMixedGraph()
    #     graph.add_directed_edge("X", "Y")
    #     graph.add_directed_edge("Z", "X")
    #     graph.add_directed_edge("Z", "Y")
    #     graph.add_undirected_edge("Y", "Z")
    #     print(identify(graph, Y @ X).to_text())
    #     expr = "[ sum_{Z} P(Z) P(Y|X,Z) ]"
    #     cond_expr = Sum[Z](P(Z) * P(Y | X, Z))
    #     frac_expr = Sum[Z](
    #         Sum[X, Y](P_XYZ) / (Sum[Z](Sum[X, Y](P_XYZ))) * (P_XYZ / Sum[Y](P_XYZ))
    #     )
    #     self.assert_identify(cond_expr, graph, Y @ X)
    #     # self.assert_identify(grammar.parseString(expr)[0], graph, Y@X)

    # def test_figure_2d(self):
    #     """Test Figure 2D.
    #     frac_expr = Sum[Z](Sum[X, Y](P_XYZ) * P_XYZ / Sum[Y](P_XYZ))
    #     """
    #     graph = NxMixedGraph()
    #     graph.add_directed_edge("X", "Y")
    #     graph.add_directed_edge("Z", "X")
    #     graph.add_directed_edge("Z", "Y")
    #     graph.add_undirected_edge("X", "Z")
    #     print(identify(graph, Y @ X).to_text())

    #     expr = "[ sum_{Z} [ sum_{} P(Y|X,Z) ] [ sum_{} [ sum_{X,Y} P(X,Y,Z) ] ] ]"
    #     self.assert_identify(parse_craig(expr), graph, Y @ X)
    #     # self.assert_identify(grammar.parseString(expr)[0], graph, Y@X)

    # def test_figure_2e(self):
    #     """Test Figure 2E.
    #     Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.
    #     Journal of Machine Learning Research.
    #     """
    #     graph = NxMixedGraph()
    #     graph.add_directed_edge("X", "Z")
    #     graph.add_directed_edge("Z", "Y")
    #     graph.add_undirected_edge("X", "Y")
    #     expr = "[ sum_{Z} [ sum_{} P(Z|X) ] [ sum_{} [ sum_{X} P(X) P(Y|X,Z) ] ] ]"
    #     cond_expr = Sum[Z](P(Z | X)) * Sum[X](P(X) * P(Y | X, Z))
    #     frac_expr = Sum[Z](Sum[Y](P_XYZ) / Sum[Z](Sum[Y](P_XYZ))) * Sum[X](
    #         P_XYZ * Sum[Y, Z](P_XYZ) / Sum[Y](P_XYZ) / Sum[X](Sum[Y, Z](P_XYZ))
    #     )
    #     self.assert_identify(parse_craig(expr), graph, Y @ X)
    #     # self.assert_identify(grammar.parseString(expr)[0], graph, Y@X)

    # def test_figure_3a(self):
    #     """Test Figure 3a. A graph hedge-less for P(y1,y2|do(x))
    #     Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.
    #     Journal of Machine Learning Research.
    #     """
    #     graph = NxMixedGraph()
    #     #W1,W2,Y1,Y2 = Variable('W1'), Variable('W2'), Variable('Y1'), Variable('Y2')
    #     graph.add_directed_edge("X", "Y1")
    #     graph.add_directed_edge("W1", "X")
    #     graph.add_directed_edge("W2", "Y2")
    #     graph.add_undirected_edge("W1", "W2")
    #     graph.add_undirected_edge("W1", "Y1")
    #     graph.add_undirected_edge("W1", "Y2")
    #     graph.add_undirected_edge("X", "W2")
    #     cond_expr = Sum[W2](P(Y1,W2))*Sum[W1](P(Y1|(X,W1))*P(W1))
    #     self.assert_identify(cond_expr, graph, P(Y1 @ X, Y2 @ X))
    # def test_taheri(self):
    #     """Test that all graphs produced by Sara's design algorithm can be run with :func:`identify`."""
    #     graph = NxMixedGraph.from_causalfusion_path(VIRAL_PATHOGENESIS_PATH)

    #     cause = 'EGFR'
    #     effect = 'CytokineStorm'
    #     stop = 5
    #     tag = DEFAULT_TAG
    #     dag = admg_to_latent_variable_dag(graph.to_admg(), tag=tag)
    #     fixed_latent = {
    #         node
    #         for node, data in dag.nodes(data=True)
    #         if data[tag]
    #     }
    #     for latents, observed, lvdag in iterate_lvdags(
    #         dag,
    #         fixed_observed={cause, effect},
    #         fixed_latents=fixed_latent,
    #         stop=stop,
    #     ):
    #         with self.subTest(latents=latents):
    #             result = _get_result(
    #                 lvdag=lvdag,
    #                 latents=latents,
    #                 observed=observed,
    #                 cause=cause,
    #                 effect=effect,
    #             )
    #             self.assertIsNotNone(result)  # throwaway test


if __name__ == "__main__":
    unittest.main()
