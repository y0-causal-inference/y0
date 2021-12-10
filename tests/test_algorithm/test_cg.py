from tests.test_algorithm import cases
from y0.algorithm.identify import lemma_24, lemma_25, make_parallel_worlds_graph
from y0.algorithm.identify.cg import (
    get_worlds,
    has_same_function,
    has_same_parents,
    make_counterfactual_graph,
)
from y0.dsl import D, P, W, X, Y, Z, Zero, Variable, CounterfactualVariable, Intervention
from y0.examples import (
    figure_9a,
    figure_9b,
    figure_9c,
    figure_9d,
    figure_11a,
    figure_11b,
    figure_11c,
)
from y0.graph import NxMixedGraph


class TestCounterfactualGraph(cases.GraphTestCase):
    """Tests parallel worlds and counterfactual graphs."""

    def test_make_parallel_worlds(self):
        """Test that parallel worlds graphs are correct."""
        expected = figure_9b.graph
        actual = make_parallel_worlds_graph(figure_9a.graph, [[~X], [D]])
        self.assert_graph_equal(expected, actual)

    def test_has_same_function(self):
        """Test that two variables have the same function."""
        self.assertTrue(has_same_function(D @ X, D))
        self.assertTrue(has_same_function(D @ D, D))
        self.assertTrue(has_same_function(X @ D, X))
        self.assertFalse(has_same_function(X, D))
        self.assertFalse(has_same_function(X @ ~X, W @ ~X))
        self.assertTrue(has_same_function(X @ ~X, X))

    def test_has_same_parents(self):
        """Test that all parents of two nodes are the same."""
        self.assertTrue(has_same_parents(figure_9b.graph, D @ ~X, D))
        self.assertTrue(has_same_parents(figure_9b.graph, X @ D, X))
        self.assertFalse(has_same_parents(figure_9b.graph, D @ D, D))
        self.assertFalse(has_same_parents(figure_9b.graph, X, D))
        self.assertFalse(has_same_parents(figure_9b.graph, X @ ~X, W @ ~X))
        self.assertFalse(has_same_parents(figure_9b.graph, X @ ~X, X))

    def test_get_worlds(self):
        """Test that all interventions within each world of a counterfactual conjunction are generated."""
        expected = [[~X], [-D]]
        self.assertEqual(
            sorted(expected), sorted(sorted(world) for world in get_worlds(P(Y @ ~X | X, Z @ D, D)))
        )

    def test_lemma_24(self):
        """Test that two nodes in a parallel world graph are the same."""
        self.assertTrue(lemma_24(figure_9b.graph, D @ ~X, D))
        self.assertTrue(lemma_24(figure_9b.graph, X @ D, X))
        self.assertFalse(lemma_24(figure_9b.graph, Z @ D, Z))
        self.assertFalse(lemma_24(figure_9b.graph, D, D @ D))
        self.assertFalse(lemma_24(figure_9b.graph, X, D))
        self.assertFalse(lemma_24(figure_9b.graph, X @ ~X, W @ ~X))
        self.assertFalse(lemma_24(figure_9b.graph, X @ ~X, X))

    def test_lemma_25(self):
        """Test that the parallel worlds graph after merging two nodes is correct."""
        cf_graph_1 = lemma_25(figure_9b.graph, D, D @ ~X)
        cf_graph_2 = lemma_25(cf_graph_1, X, X @ D)
        cf_graph_3 = lemma_25(cf_graph_2, Z, Z @ ~X)
        cf_graph_4 = lemma_25(cf_graph_3, Z, Z @ D)
        cf_graph_5 = lemma_25(cf_graph_4, W, W @ D)
        cf_graph_6 = lemma_25(cf_graph_5, D, D @ D)
        cf_graph_7 = lemma_25(cf_graph_6, Y, Y @ D)
        self.assert_graph_equal(figure_11a.graph, cf_graph_2)
        self.assert_graph_equal(figure_11b.graph, cf_graph_6)
        self.assert_graph_equal(figure_11c.graph, cf_graph_7)

    def test_make_counterfactual_graph(self):
        r"""Test making a counterfactual graph.

        The invocation of **make-cg** with the graph in Figure 9(a) and the joint distribution
        :math:`P(y_x, x', z, d)` will result in the counterfactual graph shown in Fig. 9(c).
        The invocation of **make-cg** with the graph in Figure 9(a) and the joint distribution
        :math:`P(y_{x,z},x')` will result in the counterfactual graph shown in Fig. 9(d).
        """
        actual_graph, actual_query = make_counterfactual_graph(
            figure_9a.graph, P(Y @ ~X, X, Z @ D, D)
        )
        self.assert_graph_equal(figure_9c.graph, actual_graph)
        self.assert_expr_equal(expected=P(Y @ ~X, X, Z, D), actual=actual_query)
        actual_graph2, actual_query2 = make_counterfactual_graph(figure_9a.graph, P(Y @ (Z, ~X), X))
        expected_graph2, expected_query2 = figure_9d.graph, P(Y @ (Z, ~X), X)
        self.assert_expr_equal(expected_query2, actual_query2)
        self.assert_graph_equal(expected_graph2, actual_graph2)

        # Check for inconsistent counterfactual values for merged nodes
        actual_graph3, actual_query3 = make_counterfactual_graph(
            graph=NxMixedGraph.from_edges(directed=[(D, Z), (Z, Y)]),
        query=P(CounterfactualVariable(name='Z', star=False, interventions=(Intervention('D', star=False),)), Variable(name='Z', star=True), -D)
        )
        self.assertEqual(Zero(), actual_query3)
