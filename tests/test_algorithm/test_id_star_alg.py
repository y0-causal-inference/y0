# -*- coding: utf-8 -*-

"""Tests for the ID* algorithm."""

import unittest
from y0.graph import NxMixedGraph
from y0.dsl import Variable, X, D, W, P, Y, Z, Expression, get_outcomes_and_treatments, Sum
from y0.mutate import canonicalize
from y0.algorithm.conditional_independencies import are_d_separated
from y0.algorithm.identify.id_star import (
    make_parallel_worlds_graph,
    combine_parallel_worlds,
    make_parallel_world_graph,
    has_same_function,
    has_same_parents,
    get_worlds,
    lemma_24,
    lemma_25,
    make_counterfactual_graph,
    id_star,
    idc_star,
    id_star_line_9
)
from collections import Counter

from y0.examples import figure_9a, figure_9b, figure_9c, figure_9d, figure_11a, figure_11b, figure_11c


class TestIdentifyStar(unittest.TestCase):
    """Tests parallel worlds and counterfactual graphs"""

    def assert_graph_equal(self, a: NxMixedGraph, b: NxMixedGraph, msg=None) -> None:
        """Check the graphs are equal (more nice than the builtin :meth:`NxMixedGraph.__eq__` for testing)."""
        self.assertEqual(set(a.directed.nodes()), set(b.directed.nodes()), msg=msg)
        self.assertEqual(set(a.undirected.nodes()), set(b.undirected.nodes()), msg=msg)
        self.assertEqual(set(a.directed.edges()), set(b.directed.edges()), msg=msg)
        self.assertEqual(
            set(map(frozenset, a.undirected.edges())),
            set(map(frozenset, b.undirected.edges())),
            msg=msg,
        )

    def assert_expr_equal(self, expected: Expression, actual: Expression) -> None:
        """Assert that two expressions are the same."""
        expected_outcomes, expected_treatments = get_outcomes_and_treatments(query=expected)
        actual_outcomes, actual_treatments = get_outcomes_and_treatments(query=actual)
        self.assertEqual(expected_treatments, actual_treatments)
        self.assertEqual(expected_outcomes, actual_outcomes)
        ordering = tuple(expected.get_variables())
        expected_canonical = canonicalize(expected, ordering)
        actual_canonical = canonicalize(actual, ordering)
        self.assertEqual(
            expected_canonical,
            actual_canonical,
            msg=f"\nExpected: {str(expected_canonical)}\nActual:   {str(actual_canonical)}",
        )

    def test_make_parallel_worlds(self):
        """Test that parallel worlds graphs are correct"""
        expected = figure_9b.graph
        actual = make_parallel_worlds_graph(figure_9a.graph, [[~X], [D]])
        self.assert_graph_equal(expected, actual)

    def test_has_same_function(self):
        """Test that two variables have the same function"""
        self.assertTrue(has_same_function(D @ X, D))
        self.assertTrue(has_same_function(D @ D, D))
        self.assertTrue(has_same_function(X @ D, X))
        self.assertFalse(has_same_function(X, D))
        self.assertFalse(has_same_function(X @ ~X, W @ ~X))
        self.assertTrue(has_same_function(X @ ~X, X))

    def test_has_same_parents(self):
        """Test that all parents of two nodes are the same"""
        self.assertTrue(has_same_parents(figure_9b.graph, D @ ~X, D))
        self.assertTrue(has_same_parents(figure_9b.graph, X @ D, X))
        self.assertFalse(has_same_parents(figure_9b.graph, D @ D, D))
        self.assertFalse(has_same_parents(figure_9b.graph, X, D))
        self.assertFalse(has_same_parents(figure_9b.graph, X @ ~X, W @ ~X))
        self.assertFalse(has_same_parents(figure_9b.graph, X @ ~X, X))

    def test_get_worlds(self):
        """Test that all interventions within each world of a counterfactual conjunction are generated"""
        expected = [[~X], [-D]]
        self.assertEqual(
            sorted(expected), sorted(sorted(world) for world in get_worlds(P(Y @ ~X | X, Z @ D, D)))
        )

    def test_lemma_24(self):
        """Test that two nodes in a parallel world graph are the same"""
        self.assertTrue(lemma_24(figure_9b.graph, D @ ~X, D))
        self.assertTrue(lemma_24(figure_9b.graph, X @ D, X))
        self.assertFalse(lemma_24(figure_9b.graph, D, D @ D))
        self.assertFalse(lemma_24(figure_9b.graph, X, D))
        self.assertFalse(lemma_24(figure_9b.graph, X @ ~X, W @ ~X))
        self.assertFalse(lemma_24(figure_9b.graph, X @ ~X, X))

    def test_lemma_25(self):
        """Test that the parallel worlds graph after merging two nodes is correct"""
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
        """Test that the counterfactual graph returned is correct"""
        actual_graph, actual_query = make_counterfactual_graph(
            figure_9a.graph, P(Y @ ~X, X, Z @ D, D)
        )
        self.assert_graph_equal(figure_9c.graph, actual_graph)
        self.assert_expr_equal(expected=P(Y @ ~X, X, Z, D), actual=actual_query)
        #actual_graph2, actual_query2 = make_counterfactual_graph(
        #    figure_9a.graph, P( Y @ (~X, Z) | X )
        #)
        #self.assert_graph_equal(figure_9d.graph, actual_graph2)
        #self.assert_expr_equal( P( Y @ (~X, Z)), actual_query2)

    def test_idc_star_line_2(self):
        """Check to see if Axiom of Effectiveness is violated"""
        query = P( Y @ ~X | X, Z @ D, D)
        delta = query.distribution.parents
        graph = figure_9a.graph
        #self.assert_expr_equal( P(X, Z@ D, D), P(delta) )
        #self.assert_expr_equal(expected=0, actual=id_star(graph, P(delta)))

    def test_idc_star_line_4(self):
       r"""Check that line 4 or IDC* works correctly moves :math:`Z, D` (with
       :math:`D` being redundant due to graph structure) to the
       subscript of :math:`Y_\mathbf{x}`, to obtain :math:`P(Y_{X',Z}
       | X )`, and calls IDC* with this query recursively.
       """
       input_query = P(Y @ ~X | X, Z, D)
       expected_output_query = P(Y @ (~X, Z) | X )
       new_delta = {X, Z , D}
       new_gamma = {Y @ ~X}
       graph =  figure_9c.graph
       for counterfactual in [Z, D]:
           #self.assertTrue(are_d_separated(graph.remove_outgoing_edges_from( {counterfactual} ), counterfactual, new_gamma))
           counterfactual_value = Variable(counterfactual.name)
           parents = new_delta - {counterfactual}
           children = {g.intervene(counterfactual_value) for g in new_gamma}
           #self.assert_expr_equal( P( Y @ {X, counterfactual}  | new_gamma - {counterfactual}), P(children | parents))


    def test_id_star_line_3(self):
        """Check to see if the counterfactual event is tautological"""

    def test_id_star_line_4(self):
        """Check that the counterfactual graph is correct"""

    def test_id_star_line_5(self):
        """Check whether the query is inconsistent with the counterfactual graph"""

    def test_id_star_line_6(self):
        """Check that the input to id_star from each district is properly constructed"""

    def test_id_star_line_7(self):
        """Check that the graph is entirely one c-component"""

    def test_id_star_line_8(self):
        """Attempt to generate a conflict with an inconsistent value assignment."""

    def test_id_star_line_9(self):
        """Test that estimand returned by taking the effect of all subscripts in new_gamma on variables in new_gamma is correct"""
        input_query = P(Y @ (W, Z), X)
        output_query = P[W, Z](Y, X )
        self.assert_expr_equal(output_query, id_star_line_9(input_query))

    def test_id_star(self):
        """Test that the ID* algorithm returns the correct estimand"""
        query = P(Y @ (~X, Z), X)
        #actual = id_star( figure_9a.graph, query)
        expected = Sum[W](P(Y @ (Z,W), X @ (Z, W))*P( W @ X))
        #self.assert_expr_equal(expected, actual)

    def test_idc_star(self):
        """Test that the IDC* algorithm returns the correct estimand"""
        query = P(Y @ ~X | X, Z @ D, D )
        vertices = set(figure_9a.graph.nodes())
        estimand = Sum[W](P(Y @ (Z,W), X @ (Z, W))*P( W @ X))
        expected = estimand/Sum[vertices - {X, Z @ D, D}](estimand)
        #actual = idc_star( figure_9a.graph, query)
        #self.assert_expr_equal( expected, actual )
