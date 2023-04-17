# -*- coding: utf-8 -*-

"""Tests for parallel world graphs and counterfactual graphs."""

from tests.test_algorithm import cases
from y0.algorithm.identify.cg import (
    _get_directed_edges,
    extract_interventions,
    has_same_function,
    #has_same_parents,
    is_pw_equivalent,
    make_counterfactual_graph,
    make_parallel_worlds_graph,
    merge_pw,
    node_not_an_intervention_in_world,
    stitch_counterfactual_and_doppleganger_neighbors,
    stitch_counterfactual_and_dopplegangers,
    stitch_counterfactual_and_neighbors,
    stitch_factual_and_doppleganger_neighbors,
    stitch_factual_and_dopplegangers,
    nodes_attain_same_value,
    parents_attain_same_values,
    has_same_confounders,
)
from y0.dsl import D, W, X, Y, Z, Event
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

x, y, z, d, w = -X, -Y, -Z, -D, -W


class TestCounterfactualGraph(cases.GraphTestCase):
    """Tests parallel worlds and counterfactual graphs."""

    def assert_uedge_set_equal(self, expected, actual):
        """Assert that two undirected edge sets are equal."""
        return self.assertEqual(
            {frozenset(edge) for edge in expected}, {frozenset(edge) for edge in actual}
        )

    def test_make_parallel_worlds(self):
        """Test that parallel worlds graphs are correct."""
        expected = figure_9b.graph
        actual = make_parallel_worlds_graph(figure_9a.graph, {frozenset([+x]), frozenset([-d])})
        expected2 = make_parallel_worlds_graph(figure_9a.graph, {(+x, -z)})
        self.assert_graph_equal(expected, actual)
        self.assert_graph_equal(expected2, make_parallel_worlds_graph(figure_9a.graph, {(+x, -z)}))
        self.assertFalse(Y @ (-z, +x) in expected2.nodes())
        self.assertTrue(Y @ (+x, -z) in expected2.nodes())

    def test_has_same_function(self):
        """Test that two variables have the same value."""
        self.assertTrue(has_same_function(D @ X, D))
        self.assertTrue(has_same_function(D @ D, D))
        self.assertFalse(has_same_function(X, X @ +x))
        self.assertTrue(has_same_function(X @ D, X))
        self.assertFalse(has_same_function(X, D))
        self.assertFalse(has_same_function(X @ ~X, W @ ~X))
        self.assertFalse(has_same_function(X @ ~X, X))
        self.assertTrue(has_same_function(D, D @ ~x))
        self.assertTrue(has_same_function(Z @ ~x, Z))
        self.assertTrue(has_same_function(Z @ ~x, Z @ -d))
        self.assertTrue(has_same_function(Z @ -d, Z))

    def test_nodes_attain_same_value(self):
        """Test that two variables attain the same value."""
        event: Event = {D: -d}
        self.assertTrue(nodes_attain_same_value(figure_11a.graph, event, D, D @ -d))
        self.assertTrue(nodes_attain_same_value(figure_11a.graph, event, D @ -d, D))
        self.assertTrue(nodes_attain_same_value(NxMixedGraph.from_edges(directed=[(D @ +d, Z @ +d)]),
                                                event, D @ +d, D @ +d))
        self.assertTrue(nodes_attain_same_value(figure_9b.graph, event, D, D))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D, D @ -d))
        self.assertFalse(nodes_attain_same_value(figure_11a.graph, event, Z, Z @ +x))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D, X))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, X, D))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D, X @ -d))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, X @ -d, D))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D, X @ -d))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D @ -d, D))

    def test_has_same_confounders(self):
        self.assertFalse(has_same_confounders(figure_9b.graph, D, D))
        self.assertFalse(has_same_confounders(figure_9b.graph, D, D @ -d))
        self.assertFalse(has_same_confounders(figure_9b.graph, D @ -d, D))
        self.assertTrue(has_same_confounders(figure_9b.graph, D @ +x, D))

    def test_parents_attain_same_values(self):
        """Test that the parents of two nodes attain the same value"""
        graph = figure_9b.graph
        event: Event = {Y @ +x: +y, D: -d, Z @ -d: -z, X: -x}
        self.assertTrue(parents_attain_same_values(figure_11a.graph, event, Z, Z @ -d))
        self.assertTrue(parents_attain_same_values(figure_11a.graph, event, Z, Z @ +x))
        self.assertTrue(parents_attain_same_values(figure_11a.graph, event, Z @ -d, Z @ +x))
        self.assertFalse(parents_attain_same_values(graph, event, Z, Z @ -d))
        self.assertFalse(parents_attain_same_values(figure_9b.graph, event, D, D @ -d))
        self.assertFalse(parents_attain_same_values(figure_9b.graph, event, X, X @ +x))

    # def test_has_same_parents(self):
    #     """Test that all parents of two nodes are the same."""
    #     self.assertTrue(has_same_parents(figure_9b.graph, D @ ~X, D))
    #     self.assertTrue(has_same_parents(figure_9b.graph, X @ D, X))
    #     self.assertFalse(has_same_parents(figure_9b.graph, D @ D, D))
    #     self.assertFalse(has_same_parents(figure_9b.graph, X, D))
    #     self.assertFalse(has_same_parents(figure_9b.graph, X @ ~X, W @ ~X))
    #     self.assertFalse(has_same_parents(figure_9b.graph, X @ ~X, X))
    #     self.assertFalse(has_same_parents(figure_9b.graph, Z, Z @ ~x))
    #     self.assertTrue(has_same_parents(figure_11a.graph, Z, Z @ ~x))
    #     self.assertTrue(has_same_parents(figure_11a.graph, Z @ ~x, Z))
    #     self.assertFalse(has_same_parents(figure_11a.graph, Z @ ~x, Z @ -d))
    #     self.assertFalse(has_same_parents(figure_11a.graph, Z @ -d, Z))

    def test_get_worlds(self):
        """Test that all interventions within each world of a counterfactual conjunction are generated."""
        self.assert_uedge_set_equal(
            {frozenset([-D]), frozenset([~X])}, extract_interventions([Y @ ~X, X, Z @ -D, D])
        )
        self.assert_uedge_set_equal(
            {frozenset([-D]), frozenset([~X, -Z])},
            extract_interventions([Y @ (~X, -Z), X, Z @ -D, D]),
        )

    def test_node_not_an_intervention_in_world(self):
        """Test that a node is not an intervention in a world"""
        self.assertTrue(node_not_an_intervention_in_world(world=frozenset([-x]), node=Y))
        self.assertFalse(node_not_an_intervention_in_world(world=frozenset([-x, +z]), node=X))
        self.assertTrue(node_not_an_intervention_in_world(world=frozenset([-x, +x]), node=Y))
        self.assertFalse(node_not_an_intervention_in_world(world=frozenset([-x]), node=X))
        self.assertFalse(node_not_an_intervention_in_world(world=frozenset([-x, -y]), node=Y))
        self.assertFalse(node_not_an_intervention_in_world(world=frozenset([-x, +y]), node=Y))
        self.assertFalse(node_not_an_intervention_in_world(world=frozenset([-x, +y, -y]), node=Y))

    def test_stitch_factual_and_dopplegangers(self):
        """Test that factual variables and their dopplegangers are stitched together unless it is intervened upon"""
        self.assert_uedge_set_equal(
            {(Y, Y @ -x)},
            stitch_factual_and_dopplegangers(
                graph=NxMixedGraph.from_edges(directed=[(X, Y)]), worlds=set([frozenset([-x])])
            ),
        )
        self.assert_uedge_set_equal(
            {(Y, Y @ +x)},
            stitch_factual_and_dopplegangers(
                graph=NxMixedGraph.from_edges(directed=[(X, Y)]), worlds=set([frozenset([+x])])
            ),
        )
        self.assert_uedge_set_equal(
            {(Y, Y @ +x), (Z, Z @ +x)},
            stitch_factual_and_dopplegangers(
                graph=NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)]),
                worlds=set([frozenset([+x])]),
            ),
        )
        self.assert_uedge_set_equal(
            {(Y, Y @ +x), (Z, Z @ +x)},
            stitch_factual_and_dopplegangers(
                graph=NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)], undirected=[(X, Z)]),
                worlds=set([frozenset([+x])]),
            ),
        )
        self.assert_uedge_set_equal(
            {(Y, Y @ +x), (Z, Z @ +x), (D, D @ +x), (X, X @ -d), (Y, Y @ -d), (Z, Z @ -d)},
            stitch_factual_and_dopplegangers(
                graph=NxMixedGraph.from_edges(
                    directed=[(X, Y), (Y, Z), (D, X)], undirected=[(X, Z)]
                ),
                worlds=set([frozenset([+x]), frozenset([-d])]),
            ),
        )
        self.assert_uedge_set_equal(
            {
                (Y, Y @ +x),
                (Z, Z @ +x),
                (D, D @ +x),
                (X, X @ -d),
                (Y, Y @ -d),
                (Z, Z @ -d),
                (W, W @ -d),
                (W, W @ +x),
            },
            stitch_factual_and_dopplegangers(
                graph=figure_9a.graph, worlds=set([frozenset([+x]), frozenset([-d])])
            ),
        )

    def test_stitch_factual_and_doppleganger_neighbors(self):
        """Test that factual variables and their dopplegangers are stitched together unless it is intervened upon"""
        self.assert_uedge_set_equal(
            set(),
            stitch_factual_and_doppleganger_neighbors(
                graph=NxMixedGraph.from_edges(directed=[(X, Y)]), worlds=set([frozenset([-x])])
            ),
        )
        self.assert_uedge_set_equal(
            {(X, Y @ +x)},
            stitch_factual_and_doppleganger_neighbors(
                graph=NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[(X, Y)]),
                worlds=set([frozenset([+x])]),
            ),
        )
        self.assert_uedge_set_equal(
            {(X, Z @ +x)},
            stitch_factual_and_doppleganger_neighbors(
                graph=NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)], undirected=[(X, Z)]),
                worlds=set([frozenset([+x])]),
            ),
        )
        self.assert_uedge_set_equal(
            {(X, Z @ +x), (X, Z @ -d), (Z, X @ -d)},
            stitch_factual_and_doppleganger_neighbors(
                graph=NxMixedGraph.from_edges(
                    directed=[(X, Y), (Y, Z), (D, X)], undirected=[(X, Z)]
                ),
                worlds=set([frozenset([+x]), frozenset([-d])]),
            ),
        )

    def test_stitch_counterfactual_and_dopplegangers(self):
        """Test that counterfactual variables and their dopplegangers are stitched together unless it is intervened upon"""
        self.assert_uedge_set_equal(
            set(),
            stitch_counterfactual_and_dopplegangers(
                graph=NxMixedGraph.from_edges(directed=[(X, Y)]), worlds=set([frozenset([-x])])
            ),
        )
        self.assert_uedge_set_equal(
            set(),
            stitch_counterfactual_and_dopplegangers(
                graph=NxMixedGraph.from_edges(directed=[(X, Y)]), worlds=set([frozenset([+x])])
            ),
        )
        self.assert_uedge_set_equal(
            set(),
            stitch_counterfactual_and_dopplegangers(
                graph=NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)]),
                worlds=set([frozenset([+x])]),
            ),
        )
        self.assert_uedge_set_equal(
            set(),
            stitch_counterfactual_and_dopplegangers(
                graph=NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)], undirected=[(X, Z)]),
                worlds=set([frozenset([+x])]),
            ),
        )
        self.assert_uedge_set_equal(
            {frozenset({Z @ -d, Z @ +x}), frozenset({Y @ +x, Y @ -d})},
            set(
                stitch_counterfactual_and_dopplegangers(
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, Y), (Y, Z), (D, X)], undirected=[(X, Z)]
                    ),
                    worlds=set([frozenset([+x]), frozenset([-d])]),
                )
            ),
        )
        self.assert_uedge_set_equal(
            {frozenset({Y @ -d, Y @ +x}), frozenset({W @ +x, W @ -d}), frozenset({Z @ +x, Z @ -d})},
            stitch_counterfactual_and_dopplegangers(
                graph=figure_9a.graph, worlds=set([frozenset([+x]), frozenset([-d])])
            ),
        )

    def test_stitch_counterfactual_and_doppleganger_neighbors(self):
        """
        Test that counterfactual variables and their neighbor dopplegangers are stitched together unless either are intervened upon
        """
        self.assert_uedge_set_equal(
            set(),
            set(
                stitch_counterfactual_and_doppleganger_neighbors(
                    graph=NxMixedGraph.from_edges(directed=[(X, Y)]), worlds=set([frozenset([-x])])
                )
            ),
        )
        self.assert_uedge_set_equal(
            set(),
            set(
                stitch_counterfactual_and_doppleganger_neighbors(
                    graph=NxMixedGraph.from_edges(directed=[(X, Y)]), worlds=set([frozenset([+x])])
                )
            ),
        )
        self.assert_uedge_set_equal(
            set(),
            set(
                stitch_counterfactual_and_doppleganger_neighbors(
                    graph=NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)]),
                    worlds=set([frozenset([+x])]),
                )
            ),
        )
        self.assert_uedge_set_equal(
            set(),
            set(
                stitch_counterfactual_and_doppleganger_neighbors(
                    graph=NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)], undirected=[(X, Z)]),
                    worlds=set([frozenset([+x])]),
                )
            ),
        )
        self.assert_uedge_set_equal(
            {frozenset({X @ -d, Z @ +x})},
            set(
                stitch_counterfactual_and_doppleganger_neighbors(
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, Y), (Y, Z), (D, X)], undirected=[(X, Z)]
                    ),
                    worlds=set([frozenset([+x]), frozenset([-d])]),
                )
            ),
        )
        self.assert_uedge_set_equal(
            {frozenset({X @ -d, Z @ +x}), frozenset({Y @ +x, Z @ -d}), frozenset({Y @ -d, Z @ +x})},
            set(
                stitch_counterfactual_and_doppleganger_neighbors(
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, Y), (Y, Z), (D, X)], undirected=[(X, Z), (Y, Z)]
                    ),
                    worlds=set([frozenset([+x]), frozenset([-d])]),
                )
            ),
        )
        self.assert_uedge_set_equal(
            {frozenset({X @ -d, Y @ +x})},
            stitch_counterfactual_and_doppleganger_neighbors(
                graph=figure_9a.graph, worlds=set([frozenset([+x]), frozenset([-d])])
            ),
        )

    def test_stitch_counterfactual_and_neighbors(self):
        """
        Test that counterfactual variables and their neighbors are stitched together unless either are intervened upon
        """
        self.assert_uedge_set_equal(
            set(),
            set(
                stitch_counterfactual_and_neighbors(
                    graph=NxMixedGraph.from_edges(directed=[(X, Y)]), worlds=set([frozenset([-x])])
                )
            ),
        )
        self.assert_uedge_set_equal(
            set(),
            set(
                stitch_counterfactual_and_neighbors(
                    graph=NxMixedGraph.from_edges(directed=[(X, Y)]), worlds=set([frozenset([+x])])
                )
            ),
        )
        self.assert_uedge_set_equal(
            set(),
            set(
                stitch_counterfactual_and_neighbors(
                    graph=NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)]),
                    worlds=set([frozenset([+x])]),
                )
            ),
        )
        self.assert_uedge_set_equal(
            {frozenset({Y @ +x, Z @ +x})},
            set(
                stitch_counterfactual_and_neighbors(
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, Y), (Y, Z)], undirected=[(X, Z), (Y, Z)]
                    ),
                    worlds=set([frozenset([+x])]),
                )
            ),
        )
        self.assert_uedge_set_equal(
            {frozenset({X @ -d, Z @ -d})},
            set(
                stitch_counterfactual_and_neighbors(
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, Y), (Y, Z), (D, X)], undirected=[(X, Z)]
                    ),
                    worlds=set([frozenset([+x]), frozenset([-d])]),
                )
            ),
        )
        self.assert_uedge_set_equal(
            {frozenset({Y @ +x, Z @ +x}), frozenset({Y @ -d, Z @ -d}), frozenset({Z @ -d, X @ -d})},
            set(
                stitch_counterfactual_and_neighbors(
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, Y), (Y, Z), (D, X)], undirected=[(X, Z), (Y, Z)]
                    ),
                    worlds=set([frozenset([+x]), frozenset([-d])]),
                )
            ),
        )
        self.assert_uedge_set_equal(
            {frozenset({X @ -d, Y @ -d})},
            stitch_counterfactual_and_neighbors(
                graph=figure_9a.graph, worlds=set([frozenset([+x]), frozenset([-d])])
            ),
        )

    def test_get_directed_edges(self):
        """Test that the directed edges of a parallel world graph are correctly identified."""
        self.assertEqual(
            {
                (X @ -d, Y @ -d),
                (Y @ -d, Z @ -d),
                (D @ -d, X @ -d),
                (X @ +x, Y @ +x),
                (Y @ +x, Z @ +x),
            },
            _get_directed_edges(
                NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z), (D, X)]),
                worlds=set([frozenset([+x]), frozenset([-d])]),
            ),
        )

    def test_is_pw_equivalent(self):
        """Test that two nodes in a parallel world graph are the same
        (lemma 24)."""
        event: Event = {Y @ +x: +y, D: -d, Z @ -d: -z, X: -x}
        self.assertTrue(is_pw_equivalent(figure_9b.graph, event, D @ ~X, D))
        self.assertTrue(is_pw_equivalent(figure_9b.graph, event, X @ D, X))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, Z, Z @ ~X))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, W, W @ -d))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, Z @ +x, Z))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, Z @ -d, Z))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, Z @ +x, Z @ -d))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, D @ -d, D))
        self.assertTrue(is_pw_equivalent(figure_9b.graph, event, D @ +x, D))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, Z @ -d, Z))
        self.assertFalse(is_pw_equivalent(figure_9b.graph, event, X, X @ ~X))
        self.assertFalse(is_pw_equivalent(figure_9b.graph, event, Z, Z @ ~X))
        self.assertFalse(is_pw_equivalent(figure_9b.graph, event, D, D @ -d))
        self.assertFalse(is_pw_equivalent(figure_9b.graph, event, X, X @ -x))
        self.assertFalse(is_pw_equivalent(figure_11a.graph, event, X @ +x, X))


    def test_merge_pw(self):
        """Test that the parallel worlds graph after merging two nodes is correct.
        (This is lemma 25)"""
        cf_graph_1, preferred, eliminated = merge_pw(figure_9b.graph, D, D @ ~X)
        cf_graph_2, preferred, eliminated = merge_pw(cf_graph_1, X, X @ D)
        # test that we swap the order of the nodes if the first is a counterfactual
        cf_graph_3,  preferred, eliminated = merge_pw(cf_graph_2, Z @ ~X, Z)
        cf_graph_4,  preferred, eliminated = merge_pw(cf_graph_3, Z, Z @ D)
        cf_graph_5,  preferred, eliminated = merge_pw(cf_graph_4, W, W @ D)
        cf_graph_6, preferred, eliminated = merge_pw(cf_graph_5, D @ D, D)
        cf_graph_7, preferred, eliminated = merge_pw(cf_graph_6, Y, Y @ D)
        # test that we sort the order of the nodes if both are counterfactual
        cf_graph_8, preferred8, eliminated8 = merge_pw(figure_9b.graph, W @ -d, W @ +x)
        # test that we sort the order of the nodes if the both are factual
        cf_graph_9, preferred9, eliminated9 = merge_pw(figure_9b.graph, W, Z)
        self.assert_graph_equal(figure_11a.graph, cf_graph_2)
        self.assert_graph_equal(figure_11b.graph, cf_graph_6)
        self.assert_graph_equal(figure_11c.graph, cf_graph_7)
        self.assert_graph_equal(merge_pw(figure_9b.graph, W @ +x, W @ -d)[0], cf_graph_8)
        self.assert_graph_equal(merge_pw(figure_9b.graph, Z, W)[0], cf_graph_9)


    def test_make_counterfactual_graph(self):
        r"""Test making a counterfactual graph.

        The invocation of **make-cg** with the graph in Figure 9(a) and the joint distribution
        :math:`P(y_x, x', z, d)` will result in the counterfactual graph shown in Fig. 9(c).
        The invocation of **make-cg** with the graph in Figure 9(a) and the joint distribution
        :math:`P(y_{x,z},x')` will result in the counterfactual graph shown in Fig. 9(d).
        """
        actual_graph, actual_event = make_counterfactual_graph(
            figure_9a.graph, {Y @ +x: +y, X: -x, Z @ -d: -z, D: -d}
        )
        self.assert_graph_equal(figure_9c.graph, actual_graph)
        self.assertEqual({Y @ +x: +y, X: -x, Z: -z, D: -d}, actual_event)
        actual_graph2, actual_event2 = make_counterfactual_graph(
            figure_9a.graph, {Y @ (+x, -z): +y, X: -x}
        )
        expected_graph2, expected_event2 = figure_9d.graph, {Y @ (+x, -z): +y, X: -x}
        self.assertEqual(expected_event2, actual_event2)
        self.assert_graph_equal(expected_graph2, actual_graph2)

        # Check for inconsistent counterfactual values for merged nodes
        actual_graph3, actual_event3 = make_counterfactual_graph(
            graph=NxMixedGraph.from_edges(directed=[(D, Z), (Z, Y)]),
            event={Z @ -d: -z, Z: +z, D: -d},
        )
        self.assertIsNone(actual_event3)

        # Check whether {Y_{+x,z,w): -y, X_w: -x} automatically simplifies to {Y_{z,w}: y, X: -x} (it should not)
        actual_graph4, actual_event4 = make_counterfactual_graph(
            graph=figure_9a.graph, event={X @ -W: X, Y @ (-W, +X, -Z): Y}
        )
        expected_event4 = {Y @ (-W, -Z): Y, X: X}
        expected_graph4 = NxMixedGraph.from_edges(
            nodes={W @ -W, Y @ (-W, +X, -Z), Z @ (-W, +X, -Z), X},
            directed={(W @ -W, Y @ (-W, +X, -Z)), (Z @ (-W, +X, -Z), Y @ (-W, +X, -Z))},
            undirected={frozenset({X, Y @ (-W, +X, -Z)})},
        )
        self.assertNotEqual(expected_event4, actual_event4)
        self.assert_graph_equal(expected_graph4, actual_graph4)
