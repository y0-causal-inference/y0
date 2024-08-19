# -*- coding: utf-8 -*-

"""Tests for parallel world graphs and counterfactual graphs."""

import unittest

from tests.test_algorithm import cases
from y0.algorithm.identify.cg import (
    World,
    _get_directed_edges,
    extract_interventions,
    has_same_confounders,
    has_same_function,
    is_inconsistent,
    is_not_self_intervened,
    is_pw_equivalent,
    make_counterfactual_graph,
    make_parallel_worlds_graph,
    merge_pw,
    node_not_an_intervention_in_world,
    nodes_attain_same_value,
    nodes_have_same_domain_of_values,
    parents_attain_same_values,
    stitch_counterfactual_and_doppleganger_neighbors,
    stitch_counterfactual_and_dopplegangers,
    stitch_counterfactual_and_neighbors,
    stitch_factual_and_doppleganger_neighbors,
    stitch_factual_and_dopplegangers,
    value_of_self_intervention,
)
from y0.dsl import A, B, D, Event, W, X, Y, Z
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

    def test_world(self):
        """Test that a world contains an intervention."""
        with self.assertRaises(TypeError):
            input_world1: World = World([-x])
            3 in input_world1  # noqa

        with self.assertRaises(TypeError):
            input_world1: World = World([3])
            3 in input_world1  # noqa

        input_world2 = World([-x])
        self.assertFalse(+x in input_world2)
        self.assertFalse(+y in input_world2)
        self.assertTrue(-x in input_world2)

    def assert_uedge_set_equal(self, expected, actual):
        """Assert that two undirected edge sets are equal."""
        return self.assertEqual(
            {frozenset(edge) for edge in expected}, {frozenset(edge) for edge in actual}
        )

    def test_make_parallel_worlds(self):
        """Test that parallel worlds graphs are correct."""
        expected = figure_9b.graph
        actual = make_parallel_worlds_graph(figure_9a.graph, {frozenset([-x]), frozenset([-d])})
        expected2 = make_parallel_worlds_graph(figure_9a.graph, {(-x, -z)})
        self.assert_graph_equal(expected, actual)
        self.assert_graph_equal(expected2, make_parallel_worlds_graph(figure_9a.graph, {(-x, -z)}))
        self.assertTrue(Y @ (-z, -x) in expected2.nodes())
        self.assertTrue(Y @ (-x, -z) in expected2.nodes())

    def test_has_same_function(self):
        """Test that two variables have the same value."""
        self.assertTrue(has_same_function(D @ X, D))
        self.assertFalse(has_same_function(D @ D, D))
        self.assertFalse(has_same_function(X, X @ +x))
        self.assertTrue(has_same_function(X @ D, X))
        self.assertFalse(has_same_function(X, D))
        self.assertFalse(has_same_function(X @ ~X, W @ ~X))
        self.assertFalse(has_same_function(X @ ~X, X))
        self.assertTrue(has_same_function(D, D @ ~x))
        self.assertTrue(has_same_function(Z @ ~x, Z))
        self.assertTrue(has_same_function(Z @ ~x, Z @ -d))
        self.assertTrue(has_same_function(Z @ -d, Z))
        self.assertTrue(has_same_function(Z @ (-d, -z), Z @ (-x, -z)))
        self.assertTrue(has_same_function(Z @ (-d, -z), Z @ (-x, +z)))
        self.assertFalse(has_same_function(Z @ (-d, -z), Z @ (-d, +x)))

    def test_nodes_attain_same_value(self):
        """Test that two variables attain the same value."""
        event: Event = {D: -d}
        self.assertTrue(nodes_attain_same_value(figure_11a.graph, event, D, D @ -d))
        self.assertTrue(nodes_attain_same_value(figure_11a.graph, event, D @ -d, D))
        self.assertTrue(
            nodes_attain_same_value(
                NxMixedGraph.from_edges(directed=[(D @ +d, Z @ +d)]), event, D @ +d, D @ +d
            )
        )
        self.assertTrue(nodes_attain_same_value(figure_9b.graph, event, D, D))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D @ -d, D))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D, D @ -d))
        self.assertFalse(nodes_attain_same_value(figure_11a.graph, event, Z, Z @ +x))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D, X))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, X, D))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D, X @ -d))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, X @ -d, D))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D, X @ -d))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D @ -d, D))
        # This should get us onto the return False on line 56
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, Y @ -X, X))
        # This one should trigger the False on line 59
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, {Y @ -X: -Y, Y: +Y}, Y @ -X, Y))
        # This one will trigger the True on line 61, but I'm not sure if the event really makes sense

        self.assertTrue(nodes_attain_same_value(figure_9b.graph, {Y @ -X: -Y, Y: -Y}, Y @ -X, Y))
        # I think these two should get the falses on lines 64 and 69
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, {Y @ -X: -Y}, Y @ -X, Y))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, {Y: +Y}, Y @ -X, Y))
        # These two should get the False on line 73
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, {}, Y @ -X, Y))
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, {}, Y, Y @ -X))
        # These two should get the True on line 74
        self.assertTrue(
            nodes_attain_same_value(NxMixedGraph.from_edges(undirected=[(+Y, -Y)]), {}, +Y, -Y)
        )
        self.assertFalse(nodes_attain_same_value(figure_9b.graph, event, D @ -X, D))

    def test_has_same_confounders(self):
        """Check whether two nodes have the same confounding edges."""
        self.assertFalse(has_same_confounders(figure_9b.graph, D, D))
        self.assertFalse(has_same_confounders(figure_9b.graph, D, D @ -d))
        self.assertFalse(has_same_confounders(figure_9b.graph, D @ -d, D))
        self.assertTrue(has_same_confounders(figure_9b.graph, D @ -x, D))
        self.assertTrue(has_same_confounders(figure_11a.graph, D @ -d, D))
        self.assertTrue(has_same_confounders(figure_9b.graph, D, D @ -x))

    def test_parents_attain_same_values(self):
        """Test that the parents of two nodes attain the same value."""
        graph = figure_9b.graph
        event: Event = {Y @ -x: -y, D: -d, Z @ -d: -z, X: +x}
        self.assertTrue(parents_attain_same_values(figure_11a.graph, event, Z, Z @ -d))
        self.assertTrue(parents_attain_same_values(figure_11a.graph, event, Z, Z @ -x))
        self.assertTrue(parents_attain_same_values(figure_11a.graph, event, Z @ -d, Z @ -x))
        self.assertTrue(parents_attain_same_values(figure_11a.graph, event, D, D @ -d))
        self.assertFalse(parents_attain_same_values(graph, event, Z, Z @ -d))
        self.assertFalse(parents_attain_same_values(figure_9b.graph, event, D, D @ -d))
        self.assertFalse(parents_attain_same_values(figure_9b.graph, event, X, X @ -x))
        self.assertFalse(
            parents_attain_same_values(
                NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y), (W, Y)]), event, X, Y
            )
        )
        self.assertFalse(
            parents_attain_same_values(
                NxMixedGraph.from_edges(
                    directed=[(X @ -x, Y @ -x), (X @ +x, Y @ +x)], undirected=[(Y @ -x, Y @ +x)]
                ),
                {Y @ -x: -y, Y @ +x: +y},
                Y @ -x,
                Y @ +x,
            )
        )

    def test_nodes_have_same_domain_of_values(self):
        """Test that two nodes have the same domain of values."""
        graph = figure_9b.graph
        event = {Y @ -x: -y, X: +x, D: -d, Z @ -d: -z}
        self.assertTrue(nodes_have_same_domain_of_values(graph, event, D, D @ -x))
        self.assertFalse(nodes_have_same_domain_of_values(graph, event, D, D @ -d))
        self.assertFalse(
            nodes_have_same_domain_of_values(
                NxMixedGraph.from_edges(
                    directed=[(X, Y), (X @ +x, Y @ +x)], undirected=[(Y, Y @ +x)]
                ),
                {Y @ +x: +y, Y @ -x: -y},
                X,
                X @ +x,
            )
        )
        self.assertTrue(nodes_have_same_domain_of_values(figure_9b.graph, event, D @ -X, D))
        self.assertFalse(nodes_have_same_domain_of_values(figure_9b.graph, event, X, Y))
        self.assertFalse(
            nodes_have_same_domain_of_values(
                NxMixedGraph.from_edges(
                    directed=[(X @ +x, Y @ +x), (X @ -x, Y @ -x), (X, Y)],
                    undirected=[(Y @ -x, Y @ +x), (Y @ +x, Y), (Y @ -x, Y)],
                ),
                {Y @ -x: -y, Y @ +x: +y},
                X @ -x,
                X @ +x,
            )
        )
        self.assertTrue(
            nodes_have_same_domain_of_values(
                NxMixedGraph.from_edges(
                    directed=[(X @ +x, Y @ +x), (X @ -x, Y @ -x), (X, Y)],
                    undirected=[(Y @ -x, Y @ +x), (Y @ +x, Y), (Y @ -x, Y)],
                ),
                {Y @ -x: -y, Y @ +x: +y},
                X @ +x,
                X @ +x,
            )
        )
        self.assertTrue(
            nodes_have_same_domain_of_values(
                NxMixedGraph.from_edges(
                    directed=[(X @ +x, Y @ +x), (X @ -x, Y @ -x), (X, Y)],
                    undirected=[(Y @ -x, Y @ +x), (Y @ +x, Y), (Y @ -x, Y)],
                ),
                {Y @ -x: +y, Y @ +x: +y},
                Y @ -x,
                Y @ +x,
            )
        )

    def test_value_of_self_intervention(self):
        """Return the value of a self intervention."""
        self.assertIsNone(value_of_self_intervention(X))
        self.assertEqual(+x, value_of_self_intervention(X @ +x))
        self.assertEqual(-x, value_of_self_intervention(X @ (-x, -y, -d)))
        self.assertIsNone(value_of_self_intervention(X @ -y))

    def test_extract_interventions(self):
        """Test that all interventions are extracted."""
        event = {X: +x, D: -d, Z @ -d: -z}
        expected_worlds = {frozenset({-d})}
        self.assertEqual(expected_worlds, extract_interventions(event))

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
        """Test that a node is not an intervention in a world."""
        self.assertTrue(node_not_an_intervention_in_world(world=frozenset([-x]), node=Y))
        self.assertFalse(node_not_an_intervention_in_world(world=frozenset([-x, +z]), node=X))
        self.assertTrue(node_not_an_intervention_in_world(world=frozenset([-x, +x]), node=Y))
        self.assertFalse(node_not_an_intervention_in_world(world=frozenset([-x]), node=X))
        self.assertFalse(node_not_an_intervention_in_world(world=frozenset([-x, -y]), node=Y))
        self.assertFalse(node_not_an_intervention_in_world(world=frozenset([-x, +y]), node=Y))
        self.assertFalse(node_not_an_intervention_in_world(world=frozenset([-x, +y, -y]), node=Y))
        with self.assertRaises(TypeError):
            node_not_an_intervention_in_world(world=frozenset([-x]), node=+Y)

        with self.assertRaises(TypeError):
            node_not_an_intervention_in_world(world=frozenset({-x}), node=X @ +x)

    def test_stitch_factual_and_dopplegangers(self):
        """Test that factual variables and their dopplegangers are stitched together unless it is intervened upon."""
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

    def test_is_not_self_intervened(self):
        """Test that we can detect when a counterfactual variable intervenes on itself."""
        self.assertFalse(is_not_self_intervened(Y @ (+x, -y)))
        self.assertTrue(is_not_self_intervened(Y @ (+x, -z)))
        self.assertFalse(is_not_self_intervened(Y @ (+x, +y)))

    def test_stitch_factual_and_doppleganger_neighbors(self):
        """Test that factual variables and their dopplegangers are stitched together unless it is intervened upon."""
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
        """Test counterfactual variables and their dopplegangers are stitched together unless it is intervened upon."""
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
        """Test that counterfactual variables and their neighbor dopplegangers are stitched together."""
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
        """Test counterfactual variables and their neighbors are stitched together."""
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
        """Test that two nodes in a parallel world graph are the same (lemma 24)."""
        event: Event = {Y @ -x: -y, D: -d, Z @ -d: -z, X: +x}
        self.assertTrue(is_pw_equivalent(figure_9b.graph, event, D @ -X, D))
        self.assertTrue(is_pw_equivalent(figure_9b.graph, event, X @ -D, X))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, Z, Z @ -X))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, W, W @ -d))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, Z @ -x, Z))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, Z @ -d, Z))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, Z @ -x, Z @ -d))
        self.assertFalse(is_pw_equivalent(figure_11a.graph, event, D @ -d, D))
        self.assertTrue(is_pw_equivalent(figure_9b.graph, event, D @ -x, D))
        self.assertTrue(is_pw_equivalent(figure_11a.graph, event, Z @ -d, Z))
        self.assertFalse(is_pw_equivalent(figure_9b.graph, event, X, X @ -X))
        self.assertFalse(is_pw_equivalent(figure_9b.graph, event, Z, Z @ -X))
        self.assertFalse(is_pw_equivalent(figure_9b.graph, event, D, D @ -d))
        self.assertFalse(is_pw_equivalent(figure_9b.graph, event, X, X @ -x))
        self.assertFalse(is_pw_equivalent(figure_11a.graph, event, X @ -x, X))
        self.assertFalse(
            is_pw_equivalent(
                NxMixedGraph.from_edges(
                    directed=[(X, Y), (X @ +x, Y @ +y), (X @ -x, Y @ -y)],
                    undirected=[(Y @ +y, Y @ -y), (Y @ -y, Y)],
                ),
                {Y @ +x: +y, Y @ -x: -y},
                X @ +x,
                X @ -x,
            )
        )
        self.assertFalse(
            is_pw_equivalent(
                NxMixedGraph.from_edges(
                    directed=[(X, Y), (X @ +x, Y @ +x), (X @ -x, Y @ -x)],
                    undirected=[(Y @ +x, Y @ -x), (Y @ -x, Y)],
                ),
                {Y @ +x: +y, Y @ -x: -y},
                X,
                X @ -x,
            )
        )
        self.assertFalse(
            is_pw_equivalent(
                NxMixedGraph.from_edges(
                    directed=[(X, Y), (X @ +x, Y @ +y), (X @ -x, Y @ -y)],
                    undirected=[(Y @ +y, Y @ -y), (Y @ -y, Y)],
                ),
                {Y @ +x: +y, Y @ -x: -y},
                X,
                X @ +x,
            )
        )
        self.assertFalse(
            is_pw_equivalent(
                NxMixedGraph.from_edges(
                    directed=[(X, Y), (X @ +x, Y @ +x), (X @ -x, Y @ -x)],
                    undirected=[(Y @ +y, Y @ -x), (Y @ -x, Y)],
                ),
                {Y @ +x: +y, Y @ -x: -y},
                Y @ +x,
                Y @ -x,
            )
        )
        self.assertFalse(
            is_pw_equivalent(
                NxMixedGraph.from_edges(
                    directed=[(X, Y), (X @ +x, Y @ +x), (X @ -x, Y @ -x)],
                    undirected=[(Y @ +x, Y @ -x), (Y @ -x, Y)],
                ),
                {Y @ +x: +y, Y @ -x: -y},
                Y @ +x,
                Y,
            )
        )
        self.assertFalse(
            is_pw_equivalent(
                NxMixedGraph.from_edges(
                    directed=[(X, Y), (X @ +x, Y @ +x), (X @ -x, Y @ -x)],
                    undirected=[(Y @ +x, Y @ -x), (Y @ -x, Y)],
                ),
                {Y @ +x: +y, Y @ -x: -y},
                Y,
                Y @ -x,
            )
        )
        self.assertTrue(
            is_pw_equivalent(
                NxMixedGraph.from_edges(
                    directed=[(X, Y), (X @ +x, Y @ +x), (X @ -x, Y @ -x)],
                    undirected=[(Y @ +x, Y @ -x), (Y @ -x, Y)],
                ),
                {Y @ +x: +y, Y @ -x: -y, X: -x},
                Y,
                Y @ -x,
            )
        )

    def test_merge_pw(self):
        """Test the parallel worlds graph after merging two nodes is correct (Lemma 25)."""
        cf_graph_1, preferred, eliminated = merge_pw(figure_9b.graph, D, D @ -X)
        cf_graph_2, preferred, eliminated = merge_pw(cf_graph_1, X, X @ D)
        # test that we swap the order of the nodes if the first is a counterfactual
        cf_graph_3, preferred, eliminated = merge_pw(cf_graph_2, Z @ -X, Z)
        cf_graph_4, preferred, eliminated = merge_pw(cf_graph_3, Z, Z @ D)
        cf_graph_5, preferred, eliminated = merge_pw(cf_graph_4, W, W @ D)
        cf_graph_6, preferred, eliminated = merge_pw(cf_graph_5, D @ -D, D)
        cf_graph_7, preferred, eliminated = merge_pw(cf_graph_6, Y, Y @ -D)
        self.assert_graph_equal(figure_11a.graph, cf_graph_2)
        self.assert_graph_equal(figure_11b.graph, cf_graph_6)
        self.assert_graph_equal(figure_11c.graph, cf_graph_7)
        self.assertNotIn(D @ -d, merge_pw(figure_11a.graph, Z, Z @ -d)[0].nodes())

    def test_merge_pw_both_counterfactual(self):
        """Test that we sort the order of the nodes if both are counterfactual."""
        cf_graph_1, _, _ = merge_pw(figure_9b.graph, W @ -d, W @ -x)
        cf_graph_2, _, _ = merge_pw(figure_9b.graph, W @ -x, W @ -d)
        self.assert_graph_equal(cf_graph_2, cf_graph_1)

    def test_merge_pw_both_factual(self):
        """Test that we sort the order of the nodes if the both are factual."""
        cf_graph_1, _, _ = merge_pw(figure_9b.graph, W, Z)
        cg_graph_1, _, _ = merge_pw(figure_9b.graph, Z, W)
        self.assert_graph_equal(cg_graph_1, cf_graph_1)

    def test_is_inconsistent(self):
        r"""Test whether two nodes are inconsistent."""
        self.assertTrue(is_inconsistent({D: -d, D @ +x: +d}, D, D @ ~X))
        self.assertTrue(is_inconsistent({D @ -x: -d, D @ +x: +d}, D @ +x, D @ -x))
        self.assertTrue(is_inconsistent({Y @ -x: -y, Y @ +x: +y}, Y @ +x, Y @ -x))


class TestMakeCounterfactualGraph(cases.GraphTestCase):
    r"""Test making a counterfactual graph.

    The invocation of **make-cg** with the graph in Figure 9(a) and the joint distribution
    :math:`P(y_x, x', z, d)` will result in the counterfactual graph shown in Fig. 9(c).
    The invocation of **make-cg** with the graph in Figure 9(a) and the joint distribution
    :math:`P(y_{x,z},x')` will result in the counterfactual graph shown in Fig. 9(d).
    """

    def test_1(self):
        """Check JZ scenario 1."""
        actual_graph, actual_event = make_counterfactual_graph(
            figure_9a.graph, {Y @ -x: -y, X: +x, Z @ -d: -z, D: -d}
        )
        self.assert_graph_equal(figure_9c.graph, actual_graph)
        self.assertEqual({Y @ -x: -y, X: +x, Z: -z, D: -d}, actual_event)

    def test_2(self):
        """Check JZ scenario 2."""
        cf_graph, new_event = make_counterfactual_graph(figure_9a.graph, {Y @ (-x, -z): -y, X: +x})
        self.assertEqual({Y @ (-x, -z): -y, X: +x}, new_event)
        self.assert_graph_equal(figure_9d.graph, cf_graph)

    def test_3(self):
        """Check for inconsistent counterfactual values for merged nodes."""
        _, new_event = make_counterfactual_graph(
            graph=NxMixedGraph.from_edges(directed=[(D, Z), (Z, Y)]),
            event={Z @ -d: -z, Z: +z, D: -d},
        )
        self.assertIsNone(new_event)

        # # Check whether {Y_{+x,z,w): -y, X_w: -x} automatically simplifies to {Y_{z,w}: y, X: -x} (it should not)
        # actual_graph4, actual_event4 = make_counterfactual_graph(
        #     graph=figure_9a.graph, event={X @ -W: X, Y @ (-W, +X, -Z): Y}
        # )
        # expected_event4 = {Y @ (-W, -Z): Y, X: X}
        # expected_graph4 = NxMixedGraph.from_edges(
        #     nodes={W @ -W, Y @ (-W, +X, -Z), Z @ (-W, +X, -Z), X},
        #     directed={(W @ -W, Y @ (-W, +X, -Z)), (Z @ (-W, +X, -Z), Y @ (-W, +X, -Z))},
        #     undirected={frozenset({X, Y @ (-W, +X, -Z)})},
        # )
        # self.assertNotEqual(expected_event4, actual_event4)
        # self.assert_graph_equal(expected_graph4, actual_graph4)

    def test_5(self):
        """Check whether the counterfactual graph is consistent (it is not)."""
        _, new_event = make_counterfactual_graph(
            graph=NxMixedGraph.from_edges(directed=[(W, X), (Z, X), (X, Y)]),
            event={Y @ -w: -y, Y @ -z: +y, X @ -w: +x, X @ -z: +x},
        )
        self.assertIsNone(new_event)

    def test_6(self):
        """Check whether Probability of necessary and sufficient causation induces a W graph."""
        actual_cf_graph, new_event = make_counterfactual_graph(
            graph=NxMixedGraph.from_edges(directed=[(X, Y)]), event={Y @ -x: -y, Y @ +x: +y}
        )
        expected_cf_graph = NxMixedGraph.from_edges(
            directed=[(X @ -x, Y @ -x), (X @ +x, Y @ +x)], undirected=[(Y @ -x, Y @ +x)]
        )

        self.assert_graph_equal(expected_cf_graph, actual_cf_graph)
        self.assertEqual({Y @ -x: -y, Y @ +x: +y}, new_event)

    def test_7(self):
        """Check that a triplet world graph is not inconsistent."""
        actual_cf_graph, new_event = make_counterfactual_graph(
            graph=NxMixedGraph.from_edges(
                directed=[(X, W), (W, Y), (D, Z), (Z, Y), (A, B), (B, Y)], undirected=[(X, Y)]
            ),
            event={Y @ +x: +y, X: -x, Z @ -d: -z, D: -d, A: +A},
        )
        self.assertIsNotNone(new_event)
        expected_cf_graph = NxMixedGraph.from_edges(
            directed=[(D, Z), (B, Y @ +X), (W @ +X, Y @ +X), (Z, Y @ +X), (A, B), (X @ +X, W @ +X)],
            undirected=[(X, Y @ +X)],
        )
        self.assert_graph_equal(expected_cf_graph, actual_cf_graph)

    @unittest.skip(reason="This relied on unstable sort before")
    def test_8(self):
        """Check JZ scenario 8."""
        actual_cf_graph, new_event = make_counterfactual_graph(
            graph=NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)]),
            event={Y @ -x: -y, Y @ +x: -y, Z @ +x: -z, Z @ -x: -z},
        )
        expected_cf_graph = NxMixedGraph.from_edges(
            directed=[(X @ -x, Z @ -x), (Z @ -x, Y @ -x), (X @ +x, Z @ +x)],
            undirected=[(Z @ -x, Z @ +x)],
        )
        self.assert_graph_equal(expected_cf_graph, actual_cf_graph, sort=True)
        self.assertEqual({Y @ -X: -Y, Z @ +X: -Z, Z @ -X: -Z}, new_event)
