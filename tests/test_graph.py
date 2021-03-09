# -*- coding: utf-8 -*-

"""Test graph construction and conversion."""

import unittest
from textwrap import dedent
from typing import Set, Tuple

from y0.examples import verma_1
from y0.graph import DEFAULT_TAG, DEFULT_PREFIX, NxMixedGraph


class TestGraph(unittest.TestCase):
    """Test graph construction and conversion."""

    def test_causaleffect_str_verma_1(self):
        """Test generating R code for the figure 1A graph for causaleffect."""
        expected = dedent('''
        g <- graph.formula(V1 -+ V2, V2 -+ V3, V3 -+ V4, V2 -+ V4, V4 -+ V2, simplify = FALSE)
        g <- set.edge.attribute(graph = g, name = "description", index = c(4, 5), value = "U")
        ''').strip()
        self.assertEqual(expected, verma_1.to_causaleffect_str())

    def assert_labeled_convertable(self, graph: NxMixedGraph, labeled_edges: Set[Tuple[str, str]]) -> None:
        """Test that the graph can be converted to a DAG, then back to an ADMG."""
        prefix = DEFULT_PREFIX
        tag = DEFAULT_TAG

        labeled_dag = graph.to_latent_variable_dag(prefix=prefix, tag=tag)
        for node in labeled_dag:
            self.assertIn(tag, labeled_dag.nodes[node], msg=f'Node: {node}')
            self.assertEqual(node.startswith(prefix), labeled_dag.nodes[node][tag])

        self.assertEqual(labeled_edges, set(labeled_dag.edges()))

        reconstituted = NxMixedGraph.from_latent_variable_dag(labeled_dag, tag=tag)
        self.assertEqual(set(graph.directed.nodes()), set(reconstituted.directed.nodes()))
        self.assertEqual(set(graph.undirected.nodes()), set(reconstituted.undirected.nodes()))
        self.assertEqual(set(graph.directed.edges()), set(reconstituted.directed.edges()))
        self.assertEqual(set(graph.undirected.edges()), set(reconstituted.undirected.edges()))

    def test_convertable(self):
        """Tests graphs are convertable."""
        for graph, labeled_edges in [
            (verma_1, {
                ('V1', 'V2'), ('V2', 'V3'), ('V3', 'V4'),
                (f'{DEFULT_PREFIX}0', 'V2'), (f'{DEFULT_PREFIX}0', 'V4'),
            }),
        ]:
            with self.subTest():
                self.assert_labeled_convertable(graph, labeled_edges)
