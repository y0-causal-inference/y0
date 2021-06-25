# -*- coding: utf-8 -*-

"""Test graph construction and conversion."""

import unittest
from textwrap import dedent
from typing import Set, Tuple

import networkx as nx
from ananke.graphs import ADMG

from y0.examples import verma_1
from y0.graph import DEFAULT_TAG, DEFULT_PREFIX, NxMixedGraph
from y0.resources import VIRAL_PATHOGENESIS_PATH


class TestGraph(unittest.TestCase):
    """Test graph construction and conversion."""

    def test_causaleffect_str_verma_1(self):
        """Test generating R code for the figure 1A graph for causaleffect."""
        expected = dedent(
            """
        g <- graph.formula(V1 -+ V2, V2 -+ V3, V3 -+ V4, V2 -+ V4, V4 -+ V2, simplify = FALSE)
        g <- set.edge.attribute(graph = g, name = "description", index = c(4, 5), value = "U")
        """
        ).strip()
        self.assertEqual(expected, verma_1.to_causaleffect_str())

    def assert_labeled_convertable(
        self, graph: NxMixedGraph, labeled_edges: Set[Tuple[str, str]]
    ) -> None:
        """Test that the graph can be converted to a DAG, then back to an ADMG."""
        prefix = DEFULT_PREFIX
        tag = DEFAULT_TAG

        labeled_dag = graph.to_latent_variable_dag(prefix=prefix, tag=tag)
        for node in labeled_dag:
            self.assertIn(tag, labeled_dag.nodes[node], msg=f"Node: {node}")
            self.assertEqual(node.startswith(prefix), labeled_dag.nodes[node][tag])

        self.assertEqual(labeled_edges, set(labeled_dag.edges()))

        reconstituted = NxMixedGraph.from_latent_variable_dag(labeled_dag, tag=tag)
        self.assertEqual(set(graph.directed.nodes()), set(reconstituted.directed.nodes()))
        self.assertEqual(set(graph.undirected.nodes()), set(reconstituted.undirected.nodes()))
        self.assertEqual(set(graph.directed.edges()), set(reconstituted.directed.edges()))
        self.assertEqual(set(graph.undirected.edges()), set(reconstituted.undirected.edges()))

    def test_convertable(self):
        """Test graphs are convertable."""
        for graph, labeled_edges in [
            (
                verma_1,
                {
                    ("V1", "V2"),
                    ("V2", "V3"),
                    ("V3", "V4"),
                    (f"{DEFULT_PREFIX}0", "V2"),
                    (f"{DEFULT_PREFIX}0", "V4"),
                },
            ),
        ]:
            with self.subTest():
                self.assert_labeled_convertable(graph, labeled_edges)

    def test_from_causalfusion(self):
        """Test importing a CausalFusion graph."""
        graph = NxMixedGraph.from_causalfusion_path(VIRAL_PATHOGENESIS_PATH)
        self.assertIsInstance(graph, NxMixedGraph)

    def test_from_admg(self):
        """Test that all ADMGs can be converted to NxMixedGraph."""
        expected = NxMixedGraph.from_adj(
            directed={"W": [], "X": ["Y"], "Y": ["Z"], "Z": []},
            undirected={"W": [], "X": ["Z"], "Y": [], "Z": []},
        )
        admg = ADMG(
            vertices=["W", "X", "Y", "Z"],
            di_edges=[["X", "Y"], ["Y", "Z"]],
            bi_edges=[["X", "Z"]],
        )
        self.assertEqual(expected, NxMixedGraph.from_admg(admg))

    def test_from_adj(self):
        """Test the adjacency graph is not a multigraph."""
        directed = dict([("a", ["b", "c"]), ("b", ["a"]), ("c", [])])
        expected = NxMixedGraph.from_edges(directed=[("a", "b"), ("a", "c"), ("b", "a")])
        self.assertEqual(expected, NxMixedGraph.from_adj(directed=directed))

    def test_is_acyclic(self):
        """Test the directed edges are acyclic."""
        example = NxMixedGraph.from_edges(directed=[("a", "b"), ("a", "c"), ("b", "a")])
        self.assertFalse(nx.algorithms.dag.is_directed_acyclic_graph(example.directed))

    def test_is_not_multigraph(self):
        """Test the undirected edges are not inverses of each other."""
        redundant_edges = [("a", "b"), ("b", "a")]
        directed_edges = [("a", "b")]
        expected = NxMixedGraph.from_edges(directed=[("a", "b")], undirected=[("a", "b")])
        actual = NxMixedGraph.from_edges(directed=directed_edges, undirected=redundant_edges)
        self.assertEqual(expected, actual)

    def test_subgraph(self):
        """Test generating a subgraph from a set of vertices."""
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Y")
        graph.add_directed_edge("Y", "Z")
        graph.add_undirected_edge("X", "Z")
        self.assertEqual(graph, graph.subgraph({"X", "Y", "Z"}))

        subgraph = NxMixedGraph()
        subgraph.add_directed_edge("X", "Y")
        self.assertEqual(subgraph, graph.subgraph({"X", "Y"}))

    def test_intervention(self):
        """Test generating a subgraph based on an intervention."""
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Y")
        graph.add_directed_edge("Z", "X")
        graph.add_undirected_edge("X", "Z")
        graph.add_undirected_edge("X", "Y")
        graph.add_undirected_edge("Y", "Z")
        self.assertEqual(graph, graph.intervene(set()))

        intervened_graph = NxMixedGraph()
        intervened_graph.add_directed_edge("X", "Y")
        intervened_graph.add_undirected_edge("Z", "Y")
        self.assertEqual(intervened_graph, graph.intervene({"X"}))

    def test_remove_nodes_from(self):
        """Test generating a new graph without the given nodes."""
        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Y")
        graph.add_directed_edge("Z", "X")
        graph.add_undirected_edge("X", "Z")
        graph.add_undirected_edge("X", "Y")
        graph.add_undirected_edge("Y", "Z")
        self.assertEqual(graph, graph.remove_nodes_from(set()))

        subgraph = NxMixedGraph()
        subgraph.add_undirected_edge("Z", "Y")
        self.assertEqual(subgraph, graph.remove_nodes_from({"X"}))

    def test_ancestors_inclusive(self):
        """Test getting ancestors, inclusive."""
        graph = NxMixedGraph()
        graph.add_directed_edge("C", "A")
        graph.add_directed_edge("C", "B")
        graph.add_directed_edge("D", "C")
        graph.add_directed_edge("A", "X")
        graph.add_directed_edge("A", "Y")
        graph.add_directed_edge("B", "Z")
        self.assertEqual({"A", "B", "C", "D"}, graph.ancestors_inclusive({"A", "B"}))

        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Z")
        graph.add_directed_edge("Z", "Y")
        graph.add_undirected_edge("X", "Y")
        self.assertEqual({"X", "Y", "Z"}, graph.ancestors_inclusive({"Y"}))
        self.assertEqual({"X", "Z"}, graph.ancestors_inclusive({"Z"}))
        self.assertEqual({"X"}, graph.ancestors_inclusive({"X"}))

    def test_get_c_components(self):
        """Test that get_c_components works correctly."""
        g1 = NxMixedGraph().from_edges(directed=[("X", "Y"), ("Z", "X"), ("Z", "Y")])
        c1 = [frozenset(["X"]), frozenset(["Y"]), frozenset(["Z"])]
        g2 = NxMixedGraph().from_edges(directed=[("X", "Y")], undirected=[("X", "Y")])
        c2 = [frozenset(["X", "Y"])]
        g3 = NxMixedGraph().from_edges(directed=[("X", "M"), ("M", "Y")], undirected=[("X", "Y")])
        c3 = [frozenset(["X", "Y"]), frozenset(["M"])]
        for graph, components in [(g1, c1), (g2, c2), (g3, c3)]:
            self.assertIsInstance(graph, NxMixedGraph)
            self.assertEqual(components, graph.get_c_components())
