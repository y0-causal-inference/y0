# -*- coding: utf-8 -*-

"""Test graph construction and conversion."""

import unittest
from textwrap import dedent
from typing import Set, Tuple

import networkx as nx

from y0.dsl import A, B, C, D, M, Variable, X, Y, Z
from y0.examples import Example, examples, verma_1
from y0.graph import DEFAULT_TAG, DEFULT_PREFIX, NxMixedGraph
from y0.resources import VIRAL_PATHOGENESIS_PATH


class TestGraph(unittest.TestCase):
    """Test graph construction and conversion."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.addTypeEqualityFunc(NxMixedGraph, self.assert_graph_equal)

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

    def test_example_nodes(self):
        """Test all nodes are variables in example graphs."""
        for example in examples:
            with self.subTest(name=example.name):
                self.assertIsInstance(example, Example)
                non_variables = {
                    node for node in example.graph.nodes() if not isinstance(node, Variable)
                }
                self.assertEqual(0, len(non_variables), msg=f"Found non-variables: {non_variables}")

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
        self.assertEqual(graph, reconstituted)

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
        try:
            from ananke.graphs import ADMG
        except ImportError:
            self.skipTest("ananke is not available")

        expected = NxMixedGraph.from_str_adj(
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
        expected = NxMixedGraph.from_str_edges(directed=[("a", "b"), ("a", "c"), ("b", "a")])
        self.assertEqual(expected, NxMixedGraph.from_str_adj(directed=directed))

    def test_is_acyclic(self):
        """Test the directed edges are acyclic."""
        example = NxMixedGraph.from_str_edges(directed=[("a", "b"), ("a", "c"), ("b", "a")])
        self.assertFalse(nx.algorithms.dag.is_directed_acyclic_graph(example.directed))

    def test_is_not_multigraph(self):
        """Test the undirected edges are not inverses of each other."""
        redundant_edges = [("a", "b"), ("b", "a")]
        directed_edges = [("a", "b")]
        expected = NxMixedGraph.from_str_edges(directed=[("a", "b")], undirected=[("a", "b")])
        actual = NxMixedGraph.from_str_edges(directed=directed_edges, undirected=redundant_edges)
        self.assertEqual(expected, actual)

    def test_subgraph(self):
        """Test generating a subgraph from a set of vertices."""
        graph = NxMixedGraph.from_str_edges(
            directed=[("X", "Y"), ("Y", "Z")],
            undirected=[("X", "Z")],
        )
        self.assertEqual(graph, graph.subgraph({X, Y, Z}))

        subgraph = NxMixedGraph.from_str_edges(directed=[("X", "Y")])
        self.assertEqual(subgraph, graph.subgraph({X, Y}))

    def test_intervention(self):
        """Test generating a subgraph based on an intervention."""
        graph = NxMixedGraph.from_str_edges(
            directed=[("X", "Y"), ("Z", "X")],
            undirected=[("X", "Z"), ("X", "Y"), ("Y", "Z")],
        )
        self.assertEqual(graph, graph.remove_in_edges(set()))

        intervened_graph = NxMixedGraph.from_str_edges(
            directed=[("X", "Y")],
            undirected=[("Z", "Y")],
        )
        self.assertEqual(intervened_graph, graph.remove_in_edges({X}))
        self.assertEqual(intervened_graph, graph.remove_in_edges(X))

        with self.assertRaises(TypeError):
            self.assertEqual(intervened_graph, graph.remove_in_edges({-X}))

    def test_remove_nodes_from(self):
        """Test generating a new graph without the given nodes."""
        graph = NxMixedGraph.from_str_edges(
            directed=[("X", "Y"), ("Z", "X")],
            undirected=[("X", "Z"), ("X", "Y"), ("Y", "Z")],
        )
        self.assertEqual(graph, graph.remove_nodes_from(set()))

        subgraph = NxMixedGraph.from_str_edges(undirected=[("Z", "Y")])
        self.assertEqual(subgraph, graph.remove_nodes_from({X}))

    def test_remove_outgoing_edges_from(self):
        """Test generating a new graph without the outgoing edgs from the given nodes."""
        graph = NxMixedGraph.from_str_edges(directed=[("X", "Y")])
        self.assertEqual(graph, graph.remove_out_edges(set()))

        graph = NxMixedGraph.from_str_edges(undirected=[("X", "Y")])
        self.assertEqual(graph, graph.remove_out_edges(set()))

        graph = NxMixedGraph.from_str_edges(directed=[("W", "X"), ("X", "Y"), ("Y", "Z")])
        expected = NxMixedGraph.from_str_edges(directed=[("W", "X"), ("Y", "Z")])
        self.assertEqual(expected, graph.remove_out_edges({X}))

    def test_ancestors_inclusive(self):
        """Test getting ancestors, inclusive."""
        graph = NxMixedGraph.from_str_edges(
            directed=[("C", "A"), ("C", "B"), ("D", "C"), ("A", "X"), ("A", "Y"), ("B", "Z")]
        )
        self.assertEqual({A, B, C, D}, graph.ancestors_inclusive({A, B}))

        graph = NxMixedGraph.from_str_edges(
            directed=[("X", "Z"), ("Z", "Y")], undirected=[("X", "Y")]
        )
        self.assertEqual({X, Y, Z}, graph.ancestors_inclusive({Y}))
        self.assertEqual({X, Y, Z}, graph.ancestors_inclusive(Y))
        self.assertEqual({X, Z}, graph.ancestors_inclusive({Z}))
        self.assertEqual({X}, graph.ancestors_inclusive({X}))

    def test_get_c_components(self):
        """Test that get_c_components works correctly."""
        g1 = NxMixedGraph().from_str_edges(directed=[("X", "Y"), ("Z", "X"), ("Z", "Y")])
        c1 = [frozenset([X]), frozenset([Y]), frozenset([Z])]
        g2 = NxMixedGraph().from_str_edges(directed=[("X", "Y")], undirected=[("X", "Y")])
        c2 = [frozenset([X, Y])]
        g3 = NxMixedGraph().from_edges(directed=[(X, M), (M, Y)], undirected=[(X, Y)])
        c3 = [frozenset([X, Y]), frozenset([M])]
        for graph, components in [(g1, c1), (g2, c2), (g3, c3)]:
            self.assertIsInstance(graph, NxMixedGraph)
            actual_components = graph.get_c_components()
            self.assertTrue(all(isinstance(c, frozenset) for c in actual_components))
            self.assertTrue(all(isinstance(v, Variable) for c in actual_components for v in c))
            self.assertEqual(components, actual_components)

    def test_counterfactual_predicate(self):
        """Test checking counterfactual graph."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        self.assertFalse(graph.is_counterfactual())
        graph.raise_on_counterfactual()

        graph = NxMixedGraph.from_edges(directed=[(X @ Y, Y)])
        self.assertTrue(graph.is_counterfactual())
        with self.assertRaises(ValueError):
            graph.raise_on_counterfactual()
