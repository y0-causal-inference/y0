# -*- coding: utf-8 -*-

"""Test graph construction and conversion."""

import unittest
from textwrap import dedent
from typing import Set, Tuple

import networkx as nx
from pgmpy.models import BayesianNetwork

from y0.dsl import A, B, C, D, M, Variable, X, Y, Z
from y0.examples import SARS_SMALL_GRAPH, Example, examples, napkin, verma_1
from y0.graph import (
    DEFAULT_TAG,
    DEFULT_PREFIX,
    NxMixedGraph,
    get_nodes_in_directed_paths,
    is_a_fixable,
    is_markov_blanket_shielded,
    is_p_fixable,
)
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
        self.assertEqual({A, X, Y}, graph.descendants_inclusive({A}))
        self.assertEqual({A, B, X, Y, Z}, graph.descendants_inclusive({A, B}))

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
        c1 = {frozenset([X]), frozenset([Y]), frozenset([Z])}
        g2 = NxMixedGraph().from_str_edges(directed=[("X", "Y")], undirected=[("X", "Y")])
        c2 = {frozenset([X, Y])}
        g3 = NxMixedGraph().from_edges(directed=[(X, M), (M, Y)], undirected=[(X, Y)])
        c3 = {frozenset([X, Y]), frozenset([M])}
        for graph, components in [(g1, c1), (g2, c2), (g3, c3)]:
            self.assertIsInstance(graph, NxMixedGraph)
            actual_components = graph.districts()
            self.assertTrue(
                all(isinstance(component, frozenset) for component in actual_components)
            )
            self.assertTrue(
                all(
                    isinstance(node, Variable)
                    for component in actual_components
                    for node in component
                )
            )
            self.assertEqual(components, actual_components)

    def test_get_district(self):
        """Test getting districts."""
        graph = NxMixedGraph().from_edges(directed=[(X, M), (M, Y)], undirected=[(X, Y)])
        self.assertEqual(frozenset([X, Y]), graph.get_district(X))
        self.assertEqual(frozenset([X, Y]), graph.get_district(Y))
        self.assertEqual(frozenset([M]), graph.get_district(M))
        with self.assertRaises(KeyError):
            graph.get_district(Z)

    def test_counterfactual_predicate(self):
        """Test checking counterfactual graph."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        self.assertFalse(graph.is_counterfactual())
        graph.raise_on_counterfactual()

        graph = NxMixedGraph.from_edges(directed=[(X @ Y, Y)])
        self.assertTrue(graph.is_counterfactual())
        with self.assertRaises(ValueError):
            graph.raise_on_counterfactual()

    def test_markov_blanket(self):
        """Test getting a markov blanket."""
        x = [Variable(f"X{i}") for i in range(12)]
        graph = NxMixedGraph.from_edges(
            directed=[
                (x[1], x[5]),
                (x[2], x[6]),
                (x[3], x[6]),
                (x[4], x[3]),
                (x[5], x[8]),
                (x[6], x[8]),
                (x[6], x[9]),
                (x[7], x[9]),
                (x[8], x[10]),
                (x[9], x[11]),
            ]
        )
        self.assertEqual(
            {x[2], x[3], x[5], x[7], x[8], x[9]},
            graph.get_markov_blanket(x[6]),
        )

    def test_disorient(self):
        """Test disorienting a graph."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[(Y, Z)])
        disoriented = graph.disorient()
        self.assertTrue(disoriented.has_edge(X, Y))
        self.assertTrue(disoriented.has_edge(Y, Z))

    def test_pre(self):
        """Test getting the pre-ordering for a given node or set of nodes."""
        g1 = NxMixedGraph.from_str_adj(
            directed={"1": ["2", "3"], "2": ["4", "5"], "3": ["4"], "4": ["5"]}
        )
        g1_ananke = g1.to_admg()
        g1_y0_pre = set(g1.pre(Variable("4")))
        g1_ananke_pre = set(g1_ananke.pre(vertices=["4"], top_order=g1_ananke.topological_sort()))
        g1_y0_pre = {node.name for node in g1_y0_pre}
        self.assertEqual(g1_y0_pre, g1_ananke_pre)

        g2_ananke = SARS_SMALL_GRAPH.to_admg()
        g2_y0_pre = set(SARS_SMALL_GRAPH.pre(Variable("cytok")))
        g2_ananke_pre = set(
            g2_ananke.pre(vertices=["cytok"], top_order=g2_ananke.topological_sort())
        )
        g2_y0_pre = {node.name for node in g2_y0_pre}
        self.assertEqual(g2_y0_pre, g2_ananke_pre)


class TestFixability(unittest.TestCase):
    """A test case for fixability in estimation workflows and tools."""

    def assert_mb_shielded(self, graph: NxMixedGraph):
        """Assert the graph is mb-shielded."""
        self.assertTrue(graph.to_admg().mb_shielded())
        self.assertTrue(is_markov_blanket_shielded(graph))

    def assert_mb_unshielded(self, graph: NxMixedGraph):
        """Assert the graph is not mb-shielded."""
        self.assertFalse(graph.to_admg().mb_shielded())
        self.assertFalse(is_markov_blanket_shielded(graph))

    def test_is_mb_shielded(self):
        """Test checking for markov blanket shielding."""
        # Adapted from https://gitlab.com/causal/ananke/-/blob/dev/tests/estimation/test_automated_if.py#L80-92
        graph_unshielded = NxMixedGraph.from_str_edges(
            directed=[("T", "M"), ("M", "L"), ("L", "Y")], undirected=[("M", "Y")]
        )
        with self.subTest(name="Graph 1"):
            self.assert_mb_unshielded(graph_unshielded)

        # Second test: Napkin model
        with self.subTest(name="Napkin"):
            self.assert_mb_unshielded(napkin)

        # Third test
        with self.subTest(name="Graph 3"):
            self.assert_mb_unshielded(SARS_SMALL_GRAPH)

        # Fourth test
        graph_4 = NxMixedGraph.from_str_edges(
            directed=[
                ("SOS", "Ras"),
                ("Ras", "PI3K"),
                ("Ras", "Raf"),
                ("PI3K", "AKT"),
                ("AKT", "Raf"),
            ],
            undirected=[("SOS", "PI3K")],
        )
        with self.subTest(name="Graph 4"):
            self.assert_mb_shielded(graph_4)

        # Fifth test
        graph_5 = NxMixedGraph.from_str_edges(
            directed=[
                ("Z1", "X"),
                ("X", "M1"),
                ("M1", "Y"),
                ("Z1", "Z2"),
                ("Z2", "Z3"),
                ("Z3", "Y"),
                ("Z2", "M1"),
            ],
            undirected=[("Z1", "X"), ("Z2", "M1")],
        )
        with self.subTest(name="Graph 5"):
            self.assert_mb_unshielded(graph_5)

        # Sixth test
        graph_6 = NxMixedGraph.from_str_edges(
            directed=[("Z1", "X"), ("X", "M1"), ("M1", "Y"), ("Z1", "Y")]
        )
        with self.subTest(name="Graph 6"):
            self.assert_mb_shielded(graph_6)

        # Seventh test
        graph_7 = NxMixedGraph.from_str_edges(directed=[("A", "B"), ("B", "C")])
        with self.subTest(name="Graph 6"):
            self.assert_mb_shielded(graph_7)

    def assert_a_fixable(self, graph: NxMixedGraph, treatment: Variable):
        """Assert that the graph is a-fixable."""
        self.assertTrue(_ananke_a_fixable(graph, treatment))
        self.assertTrue(is_a_fixable(graph, treatment))

    def assert_not_a_fixable(self, graph: NxMixedGraph, treatment: Variable):
        """Assert that the graph is not a-fixable."""
        self.assertFalse(_ananke_a_fixable(graph, treatment))
        self.assertFalse(is_a_fixable(graph, treatment))

    def test_is_a_fixable(self):
        """Test checking for a-fixability.

        Graphs 9, 10, and 11 are from https://gitlab.com/causal/ananke/-/blob/dev/tests/\
        estimation/test_counterfactual_mean.py?ref_type=heads#L20-47
        """
        graph_1 = NxMixedGraph.from_str_edges(
            directed=[("T", "M"), ("M", "L"), ("L", "Y")], undirected=[("M", "Y")]
        )
        treatment_1 = Variable("T")
        self.assert_a_fixable(graph_1, treatment_1)

        self.assert_not_a_fixable(napkin, X)

        graph_3 = NxMixedGraph.from_str_edges(
            directed=[
                ("ADAM17", "EGFR"),
                ("ADAM17", "TNF"),
                ("ADAM17", "Sil6r"),
                ("EGFR", "cytok"),
                ("TNF", "cytok"),
                ("Sil6r", "IL6STAT3"),
                ("IL6STAT3", "cytok"),
            ],
            undirected=[
                ("ADAM17", "cytok"),
                ("ADAM17", "Sil6r"),
                ("EGFR", "TNF"),
                ("EGFR", "IL6STAT3"),
            ],
        )
        treatment_3 = Variable("EGFR")
        self.assert_a_fixable(graph_3, treatment_3)

        graph_4 = NxMixedGraph.from_str_edges(
            directed=[
                ("SOS", "Ras"),
                ("Ras", "PI3K"),
                ("Ras", "Raf"),
                ("PI3K", "AKT"),
                ("AKT", "Raf"),
            ],
            undirected=[("SOS", "PI3K")],
        )
        treatment_4 = Variable("PI3K")
        self.assert_a_fixable(graph_4, treatment_4)

        graph_5 = NxMixedGraph.from_str_edges(
            directed=[
                ("Z1", "X"),
                ("X", "M1"),
                ("M1", "Y"),
                ("Z1", "Z2"),
                ("Z2", "Z3"),
                ("Z3", "Y"),
                ("Z2", "M1"),
            ],
            undirected=[("Z1", "X"), ("Z2", "M1")],
        )
        self.assert_a_fixable(graph_5, X)

        graph_6 = NxMixedGraph.from_str_edges(
            directed=[("Z1", "X"), ("X", "M1"), ("M1", "Y"), ("Z1", "Y")], undirected=[]
        )
        self.assert_a_fixable(graph_6, X)

        graph_7 = NxMixedGraph.from_str_edges(directed=[("A", "B"), ("B", "C")], undirected=[])
        treatment_7 = Variable("A")
        self.assert_a_fixable(graph_7, treatment_7)

        graph_8 = graph_4
        treatment_8 = Variable("SOS")
        self.assert_not_a_fixable(graph_8, treatment_8)

    def assert_p_fixable(self, graph: NxMixedGraph, treatment: Variable):
        """Assert that the graph is p-fixable."""
        self.assertTrue(_ananke_p_fixable(graph, treatment))
        self.assertTrue(is_p_fixable(graph, treatment))

    def assert_not_p_fixable(self, graph: NxMixedGraph, treatment: Variable):
        """Assert that the graph is not p-fixable."""
        self.assertFalse(_ananke_p_fixable(graph, treatment))
        self.assertFalse(is_p_fixable(graph, treatment))

    def test_is_p_fixable(self):
        """Test checking for p-fixability.

        .. seealso:: https://gitlab.com/causal/ananke/-/blob/dev/tests/\
        estimation/test_counterfactual_mean.py?ref_type=heads#L151-212
        """
        graph_1 = NxMixedGraph.from_str_edges(
            directed=[("T", "M"), ("M", "L"), ("L", "Y")], undirected=[("M", "Y")]
        )
        treatment_1 = Variable("T")
        with self.subTest(name="Graph 1"):
            self.assert_p_fixable(graph_1, treatment_1)

        with self.subTest(name="Napkin"):
            self.assert_not_p_fixable(napkin, X)

        treatment_3 = Variable("EGFR")
        with self.subTest(name="Graph 3"):
            self.assert_p_fixable(SARS_SMALL_GRAPH, treatment_3)

        graph_4 = NxMixedGraph.from_str_edges(
            directed=[
                ("SOS", "Ras"),
                ("Ras", "PI3K"),
                ("Ras", "Raf"),
                ("PI3K", "AKT"),
                ("AKT", "Raf"),
            ],
            undirected=[("SOS", "PI3K")],
        )
        treatment_4 = Variable("PI3K")
        with self.subTest(name="Graph 4"):
            self.assert_p_fixable(graph_4, treatment_4)

        graph_5 = NxMixedGraph.from_str_edges(
            directed=[
                ("Z1", "X"),
                ("X", "M1"),
                ("M1", "Y"),
                ("Z1", "Z2"),
                ("Z2", "Z3"),
                ("Z3", "Y"),
                ("Z2", "M1"),
            ],
            undirected=[("Z1", "X"), ("Z2", "M1")],
        )
        with self.subTest(name="Graph 5"):
            self.assert_p_fixable(graph_5, X)

        graph_6 = NxMixedGraph.from_str_edges(
            directed=[("Z1", "X"), ("X", "M1"), ("M1", "Y"), ("Z1", "Y")], undirected=[]
        )
        with self.subTest(name="Graph 6"):
            self.assert_p_fixable(graph_6, X)

        graph_7 = NxMixedGraph.from_str_edges(directed=[("A", "B"), ("B", "C")], undirected=[])
        treatment_7 = Variable("A")
        with self.subTest(name="Graph 7"):
            self.assert_p_fixable(graph_7, treatment_7)

        graph_8 = graph_4
        treatment_8 = Variable("SOS")
        with self.subTest(name="Graph 8"):
            self.assert_p_fixable(graph_8, treatment_8)

        graph_9 = NxMixedGraph.from_str_edges(
            directed=[
                ("C", "T"),
                ("C", "M"),
                ("C", "L"),
                ("C", "Y"),
                ("T", "M"),
                ("M", "L"),
                ("M", "Y"),
                ("L", "Y"),
            ],
            undirected=[("T", "L"), ("T", "Y")],
        )
        treatment_9 = Variable("T")
        with self.subTest(name="Graph 9"):
            self.assert_p_fixable(graph_9, treatment_9)

        graph_10 = NxMixedGraph.from_str_edges(
            directed=[
                ("C", "T"),
                ("C", "M"),
                ("C", "L"),
                ("C", "Y"),
                ("T", "M"),
                ("M", "L"),
                ("T", "Y"),
                ("L", "Y"),
            ],
            undirected=[("T", "L"), ("M", "Y")],
        )
        treatment_10 = Variable("T")
        with self.subTest(name="Graph 10"):
            self.assert_p_fixable(graph_10, treatment_10)

        graph_11 = NxMixedGraph.from_str_edges(
            directed=[
                ("C1", "T"),
                ("C1", "L"),
                ("C2", "M"),
                ("C2", "L"),
                ("C2", "Y"),
                ("T", "M"),
                ("M", "L"),
                ("L", "Y"),
            ],
            undirected=[("Z1", "C1"), ("Z2", "C2"), ("T", "L")],
        )
        treatment_11 = Variable("T")
        with self.subTest(name="Graph 11"):
            self.assert_p_fixable(graph_11, treatment_11)

        graph_12 = NxMixedGraph.from_str_edges(
            directed=[
                ("C2", "T"),
                ("C2", "M1"),
                ("C2", "M2"),
                ("C2", "Y"),
                ("C1", "T"),
                ("C1", "M2"),
                ("T", "M1"),
                ("M1", "M2"),
                ("M2", "Y"),
                ("T", "Y"),
            ],
            undirected=[("T", "M2"), ("M1", "Y")],
        )
        treatment_12 = Variable("T")
        with self.subTest(name="Graph 12"):
            self.assert_p_fixable(graph_12, treatment_12)

        graph_13 = NxMixedGraph.from_str_edges(
            directed=[
                ("ViralLoad", "Income"),
                ("ViralLoad", "T"),
                ("ViralLoad", "Toxicity"),
                ("Education", "Income"),
                ("Education", "T"),
                ("Education", "Toxicity"),
                ("Income", "Insurance"),
                ("Insurance", "T"),
                ("T", "Toxicity"),
                ("Toxicity", "CD4"),
                ("T", "CD4"),
            ],
            undirected=[
                ("Income", "T"),
                ("Insurance", "ViralLoad"),
                ("Education", "CD4"),
                ("Income", "CD4"),
            ],
        )
        treatment_13 = Variable("T")
        with self.subTest(name="Graph 13"):
            self.assert_not_p_fixable(graph_13, treatment_13)


def _ananke_a_fixable(graph: NxMixedGraph, treatment: Variable) -> bool:
    admg = graph.to_admg()
    return 1 == len(admg.district(treatment.name).intersection(admg.descendants([treatment.name])))


def _ananke_p_fixable(graph: NxMixedGraph, treatment: Variable) -> bool:
    admg = graph.to_admg()
    return 0 == len(admg.district(treatment.name).intersection(admg.children([treatment.name])))


class TestToBayesianNetwork(unittest.TestCase):
    """Tests converting a mixed graph to an equivalent :class:`pgmpy.BayesianNetwork`."""

    def assert_bayesian_equal(self, expected: BayesianNetwork, actual: BayesianNetwork) -> None:
        """Compare two instances of :class:`pgmpy.BayesianNetwork`."""
        self.assertEqual(set(expected.edges), set(actual.edges))
        self.assertEqual(expected.latents, actual.latents)

    def test_graph_with_latents(self):
        """Tests converting a mixed graph with latents to an equivalent :class:`pgmpy.BayesianNetwork`."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[(X, Y)])
        expected = BayesianNetwork(
            ebunch=[("X", "Y"), ("U_X_Y", "X"), ("U_X_Y", "Y")], latents=["U_X_Y"]
        )
        actual = graph.to_pgmpy_bayesian_network()
        self.assert_bayesian_equal(expected, actual)

    def test_graph_without_latents(self):
        """Tests converting a mixed graph without latents to an equivalent :class:`pgmpy.BayesianNetwork`."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        expected = BayesianNetwork(ebunch=[("X", "Y")])
        actual = graph.to_pgmpy_bayesian_network()
        self.assert_bayesian_equal(expected, actual)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""

    def test_nodes_in_paths(self):
        """Test getting nodes in paths."""
        graph = NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)])
        self.assertEqual({X, Y, Z}, get_nodes_in_directed_paths(graph, X, Y))
        self.assertEqual({X, Z}, get_nodes_in_directed_paths(graph, X, Z))
        self.assertEqual({Z, Y}, get_nodes_in_directed_paths(graph, Z, Y))
        self.assertEqual(set(), get_nodes_in_directed_paths(graph, Z, X))
        self.assertEqual(set(), get_nodes_in_directed_paths(graph, Y, Z))
        self.assertEqual(set(), get_nodes_in_directed_paths(graph, Y, X))
