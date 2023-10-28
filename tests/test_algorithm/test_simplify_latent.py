# -*- coding: utf-8 -*-

"""Tests for simplifying latent variable DAGs."""

import unittest

import networkx as nx

from y0.algorithm.simplify_latent import (
    DEFAULT_SUFFIX,
    iter_latents,
    remove_redundant_latents,
    remove_unidirectional_latents,
    remove_widow_latents,
    simplify_latent_dag,
    transform_latents_with_parents,
)
from y0.algorithm.taheri_design import taheri_design_dag
from y0.dsl import U1, U2, U3, Y1, Y2, Y3, U, Variable, W
from y0.examples import igf_example
from y0.graph import set_latent

X1, X2, X3 = map(Variable, ["X1", "X2", "X3"])
U_LATENT = Variable(f"U{DEFAULT_SUFFIX}")


class TestDesign(unittest.TestCase):
    """Test the design algorithm."""

    def test_design(self):
        """Test the design algorithm."""
        results = taheri_design_dag(igf_example.graph.directed, cause="PI3K", effect="Erk", stop=3)
        self.assertIsNotNone(results)
        # FIXME do better than this.


def _dag_from_adj_str(directed):
    rv = nx.DiGraph()
    rv.add_edges_from((k, value) for k, values in directed.items() for value in values)
    return rv


class TestSimplify(unittest.TestCase):
    """Tests for the Robin Evans simplification algorithms."""

    def assert_latent_variable_dag_equal(self, expected, actual) -> None:
        """Check two latent variable DAGs are the same."""
        self.assertEqual(
            sorted(expected),
            sorted(actual),
            msg="Nodes are incorrect",
        )
        self.assertEqual(
            dict(expected.nodes.items()),
            dict(actual.nodes.items()),
            msg="Tags are incorrect",
        )
        self.assertEqual(
            sorted(expected.edges()),
            sorted(actual.edges()),
        )

    def test_remove_widows(self):
        """Test simplification 1 - removing widows."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                (X1, X2),
                (X2, W),
                (U, X2),
                (U, X3),
                (U, W),
            ]
        )
        latents = {U, W}
        set_latent(graph, latents)
        self.assertEqual(latents, set(iter_latents(graph)))

        # Apply the simplification
        _, removed = remove_widow_latents(graph)
        self.assertEqual({W}, removed)

        expected = nx.DiGraph()
        expected.add_edges_from(
            [
                (X1, X2),
                (U, X2),
                (U, X3),
            ]
        )
        set_latent(expected, U)

        self.assert_latent_variable_dag_equal(expected, graph)

    def test_transform_latents_with_parents(self):
        """Test simplification 2: latents with parents can be transformed."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                (X1, U),
                (X2, U),
                (U, Y1),
                (U, Y2),
                (U, Y3),
            ]
        )
        set_latent(graph, U)
        # Apply the simplification
        transform_latents_with_parents(graph)

        expected = nx.DiGraph()
        expected.add_edges_from(
            [
                (X1, Y1),
                (X1, Y2),
                (X1, Y3),
                (X2, Y1),
                (X2, Y2),
                (X2, Y3),
                (Variable(f"U{DEFAULT_SUFFIX}"), Y1),
                (Variable(f"U{DEFAULT_SUFFIX}"), Y2),
                (Variable(f"U{DEFAULT_SUFFIX}"), Y3),
            ]
        )
        set_latent(expected, Variable(f"U{DEFAULT_SUFFIX}"))

        self.assert_latent_variable_dag_equal(expected, graph)

    def test_remove_redundant_latents(self):
        """Test simplification 3 - remove redundant latents."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                (U, X1),
                (U, X2),
                (U, X3),
                (W, X1),
                (W, X2),
            ]
        )
        set_latent(graph, [U, W])
        # Apply the simplification
        remove_redundant_latents(graph)

        expected = nx.DiGraph()
        expected.add_edges_from(
            [
                (U, X1),
                (U, X2),
                (U, X3),
            ]
        )
        set_latent(expected, [U])

        self.assert_latent_variable_dag_equal(expected, graph)

    def test_remove_unidirectional_latents(self):
        """Test simplification 4 - remove unidirectional latents."""
        graph = nx.DiGraph()
        graph.add_edges_from([(U1, U2), (U3, U2)])
        set_latent(graph, [U3])
        _, actual_unidirectional_latents = remove_unidirectional_latents(graph)
        expected_unidirectional_latents = {U3}
        self.assertEqual(expected_unidirectional_latents, actual_unidirectional_latents)

    def test_unidirectional_latents_amidst_other_rules(self):
        """Test remove unidirectional latents amidst other rules."""
        # Original graph
        actual_graph = _dag_from_adj_str(
            directed={
                "EGF": ["SOS", "PI3K"],
                "IGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Mark the latent nodes
        set_latent(
            actual_graph,
            [Variable("EGF"), Variable("IGF"), Variable("Akt"), Variable("Erk")],
        )
        # Simplify the network
        simplify_latent_dag(actual_graph)
        # Expected graph after simplification
        expected_graph = _dag_from_adj_str(
            directed={
                "EGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Raf"],
                "Raf": ["Mek"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph, [Variable("EGF")])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_sample_graph_0(self):
        """Test latent simplification for a simple network."""
        # Original graph
        actual_graph = _dag_from_adj_str(
            directed={
                "U1": ["V1", "V2", "V3"],
                "U2": ["V2", "V3"],
                "U3": ["V4", "V5"],
                "U4": ["V5"],
                "V1": ["U3", "U5"],
                "V2": ["U3"],
                "V3": ["U3"],
            }
        )
        # Mark the latent nodes
        set_latent(actual_graph, [Variable(f"U{num}") for num in range(1, 6)])
        # Simplify the network
        simplify_latent_dag(actual_graph)
        # Expected graph after simplification
        expected_graph = _dag_from_adj_str(
            directed={
                "U1": ["V1", "V2", "V3"],
                "U3": ["V4", "V5"],
                "V1": ["V4", "V5"],
                "V2": ["V4", "V5"],
                "V3": ["V4", "V5"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph, [Variable(f"U{num}") for num in range(1, 6)])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_sample_graph_1(self):
        """Test latent simplification for a simple network."""
        # Original graph
        actual_graph = _dag_from_adj_str(
            directed={
                "EGF": ["SOS", "PI3K"],
                "IGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Mark the latent nodes
        set_latent(actual_graph, [Variable("EGF"), Variable("IGF")])
        # Simplify the network
        simplify_latent_dag(actual_graph)
        # Expected graph after simplification
        expected_graph = _dag_from_adj_str(
            directed={
                "EGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph, [Variable("EGF")])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_sample_graph_2(self):
        """Test latent simplification for a simple network."""
        # Original graph
        actual_graph = _dag_from_adj_str(
            directed={
                "EGF": ["SOS", "PI3K"],
                "IGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Mark the latent nodes
        set_latent(actual_graph, [Variable("EGF"), Variable("IGF"), Variable("PI3K")])
        # Simplify the network
        simplify_latent_dag(actual_graph)
        # Expected graph after simplification
        expected_graph = _dag_from_adj_str(
            directed={
                "EGF": ["SOS", "Akt"],
                "SOS": ["Ras"],
                "Ras": ["Akt", "Raf"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph, [Variable("EGF")])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_sample_graph_3(self):
        """Test latent simplification for a simple network."""
        # Original graph
        actual_graph = _dag_from_adj_str(
            directed={
                "EGF": ["SOS", "PI3K"],
                "IGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Mark the latent nodes
        set_latent(actual_graph, [Variable("EGF"), Variable("IGF"), Variable("Ras")])
        # Simplify the network
        simplify_latent_dag(actual_graph)
        # Expected graph after simplification
        expected_graph = _dag_from_adj_str(
            directed={
                "EGF": ["SOS", "PI3K"],
                "SOS": ["PI3K", "Raf"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph, [Variable("EGF"), Variable("Ras")])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_sample_graph_4(self):
        """Test latent simplification for a simple network."""
        # Original graph
        actual_graph = _dag_from_adj_str(
            directed={
                "EGF": ["SOS", "PI3K"],
                "IGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Mark the latent nodes
        set_latent(
            actual_graph,
            [Variable("EGF"), Variable("IGF"), Variable("Raf"), Variable("Akt")],
        )
        # Simplify the network
        simplify_latent_dag(actual_graph)
        # Expected graph after simplification
        expected_graph = _dag_from_adj_str(
            directed={
                "EGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Mek"],
                "PI3K": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph, [Variable("EGF")])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_sample_graph_5(self):
        """Test latent simplification for a simple network."""
        # Original graph
        actual_graph = _dag_from_adj_str(
            directed={
                "Plcg": ["PKC", "PIP2", "PIP3"],
                "PIP3": ["PIP2", "Akt"],
                "PIP2": ["PKC"],
                "PKC": ["PKA", "Raf", "Mek", "Jnk", "P38"],
                "PKA": ["Raf", "Mek", "Erk", "Akt", "Jnk", "P38"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
                "Erk": ["Akt"],
            }
        )
        # Mark the latent nodes
        set_latent(
            actual_graph,
            [Variable("PKA"), Variable("Jnk"), Variable("P38"), Variable("Akt"), Variable("Raf")],
        )
        # Simplify the network
        simplify_latent_dag(actual_graph)
        # Expected graph after simplification
        expected_graph = _dag_from_adj_str(
            directed={
                "Plcg": ["PKC", "PIP2", "PIP3"],
                "PIP3": ["PIP2"],
                "PIP2": ["PKC"],
                "PKC": ["Mek", "Erk"],
                "PKA": ["Mek", "Erk"],
                "Mek": ["Erk"],
            }
        )

        # Expected latent nodes after simplification
        set_latent(expected_graph, [Variable("PKA")])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_sample_graph_6(self):
        """Test latent simplification for a simple network."""
        # Original graph
        actual_graph = _dag_from_adj_str(
            directed={
                "Plcg": ["PKC", "PIP2", "PIP3"],
                "PIP3": ["PIP2", "Akt"],
                "PIP2": ["PKC"],
                "PKC": ["PKA", "Raf", "Mek", "Jnk", "P38"],
                "PKA": ["Raf", "Mek", "Erk", "Akt", "Jnk", "P38"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
                "Erk": ["Akt"],
            }
        )
        # Mark the latent nodes
        set_latent(
            actual_graph,
            [Variable("PKA"), Variable("PKC"), Variable("Akt")],
        )
        # Simplify the network
        simplify_latent_dag(actual_graph)
        # Expected graph after simplification
        expected_graph = _dag_from_adj_str(
            directed={
                "Plcg": ["Raf", "Mek", "Erk", "Jnk", "P38", "PIP2", "PIP3"],
                "PIP3": ["PIP2"],
                "PIP2": ["Raf", "Mek", "Jnk", "P38", "Erk"],
                "PKA": ["Raf", "Mek", "Erk", "Jnk", "P38"],
                "Mek": ["Erk"],
                "Raf": ["Mek"],
            }
        )

        # Expected latent nodes after simplification
        set_latent(expected_graph, [Variable("PKA"), Variable("PKC")])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_sample_graph_7(self):
        """Test latent simplification for a simple network."""
        # Original graph
        actual_graph = _dag_from_adj_str(
            directed={
                "Plcg": ["PKC", "PIP2", "PIP3"],
                "PIP3": ["PIP2", "Akt"],
                "PIP2": ["PKC"],
                "PKC": ["PKA", "Raf", "Mek", "Jnk", "P38"],
                "PKA": ["Raf", "Mek", "Erk", "Akt", "Jnk", "P38"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
                "Erk": ["Akt"],
            }
        )
        # Mark the latent nodes
        set_latent(
            actual_graph,
            [Variable("Plcg"), Variable("PKA")],
        )
        # Simplify the network
        simplify_latent_dag(actual_graph)
        # Expected graph after simplification
        expected_graph = _dag_from_adj_str(
            directed={
                "Plcg": ["PKC", "PIP2", "PIP3"],
                "PIP3": ["PIP2", "Akt"],
                "PIP2": ["PKC"],
                "PKC": ["Mek", "Raf", "Erk", "Jnk", "P38", "Akt"],
                "PKA": ["Raf", "Mek", "Erk", "Jnk", "P38", "Akt"],
                "Mek": ["Erk"],
                "Raf": ["Mek"],
                "Erk": ["Akt"],
            }
        )

        # Expected latent nodes after simplification
        set_latent(expected_graph, [Variable("Plcg"), Variable("PKA")])
        self.assertEqual(actual_graph, expected_graph)
