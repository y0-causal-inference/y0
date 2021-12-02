# -*- coding: utf-8 -*-

"""Tests for simplifying latent variable DAGs."""

import unittest

import networkx as nx

from y0.algorithm.simplify_latent import (
    DEFAULT_SUFFIX,
    iter_latents,
    remove_redundant_latents,
    remove_widow_latents,
    transform_latents_with_parents,
)
from y0.algorithm.taheri_design import taheri_design_dag
from y0.dsl import Y1, Y2, Y3, U, Variable, W
from y0.examples import igf_example
from y0.graph import set_latent

X1, X2, X3 = map(Variable, ["X1", "X2", "X3"])


class TestDesign(unittest.TestCase):
    """Test the design algorithm."""

    def test_design(self):
        """Test the design algorithm."""
        results = taheri_design_dag(igf_example.graph.directed, cause="PI3K", effect="Erk", stop=3)
        self.assertIsNotNone(results)
        # FIXME do better than this.


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
