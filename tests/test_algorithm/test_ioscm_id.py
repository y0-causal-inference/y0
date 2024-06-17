# -*- coding: utf-8 -*-

"""Tests for the ID algorithm for input-output structural causal models (ioSCMs).

.. [forré20a] http://proceedings.mlr.press/v115/forre20a/forre20a.pdf.
.. [forré20b] http://proceedings.mlr.press/v115/forre20a/forre20a-supp.pdf
"""

# import logging
import unittest

from y0.algorithm.ioscm_id import (
    get_consolidated_district,
    get_strongly_connected_component,
    get_vertex_consolidated_district,
)
from y0.dsl import (  # Fraction,; One,; P,; Pi1,; Pi2,; Product,; Sum,; Zero,; Intervention,; Variable,
    R,
    W,
    X,
    Y,
    Z,
)
from y0.graph import NxMixedGraph

# From [correa20a]_, Figure 2c.
simple_cyclic_graph_1 = NxMixedGraph.from_edges(
    directed=[
        (R, X),
        (X, W),
        (W, Z),
        (Z, X),
        (W, Y),
    ],
)

simple_cyclic_graph_2 = NxMixedGraph.from_edges(
    directed=[
        (X, W),
        (W, Z),
        (Z, X),
        (W, Y),
    ],
    undirected=[
        (R, X),
    ],
)


class TestGetStronglyConnectedComponent(unittest.TestCase):
    """Test retrieving a strongly connected component in a graph with or without cycles."""

    # TODO: Implement type checking on the graph and the input variable.
    # TODO: Also check that we can't pass in multiple vertices or an empty graph.

    def test_get_strongly_connected_component_1(self):
        """First test for strongly connected components.

        This is a simple graph with a cycle.
        """
        result_1 = get_strongly_connected_component(simple_cyclic_graph_1, X)
        self.assertSetEqual(result_1, {X, W, Z})
        result_2 = get_strongly_connected_component(simple_cyclic_graph_1, R)
        self.assertSetEqual(result_2, {R})
        result_3 = get_strongly_connected_component(simple_cyclic_graph_1, Y)
        self.assertSetEqual(result_3, {Y})


class TestGetConsolidatedDistrict(unittest.TestCase):
    """Test retrieving a consolidated district in a graph with or without cycles."""

    # TODO: Implement type checking on the graph and the input variable.
    # TODO: Also check that we can't pass in multiple vertices or an empty graph.
    # TODO: Add more tests.

    def test_get_vertex_consolidated_district_1(self):
        """First test for getting the consolidated district for a single vertex."""
        result_1 = get_vertex_consolidated_district(simple_cyclic_graph_2, X)
        self.assertSetEqual(result_1, {X, W, Z, R})
        result_2 = get_vertex_consolidated_district(simple_cyclic_graph_2, R)
        self.assertSetEqual(result_2, {X, W, Z, R})
        result_3 = get_vertex_consolidated_district(simple_cyclic_graph_2, Y)
        self.assertSetEqual(result_3, {Y})

    def test_get_consolidated_district_1(self):
        """First test for getting the consolidated districts for multiple vertices.

        Testing inputs that are single vertices.
        """
        result_1 = get_consolidated_district(simple_cyclic_graph_2, {X})
        self.assertSetEqual(result_1, frozenset(frozenset({X, W, Z, R})))
        result_2 = get_consolidated_district(simple_cyclic_graph_2, {R})
        self.assertSetEqual(result_2, frozenset(frozenset({X, W, Z, R})))
        result_3 = get_consolidated_district(simple_cyclic_graph_2, {Y})
        self.assertSetEqual(result_3, frozenset(frozenset({Y})))

    def test_get_consolidated_district_2(self):
        """Second test for getting the consolidated districts for multiple vertices.

        Testing inputs that are multiple vertices.
        """
        result_1 = get_consolidated_district(simple_cyclic_graph_2, {X, R})
        self.assertSetEqual(result_1, frozenset(frozenset({X, W, Z, R})))
        result_2 = get_consolidated_district(simple_cyclic_graph_2, {R})
        self.assertSetEqual(result_2, frozenset(frozenset({X, W, Z, R})))
        result_3 = get_consolidated_district(simple_cyclic_graph_2, {X, Y})
        self.assertSetEqual(result_3, frozenset(frozenset({X, W, R, Z}), frozenset({Y})))
