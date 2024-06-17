# -*- coding: utf-8 -*-

"""Tests for the ID algorithm for input-output structural causal models (ioSCMs).

.. [forré20a] http://proceedings.mlr.press/v115/forre20a/forre20a.pdf.
.. [forré20b] http://proceedings.mlr.press/v115/forre20a/forre20a-supp.pdf
"""

# import logging
import unittest

from y0.algorithm.ioscm_id import get_strongly_connected_component
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
