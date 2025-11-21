"""Tests for the ID algorithm for input-output structural causal models (ioSCMs).

.. [forré20a] http://proceedings.mlr.press/v115/forre20a/forre20a.pdf.
.. [forré20b] http://proceedings.mlr.press/v115/forre20a/forre20a-supp.pdf
"""

import logging
import unittest

from y0.algorithm.ioscm_id import (
    _simplify_strongly_connected_components,
    get_apt_order,
    get_consolidated_district,
    get_graph_consolidated_districts,
    get_vertex_consolidated_district,
    is_apt_order,
    scc_to_bidirected,
)
from y0.dsl import R, W, X, Y, Z
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

logger = logging.getLogger(__name__)


class TestConvertStronglyConnectedComponents(unittest.TestCase):
    """Tests converting strongly connected components in a graph to bidirected edges."""

    def test_convert_strongly_connected_components_1(self):
        """First test converting strongly connected components in a graph to bidirected edges."""
        result_1 = scc_to_bidirected(simple_cyclic_graph_1)
        self.assertSetEqual(set(result_1.undirected.edges), {(X, W), (W, Z), (X, Z)})
        self.assertSetEqual(set(result_1.directed.edges), {(R, X), (W, Y)})

    def test_convert_strongly_connected_components_2(self):
        """Second test converting strongly connected components in a graph to bidirected edges."""
        result_2 = scc_to_bidirected(simple_cyclic_graph_2)
        self.assertSetEqual(set(result_2.undirected.edges), {(X, R), (X, W), (W, Z), (X, Z)})
        self.assertSetEqual(set(result_2.directed.edges), {(W, Y)})


class TestGetConsolidatedDistrict(unittest.TestCase):
    """Test retrieving a consolidated district in a graph with or without cycles."""

    # TODO: Implement type checking on the graph and the input variable.
    # TODO: Also check that we can't pass in multiple vertices or an empty graph.
    # TODO: Add more tests.

    def test_get_vertex_consolidated_district_1(self):
        """First test for getting the consolidated district for a single vertex."""
        result_1 = get_vertex_consolidated_district(simple_cyclic_graph_2, X)
        logger.warning(f"In test_get_vertex_consolidated_district_1: result_1 = {result_1!s}")
        self.assertSetEqual(result_1, {X, W, Z, R})
        result_2 = get_vertex_consolidated_district(simple_cyclic_graph_2, R)
        logger.warning(f"In test_get_vertex_consolidated_district_1: result_2 = {result_2!s}")
        self.assertSetEqual(result_2, {X, W, Z, R})
        result_3 = get_vertex_consolidated_district(simple_cyclic_graph_2, Y)
        logger.warning(f"In test_get_vertex_consolidated_district_1: result_3 = {result_3!s}")
        self.assertSetEqual(result_3, {Y})

    def test_get_consolidated_district_1(self):
        """First test for getting the consolidated districts for multiple vertices.

        Testing inputs that are single vertices.
        """
        result_1 = get_consolidated_district(simple_cyclic_graph_2, {X})
        logger.warning(f"In test_get_consolidated_district_1: result_1 = {result_1!s}")
        self.assertSetEqual(result_1, {X, W, Z, R})
        result_2 = get_consolidated_district(simple_cyclic_graph_2, {R})
        logger.warning(f"In test_get_consolidated_district_1: result_2 = {result_2!s}")
        self.assertSetEqual(result_2, {X, W, Z, R})
        result_3 = get_consolidated_district(simple_cyclic_graph_2, {Y})
        logger.warning(f"In test_get_consolidated_district_1: result_3 = {result_3!s}")
        self.assertSetEqual(result_3, {Y})

    def test_get_consolidated_district_2(self):
        """Second test for getting the consolidated districts for multiple vertices.

        Testing inputs that are multiple vertices.
        """
        result_1 = get_consolidated_district(simple_cyclic_graph_2, {X, R})
        self.assertSetEqual(result_1, {X, W, Z, R})
        result_2 = get_consolidated_district(simple_cyclic_graph_2, {R})
        self.assertSetEqual(result_2, {X, W, Z, R})
        result_3 = get_consolidated_district(simple_cyclic_graph_2, {X, Y})
        self.assertSetEqual(result_3, {X, W, R, Z, Y})

    def test_get_graph_consolidated_district_1(self):
        """First test for getting the consolidated districts for a graph."""
        result_1 = get_graph_consolidated_districts(simple_cyclic_graph_1)
        logger.warning(f"In test_get_graph_consolidated_district_1: result_1 = {result_1!s}")
        self.assertSetEqual(result_1, {frozenset({R}), frozenset({X, W, Z}), frozenset({Y})})

    def test_get_graph_consolidated_district_2(self):
        """Second test for getting the consolidated districts for a graph."""
        result_2 = get_graph_consolidated_districts(simple_cyclic_graph_2)
        logger.warning(f"In test_get_graph_consolidated_district_1: result_2 = {result_2!s}")
        self.assertSetEqual(result_2, {frozenset({R, X, W, Z}), frozenset({Y})})


class TestAptOrder(unittest.TestCase):
    """Test retrieving an apt-order for a graph G."""

    # TODO: Implement type checking on the graph and the input variable.
    # TODO: Also check that we can't pass in multiple vertices or an empty graph.
    # TODO: Add more tests.

    def test_simplify_strongly_connected_components_1(self):
        """Test a utility function to simplify strongly-connected components for a graph."""
        # TODO: Check the returned dictionaries.
        result_1, _result_1_dict = _simplify_strongly_connected_components(simple_cyclic_graph_1)
        # Make the representative vertex W
        expected_result = NxMixedGraph.from_edges(
            directed=[
                (R, W),
                (W, Y),
            ],
        )
        self.assertListEqual(
            sorted(result_1.directed.edges), sorted(expected_result.directed.edges)
        )
        self.assertListEqual(
            sorted(result_1.undirected.edges), sorted(expected_result.undirected.edges)
        )

    def test_simplify_strongly_connected_components_2(self):
        """Test a utility function to simplify strongly-connected components for a graph."""
        # TODO: Check the returned dictionaries.
        result_2, _result_2_dict = _simplify_strongly_connected_components(simple_cyclic_graph_2)
        # Make the representative vertex W
        expected_result_2 = NxMixedGraph.from_edges(
            directed=[
                (W, Y),
            ],
            undirected=[
                (R, W),
            ],
        )
        self.assertListEqual(
            sorted(expected_result_2.directed.edges), sorted(result_2.directed.edges)
        )
        self.assertListEqual(
            sorted(expected_result_2.undirected.edges),
            sorted(result_2.undirected.edges),
        )

    def test_get_apt_order_1(self):
        """First test for getting an assembling pseudo-topological order for a graph."""
        result_1 = get_apt_order(simple_cyclic_graph_1)
        self.assertListEqual(result_1, [R, W, X, Z, Y])
        result_2 = get_apt_order(simple_cyclic_graph_2)
        # For this second test, R is allowed to occur before W, X, Z as well but it comes after
        # just based on the way the sorting works out.
        self.assertListEqual(result_2, [W, X, Z, R, Y])

    def test_is_apt_order_1(self):
        """First test for verifying an assembling pseudo-topological order for a graph."""
        self.assertTrue(is_apt_order([R, X, W, Z, Y], simple_cyclic_graph_1))
        self.assertTrue(is_apt_order([R, X, Z, W, Y], simple_cyclic_graph_1))
        self.assertTrue(is_apt_order([R, W, X, Z, Y], simple_cyclic_graph_1))
        self.assertTrue(is_apt_order([R, W, Z, X, Y], simple_cyclic_graph_1))
        self.assertTrue(is_apt_order([R, Z, X, W, Y], simple_cyclic_graph_1))
        self.assertTrue(is_apt_order([R, Z, W, X, Y], simple_cyclic_graph_1))
        self.assertFalse(is_apt_order([Y, Z, W, X, R], simple_cyclic_graph_1))
        self.assertFalse(is_apt_order([Y, Z, W, R, X], simple_cyclic_graph_1))
        self.assertFalse(is_apt_order([Y, Z, X, R, W], simple_cyclic_graph_1))
        self.assertFalse(is_apt_order([Y, Z, X, W, R], simple_cyclic_graph_1))
        self.assertFalse(is_apt_order([Y, Z, R, X, W], simple_cyclic_graph_1))
        self.assertFalse(is_apt_order([Y, Z, R, W, X], simple_cyclic_graph_1))
    # TODO: Use itertools.permutations to test every permutation of the vertices for this small graph

    def test_check_scc_consecutiveness(self):
        """Test Condition 2: SCC members must be consecutive in the apt_order."""
    
        # simple_cyclic_graph_1 has:
        # - R → X → W → Z (with Z → X creating cycle)
        # - W → Y
        # - SCC: {X, W, Z}
        # - Single-node SCCs: {R}, {Y}
    
        # VALID: SCC {X, W, Z} is consecutive
        self.assertTrue(is_apt_order([R, X, W, Z, Y], simple_cyclic_graph_1))
        self.assertTrue(is_apt_order([R, W, Z, X, Y], simple_cyclic_graph_1))
        self.assertTrue(is_apt_order([R, Z, X, W, Y], simple_cyclic_graph_1))
        self.assertTrue(is_apt_order([R, W, X, Z, Y], simple_cyclic_graph_1))
    
        # INVALID: SCC {X, W, Z} is broken up
        # Note: These also violate Condition 1 (ancestry constraint)
        # because Y appears before its ancestors
        self.assertFalse(is_apt_order([R, X, Y, W, Z], simple_cyclic_graph_1))  # Y breaks SCC
        self.assertFalse(is_apt_order([R, X, W, Y, Z], simple_cyclic_graph_1))  # Y breaks SCC
        self.assertFalse(is_apt_order([R, Y, X, W, Z], simple_cyclic_graph_1))  # Y breaks SCC
   
        