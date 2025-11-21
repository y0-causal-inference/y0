"""Tests for the ID algorithm for input-output structural causal models (ioSCMs).

.. [forré20a] http://proceedings.mlr.press/v115/forre20a/forre20a.pdf.

.. [forré20b] http://proceedings.mlr.press/v115/forre20a/forre20a-supp.pdf
"""

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
from y0.dsl import A, B, C, R, W, X, Y, Z
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


class TestConvertStronglyConnectedComponents(unittest.TestCase):
    """Tests converting strongly connected components in a graph to bidirected edges."""

    def test_convert_strongly_connected_components_1(self) -> None:
        """First test converting strongly connected components in a graph to bidirected edges."""
        result_1 = scc_to_bidirected(simple_cyclic_graph_1)
        self.assertSetEqual(set(result_1.undirected.edges), {(X, W), (W, Z), (X, Z)})
        self.assertSetEqual(set(result_1.directed.edges), {(R, X), (W, Y)})

    def test_convert_strongly_connected_components_2(self) -> None:
        """Second test converting strongly connected components in a graph to bidirected edges."""
        result_2 = scc_to_bidirected(simple_cyclic_graph_2)
        self.assertSetEqual(set(result_2.undirected.edges), {(X, R), (X, W), (W, Z), (X, Z)})
        self.assertSetEqual(set(result_2.directed.edges), {(W, Y)})


class TestGetConsolidatedDistrict(unittest.TestCase):
    """Test retrieving a consolidated district in a graph with or without cycles."""

    # TODO: Implement type checking on the graph and the input variable.
    # TODO: Also check that we can't pass in multiple vertices or an empty graph.
    # TODO: Add more tests.

    def test_get_vertex_consolidated_district_1(self) -> None:
        """First test for getting the consolidated district for a single vertex."""
        for vertex, result in [(X, {X, W, Z, R}), (R, {X, W, Z, R}), (Y, {Y})]:
            self.assertSetEqual(
                result, get_vertex_consolidated_district(simple_cyclic_graph_2, vertex)
            )

    def test_get_consolidated_district_1(self) -> None:
        """Test getting consolidated districts."""
        for vertices, result in [
            # Single vertex input
            ({X}, {X, W, Z, R}),
            ({R}, {X, W, Z, R}),
            ({Y}, {Y}),
            # Multiple vertex input
            ({X, R}, {X, W, Z, R}),
            ({R}, {X, W, Z, R}),
            ({X, Y}, {X, W, R, Z, Y}),
        ]:
            self.assertSetEqual(get_consolidated_district(simple_cyclic_graph_2, vertices), result)

    def test_get_graph_consolidated_district_1(self) -> None:
        """First test for getting the consolidated districts for a graph."""
        # create a graph where multiple nodes belong to the same consolidated district
        # create a graph with bidirected edges forming one district
        graph_m = NxMixedGraph.from_edges(undirected=[(A, B), (B, C)])

        for graph, result in [
            (simple_cyclic_graph_1, {frozenset({R}), frozenset({X, W, Z}), frozenset({Y})}),
            (simple_cyclic_graph_2, {frozenset({R, X, W, Z}), frozenset({Y})}),
            # This test ensures that when multiple nodes belong to the same consolidated district,
            # the district appears only once in the returned set (automatic deduplication by the set data structure).
            (graph_m, {frozenset({A, B, C})}),
        ]:
            self.assertSetEqual(result, get_graph_consolidated_districts(graph))


class TestAptOrder(unittest.TestCase):
    """Test retrieving an apt-order for a graph G."""

    # TODO: Implement type checking on the graph and the input variable.
    # TODO: Also check that we can't pass in multiple vertices or an empty graph.
    # TODO: Add more tests.

    def test_simplify_strongly_connected_components_1(self) -> None:
        """Test a utility function to simplify strongly-connected components for a graph."""
        # TODO: Check the returned dictionaries.
        result_1, _result_1_dict = _simplify_strongly_connected_components(simple_cyclic_graph_1)
        # Make the representative vertex W
        expected_result = NxMixedGraph.from_edges(directed=[(R, W), (W, Y)])
        self.assertListEqual(
            sorted(result_1.directed.edges), sorted(expected_result.directed.edges)
        )
        self.assertListEqual(
            sorted(result_1.undirected.edges), sorted(expected_result.undirected.edges)
        )

    def test_simplify_strongly_connected_components_2(self) -> None:
        """Test a utility function to simplify strongly-connected components for a graph."""
        # TODO: Check the returned dictionaries.
        result_2, _result_2_dict = _simplify_strongly_connected_components(simple_cyclic_graph_2)
        # Make the representative vertex W
        expected_result_2 = NxMixedGraph.from_edges(directed=[(W, Y)], undirected=[(R, W)])
        self.assertListEqual(
            sorted(expected_result_2.directed.edges), sorted(result_2.directed.edges)
        )
        self.assertListEqual(
            sorted(expected_result_2.undirected.edges),
            sorted(result_2.undirected.edges),
        )

    def test_simplify_strongly_connected_components_3(self) -> None:
        """Test a utility function to simplify strongly-connected components for a graph.

        This test covers the case where an undirected edge exists between two nodes in
        the same strongly connected component. The edge should be removed during
        simplification since both nodes get collapsed into a single representative node.
        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, W),
                (W, Z),
                (Z, X),
                (W, Y),
            ],
            undirected=[
                (X, W)  # this undirected edge is within the SCC
            ],
        )

        simplified_graph, result_dict = _simplify_strongly_connected_components(graph)

        # The SCC {X, W, Z} should be collapsed to one representative vertex (W is min)
        # Since X and W are in the same component, the undirected edge should NOT appear
        # in the simplified graph

        # check that the result has 2 nodes: SCC and Y
        self.assertEqual(len(simplified_graph.nodes()), 2)

        # check that there are no undirected edges in the result
        # (the X <-> W edge should be removed via the continue statement)
        self.assertEqual(len(list(simplified_graph.undirected.edges)), 0)

        # FIXME there's a logical error in the tests, just above it's checking
        #  that there are 2 nodes. Check this.
        # checking that the result has only one node (the representattive to Y)
        # self.assertEqual(len(simplified_graph.nodes()), 1)

        # verify the representative node maps to all three nodes in the component
        representative = min([X, W, Z])  # W is the min = representative
        self.assertIn(representative, result_dict)
        self.assertEqual(result_dict[representative], frozenset({X, W, Z}))

    def test_get_apt_order_1(self) -> None:
        """First test for getting an assembling pseudo-topological order for a graph."""
        for graph, result in [
            (simple_cyclic_graph_1, [R, W, X, Z, Y]),
            # For this second test, R is allowed to occur before W, X, Z as well, but it comes after
            # just based on the way the sorting works out.
            (simple_cyclic_graph_2, [W, X, Z, R, Y]),
        ]:
            self.assertListEqual(result, get_apt_order(graph))

    @unittest.skip(reason="not implemented")
    def test_is_apt_order_1(self) -> None:
        """First test for verifying an assembling pseudo-topological order for a graph."""
        for order in [
            [R, X, W, Z, Y],
            [R, X, Z, W, Y],
            [R, W, X, Z, Y],
            [R, W, Z, X, Y],
            [R, Z, X, W, Y],
            [R, Z, W, X, Y],
        ]:
            self.assertTrue(is_apt_order(order, simple_cyclic_graph_1))

        for order in [
            [Y, Z, W, X, R],
            [Y, Z, W, R, X],
            [Y, Z, X, R, W],
            [Y, Z, X, W, R],
            [Y, Z, R, X, W],
            [Y, Z, R, W, X],
        ]:
            self.assertFalse(is_apt_order(order, simple_cyclic_graph_1))
