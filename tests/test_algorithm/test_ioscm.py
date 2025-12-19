"""Tests for the ID algorithm for input-output structural causal models (ioSCMs).

.. [forré20a] http://proceedings.mlr.press/v115/forre20a/forre20a.pdf.

.. [forré20b] http://proceedings.mlr.press/v115/forre20a/forre20a-supp.pdf
"""

from tests.test_algorithm import cases
from y0.algorithm.ioscm.utils import (
    _check_members_of_scc_are_consecutive,
    get_apt_order,
    get_consolidated_district,
    get_graph_consolidated_districts,
    get_vertex_consolidated_district,
    is_apt_order,
    scc_to_bidirected,
    simplify_strongly_connected_components,
)
from y0.dsl import A, B, C, R, Variable, W, X, Y, Z
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


def _fs(*args: Variable) -> frozenset[Variable]:
    return frozenset(args)


class TestIOSCMUtils(cases.GraphTestCase):
    """Test ioSCMs utilities."""

    def test_convert_strongly_connected_components(self) -> None:
        """Test converting strongly connected components in a graph to bidirected edges."""
        for graph, undirected, directed in [
            (simple_cyclic_graph_1, {(X, W), (W, Z), (X, Z)}, {(R, X), (W, Y)}),
            (simple_cyclic_graph_2, {(X, R), (X, W), (W, Z), (X, Z)}, {(W, Y)}),
        ]:
            bidirected_graph = scc_to_bidirected(graph)
            self.assertEqual(undirected, set(bidirected_graph.undirected.edges))
            self.assertEqual(directed, set(bidirected_graph.directed.edges))

    def test_get_vertex_consolidated_district(self) -> None:
        """Test for getting the consolidated district for a single vertex."""
        for vertex, expected in [(X, {X, W, Z, R}), (R, {X, W, Z, R}), (Y, {Y})]:
            self.assertEqual(
                expected, get_vertex_consolidated_district(simple_cyclic_graph_2, vertex)
            )

    def test_get_consolidated_district(self) -> None:
        """Test getting consolidated districts."""
        for vertices, expected in [
            # Single vertex input
            ({X}, {X, W, Z, R}),
            ({R}, {X, W, Z, R}),
            ({Y}, {Y}),
            # Multiple vertex input
            ({X, R}, {X, W, Z, R}),
            ({R}, {X, W, Z, R}),
            ({X, Y}, {frozenset({X, W, R, Z}), frozenset({Y})}),
        ]:
            self.assertEqual(expected, get_consolidated_district(simple_cyclic_graph_2, vertices))

        for vertices, expected in [
            # Single vertex input
            ({X}, {X, W, Z}),
            ({R}, {R}),
            ({Y}, {Y}),
            # Multiple vertex input
            ({X, R}, {frozenset({X, W, Z}), frozenset({R})}),
            ({R, Y}, {frozenset({R}), frozenset({Y})}),
            ({X, Y}, {frozenset({X, W, Z}), frozenset({Y})}),
        ]:
            self.assertEqual(expected, get_consolidated_district(simple_cyclic_graph_1, vertices))

    def test_get_consolidated_district_all_nodes_same_district(self) -> None:
        """Test querying all nodes when they belong to the same consolidated district.

        Graph: simple_cyclic_graph_2
        - Structure: R <-> X -> W -> Z -> X, W -> Y
        - Districs: {R, X, W, Z}, {Y}

        Input: {R, X, W, Z}
        Expected Output: {R, X, W, Z} flat set of Variables

        Reasoning: Since all queried nodes belong to the same consolidated district,
        the function should return a flat set of Variables representing that district.
        """
        # query all nodes from the big district
        query = {R, X, W, Z}
        result = get_consolidated_district(simple_cyclic_graph_2, query)
        expected = {R, X, W, Z}

        self.assertEqual(
            expected,
            result,
            "All nodes in the same consolidated district should return that district.",
        )

    def test_get_consolidated_district_all_nodes_different_districts(self) -> None:
        """Test querying all nodes when they span all districts.

        Graph: simple_cyclic_graph_1
        - Structure: R -> X -> W -> Z -> X, (cycle) W -> Y
        - Districts: {R}, {X, W, Z}, {Y}

        Input: {R, X, Y}
        Expected Output: {frozenset({R}), frozenset({X, W, Z}), frozenset({Y})}

        Reasoning: Each queried node belongs to a different consolidated district, so return
        a set of frozensets representing each district to preserve district boundaries.
        """
        # query one node from each district
        query = {R, X, Y}
        expected = {frozenset({R}), frozenset({X, W, Z}), frozenset({Y})}

        result = get_consolidated_district(simple_cyclic_graph_1, query)
        self.assertEqual(
            expected, result, "Querying nodes from all districts should return all districts."
        )

    def test_get_consolidated_district_bidirected_edge_within_scc(self) -> None:
        """Test graph with bidirected edge within a SCC.

        Graph Structure:

        - Directed edges: X -> Y -> Z -> X (forming a cycle/SCC)
        - Bidirected edge: X <-> Y (within the cycle)
        - Districts: {X, Y, Z}

        Test Case 1:
        Input: {X}
        Expected Output: {X, Y, Z}

        Test Case 2:
        Input: {X, Z}
        Expected Output: {X, Y, Z}

        Reasoning: Bidirected edges within SCCs do not create separate districts.
        The entire cycle remains one consolidated district.

        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, Z),
                (Z, X),
            ],
            undirected=[
                (X, Y),
            ],
        )

        # Test 1: all nodes in same SCC + bidirected edge = one consolidated district (Single node query)
        result = get_consolidated_district(graph, {X})
        expected = {X, Y, Z}
        self.assertEqual(expected, result)

        # Test 2: query multiple nodes from same district
        result = get_consolidated_district(graph, {X, Z})
        expected = {X, Y, Z}
        self.assertEqual(expected, result)

    def test_get_consolidated_district_chain_of_bidirected_edges(self) -> None:
        """Test graph with chain of bidirected edges that connect nodes.

        A <-> B <-> C should form one consolidated district.

        Graph Structure:
        - Bidirected edges: A <-> B, B <-> C
        - No directed edges.
        - Districts: {A, B, C}

        Test Case 1:
        Input: {A}
        Expected Output: {A, B, C}
        Reasoning: A is connected to B and C via bidirected edges, so all belong to the same district.


        Test Case 2:
        Input: {A, C}
        Expected Output: {A, B, C}
        Reasoning: Both A and C are in the same consolidated district via B, so return the full district. (flat set)
        """
        graph = NxMixedGraph.from_edges(
            undirected=[
                (A, B),
                (B, C),
            ]
        )

        # Test 1: all are connected via bidirected edges = one consolidated district
        result = get_consolidated_district(graph, {A})
        expected = {A, B, C}
        self.assertEqual(expected, result)

        # Test 2
        result = get_consolidated_district(graph, {A, C})
        expected = {A, B, C}
        self.assertEqual(expected, result)

    def test_get_consolidated_district_disconnected_components(self) -> None:
        """Test graph with disconnected components.

        Graph Structure:
        - Component 1: A -> B
        - Component 2: X -> Y -> Z -> X (cycle)
        - No connections between components.
        - Districts: {A}, {X, Y, Z}

        Test Case 1:
        Input: {A, X}
        Expected Output: {frozenset({A}), frozenset({X, Y, Z})}
        Reasoning: Querying nodes from different disconnected components should return separate frozensets.

        Test Case 2:
        Input: {X, Y}
        Expected Output: {X, Y, Z}
        Reasoning: Both are in the same district (cycle).
        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (A, B),  # component 1
                (X, Y),  # component 2
                (Y, Z),
                (Z, X),
            ]
        )

        # Test 1: query from different components
        result = get_consolidated_district(graph, {A, X})
        expected = {frozenset({A}), frozenset({X, Y, Z})}
        self.assertEqual(
            expected, result, "Disconnected components should return separate frozensets."
        )

        # Test 2: query from single component
        result_2 = get_consolidated_district(graph, {X, Y})
        expected_2 = {X, Y, Z}
        self.assertEqual(
            expected_2, result_2, "Nodes from same component/district should return flat set."
        )

    def test_get_consolidated_district_single_node_queries(self) -> None:
        """Test querying single nodes always returns flat set.

        Note: Single-node queries should always return a flat set of Variables, regardless of how many districts exist
        in the graph.

        Test Cases:
        Graph 1: simple_cyclic_graph_1: R -> X -> W -> Z -> X, W -> Y
        Districts: {R}, {X, W, Z}, {Y}
        """
        # single node queries should just return the flat set

        # from the simple_cyclic_graph_1
        test_cases = [
            (simple_cyclic_graph_1, {R}, {R}),  # R would be an isolated district
            (simple_cyclic_graph_1, {X}, {X, W, Z}),  # X is in cycle district
            (simple_cyclic_graph_1, {Y}, {Y}),  # Y is isolated district
            (simple_cyclic_graph_1, {W}, {X, W, Z}),  # W is in cycle district
            # from the simple_cyclic_graph_2
            (simple_cyclic_graph_2, {R}, {R, X, W, Z}),  # R is in cycle district
            (simple_cyclic_graph_2, {Y}, {Y}),  # Y is isolated district
        ]

        for graph, query, expected in test_cases:
            with self.subTest(query=query):
                self.assertEqual(expected, get_consolidated_district(graph, query))

    def test_get_consolidated_district_preserves_district_membership(self) -> None:
        """Test that function correctly identifies district membership. (i.e. which nodes belong to which district).

        Graph: simple_cyclic_graph_1
        Structure: R -> X -> W -> Z -> X (cycle), W -> Y
        Districts: {R}, {X, W, Z}, {Y}
        """
        different_district_pairs = [
            ({R, X}, {frozenset({R}), frozenset({X, W, Z})}),
            ({R, Y}, {frozenset({R}), frozenset({Y})}),
            ({X, Y}, {frozenset({X, W, Z}), frozenset({Y})}),
        ]

        for query, expected in different_district_pairs:
            with self.subTest(query=query):
                result = get_consolidated_district(simple_cyclic_graph_1, query)
                self.assertEqual(expected, result, f"Nodes {query} are in different districts.")

        same_district_pairs = [
            ({X, W}, {X, W, Z}),
            ({W, Z}, {X, W, Z}),
            ({X, Z}, {X, W, Z}),
        ]
        for query, expected_2 in same_district_pairs:
            with self.subTest(query=query):
                result = get_consolidated_district(simple_cyclic_graph_1, query)
                self.assertEqual(expected_2, result, f"Nodes {query} are in the same district.")

    def test_get_consolidated_district_return_type_consistency(self) -> None:
        """Test that return type is predictable based on district count.

        Rule:
        - 1 district -> set[Variable]
        - 2+ districts -> set[frozenset[Variable]]

        Test Cases for Single district:

        Graph 1: simple_cyclic_graph_1
        1. Query: {X} -> Output: set of Variables {X, W, Z}
        2. Query: {X, W, Z} -> Output: set of Variables {X, W, Z} (all from the same district)


        Graph 2: simple_cyclic_graph_2
        3. Query: {R, X} -> Output: set of Variables {R, X, W, Z}
        4. Query: {R, X, W, Z} -> Output: set of Variables {R, X, W, Z} (all from the same district)

        Test Cases for Multiple Districts:

        Graph 1 (simple_cyclic_graph_1):
        5. Query {X, R} → Returns {frozenset({X,W,Z}), frozenset({R})} (2 districts)
        6. Query {R, Y} → Returns {frozenset({R}), frozenset({Y})} (2 districts)
        7. Query {X, Y} → Returns {frozenset({X,W,Z}), frozenset({Y})} (2 districts)
        8. Query {R, X, Y} → Returns {frozenset({R}), frozenset({X,W,Z}), frozenset({Y})} (3 districts)

        Graph 2 (simple_cyclic_graph_2):
        9. Query {X, Y} → Returns {frozenset({R,X,W,Z}), frozenset({Y})} (2 districts)
        """
        # 1 district cases (should return set of Variables)
        one_district = [
            (simple_cyclic_graph_1, {X}, {X, W, Z}),
            (simple_cyclic_graph_1, {X, W, Z}, {X, W, Z}),
            (simple_cyclic_graph_2, {R, X}, {R, X, W, Z}),
            (simple_cyclic_graph_2, {R, X, W, Z}, {R, X, W, Z}),
        ]
        for graph, query, expected in one_district:
            with self.subTest(query=query, expected_districts="1"):
                self.assertEqual(expected, get_consolidated_district(graph, query))

        # 2+ district cases (should return set of frozen sets)
        multiple_districts = [
            (simple_cyclic_graph_1, {X, R}, {frozenset({X, W, Z}), frozenset({R})}),
            (simple_cyclic_graph_1, {R, Y}, {frozenset({R}), frozenset({Y})}),
            (simple_cyclic_graph_1, {X, Y}, {frozenset({X, W, Z}), frozenset({Y})}),
            (
                simple_cyclic_graph_1,
                {R, X, Y},
                {frozenset({R}), frozenset({X, W, Z}), frozenset({Y})},
            ),
            (simple_cyclic_graph_2, {X, Y}, {frozenset({R, X, W, Z}), frozenset({Y})}),
        ]
        for graph, query, expected_2 in multiple_districts:
            with self.subTest(query=query, expected_districts="2+"):
                self.assertEqual(expected_2, get_consolidated_district(graph, query))

    def test_get_graph_consolidated_district(self) -> None:
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
            self.assertEqual(result, get_graph_consolidated_districts(graph))

    def test_simplify_strongly_connected_components(self) -> None:
        """Test simplify strongly-connected components for a graph."""
        for graph, expected in [
            (simple_cyclic_graph_1, NxMixedGraph.from_edges(directed=[(R, W), (W, Y)])),
            (
                simple_cyclic_graph_2,
                NxMixedGraph.from_edges(directed=[(W, Y)], undirected=[(R, W)]),
            ),
        ]:
            # TODO test result dict
            actual, _result_dict = simplify_strongly_connected_components(graph)
            self.assert_graph_equal(expected, actual)

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

        simplified_graph, result_dict = simplify_strongly_connected_components(graph)

        # The SCC {X, W, Z} should be collapsed to one representative vertex (W is min)
        # Since X and W are in the same component, the undirected edge should NOT appear
        # in the simplified graph

        # check that the result has 2 nodes: SCC and Y
        self.assertEqual(len(simplified_graph.nodes()), 2)

        # check that there are no undirected edges in the result
        # (the X <-> W edge should be removed via the continue statement)
        self.assertEqual(len(list(simplified_graph.undirected.edges)), 0)

        # verify the representative node maps to all three nodes in the component
        representative = min([X, W, Z])  # W is the min = representative
        self.assertIn(representative, result_dict)
        self.assertEqual(result_dict[representative], frozenset({X, W, Z}))

    def test_get_apt_order(self) -> None:
        """Test getting an assembling pseudo-topological order for a graph."""
        for graph, result in [
            (simple_cyclic_graph_1, [R, W, X, Z, Y]),
            # For this second test, R is allowed to occur before W, X, Z as well, but it comes after
            # just based on the way the sorting works out.
            (simple_cyclic_graph_2, [W, X, Z, R, Y]),
        ]:
            self.assertListEqual(result, get_apt_order(graph))

    def test_is_apt_order_1(self) -> None:
        """Test verifying an assembling pseudo-topological order for a graph."""
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

    def test_check_scc_consecutiveness(self) -> None:
        """Test Condition 2: SCC members must be consecutive in the apt_order."""
        # simple_cyclic_graph_1 has:
        # - R → X → W → Z (with Z → X creating cycle)
        # - W → Y
        # - SCC: {X, W, Z}
        # - Single-node SCCs: {R}, {Y}

        # VALID: SCC {X, W, Z} is consecutive
        for candidate_order in [[R, X, W, Z, Y], [R, W, Z, X, Y], [R, Z, X, W, Y], [R, W, X, Z, Y]]:
            self.assertTrue(is_apt_order(candidate_order, simple_cyclic_graph_1))

        # INVALID: SCC {X, W, Z} is broken up
        # Note: These also violate Condition 1 (ancestry constraint)
        # because Y appears before its ancestors
        for candidate_order in [
            # Y breaks SCC
            [R, X, Y, W, Z],
            [R, X, W, Y, Z],
            [R, Y, X, W, Z],
        ]:
            self.assertFalse(is_apt_order(candidate_order, simple_cyclic_graph_1))

    def test_check_members_of_scc_are_consecutive_valid(self) -> None:
        """Test that consecutive SCC members pass validation."""
        cases = [
            # all SCC members are consecutive
            ([R, X, W, Z, Y], {_fs(X, W, Z), _fs(R), _fs(Y)}),
            # all single node SCCS are always valid
            ([R, X, Y, W, Z], {_fs(R), _fs(X), _fs(Y), _fs(W), _fs(Z)}),
            # "no SCCs returns true"
            ([R, X, W, Z, Y], set()),
            # single-node SCCS is executed while still testing multi-node SCC consecutiveness logic.
            ([R, X, W, Z, Y], {_fs(X, W, Z), _fs(R), _fs(Y)}),
            # multiple SCCs that are all consecutive
            ([R, X, W, A, Y, Z, B], {_fs(X, W), _fs(Y, Z), _fs(R), _fs(A), _fs(B)}),
        ]
        for candidate_order, sccs in cases:
            with self.subTest():
                self.assertTrue(_check_members_of_scc_are_consecutive(candidate_order, sccs))

        false_cases = [
            # SCC members are not consecutive
            ([R, X, Y, W, Z], {_fs(X, W, Z), _fs(R), _fs(Y)}),
            # multiple SCCs where one is not consecutive
            ([R, X, W, A, Y, B, Z], {_fs(X, W), _fs(Y, Z), _fs(R), _fs(A), _fs(B)}),
        ]
        for candidate_order, sccs in false_cases:
            with self.subTest():
                self.assertFalse(_check_members_of_scc_are_consecutive(candidate_order, sccs))
