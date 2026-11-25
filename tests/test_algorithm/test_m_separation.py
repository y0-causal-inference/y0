"""Test m-separation."""

import unittest

from y0.algorithm.conditional_independencies import are_d_separated
from y0.algorithm.separation import are_m_separated, find_minimal_m_separator
from y0.dsl import Variable, X, Y, Z
from y0.graph import NxMixedGraph


class TestMSeparation(unittest.TestCase):
    """Test m-separation for acyclic directed mixed graphs."""

    def test_bidirected_chain(self) -> None:
        """Conditioning on a collider in a bidirected chain opens the path."""
        graph = NxMixedGraph.from_edges(undirected=[(X, Z), (Z, Y)])

        self.assertTrue(are_m_separated(graph, X, Y))
        self.assertFalse(are_m_separated(graph, X, Y, conditions=[Z]))

    def test_directed_graphs_match_d_separation(self) -> None:
        """m-separation reduces to d-separation when there are no bidirected edges."""
        cases = [
            (NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)]), None, False),
            (NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y)]), [Z], True),
            (NxMixedGraph.from_edges(directed=[(X, Z), (Y, Z)]), None, True),
            (NxMixedGraph.from_edges(directed=[(X, Z), (Y, Z)]), [Z], False),
        ]

        for graph, conditions, expected in cases:
            with self.subTest(conditions=conditions, expected=expected):
                self.assertEqual(
                    expected,
                    are_d_separated(graph, X, Y, conditions=conditions).separated,
                )
                self.assertEqual(
                    expected,
                    are_m_separated(graph, X, Y, conditions=conditions).separated,
                )

    def test_mixed_paths(self) -> None:
        """Directed and bidirected edges obey collider and non-collider rules."""
        collider = NxMixedGraph.from_edges(directed=[(X, Z)], undirected=[(Z, Y)])
        self.assertTrue(are_m_separated(collider, X, Y))
        self.assertFalse(are_m_separated(collider, X, Y, conditions=[Z]))

        non_collider = NxMixedGraph.from_edges(directed=[(Z, Y)], undirected=[(X, Z)])
        self.assertFalse(are_m_separated(non_collider, X, Y))
        self.assertTrue(are_m_separated(non_collider, X, Y, conditions=[Z]))

    def test_direct_bidirected_edge_remains_open(self) -> None:
        """Conditioning on an unrelated node does not block a direct edge."""
        graph = NxMixedGraph.from_edges(
            nodes=[X, Y, Z],
            undirected=[(X, Y)],
        )

        self.assertFalse(are_m_separated(graph, X, Y))
        self.assertFalse(are_m_separated(graph, X, Y, conditions=[Z]))

    def test_isolated_node_and_same_node(self) -> None:
        """Isolated nodes are separated, while a node is not separated from itself."""
        isolated = Variable("isolated")
        graph = NxMixedGraph.from_edges(nodes=[X, Y, isolated], directed=[(X, Y)])

        self.assertTrue(are_m_separated(graph, X, isolated))
        self.assertFalse(are_m_separated(graph, X, X))

    def test_unknown_nodes_raise_value_error(self) -> None:
        """Query and conditioning nodes must belong to the graph."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        unknown = Variable("unknown")

        with self.assertRaises(ValueError):
            are_m_separated(graph, X, unknown)
        with self.assertRaises(ValueError):
            are_m_separated(graph, X, Y, conditions=[unknown])


class TestMinimalMSeparator(unittest.TestCase):
    """Test minimal m-separator search and its constraints."""

    def test_default_pool_returns_minimal_separator(self) -> None:
        """The default ancestor pool shrinks to the two necessary blockers."""
        a, b, irrelevant = map(Variable, ["a", "b", "irrelevant"])
        graph = NxMixedGraph.from_edges(
            nodes=[X, Y, a, b, irrelevant],
            directed=[(X, a), (a, Y), (X, b), (b, Y), (irrelevant, a)],
        )

        separator = find_minimal_m_separator(graph, X, Y)

        self.assertEqual(frozenset({a, b}), separator)
        self.assertTrue(are_m_separated(graph, X, Y, conditions=separator))
        for node in separator:
            self.assertFalse(are_m_separated(graph, X, Y, conditions=separator - {node}))

    def test_restricted_pool_and_included_nodes(self) -> None:
        """Keep mandatory nodes while removing unnecessary allowed nodes."""
        a, required, irrelevant = map(Variable, ["a", "required", "irrelevant"])
        graph = NxMixedGraph.from_edges(
            nodes=[X, Y, a, required, irrelevant],
            directed=[(X, a), (a, Y)],
        )

        separator = find_minimal_m_separator(
            graph,
            X,
            Y,
            included=[required],
            restricted=[a, required, irrelevant],
        )

        self.assertEqual(frozenset({a, required}), separator)

    def test_included_ancestor_enters_default_pool(self) -> None:
        """An included node is retained even when it is not needed to block a path."""
        included = Variable("included")
        graph = NxMixedGraph.from_edges(
            nodes=[X, Y, Z, included],
            directed=[(X, Z), (Z, Y)],
        )

        self.assertEqual(
            frozenset({Z, included}),
            find_minimal_m_separator(graph, X, Y, included=[included]),
        )

    def test_no_separator_with_available_candidates(self) -> None:
        """A direct edge cannot be blocked by the available restricted nodes."""
        irrelevant = Variable("irrelevant")
        graph = NxMixedGraph.from_edges(
            nodes=[X, Y, irrelevant],
            directed=[(X, Y)],
        )

        self.assertIsNone(find_minimal_m_separator(graph, X, Y))
        self.assertIsNone(find_minimal_m_separator(graph, X, Y, restricted=[irrelevant]))

    def test_included_node_outside_restricted_pool_is_infeasible(self) -> None:
        """An explicitly required node must be allowed by the restricted pool."""
        required = Variable("required")
        graph = NxMixedGraph.from_edges(nodes=[X, Y, required], directed=[(X, Y)])

        self.assertIsNone(find_minimal_m_separator(graph, X, Y, included=[required], restricted=[]))

    def test_isolated_endpoint_has_empty_separator(self) -> None:
        """An isolated query endpoint is already separated from the other endpoint."""
        isolated = Variable("isolated")
        graph = NxMixedGraph.from_edges(nodes=[X, Y, isolated], directed=[(X, Y)])

        self.assertEqual(frozenset(), find_minimal_m_separator(graph, X, isolated))

    def test_unknown_query_included_and_restricted_nodes_raise(self) -> None:
        """All query and constraint nodes must be observed graph nodes."""
        unknown = Variable("unknown")
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])

        with self.assertRaises(ValueError):
            find_minimal_m_separator(graph, X, unknown)
        with self.assertRaises(ValueError):
            find_minimal_m_separator(graph, X, Y, included=[unknown])
        with self.assertRaises(ValueError):
            find_minimal_m_separator(graph, X, Y, restricted=[unknown])
