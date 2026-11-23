"""Test m-separation."""

import unittest

from y0.algorithm.conditional_independencies import are_d_separated
from y0.algorithm.separation import are_m_separated
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
