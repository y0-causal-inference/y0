"""Tests for _get_projected_subgraph_ function inside the cyclic ID algorithm."""


from tests.test_algorithm import cases
from y0.algorithm.identify.cyclic_id import _get_projected_subgraph
from y0.dsl import Variable, X, Y, A, B, W1, W2, W3, Z, Z2
from y0.graph import NxMixedGraph


class TestGetProjectedSubgraph(cases.GraphTestCase):
    """Projection correctly adds a new bidirected edge when path exists through marginalized nodes."""

    def test_projection_adds_edge(self):
        cases = [
            ([], [(X, Z), (Z, Y)], {X, Y}, X, Y),                         # single hop
            ([], [(A, W1), (W1, W2), (W2, W3), (W3, B)], {A, B}, A, B),  # long chain
            ([], [(X, W1), (W1, W2), (W1, Y)], {X, Y}, X, Y),            # cycle in marginalized
            ([], [(X, Y)], {X, Y}, X, Y),                                  # empty marginalized set
            ([(Z2, X), (Z2, Y)], [(Z2, X), (Z2, Y)], {X, Y}, X, Y),      # napkin structure
        ]
        for directed, undirected, vertices, u, v in cases:
            with self.subTest(vertices=vertices, u=u, v=v):
                graph = NxMixedGraph.from_edges(directed=directed, undirected=undirected)
                result = _get_projected_subgraph(graph, frozenset(vertices))
                self.assertTrue(result.undirected.has_edge(u, v))

    def test_projection_does_not_add_edge(self):
        """No projection when path doesn't exist through only marginalized nodes."""
        cases = [
            ([], [(X, W1), (Z, Y)], {X, Y}, X, Y),   # disconnected components
            ([(X, W1)], [(W1, Y)], {X, Y}, X, Y),     # directed edge, not bidirected
        ]
        for directed, undirected, vertices, u, v in cases:
            with self.subTest(vertices=vertices, u=u, v=v):
                graph = NxMixedGraph.from_edges(directed=directed, undirected=undirected)
                result = _get_projected_subgraph(graph, frozenset(vertices))
                self.assertFalse(result.undirected.has_edge(u, v))

# ------------------------------------------------------------------------------

    def test_preserved_directed_edges(self):
        """Case 2: Directed edges between vertices in A are preserved as-is.

        Graph: X→Y
        Vertices (A): {X, Y}
        Expected: X→Y is preserved.
        """
        graph = NxMixedGraph.from_edges(
            directed=[(X, Y)],
            undirected=[],
        )

        result = _get_projected_subgraph(graph, frozenset({X, Y}))

        self.assertTrue(result.directed.has_edge(X, Y))
        self.assertEqual(set(result.nodes()), {X, Y})
        

    def test_path_blocked_by_observed_node(self):
        """Case 4: No projection when intermediate node is in A (not marginalized).

        Graph: X↔Y↔Z
        Vertices (A): {X, Y, Z}
        Expected: Original edges X↔Y and Y↔Z preserved, but no X↔Z added.
        """
        graph = NxMixedGraph.from_edges(
            directed=[],
            undirected=[(X, Y), (Y, Z)],
        )

        result = _get_projected_subgraph(graph, frozenset({X, Y, Z}))

        self.assertTrue(result.undirected.has_edge(X, Y))
        self.assertTrue(result.undirected.has_edge(Y, Z))
        self.assertFalse(result.undirected.has_edge(X, Z))
