# -*- coding: utf-8 -*-

"""Graph data structures."""

from dataclasses import dataclass, field

import networkx as nx
from ananke.graphs import ADMG

__all__ = [
    'NxMixedGraph',
    'napkin_graph',
]


@dataclass
class NxMixedGraph:
    """A mixed graph based on a :class:`networkx.Graph` and a :class:`networkx.DiGraph`.

    Example usage:

    .. code-block:: python

        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Y')
        graph.add_undirected_edge('X', 'Y')

        # Convert to an Ananke acyclic directed mixed graph
        admg_graph = graph.to_admg()
    """

    #: A directed graph
    directed: nx.DiGraph = field(default_factory=nx.DiGraph)
    #: A undirected graph
    undirected: nx.Graph = field(default_factory=nx.Graph)

    def add_directed_edge(self, u, v, **attr) -> None:
        """Add a directed edge from u to v."""
        self.directed.add_edge(u, v, **attr)
        self.undirected.add_node(u)
        self.undirected.add_node(v)

    def add_undirected_edge(self, u, v, **attr) -> None:
        """Add an undirected edge between u and v."""
        self.undirected.add_edge(u, v, **attr)
        self.directed.add_node(u)
        self.directed.add_node(v)

    def to_admg(self) -> ADMG:
        """Get an ADMG instance."""
        di_edges = list(self.directed.edges())
        bi_edges = list(self.undirected.edges())
        vertices = list(self.directed)  # could be either since they're maintained together
        return ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)

    def to_causaleffect_str(self) -> str:
        """Get a string to be imported by R."""
        if not self.directed:
            raise ValueError('graph must have some directed edges')

        formula = ', '.join(
            f'{u} -+ {v}'
            for u, v in self.directed.edges()
        )
        if self.undirected:
            formula += ''.join(
                f', {u} -+ {v}, {v} -+ {u}'
                for u, v in self.undirected.edges()
            )

        rv = f'g <- graph.formula({formula}, simplify = FALSE)'
        for i in range(self.undirected.number_of_edges()):
            idx = 2 * i + self.directed.number_of_edges() + 1
            rv += (
                f'\ng <- set.edge.attribute(graph = g, name = "description",'
                f' index = c({idx}, {idx + 1}), value = "U")'
            )

        return rv


napkin_graph = NxMixedGraph()
napkin_graph.add_directed_edge('W', 'R')
napkin_graph.add_directed_edge('R', 'X')
napkin_graph.add_directed_edge('X', 'Y')
napkin_graph.add_directed_edge('W', 'V1')
napkin_graph.add_directed_edge('V1', 'Y')
napkin_graph.add_undirected_edge('W', 'X')
napkin_graph.add_undirected_edge('W', 'Y')
