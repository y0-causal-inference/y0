# -*- coding: utf-8 -*-

"""Graph data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Generic, Iterable, Mapping, Optional, Tuple, TypeVar

import networkx as nx
from ananke.graphs import ADMG

__all__ = [
    'NxMixedGraph',
    'CausalEffectGraph',
]

X = TypeVar('X')
CausalEffectGraph = Any


@dataclass
class NxMixedGraph(Generic[X]):
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

    def add_directed_edge(self, u: X, v: X, **attr) -> None:
        """Add a directed edge from u to v."""
        self.directed.add_edge(u, v, **attr)
        self.undirected.add_node(u)
        self.undirected.add_node(v)

    def add_undirected_edge(self, u: X, v: X, **attr) -> None:
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

    def to_causaleffect(self) -> CausalEffectGraph:
        """Get a causaleffect R object.

        :returns: A causaleffect R object.

        .. warning:: Appropriate R imports need to be done first for 'causaleffect' and 'igraph'.
        """
        import rpy2.robjects
        return rpy2.robjects.r(self.to_causaleffect_str())

    @classmethod
    def from_causaleffect(cls, graph) -> NxMixedGraph:
        """Construct an instance from a causaleffect R graph."""
        raise NotImplementedError

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

    @classmethod
    def from_edges(
        cls,
        directed: Iterable[Tuple[X, X]],
        undirected: Optional[Iterable[Tuple[X, X]]] = None,
    ) -> NxMixedGraph:
        """Make a mixed graph from a pair of edge lists."""
        rv = cls()
        for u, v in directed:
            rv.add_directed_edge(u, v)
        for u, v in undirected or []:
            rv.add_undirected_edge(u, v)
        return rv

    @classmethod
    def from_adj(
        cls,
        directed: Mapping[X, Collection[X]],
        undirected: Mapping[X, Collection[X]],
    ) -> NxMixedGraph:
        """Make a mixed graph from a pair of adjacency lists."""
        rv = cls()
        for u, vs in directed.items():
            for v in vs:
                rv.add_directed_edge(u, v)
        for u, vs in undirected.items():
            for v in vs:
                rv.add_undirected_edge(u, v)
        return rv
