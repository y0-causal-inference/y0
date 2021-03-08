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

DEFAULT_TAG = 'hidden'


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

    def to_labeled_dag(self, prefix: str = 'u_', start: int = 0, tag: Optional[str] = None) -> nx.DiGraph:
        """Create a labeled DAG where bi-directed edges are assigned as nodes upstream of their two incident nodes.

        :param prefix: The prefix for latent variables.
        :param start: The starting number for latent variables (defaults to 0, could be changed to 1 if desired)
        :param tag: The key for node data describing whether it is latent.
        :return: A labeled DAG.
        """
        if tag is None:
            tag = DEFAULT_TAG
        rv = self.directed.copy()
        nx.set_node_attributes(rv, False, tag)
        for i, (u, v) in enumerate(sorted(self.undirected.edges()), start=start):
            latent_node = f'{prefix}{i}'
            rv.add_node(latent_node, **{tag: True})
            rv.add_edge(latent_node, u)
            rv.add_edge(latent_node, v)
        return rv

    @classmethod
    def from_labeled_dag(cls, graph: nx.DiGraph, tag: Optional[str] = None) -> NxMixedGraph:
        """Load a labeled DAG."""
        if tag is None:
            tag = DEFAULT_TAG
        if any(tag not in data for data in graph.nodes.values()):
            raise ValueError(f'missing label {tag} in one or more nodes.')

        rv = cls()
        for node, data in graph.nodes.items():
            if data[tag]:
                # this works because there are always exactly 2 children of a latent node
                (_, a), (_, b) = list(graph.out_edges(node))
                rv.add_undirected_edge(a, b)
            else:
                for _, child in graph.out_edges(node):
                    rv.add_directed_edge(node, child)
        return rv

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
