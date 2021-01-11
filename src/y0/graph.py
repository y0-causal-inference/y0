# -*- coding: utf-8 -*-

"""Graph data structures."""

import networkx as nx
from ananke.graphs import ADMG


# TODO the actual abstract implementation should evolve to meet what is
#  required to implement the identify() algorithm.

class NxMixedGraph:
    """A mixed graph based on :mod:`networkx`."""

    def __init__(self):
        """Initialize the networkx mixed graph."""
        self.directed = nx.DiGraph()
        self.undirected = nx.Graph()

    def add_directed_edge(self, u, v, **attr):
        """Add a directed edge from u to v."""
        self.directed.add_edge(u, v, **attr)
        self.undirected.add_node(u)
        self.undirected.add_node(v)

    def add_undirected_edge(self, u, v, **attr):
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
