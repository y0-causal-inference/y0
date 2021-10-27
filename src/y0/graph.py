# -*- coding: utf-8 -*-

"""Graph data structures."""

from __future__ import annotations

import itertools as itt
import json
from dataclasses import dataclass, field
from typing import Any, Collection, Iterable, List, Mapping, Optional, Tuple, Union

import networkx as nx
from ananke.graphs import ADMG
from networkx.classes.reportviews import NodeView
from networkx.utils import open_file

from .dsl import CounterfactualVariable, Variable

__all__ = [
    "NxMixedGraph",
    "CausalEffectGraph",
    "DEFULT_PREFIX",
    "DEFAULT_TAG",
    "admg_to_latent_variable_dag",
    "admg_from_latent_variable_dag",
    "set_latent",
]

CausalEffectGraph = Any

#: The default key in a latent variable DAG represented as a :class:`networkx.DiGraph`
#: for nodes that corresond to "latent" variables
DEFAULT_TAG = "hidden"
#: The default prefix for latent variables in a latent variable DAG represented. After the prefix,
#: there will be a number assigned that's incremented during construction.
DEFULT_PREFIX = "u_"


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

    def __eq__(self, other: Any) -> bool:
        """Check for equality of nodes, directed edges, and undirected edges."""
        return (
            isinstance(other, NxMixedGraph)
            and self.nodes() == other.nodes()
            and (self.directed.edges() == other.directed.edges())
            and (self.undirected.edges() == other.undirected.edges())
        )

    def __iter__(self) -> Iterable[Variable]:
        """Iterate over nodes in the graph."""
        return iter(self.directed)

    def __len__(self) -> int:
        """Count the nodes in the graph."""
        return len(self.directed)

    def __contains__(self, item: Variable) -> bool:
        """Check if the given item is a node in the graph."""
        return item in self.directed

    def is_counterfactual(self) -> bool:
        """Check if this is a counterfactual graph."""
        return any(isinstance(n, CounterfactualVariable) for n in self.nodes())

    def add_node(self, n: Variable) -> None:
        """Add a node."""
        n = Variable.norm(n)
        self.directed.add_node(n)
        self.undirected.add_node(n)

    def add_directed_edge(self, u: Union[str, Variable], v: Union[str, Variable], **attr) -> None:
        """Add a directed edge from u to v."""
        u = Variable.norm(u)
        v = Variable.norm(v)
        self.directed.add_edge(u, v, **attr)
        self.undirected.add_node(u)
        self.undirected.add_node(v)

    def add_undirected_edge(self, u: Variable, v: Variable, **attr) -> None:
        """Add an undirected edge between u and v."""
        u = Variable.norm(u)
        v = Variable.norm(v)
        self.undirected.add_edge(u, v, **attr)
        self.directed.add_node(u)
        self.directed.add_node(v)

    def nodes(self) -> NodeView:
        """Get the nodes in the graph."""
        return self.directed.nodes()

    def to_admg(self) -> ADMG:
        """Get an ADMG instance."""
        if self.is_counterfactual():
            raise NotImplementedError("Can not directly convert counterfactual graph to ADMG")
        return ADMG(
            vertices=[n.name for n in self.nodes()],
            di_edges=[(u.name, v.name) for u, v in self.directed.edges()],
            bi_edges=[(u.name, v.name) for u, v in self.undirected.edges()],
        )

    @classmethod
    def from_admg(cls, admg: ADMG) -> NxMixedGraph:
        """Create from an ADMG."""
        return cls.from_str_edges(
            nodes=admg.vertices,
            directed=admg.di_edges,
            undirected=admg.bi_edges,
        )

    def to_latent_variable_dag(
        self,
        *,
        prefix: Optional[str] = None,
        start: int = 0,
        tag: Optional[str] = None,
    ) -> nx.DiGraph:
        """Create a labeled DAG where bi-directed edges are assigned as nodes upstream of their two incident nodes.

        :param prefix: The prefix for latent variables. If none, defaults to :data:`y0.graph.DEFAULT_PREFIX`.
        :param start: The starting number for latent variables (defaults to 0, could be changed to 1 if desired)
        :param tag: The key for node data describing whether it is latent.
            If None, defaults to :data:`y0.graph.DEFAULT_TAG`.
        :return: A latent variable DAG.
        """
        if self.is_counterfactual():
            raise NotImplementedError
        return _latent_dag(
            di_edges=self.directed.edges(),
            bi_edges=self.undirected.edges(),
            prefix=prefix,
            start=start,
            tag=tag,
        )

    @classmethod
    def from_latent_variable_dag(cls, graph: nx.DiGraph, tag: Optional[str] = None) -> NxMixedGraph:
        """Load a labeled DAG."""
        if tag is None:
            tag = DEFAULT_TAG
        if any(tag not in data for data in graph.nodes.values()):
            raise ValueError(f"missing label {tag} in one or more nodes.")

        rv = cls()
        for node, data in graph.nodes.items():
            if data[tag]:
                for a, b in itt.combinations(graph.successors(node), 2):
                    rv.add_undirected_edge(a, b)
            else:
                for child in graph.successors(node):
                    rv.add_directed_edge(node, child)
        return rv

    def to_causaleffect(self) -> CausalEffectGraph:
        """Get a causaleffect R object.

        :returns: A causaleffect R object.

        .. warning:: Appropriate R imports need to be done first for 'causaleffect' and 'igraph'.
        """
        import rpy2.robjects

        return rpy2.robjects.r(self.to_causaleffect_str())

    def joint(self) -> nx.MultiGraph:
        """Return a joint graph."""
        rv = nx.MultiGraph()
        rv.add_nodes_from(self.directed)
        rv.add_edges_from(self.directed.edges)
        rv.add_edges_from(self.undirected.edges)
        return rv

    def draw(
        self, ax=None, title: Optional[str] = None, prog: str = "dot", latex: bool = True
    ) -> None:
        """Render the graph using matplotlib.

        :param ax: Axis to draw on (if none specified, makes a new one)
        :param title: The optional title to show with the graph
        :param prog: The pydot program to use, like dot, neato, etc.
        :param latex: Parse string variables as y0 if possible to make pretty latex output
        """
        import matplotlib.pyplot as plt

        layout = nx.nx_pydot.graphviz_layout(self.joint(), prog=prog)
        u_proxy = nx.DiGraph(self.undirected.edges)
        labels = None if not latex else {node: _get_latex(node) for node in self.directed}

        if ax is None:
            ax = plt.gca()
        nx.draw_networkx_nodes(self.directed, pos=layout, node_color="white", node_size=350, ax=ax)
        nx.draw_networkx_labels(self.directed, pos=layout, ax=ax, labels=labels)
        nx.draw_networkx_edges(self.directed, pos=layout, edge_color="b", ax=ax)
        nx.draw_networkx_edges(
            u_proxy,
            pos=layout,
            ax=ax,
            connectionstyle="arc3, rad=0.2",
            arrowstyle="-",
            edge_color="r",
        )

        if title:
            ax.set_title(title)
        ax.axis("off")

    @classmethod
    def from_causaleffect(cls, graph) -> NxMixedGraph:
        """Construct an instance from a causaleffect R graph."""
        raise NotImplementedError

    def to_causaleffect_str(self) -> str:
        """Get a string to be imported by R."""
        if not self.directed:
            raise ValueError("graph must have some directed edges")

        formula = ", ".join(f"{u} -+ {v}" for u, v in self.directed.edges())
        if self.undirected:
            formula += "".join(f", {u} -+ {v}, {v} -+ {u}" for u, v in self.undirected.edges())

        rv = f"g <- graph.formula({formula}, simplify = FALSE)"
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
        nodes: Optional[Iterable[Variable]] = None,
        directed: Optional[Iterable[Tuple[Variable, Variable]]] = None,
        undirected: Optional[Iterable[Tuple[Variable, Variable]]] = None,
    ) -> NxMixedGraph:
        """Make a mixed graph from a pair of edge lists."""
        if directed is None and undirected is None:
            raise ValueError("must provide at least one of directed/undirected edge lists")
        rv = cls()
        for n in nodes or []:
            rv.add_node(n)
        for u, v in directed or []:
            rv.add_directed_edge(u, v)
        for u, v in undirected or []:
            rv.add_undirected_edge(u, v)
        return rv

    @classmethod
    def from_str_edges(
        cls,
        nodes: Optional[Iterable[str]] = None,
        directed: Optional[Iterable[Tuple[str, str]]] = None,
        undirected: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> NxMixedGraph:
        """Make a mixed graph from a pair of edge lists where nodes are strings."""
        return cls.from_edges(
            nodes=None if nodes is None else [Variable(n) for n in nodes],
            directed=None if directed is None else vmap_pairs(directed),
            undirected=None if undirected is None else vmap_pairs(undirected),
        )

    @classmethod
    def from_adj(
        cls,
        nodes: Optional[Iterable[Variable]] = None,
        directed: Optional[Mapping[Variable, Collection[Variable]]] = None,
        undirected: Optional[Mapping[Variable, Collection[Variable]]] = None,
    ) -> NxMixedGraph:
        """Make a mixed graph from a pair of adjacency lists."""
        rv = cls()
        for n in nodes or []:
            rv.add_node(n)
        for u, vs in (directed or {}).items():
            rv.add_node(u)
            for v in vs:
                rv.add_directed_edge(u, v)
        for u, vs in (undirected or {}).items():
            rv.add_node(u)
            for v in vs:
                rv.add_undirected_edge(u, v)
        return rv

    @classmethod
    def from_str_adj(
        cls,
        nodes: Optional[Iterable[str]] = None,
        directed: Optional[Mapping[str, Collection[str]]] = None,
        undirected: Optional[Mapping[str, Collection[str]]] = None,
    ) -> NxMixedGraph:
        return cls.from_adj(
            nodes=None if nodes is None else [Variable(n) for n in nodes],
            directed=None if directed is None else vmap_adj(directed),
            undirected=None if undirected is None else vmap_adj(undirected),
        )

    @classmethod
    @open_file(1)
    def from_causalfusion_path(cls, file) -> NxMixedGraph:
        """Load a graph from a CausalFusion JSON file."""
        return cls.from_causalfusion_json(json.load(file))

    @classmethod
    def from_causalfusion_json(cls, data: Mapping[str, Any]) -> NxMixedGraph:
        """Load a graph from a CausalFusion JSON object."""
        rv = cls()
        for edge in data["edges"]:
            u, v = edge["from"], edge["to"]
            if edge["type"] == "directed":
                rv.add_directed_edge(u, v)
            elif edge["type"] == "bidirected":
                rv.add_undirected_edge(u, v)
            else:
                raise ValueError(f'unhandled edge type: {edge["type"]}')
        return rv

    def subgraph(self, vertices: Collection[Variable]) -> NxMixedGraph:
        """Return a subgraph given a set of vertices.

        :param vertices: a subset of nodes
        :returns: A NxMixedGraph subgraph
        """
        vertices = set(vertices)
        return self.from_edges(
            nodes=vertices,
            directed=_include_adjacent(self.directed, vertices),
            undirected=_include_adjacent(self.undirected, vertices),
        )

    def intervene(self, vertices: Collection[Variable]) -> NxMixedGraph:
        """Return a mutilated graph given a set of interventions.

        :param vertices: a subset of nodes from which to remove incoming edges
        :returns: A NxMixedGraph subgraph
        """
        vertices = set(vertices)
        return self.from_edges(
            nodes=vertices,
            directed=_exclude_target(self.directed, vertices),
            undirected=_exclude_adjacent(self.undirected, vertices),
        )

    def remove_nodes_from(self, vertices: Collection[Variable]) -> NxMixedGraph:
        """Return a subgraph that does not contain any of the specified vertices.

        :param vertices: a set of nodes to remove from graph
        :returns: A NxMixedGraph subgraph
        """
        vertices = set(vertices)
        return self.from_edges(
            nodes=self.nodes() - vertices,
            directed=_exclude_adjacent(self.directed, vertices),
            undirected=_exclude_adjacent(self.undirected, vertices),
        )

    def remove_outgoing_edges_from(self, vertices: Collection[Variable]) -> NxMixedGraph:
        """Return a subgraph that does not have any outgoing edges from any of the given vertices.

        :param vertices: a set of nodes whose outgoing edges get removed from the graph
        :returns: NxMixedGraph subgraph
        """
        vertices = set(vertices)
        return self.from_edges(
            nodes=self.nodes(),
            directed=_exclude_source(self.directed, vertices),
            undirected=self.undirected.edges(),
        )

    def ancestors_inclusive(self, sources: Iterable[Variable]) -> set[Variable]:
        """Ancestors of a set include the set itself."""
        return _ancestors_inclusive(self.directed, sources)

    def topological_sort(self) -> Iterable[Variable]:
        """Get a topological sort from the directed component of the mixed graph."""
        return nx.topological_sort(self.directed)

    def connected_components(self) -> Iterable[set[Variable]]:
        """Iterate over the connected components in the undirected graph."""
        return nx.connected_components(self.undirected)

    def get_c_components(self) -> list[frozenset[Variable]]:
        """Get the C-components in the undirected portion of the graph."""
        return [frozenset(c) for c in self.connected_components()]

    def is_connected(self) -> bool:
        """Return if there is only a single connected component in the undirected graph."""
        return nx.is_connected(self.undirected)


def _ancestors_inclusive(graph: nx.DiGraph, sources: Iterable[Variable]) -> set[Variable]:
    rv = set(sources)
    for source in sources:
        rv.update(nx.algorithms.dag.ancestors(graph, source))
    return rv


def _include_adjacent(
    graph: nx.Graph, vertices: Collection[Variable]
) -> Collection[Tuple[Variable, Variable]]:
    return [(u, v) for u, v in graph.edges() if u in vertices and v in vertices]


def _exclude_source(
    graph: nx.Graph, vertices: Collection[Variable]
) -> Collection[Tuple[Variable, Variable]]:
    return [(u, v) for u, v in graph.edges() if u not in vertices]


def _exclude_target(
    graph: nx.Graph, vertices: Collection[Variable]
) -> Collection[Tuple[Variable, Variable]]:
    return [(u, v) for u, v in graph.edges() if v not in vertices]


def _exclude_adjacent(
    graph: nx.Graph, vertices: Collection[Variable]
) -> Collection[Tuple[Variable, Variable]]:
    return [(u, v) for u, v in graph.edges() if u not in vertices and v not in vertices]


def admg_to_latent_variable_dag(
    graph: ADMG,
    prefix: Optional[str] = None,
    start: int = 0,
    tag: Optional[str] = None,
) -> nx.DiGraph:
    """Convert an ADMG to a latent variable DAG.

    :param graph: An ADMG
    :param prefix: The prefix for latent variables. If none, defaults to :data:`y0.graph.DEFAULT_PREFIX`.
    :param start: The starting number for latent variables (defaults to 0, could be changed to 1 if desired)
    :param tag: The key for node data describing whether it is latent.
        If None, defaults to :data:`y0.graph.DEFAULT_TAG`.
    :return: A latent variable DAG.
    """
    return _latent_dag(
        graph.di_edges,
        graph.bi_edges,
        prefix=prefix,
        start=start,
        tag=tag,
    )


def admg_from_latent_variable_dag(graph: nx.DiGraph, *, tag: Optional[str] = None) -> ADMG:
    """Convert a latent variable DAG to an ADMG.

    :param graph: A latent variable directed acyclic graph (LV-DAG)
    :param tag: The key for node data describing whether it is latent.
        If None, defaults to :data:`y0.graph.DEFAULT_TAG`.
    :return: An ADMG
    """
    return NxMixedGraph.from_latent_variable_dag(graph, tag=tag).to_admg()


def _latent_dag(
    di_edges: Iterable[Tuple[Variable, Variable]],
    bi_edges: Iterable[Tuple[Variable, Variable]],
    *,
    prefix: Optional[str] = None,
    start: int = 0,
    tag: Optional[str] = None,
) -> nx.DiGraph:
    """Create a labeled DAG where bi-directed edges are assigned as nodes upstream of their two incident nodes.

    :param di_edges: A list of directional edges
    :param bi_edges: A list of bi-directional edges
    :param prefix: The prefix for latent variables. If none, defaults to :data:`y0.graph.DEFAULT_PREFIX`.
    :param start: The starting number for latent variables (defaults to 0, could be changed to 1 if desired)
    :param tag: The key for node data describing whether it is latent.
        If None, defaults to :data:`y0.graph.DEFAULT_TAG`.
    :return: A latent variable DAG.
    """
    if tag is None:
        tag = DEFAULT_TAG
    if prefix is None:
        prefix = DEFULT_PREFIX

    di_edges = [(u.name, v.name) for u, v in di_edges]
    bi_edges = [(u.name, v.name) for u, v in bi_edges]

    rv = nx.DiGraph()
    rv.add_nodes_from(itt.chain.from_iterable(bi_edges))
    rv.add_edges_from(di_edges)
    nx.set_node_attributes(rv, False, tag)
    for i, (u, v) in enumerate(sorted(bi_edges), start=start):
        latent_node = f"{prefix}{i}"
        rv.add_node(latent_node, **{tag: True})
        rv.add_edge(latent_node, u)
        rv.add_edge(latent_node, v)
    return rv


def set_latent(
    graph: nx.DiGraph,
    latent_nodes: Union[str, Iterable[str]],
    tag: Optional[str] = None,
) -> None:
    """Quickly set the latent variables in a graph."""
    if tag is None:
        tag = DEFAULT_TAG
    if isinstance(latent_nodes, str):
        latent_nodes = [latent_nodes]

    latent_nodes = set(latent_nodes)
    for node, data in graph.nodes(data=True):
        data[tag] = node in latent_nodes


def _get_latex(node) -> str:
    if isinstance(node, str):
        from y0.parser import parse_y0

        try:
            expr = parse_y0(node)
        except Exception:
            return node
        else:
            return expr._repr_latex_()

    from y0.dsl import Variable

    if isinstance(node, Variable):
        return node._repr_latex_()
    raise TypeError


def vmap_pairs(edges: Iterable[Tuple[str, str]]) -> List[Tuple[Variable, Variable]]:
    return [(Variable(source), Variable(target)) for source, target in edges]


def vmap_adj(adjacency_dict):
    return {
        Variable(source): [Variable(target) for target in targets]
        for source, targets in adjacency_dict.items()
    }
