"""Graph data structures."""

from __future__ import annotations

import importlib.util
import itertools as itt
import json
import unittest
import warnings
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain, combinations
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import networkx as nx
from networkx.classes.reportviews import NodeView
from networkx.utils import open_file

from .dsl import (
    CounterfactualVariable,
    Intervention,
    P,
    Probability,
    Variable,
    vmap_adj,
    vmap_pairs,
)

if TYPE_CHECKING:
    import ananke.graphs
    import matplotlib.axes
    import pgmpy.inference.CausalInference
    import pgmpy.models
    import sympy

__all__ = [
    "DEFAULT_TAG",
    "DEFULT_PREFIX",
    "CausalEffectGraph",
    "NxMixedGraph",
    "set_latent",
]

CausalEffectGraph = Any

#: The default key in a latent variable DAG represented as a :class:`networkx.DiGraph`
#: for nodes that correspond to "latent" variables
DEFAULT_TAG = "hidden"
#: The default prefix for latent variables in a latent variable DAG represented. After the prefix,
#: there will be a number assigned that's incremented during construction.
DEFULT_PREFIX = "u_"
NO_SET_LATENT_FLAG = "no_set_latent"

ANANKE_AVAILABLE = bool(importlib.util.find_spec("ananke"))
ANANKE_REQUIRED = unittest.skipUnless(ANANKE_AVAILABLE, reason="Ananke is not installed")

LayoutDict = dict[Variable, tuple[float, float]]


@dataclass
class NxMixedGraph:
    """A mixed graph based on a :class:`networkx.Graph` and a :class:`networkx.DiGraph`.

    Example usage:

    .. code-block:: python

        graph = NxMixedGraph()
        graph.add_directed_edge("X", "Y")
        graph.add_undirected_edge("X", "Y")
    """

    #: A directed graph
    directed: nx.DiGraph = field(default_factory=nx.DiGraph)
    #: A undirected graph
    undirected: nx.Graph = field(default_factory=nx.Graph)

    def __post_init__(self) -> None:
        """Process the graphs."""
        self.directed.graph[NO_SET_LATENT_FLAG] = True
        self.undirected.graph[NO_SET_LATENT_FLAG] = True

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
        return iter(self.directed.nodes())  # type:ignore

    def __len__(self) -> int:
        """Count the nodes in the graph."""
        return len(self.directed)

    def __contains__(self, item: Variable) -> bool:
        """Check if the given item is a node in the graph."""
        return item in self.directed

    def copy(self) -> NxMixedGraph:
        """Get a copy of the graph."""
        return self.__class__(
            directed=self.directed.copy(),
            undirected=self.undirected.copy(),
        )

    def is_counterfactual(self) -> bool:
        """Check if this is a counterfactual graph."""
        return any(isinstance(n, CounterfactualVariable) for n in self.nodes())

    def raise_on_counterfactual(self) -> None:
        """Raise an error if this is a counterfactual graph.

        :raises ValueError: if this graph is a counterfactual graph
        """
        if self.is_counterfactual():
            raise ValueError("This operation is not available for counterfactual graphs")

    def add_node(self, n: Variable) -> None:
        """Add a node."""
        n = Variable.norm(n)
        self.directed.add_node(n)
        self.undirected.add_node(n)

    def add_directed_edge(self, u: str | Variable, v: str | Variable, **attr: Any) -> None:
        """Add a directed edge from u to v."""
        u = Variable.norm(u)
        v = Variable.norm(v)
        self.directed.add_edge(u, v, **attr)
        self.undirected.add_node(u)
        self.undirected.add_node(v)

    def add_undirected_edge(self, u: str | Variable, v: str | Variable, **attr: Any) -> None:
        """Add an undirected edge between u and v."""
        u = Variable.norm(u)
        v = Variable.norm(v)
        self.undirected.add_edge(u, v, **attr)
        self.directed.add_node(u)
        self.directed.add_node(v)

    def nodes(self) -> NodeView[Variable]:
        """Get the nodes in the graph."""
        return self.directed.nodes()

    def joint_probability(self) -> Probability:
        """Get the joint probability over all nodes."""
        return P(self.nodes())

    def to_admg(self) -> ananke.graphs.ADMG:
        """Get an ananke ADMG."""
        self.raise_on_counterfactual()
        from ananke.graphs import ADMG

        # update the way stringification happens so this
        # can support arbitrary variables, like counterfactuals
        return ADMG(
            vertices=[n.name for n in self.nodes()],
            di_edges=[(u.name, v.name) for u, v in self.directed.edges()],
            bi_edges=[(u.name, v.name) for u, v in self.undirected.edges()],
        )

    def to_pgmpy_bayesian_network(self) -> pgmpy.models.DiscreteBayesianNetwork:
        """Convert a mixed graph to an equivalent :class:`pgmpy.DiscreteBayesianNetwork`."""
        from pgmpy.models import DiscreteBayesianNetwork

        edges = [(u.name, v.name) for u, v in self.directed.edges()]
        latents = set()
        for u, v in self.undirected.edges():
            latent = f"U_{u.name}_{v.name}"
            latents.add(latent)
            edges.append((latent, u.name))
            edges.append((latent, v.name))
        model = DiscreteBayesianNetwork(ebunch=edges, latents=latents)
        return model

    def to_pgmpy_causal_inference(self) -> pgmpy.inference.CausalInference.CausalInference:
        """Get a pgmpy causal inference object."""
        from pgmpy.inference.CausalInference import CausalInference

        return CausalInference(self.to_pgmpy_bayesian_network())

    def to_linear_scm_sympy(self) -> dict[Variable, sympy.Expr]:
        """Generate a Sympy system of equations."""
        import sympy

        variable_to_equation = {}
        for node in self.topological_sort():
            terms = []

            # Add parent edges
            for parent in self.directed.predecessors(node):
                beta = sympy_nested(r"\beta", parent, node)
                terms.append(beta * parent.to_sympy())

            # Add noise term
            epsilon_symbol = sympy_nested(r"\epsilon", node)
            terms.append(epsilon_symbol)

            # get bidirected edges
            for u, v in self.undirected.edges(node):
                u, v = sorted([u, v])
                gamma_symbol = sympy_nested(r"\gamma", u, v)
                terms.append(gamma_symbol)

            variable_to_equation[node] = cast(sympy.Expr, sum(terms))
        return variable_to_equation

    def to_linear_scm_latex(self) -> str:
        """Generate a Sympy system of equations."""
        import sympy

        equations_dict = self.to_linear_scm_sympy()
        latex_equations = [
            rf"{variable.to_latex()} &= {sympy.latex(expression)} \\"
            for variable, expression in equations_dict.items()
        ]
        return _LatexStr(r"\begin{align*}" + "\n ".join(latex_equations) + r"\end{align*}")

    @classmethod
    def from_admg(cls, admg: ananke.graphs.ADMG) -> NxMixedGraph:
        """Create from an ananke ADMG."""
        return cls.from_str_edges(
            nodes=admg.vertices,
            directed=admg.di_edges,
            undirected=admg.bi_edges,
        )

    def to_latent_variable_dag(
        self,
        *,
        prefix: str | None = None,
        start: int = 0,
        tag: str | None = None,
    ) -> nx.DiGraph:
        """Create a labeled DAG where bi-directed edges are assigned as nodes upstream of their two incident nodes.

        :param prefix: The prefix for latent variables. If none, defaults to :data:`y0.graph.DEFAULT_PREFIX`.
        :param start: The starting number for latent variables (defaults to 0, could be changed to 1 if desired)
        :param tag: The key for node data describing whether it is latent.
            If None, defaults to :data:`y0.graph.DEFAULT_TAG`.
        :return: A latent variable DAG.
        """
        self.raise_on_counterfactual()
        return _latent_dag(
            di_edges=self.directed.edges(),
            bi_edges=self.undirected.edges(),
            prefix=prefix,
            start=start,
            tag=tag,
        )

    @classmethod
    def from_latent_variable_dag(cls, graph: nx.DiGraph, tag: str | None = None) -> NxMixedGraph:
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

    def moralize(self) -> NxMixedGraph:
        """Moralize the graph.

        :returns: A moralized ADMG in which all nodes $U$ and $v$ that are parents of some
            node $N$ are connected with an undirected edge.

        .. seealso:: https://en.wikipedia.org/wiki/Moral_graph
        """
        rv = NxMixedGraph(directed=self.directed.copy(), undirected=self.undirected.copy())
        # Moralize (link parents of mentioned nodes)
        for u, v in iter_moral_links(self):
            rv.add_undirected_edge(u, v)
        return rv

    def _layout(self, prog: str) -> LayoutDict:
        joint = self.joint()
        try:
            layout = nx.nx_agraph.pygraphviz_layout(joint, prog=prog)
        except ImportError:
            pass
        else:
            return cast(LayoutDict, layout)
        try:
            layout = nx.nx_pydot.pydot_layout(joint, prog=prog)
        except (ImportError, IndexError):
            pass
        else:
            return cast(LayoutDict, layout)

        return cast(LayoutDict, nx.spring_layout(joint))

    def draw(
        self,
        ax: None | matplotlib.axes.Axes = None,
        title: str | None = None,
        prog: str | None = None,
        latex: bool = True,
        layout: LayoutDict | None = None,
        node_size: int = 1_500,
        node_size_offset: int = 500,
        line_widths: int = 2,
        margins: float = 0.3,
        font_size: int = 20,
        arrow_size: int = 20,
        radius: float = 0.3,
    ) -> None:
        """Render the graph using matplotlib.

        :param ax: Axis to draw on (if none specified, makes a new one)
        :param title: The optional title to show with the graph
        :param prog: The pydot program to use, like dot, neato, osage, etc.
            If none is given, uses osage for small graphs and dot for larger ones.
        :param node_size: node size
        :param node_size_offset: node size offset
        :param line_widths: line widths
        :param margins: margins
        :param font_size: node text font size
        :param arrow_size: edge arrow size
        :param radius: bidirectional edge radius
        :param latex: Parse string variables as y0 if possible to make pretty latex output
        """
        import matplotlib.pyplot as plt

        if prog is None:
            if self.directed.number_of_nodes() > 6:
                prog = "dot"
            else:
                prog = "osage"

        if layout is None:
            layout = self._layout(prog=prog)
        u_proxy = nx.DiGraph(self.undirected.edges)
        labels = None if not latex else {node: _get_latex(node) for node in self.directed}

        if ax is None:
            ax = plt.gca()

        # TODO choose sizes based on size of axis

        nx.draw_networkx_nodes(
            self.directed,
            pos=layout,
            node_color="white",
            node_size=node_size,
            edgecolors="black",
            linewidths=line_widths,
            ax=ax,
            margins=margins,
        )
        nx.draw_networkx_labels(
            self.directed, pos=layout, ax=ax, labels=labels, font_size=font_size
        )
        nx.draw_networkx_edges(
            self.directed,
            pos=layout,
            edge_color="black",
            ax=ax,
            node_size=node_size + node_size_offset,
            width=line_widths,
            arrowsize=arrow_size,
        )
        nx.draw_networkx_edges(
            u_proxy,
            pos=layout,
            node_size=node_size + node_size_offset,
            ax=ax,
            style=":",
            width=line_widths,
            connectionstyle=f"arc3, rad={radius}",
            arrowstyle="-",
            edge_color="grey",
        )

        if title:
            ax.set_title(title)
        ax.axis("off")

    @classmethod
    def from_causaleffect(cls, graph: Any) -> NxMixedGraph:
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
        nodes: Iterable[Variable] | None = None,
        directed: Iterable[tuple[Variable, Variable]] | None = None,
        undirected: Iterable[tuple[Variable, Variable]] | None = None,
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
        nodes: Iterable[str] | None = None,
        directed: Iterable[tuple[str, str]] | None = None,
        undirected: Iterable[tuple[str, str]] | None = None,
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
        nodes: Iterable[Variable] | None = None,
        directed: Mapping[Variable, Collection[Variable]] | None = None,
        undirected: Mapping[Variable, Collection[Variable]] | None = None,
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
        nodes: Iterable[str] | None = None,
        directed: Mapping[str, Collection[str]] | None = None,
        undirected: Mapping[str, Collection[str]] | None = None,
    ) -> NxMixedGraph:
        """Make a mixed graph from a pair of adjacency lists of strings."""
        return cls.from_adj(
            nodes=None if nodes is None else [Variable(n) for n in nodes],
            directed=None if directed is None else vmap_adj(directed),
            undirected=None if undirected is None else vmap_adj(undirected),
        )

    @classmethod
    @open_file(1)  # type:ignore
    def from_causalfusion_path(cls, file) -> NxMixedGraph:  # type:ignore[no-untyped-def]
        """Load a graph from a CausalFusion JSON file."""
        data = json.load(file)
        return cls.from_causalfusion_json(data)

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
                raise ValueError(f"unhandled edge type: {edge['type']}")
        return rv

    def subgraph(self, vertices: Variable | Iterable[Variable]) -> NxMixedGraph:
        """Return a subgraph given a set of vertices.

        :param vertices: a subset of nodes
        :returns: A NxMixedGraph subgraph
        """
        vertices = _ensure_set(vertices)
        return self.from_edges(
            nodes=vertices,
            directed=_include_adjacent(self.directed, vertices),
            undirected=_include_adjacent(self.undirected, vertices),
        )

    def remove_in_edges(self, vertices: Variable | Iterable[Variable]) -> NxMixedGraph:
        """Return a mutilated graph given a set of interventions.

        :param vertices: a subset of nodes from which to remove incoming edges
        :returns: A NxMixedGraph subgraph
        """
        vertices = _ensure_set(vertices)
        return self.from_edges(
            nodes=vertices,
            directed=_exclude_target(self.directed, vertices),
            undirected=_exclude_adjacent(self.undirected, vertices),
        )

    def get_intervened_ancestors(
        self, interventions: Variable | set[Variable], outcomes: Variable | set[Variable]
    ) -> set[Variable]:
        """Get the ancestors of outcomes in a graph that has been intervened on.

        :param interventions: a set of interventions in the graph
        :param outcomes: a set of outcomes in the graph
        :returns: Set of nodes
        """
        return self.remove_in_edges(interventions).ancestors_inclusive(outcomes)

    def get_no_effect_on_outcomes(
        self, interventions: set[Variable], outcomes: set[Variable]
    ) -> set[Variable]:
        """Find nodes in the graph which have no effect on the outcomes.

        :param interventions: a set of interventions in the graph
        :param outcomes: a set of outcomes in the graph
        :returns: Set of nodes
        """
        return (
            set(self.nodes())
            - interventions
            - self.get_intervened_ancestors(interventions, outcomes)
        )

    def remove_nodes_from(self, vertices: Variable | Iterable[Variable]) -> NxMixedGraph:
        """Return a subgraph that does not contain any of the specified vertices.

        :param vertices: a set of nodes to remove from graph
        :returns: A NxMixedGraph subgraph
        """
        vertices = _ensure_set(vertices)
        return self.from_edges(
            nodes=self.nodes() - vertices,
            directed=_exclude_adjacent(self.directed, vertices),
            undirected=_exclude_adjacent(self.undirected, vertices),
        )

    def remove_out_edges(self, vertices: Variable | Iterable[Variable]) -> NxMixedGraph:
        """Return a subgraph that does not have any outgoing edges from any of the given vertices.

        :param vertices: a set of nodes whose outgoing edges get removed from the graph
        :returns: NxMixedGraph subgraph
        """
        vertices = _ensure_set(vertices)
        return self.from_edges(
            nodes=self.nodes(),
            directed=_exclude_source(self.directed, vertices),
            undirected=self.undirected.edges(),
        )

    def ancestors_inclusive(self, sources: Variable | Iterable[Variable]) -> set[Variable]:
        """Ancestors of a set include the set itself."""
        sources = _ensure_set(sources)
        return _ancestors_inclusive(self.directed, sources)

    def descendants_inclusive(self, sources: Variable | Iterable[Variable]) -> set[Variable]:
        """Descendants of a set include the set itself."""
        sources = _ensure_set(sources)
        return _descendants_inclusive(self.directed, sources)

    def topological_sort(self) -> list[Variable]:
        """Get a topological sort from the directed component of the mixed graph."""
        return list(nx.topological_sort(self.directed))

    def get_c_components(self) -> list[frozenset[Variable]]:
        """Get the co-components (i.e., districts) in the undirected portion of the graph."""
        warnings.warn("use NxMixedGraph.districts()", DeprecationWarning, stacklevel=2)
        return list(self.districts())

    def districts(self) -> set[frozenset[Variable]]:
        """Get the districts."""
        return {frozenset(c) for c in nx.connected_components(self.undirected)}

    def get_district(self, node: Variable) -> frozenset[Variable]:
        """Get the district the node is in."""
        for district in self.districts():
            if node in district:
                return district
        raise KeyError(f"{node} not found in graph")

    def is_connected(self) -> bool:
        """Return if there is only a single connected component in the undirected graph."""
        return cast(bool, nx.is_connected(self.undirected))

    def intervene(self, variables: set[Intervention]) -> NxMixedGraph:
        """Intervene on the given variables.

        :param variables: A set of interventions
        :returns: A graph that has been intervened on the given variables, with edges into the intervened nodes removed
        """
        return self.from_edges(
            nodes=[node.intervene(variables) for node in self.nodes()],
            directed=[
                (u.intervene(variables), v.intervene(variables))
                for u, v in self.directed.edges()
                if _node_not_an_intervention(v, variables)
            ],
            undirected=[
                (u.intervene(variables), v.intervene(variables))
                for u, v in self.undirected.edges()
                if _node_not_an_intervention(u, variables)
                and _node_not_an_intervention(v, variables)
            ],
        )

    def get_markov_pillow(self, nodes: Collection[Variable]) -> set[Variable]:
        """For each district, intervene on the domain of each parent not in the district."""
        parents_of_district: set[Variable] = set()
        for node in nodes:
            parents_of_district |= set(self.directed.predecessors(node))
        return parents_of_district - set(nodes)

    def get_markov_blanket(self, nodes: Variable | Iterable[Variable]) -> set[Variable]:
        """Get the Markov blanket for a set of nodes.

        The Markov blanket in a directed graph is the union of the parents, children,
        and parents of children of a given node.

        :param nodes: A node or nodes to get the Markov blanket from
        :return: A set of variables comprising the Markov blanket
        """
        if isinstance(nodes, Variable):
            nodes = {nodes}
        else:
            nodes = set(nodes)
        blanket = set()
        for node in nodes:
            blanket.update(self.directed.predecessors(node))
            for successor in self.directed.successors(node):
                blanket.add(successor)
                blanket.update(self.directed.predecessors(successor))
        return blanket.difference(nodes)

    def disorient(self) -> nx.Graph:
        """Return a graph with all edges converted to a flat undirected graph."""
        rv = nx.Graph()
        rv.add_nodes_from(self.nodes())
        rv.add_edges_from(self.directed.edges())
        rv.add_edges_from(self.undirected.edges())
        return rv

    def pre(
        self,
        nodes: Variable | Iterable[Variable],
        topological_sort_order: Sequence[Variable] | None = None,
    ) -> list[Variable]:
        """Find all nodes prior to the given set of nodes under a topological sort order.

        :param nodes: iterable of nodes.
        :param topological_sort_order: A valid topological sort order. If none given, calculates from the graph.
        :return: list corresponding to the order up until the given nodes.
            This does not include any of the nodes from the query.
        """
        if not topological_sort_order:
            topological_sort_order = list(self.topological_sort())
        node_set = _ensure_set(nodes)
        pre = []
        for node in topological_sort_order:
            if node in node_set:
                break
            pre.append(node)
        return pre


class _LatexStr(str):
    def _repr_latex_(self) -> str:
        return self


def _node_not_an_intervention(node: Variable, interventions: set[Intervention]) -> bool:
    """Confirm that node is not an intervention."""
    if isinstance(node, Intervention | CounterfactualVariable):
        raise TypeError(
            "this shouldn't happen since the graph should not have interventions as nodes"
        )
    return (+node not in interventions) and (-node not in interventions)


def _ancestors_inclusive(graph: nx.DiGraph, sources: set[Variable]) -> set[Variable]:
    ancestors = set(
        itt.chain.from_iterable(nx.algorithms.dag.ancestors(graph, source) for source in sources)
    )
    return sources | ancestors


def _descendants_inclusive(graph: nx.DiGraph, sources: set[Variable]) -> set[Variable]:
    descendants = set(
        itt.chain.from_iterable(nx.algorithms.dag.descendants(graph, source) for source in sources)
    )
    return sources | descendants


def _include_adjacent(
    graph: nx.Graph, vertices: set[Variable]
) -> Collection[tuple[Variable, Variable]]:
    vertices = _ensure_set(vertices)
    return [(u, v) for u, v in graph.edges() if u in vertices and v in vertices]


def _exclude_source(
    graph: nx.Graph, vertices: set[Variable]
) -> Collection[tuple[Variable, Variable]]:
    return [(u, v) for u, v in graph.edges() if u not in vertices]


def _exclude_target(
    graph: nx.Graph, vertices: set[Variable]
) -> Collection[tuple[Variable, Variable]]:
    return [(u, v) for u, v in graph.edges() if v not in vertices]


def _exclude_adjacent(
    graph: nx.Graph, vertices: set[Variable]
) -> Collection[tuple[Variable, Variable]]:
    return [(u, v) for u, v in graph.edges() if u not in vertices and v not in vertices]


def _latent_dag(
    di_edges: Iterable[tuple[Variable, Variable]],
    bi_edges: Iterable[tuple[Variable, Variable]],
    *,
    prefix: str | None = None,
    start: int = 0,
    tag: str | None = None,
) -> nx.DiGraph:
    """Create a labeled DAG where bi-directed edges are assigned as nodes upstream of their two incident nodes.

    :param di_edges: A list of directional edges
    :param bi_edges: A list of bidirectional edges
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

    bi_edges_list = list(bi_edges)

    rv = nx.DiGraph()
    rv.add_nodes_from(itt.chain.from_iterable(bi_edges_list))
    rv.add_edges_from(di_edges)
    nx.set_node_attributes(rv, False, tag)
    for i, (u, v) in enumerate(sorted(bi_edges_list), start=start):
        latent_node = Variable(f"{prefix}{i}")
        rv.add_node(latent_node, **{tag: True})
        rv.add_edge(latent_node, u)
        rv.add_edge(latent_node, v)
    return rv


def set_latent(
    graph: nx.DiGraph,
    latent_nodes: Variable | Iterable[Variable],
    tag: str | None = None,
) -> None:
    """Quickly set the latent variables in a graph."""
    if graph.graph.get(NO_SET_LATENT_FLAG):
        raise RuntimeError(
            "Do not set latent variables on graphs inside a NxMixedGraph using set_latent().\n"
            "This function is strictly only for nx.DiGraphs that have been constructed based on "
            "a NxMixedGraph, but not the NxMixedGraph itself."
        )
    if tag is None:
        tag = DEFAULT_TAG
    if isinstance(latent_nodes, Variable):
        latent_nodes = [latent_nodes]

    latent_nodes = set(latent_nodes)
    for node, data in graph.nodes(data=True):
        data[tag] = node in latent_nodes


def _get_latex(node: str | Variable) -> str:
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


def _ensure_set(vertices: Variable | Iterable[Variable]) -> set[Variable]:
    rv = {vertices} if isinstance(vertices, Variable) else set(vertices)
    if any(isinstance(v, Intervention) for v in rv):
        raise TypeError("can not use interventions here")
    return rv


def is_a_fixable(graph: NxMixedGraph, treatments: Variable | Collection[Variable]) -> bool:
    """Check if the treatments are a-fixable.

    A treatment is said to be a-fixable if it can be fixed by removing a single directed edge from the graph.
    In other words, a treatment is a-fixable if it has exactly one descendant in its district.

    This code was adapted from :mod:`ananke` ananke code at:
    https://gitlab.com/causal/ananke/-/blob/dev/ananke/estimation/counterfactual_mean.py?ref_type=heads#L58-65

    :param graph: A NxMixedGraph
    :param treatments: A list of treatments
    :raises NotImplementedError: a-fixability on multiple treatments is an open research question
    :returns: bool
    """
    if not isinstance(treatments, Variable):
        raise NotImplementedError(
            "a-fixability on multiple treatments is an open research question"
        )
    descendants = graph.descendants_inclusive(treatments)
    descendants_in_district = graph.get_district(treatments).intersection(descendants)
    return 1 == len(descendants_in_district)


def is_p_fixable(graph: NxMixedGraph, treatments: Variable | Collection[Variable]) -> bool:
    """Check if the treatments are p-fixable.

    This code was adapted from :mod:`ananke` ananke code at:
    https://gitlab.com/causal/ananke/-/blob/dev/ananke/estimation/counterfactual_mean.py?ref_type=heads#L85-92

    :param graph: A NxMixedGraph
    :param treatments: A list of treatments
    :raises NotImplementedError: p-fixability on multiple treatments is an open research question
    :returns: bool
    """
    if not isinstance(treatments, Variable):
        raise NotImplementedError(
            "p-fixability on multiple treatments is an open research question"
        )
    children = set(graph.directed.successors(treatments))
    children_in_district = graph.get_district(treatments).intersection(children)
    return 0 == len(children_in_district)


def is_markov_blanket_shielded(graph: NxMixedGraph) -> bool:
    """Check if the ADMG is a Markov blanket shielded.

    Being Markov blanket (Mb) shielded means that two vertices are non-adjacent
    only when they are absent from each others' Markov blankets.

    This code was adapted from :mod:`ananke` ananke code at:
    https://gitlab.com/causal/ananke/-/blob/dev/ananke/graphs/admg.py?ref_type=heads#L381-403

    :param graph: A NxMixedGraph
    :returns: bool
    """
    # Iterate over all pairs of vertices
    for u, v in itt.combinations(graph.nodes(), 2):
        # Check if the pair is not adjacent
        if not (
            any(
                [
                    graph.directed.has_edge(u, v),
                    graph.directed.has_edge(v, u),
                    graph.undirected.has_edge(u, v),
                ]
            )
        ):
            # If one is in the Markov blanket of the other, then it is not mb-shielded
            if _markov_blanket_overlap(graph, u, v):
                return False
    return True


def get_district_and_predecessors(
    graph: NxMixedGraph,
    nodes: Iterable[Variable],
    topological_sort_order: Sequence[Variable] | None = None,
) -> set[Variable]:
    """Get the union of district, predecessors and predecessors of district for a given set of nodes.

    This code was adapted from :mod:`ananke` ananke code at:
    https://gitlab.com/causal/ananke/-/blob/dev/ananke/graphs/admg.py?ref_type=heads#L96-117

    :param graph: A NxMixedGraph
    :param nodes: List of nodes
    :param topological_sort_order: A valid topological sort order

    :return: Set corresponding to union of district, predecessors and predecessors of district of a given set of nodes
    """
    if not topological_sort_order:
        topological_sort_order = list(graph.topological_sort())

    # Get the subgraph corresponding to the nodes and nodes prior to them
    pre = graph.pre(nodes, topological_sort_order)
    sub_graph = graph.subgraph(pre + list(nodes))

    result: set[Variable] = set()
    for node in nodes:
        result.update(sub_graph.get_district(node))
    for node in result.copy():
        result.update(sub_graph.directed.predecessors(node))
    return result - set(nodes)


def _markov_blanket_overlap(graph: NxMixedGraph, u: Variable, v: Variable) -> bool:
    return u in get_district_and_predecessors(graph, [v]) or v in get_district_and_predecessors(
        graph, [u]
    )


def iter_moral_links(graph: NxMixedGraph) -> Iterable[tuple[Variable, Variable]]:
    """Generate links to ensure all co-parents in a graph are linked.

    May generate links that already exist as we assume we are not working on a multi-graph.

    :param graph: Graph to process
    :yields: An collection of edges to add.
    """
    #  note that combinations(x, 2) returns an empty list when len(x) == 1
    yield from chain.from_iterable(
        combinations(graph.directed.predecessors(node), 2) for node in graph.nodes()
    )


def get_nodes_in_directed_paths(
    graph: NxMixedGraph,
    sources: Variable | set[Variable],
    targets: Variable | set[Variable],
) -> set[Variable]:
    """Get all nodes appearing in directed paths from sources to targets.

    :param graph: an NxMixedGraph
    :param sources: source nodes
    :param targets: target nodes
    :return: the nodes on all causal paths from sources to targets
    """
    sources = _ensure_set(sources)
    targets = _ensure_set(targets)
    if nx.is_directed_acyclic_graph(graph.directed):
        return _get_nodes_in_directed_paths_dag(graph.directed, sources, targets)
    else:
        # note, this is a simpler implementation can use :func:`nx.all_simple_paths`,
        # but it is less efficient since it requires potentially calculating the same
        # paths over and over again.
        return _get_nodes_in_directed_paths_cyclic(graph.directed, sources, targets)


def _get_nodes_in_directed_paths_dag(
    graph: nx.DiGraph, sources: set[Variable], targets: set[Variable]
) -> set[Variable]:
    tc: nx.DiGraph = nx.transitive_closure_dag(graph)
    rv = {
        node
        for node in graph.nodes()
        if any(
            tc.has_edge(source, node) and tc.has_edge(node, target)
            for source, target in itt.product(sources, targets)
        )
    }
    for source, target in itt.product(sources, targets):
        if tc.has_edge(source, target):
            rv.add(source)
            rv.add(target)
    return rv


def _get_nodes_in_directed_paths_cyclic(
    graph: nx.DiGraph, sources: set[Variable], targets: set[Variable]
) -> set[Variable]:
    return {
        node
        for source, target in itt.product(sources, targets)
        for causal_path in nx.all_simple_paths(graph, source, target)
        for node in causal_path
    }


def sympy_nested(glyph: str, *variables: Variable) -> sympy.Symbol:
    """Create a sympy nested symbol."""
    import sympy

    inner_latex = ",".join(variable.to_latex() for variable in variables)
    return sympy.Symbol(rf"{glyph}_{{{inner_latex}}}")
