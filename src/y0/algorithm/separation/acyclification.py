r"""Acyclification of directed mixed graphs (Forre & Mooij 2017).

The **acyclification** of a directed mixed graph *G* is an acyclic directed
mixed graph *G*\ :sup:`ac` that preserves the \sigma-separation structure of *G*:

    *X* and *Y* are \sigma-separated given *Z* in *G*
    **if and only if**
    *X* and *Y* are m-separated given *Z* in *G*\ :sup:`ac`.

Construction (Forre & Mooij 2017, Definition 3.4):

1. Compute the strongly-connected components (SCCs) of the directed subgraph.
2. For each SCC *S* with |S| > 1 (non-trivial):
   a. Remove all directed edges *within* the SCC.
   b. Add a bidirected edge *u* <-> *v* for every pair *u*, *v* in *S* (u != v).
3. For each directed edge *p* -> *q* where *p* is outside SCC *S* and *q* in *S*,
   add *p* -> *w* for every *w* in *S* (the parent is "broadcast" to all SCC members).
4. Keep all inter-SCC directed edges (edges between different SCCs) and all
   existing bidirected edges.

The result is acyclic (the condensation DAG structure is preserved) and has the
same \sigma-separation as the original graph.

Reference
---------
P. Forre, J. M. Mooij (2017). Markov Properties for Acyclic Directed Mixed
Graphs. *Proceedings of the 33rd Conference on Uncertainty in Artificial
Intelligence (UAI)*.
"""

from __future__ import annotations

import networkx as nx

from y0.dsl import Variable
from y0.graph import NxMixedGraph

__all__ = ["acyclify"]


def _build_scc_index(
    directed: nx.DiGraph,
) -> tuple[list[set[Variable]], dict[Variable, int], dict[int, set[Variable]]]:
    """Compute SCCs and return (sccs, node_to_scc, nontrivial).

    :param directed: The directed subgraph of the mixed graph.
    :returns: A 3-tuple of (all SCCs as a list, mapping from node to SCC id,
        mapping from SCC id to SCC for non-trivial SCCs with size > 1).
    """
    sccs: list[set[Variable]] = list(nx.strongly_connected_components(directed))
    node_to_scc: dict[Variable, int] = {}
    for scc_id, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = scc_id
    nontrivial: dict[int, set[Variable]] = {
        scc_id: scc for scc_id, scc in enumerate(sccs) if len(scc) > 1
    }
    return sccs, node_to_scc, nontrivial


def _add_directed_edges(
    result: NxMixedGraph,
    directed: nx.DiGraph,
    node_to_scc: dict[Variable, int],
    nontrivial: dict[int, set[Variable]],
) -> None:
    """Add directed edges to *result* according to the acyclification rules.

    Inter-SCC edges are kept; intra-SCC edges are dropped.  When the target
    node belongs to a non-trivial SCC, the edge is broadcast to all SCC members.

    :param result: The graph being built (modified in place).
    :param directed: The directed subgraph of the original graph.
    :param node_to_scc: Mapping from each node to its SCC id.
    :param nontrivial: Mapping from SCC id to SCC for non-trivial SCCs.
    """
    for u, v in directed.edges():
        scc_u = node_to_scc[u]
        scc_v = node_to_scc[v]
        if scc_u == scc_v:
            # Intra-SCC: drop.
            continue
        # Inter-SCC: u -> v.  If v's SCC is non-trivial, broadcast to all members.
        if scc_v in nontrivial:
            for w in nontrivial[scc_v]:
                result.add_directed_edge(u, w)
        else:
            result.add_directed_edge(u, v)


def _add_bidirected_edges(
    result: NxMixedGraph,
    graph: NxMixedGraph,
    nontrivial: dict[int, set[Variable]],
) -> None:
    """Add bidirected edges to *result* according to the acyclification rules.

    Preserves all existing bidirected edges and adds new ones for every pair
    within each non-trivial SCC.

    :param result: The graph being built (modified in place).
    :param graph: The original mixed graph.
    :param nontrivial: Mapping from SCC id to SCC for non-trivial SCCs.
    """
    for u, v in graph.undirected.edges():
        result.add_undirected_edge(u, v)
    for scc in nontrivial.values():
        scc_list = sorted(scc, key=lambda n: n.name)
        for i, u in enumerate(scc_list):
            for v in scc_list[i + 1 :]:
                result.add_undirected_edge(u, v)


def acyclify(graph: NxMixedGraph) -> NxMixedGraph:
    r"""Return the acyclification *G*\ :sup:`ac` of *graph* (Forre & Mooij 2017).

    The acyclification is an acyclic directed mixed graph with the same
    \sigma-separation structure as *graph*:

    .. math::

        X \perp_\sigma Y \mid Z \text{ in } G
        \iff
        X \perp_m Y \mid Z \text{ in } G^{\mathrm{ac}}

    :param graph: A directed mixed graph (may contain directed cycles).
    :returns: A new :class:`~y0.graph.NxMixedGraph` that is acyclic and
        \sigma-separation-equivalent to *graph*.

    Examples
    --------
    A 2-cycle ``X -> Y -> X`` acyclifies to ``X <-> Y`` (no directed edges):

    >>> from y0.dsl import Variable
    >>> from y0.graph import NxMixedGraph
    >>> X, Y = Variable("X"), Variable("Y")
    >>> g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
    >>> gac = acyclify(g)
    >>> gac.directed.has_edge(X, Y)
    False
    >>> gac.undirected.has_edge(X, Y)
    True

    A DAG is unchanged:

    >>> Z = Variable("Z")
    >>> g2 = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y)])
    >>> gac2 = acyclify(g2)
    >>> list(gac2.directed.edges()) == list(g2.directed.edges())
    True
    """
    directed = graph.directed
    _sccs, node_to_scc, nontrivial = _build_scc_index(directed)

    result = NxMixedGraph()
    for node in graph.nodes():
        result.add_node(node)

    _add_directed_edges(result, directed, node_to_scc, nontrivial)
    _add_bidirected_edges(result, graph, nontrivial)

    return result
