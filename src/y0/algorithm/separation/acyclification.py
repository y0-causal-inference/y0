r"""Acyclification of directed mixed graphs (Forré & Mooij 2017).

The **acyclification** of a directed mixed graph *G* is an acyclic directed
mixed graph *G*\ :sup:`ac` that preserves the σ-separation structure of *G*:

    *X* and *Y* are σ-separated given *Z* in *G*
    **if and only if**
    *X* and *Y* are m-separated given *Z* in *G*\ :sup:`ac`.

Construction (Forré & Mooij 2017, Definition 3.4):

1. Compute the strongly-connected components (SCCs) of the directed subgraph.
2. For each SCC *S* with |S| > 1 (non-trivial):
   a. Remove all directed edges *within* the SCC.
   b. Add a bidirected edge *u* ↔ *v* for every pair *u*, *v* ∈ *S* (u ≠ v).
3. For each directed edge *p* → *q* where *p* is outside SCC *S* and *q* ∈ *S*,
   add *p* → *w* for every *w* ∈ *S* (the parent is "broadcast" to all SCC members).
4. Keep all inter-SCC directed edges (edges between different SCCs) and all
   existing bidirected edges.

The result is acyclic (the condensation DAG structure is preserved) and has the
same σ-separation as the original graph.

Reference
---------
P. Forré, J. M. Mooij (2017). Markov Properties for Acyclic Directed Mixed
Graphs. *Proceedings of the 33rd Conference on Uncertainty in Artificial
Intelligence (UAI)*.
"""

from __future__ import annotations

import networkx as nx

from y0.dsl import Variable
from y0.graph import NxMixedGraph

__all__ = ["acyclify"]


def acyclify(graph: NxMixedGraph) -> NxMixedGraph:
    r"""Return the acyclification *G*\ :sup:`ac` of *graph* (Forré & Mooij 2017).

    The acyclification is an acyclic directed mixed graph with the same
    σ-separation structure as *graph*:

    .. math::

        X \perp_\sigma Y \mid Z \text{ in } G
        \iff
        X \perp_m Y \mid Z \text{ in } G^{\mathrm{ac}}

    :param graph: A directed mixed graph (may contain directed cycles).
    :returns: A new :class:`~y0.graph.NxMixedGraph` that is acyclic and
        σ-separation-equivalent to *graph*.

    Examples
    --------
    A 2-cycle ``X → Y → X`` acyclifies to ``X ↔ Y`` (no directed edges):

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
    sccs: list[set[Variable]] = list(nx.strongly_connected_components(directed))

    # Map each node to its SCC index.
    node_to_scc: dict[Variable, int] = {}
    for scc_id, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = scc_id

    # Collect non-trivial SCCs (size > 1).
    nontrivial: dict[int, set[Variable]] = {
        scc_id: scc for scc_id, scc in enumerate(sccs) if len(scc) > 1
    }

    # Start with all nodes, no edges.
    result = NxMixedGraph()
    for node in graph.nodes():
        result.add_node(node)

    # --- Directed edges ---
    # Keep inter-SCC directed edges; drop intra-SCC directed edges.
    # For each inter-SCC edge p -> q where q is in a non-trivial SCC S,
    # broadcast p -> w for every w in S.
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

    # --- Bidirected edges ---
    # Preserve all existing bidirected edges.
    for u, v in graph.undirected.edges():
        result.add_undirected_edge(u, v)

    # Add bidirected edges for all pairs within each non-trivial SCC.
    for scc in nontrivial.values():
        scc_list = sorted(scc, key=lambda n: n.name)
        for i, u in enumerate(scc_list):
            for v in scc_list[i + 1 :]:
                result.add_undirected_edge(u, v)

    return result
