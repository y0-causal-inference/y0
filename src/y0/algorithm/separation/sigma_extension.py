r"""σ-extension graph construction for efficient cyclic adjustment-set computation.

The **σ-extension** (Rantanen et al. 2020) transforms a cyclic directed mixed graph
*G* into an extended graph *G*\ :sup:`σ` by adding a bidirected edge for every
directed edge whose endpoints belong to the same strongly-connected component (SCC).
The key theorem is:

    *X* and *Y* are σ-separated given *Z* in *G*
    **if and only if**
    *X* and *Y* are d-separated given *Z* in *G*\ :sup:`σ`.

This reduction lets us replace the expensive path-augmentation σ-separation oracle
(:func:`~y0.algorithm.separation.sigma_separation.are_sigma_separated`) with the
standard, linear-time d-separation algorithm
(:func:`~y0.algorithm.conditional_independencies.are_d_separated`) for all
adjustment-set computations.

**Complexity:** Building *G*\ :sup:`σ` is *O(V + E)* (one SCC pass + one edge scan).
The per-edge adjustment-set computation in
:func:`~y0.algorithm.separation.sigma_single_door.find_sigma_single_door_set` then
runs in *O(V + E)* per edge, giving *O(E · (V + E))* total — tractable for large
regulatory networks.

Reference
---------
K. Rantanen, A. Hyttinen, M. Järvisalo (2020). Learning Optimal Cyclic Causal
Graphs from Interventional Data. *Probabilistic Graphical Models*.
"""

from __future__ import annotations

import networkx as nx

from y0.dsl import Variable
from y0.graph import NxMixedGraph

__all__ = [
    "sigma_extension",
]


def sigma_extension(graph: NxMixedGraph) -> NxMixedGraph:
    r"""Build the σ-extension *G*\ :sup:`σ` of a directed mixed graph.

    For every directed edge :math:`u \to v` in *graph* where *u* and *v* belong
    to the **same strongly-connected component** (SCC), a bidirected edge
    :math:`u \leftrightarrow v` is added to the copy.  All other nodes and edges
    are preserved unchanged.

    The resulting graph satisfies the equivalence:

    .. math::

        X \perp_\sigma Y \mid Z \text{ in } G
        \iff
        X \perp_d Y \mid Z \text{ in } G^\sigma

    so standard d-separation algorithms can be used in place of the more expensive
    σ-separation oracle.

    :param graph: A directed mixed graph (may contain directed cycles).
    :returns: A new :class:`~y0.graph.NxMixedGraph` equal to *graph* plus
        bidirected edges for all intra-SCC directed edges.

    .. note::
        On a DAG (no non-trivial SCCs), the σ-extension is identical to the
        original graph — no bidirected edges are added.

    Examples
    --------
    A simple 2-cycle ``X → Y → X`` gains one bidirected edge ``X ↔ Y``:

    >>> from y0.dsl import Variable
    >>> from y0.graph import NxMixedGraph
    >>> X, Y = Variable("X"), Variable("Y")
    >>> g = NxMixedGraph.from_edges(directed=[(X, Y), (Y, X)])
    >>> gs = sigma_extension(g)
    >>> gs.undirected.has_edge(X, Y)
    True
    """
    # Compute SCCs on the directed subgraph only (bidirected edges are ignored,
    # matching the definition in Bongers et al. 2021 / Forré & Mooij 2018).
    sccs: list[set[Variable]] = list(
        nx.strongly_connected_components(graph.directed)
    )
    # Build a node → SCC-id mapping for O(1) same-SCC lookup.
    node_to_scc: dict[Variable, int] = {}
    for scc_id, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = scc_id

    extended = graph.copy()
    for u, v in graph.directed.edges():
        # Add a bidirected edge iff u and v are in the same SCC (i.e. there is
        # a directed path back from v to u as well).
        if node_to_scc[u] == node_to_scc[v]:
            # NxMixedGraph.add_undirected_edge is idempotent for existing edges.
            extended.add_undirected_edge(u, v)

    return extended
