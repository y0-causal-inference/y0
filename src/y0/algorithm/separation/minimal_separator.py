r"""m-separation and minimal m-separator for acyclic directed mixed graphs (ADMGs).

m-separation (Drton & Richardson 2003) is the standard separation criterion for
acyclic directed mixed graphs (ADMGs) -- DAGs with bidirected edges representing
latent common causes.  It generalises d-separation to graphs with bidirected
edges by reducing to d-separation on the canonical latent-variable DAG (LVM)
obtained by replacing each bidirected edge ``u <-> v`` with a fresh latent node
``L`` and directed edges ``L -> u``, ``L -> v``.

This module provides:

- :func:`are_m_separated` -- m-separation oracle for an :class:`~y0.graph.NxMixedGraph`.
  Operates directly on the ADMG; the latent-projection DAG is built internally.
- :func:`find_minimal_m_separator` -- finds a minimal m-separator between two
  nodes within optional ``included`` / ``restricted`` constraints.

Both functions work on **acyclic** ADMGs.  For cyclic graphs, use
:func:`~y0.algorithm.separation.sigma_extension.sigma_extension` to first
convert to an acyclic ADMG and then call these functions on the result.

References
----------
M. Drton, T. S. Richardson (2003). Iterative Conditional Fitting for Gaussian
Ancestral Graph Models. *Technical Report 437*, University of Washington.

J. Pearl (2009). *Causality: Models, Reasoning and Inference* (2nd ed.).
Cambridge University Press.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

from y0.algorithm.conditional_independencies import are_d_separated
from y0.dsl import Variable
from y0.graph import NxMixedGraph
from y0.struct import DSeparationJudgement

__all__ = [
    "are_m_separated",
    "find_minimal_m_separator",
]

# The default tag key used by NxMixedGraph.to_latent_variable_dag() to mark
# latent nodes.  Kept in sync with y0.graph.DEFAULT_TAG.
_DEFAULT_LATENT_TAG = "hidden"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_lvm(graph: NxMixedGraph) -> tuple[NxMixedGraph, frozenset[Variable]]:
    """Build the canonical latent-variable DAG for *graph*.

    Each bidirected edge ``u <-> v`` becomes a fresh latent node ``L`` with
    directed edges ``L -> u`` and ``L -> v``.  The returned :class:`NxMixedGraph`
    is a pure DAG (no bidirected edges); the second element is the frozenset of
    latent node names.

    :param graph: An acyclic directed mixed graph.
    :returns: ``(lvm, latent_nodes)`` where *lvm* is the latent-variable DAG
        as an :class:`~y0.graph.NxMixedGraph` and *latent_nodes* is the set of
        latent :class:`~y0.dsl.Variable` objects.
    """
    lvdag: nx.DiGraph = graph.to_latent_variable_dag()
    tag = _DEFAULT_LATENT_TAG
    latent_nodes: frozenset[Variable] = frozenset(
        node for node in lvdag.nodes() if lvdag.nodes[node].get(tag, False)
    )
    # Wrap as NxMixedGraph (pure DAG -- only directed edges)
    lvm = NxMixedGraph.from_edges(directed=list(lvdag.edges()))
    return lvm, latent_nodes


def _ancestors_of(graph: NxMixedGraph, nodes: Iterable[Variable]) -> frozenset[Variable]:
    """Return the set of ancestors (inclusive) of *nodes* in *graph*.

    Uses the directed part of the graph only.

    :param graph: The graph to query.
    :param nodes: Seed nodes.
    :returns: All ancestors of *nodes* including *nodes* themselves.
    """
    result: set[Variable] = set()
    queue = list(nodes)
    while queue:
        node = queue.pop()
        if node in result:
            continue
        result.add(node)
        queue.extend(graph.directed.predecessors(node))
    return frozenset(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def are_m_separated(
    graph: NxMixedGraph,
    x: Variable,
    y: Variable,
    *,
    conditions: Collection[Variable] | None = None,
) -> DSeparationJudgement:
    """Test whether *x* and *y* are m-separated given *conditions* in *graph*.

    m-separation is the standard separation criterion for acyclic directed
    mixed graphs (ADMGs).  It reduces to d-separation on the canonical
    latent-variable DAG (LVM) obtained by replacing each bidirected edge
    ``u <-> v`` with a fresh latent node and two directed edges.

    The latent-projection DAG is built internally; callers never see it.
    Conditioning nodes must be observed (non-latent) nodes of *graph* -- the
    function raises :class:`ValueError` if a conditioning node is not in the
    graph.

    The return value is a :class:`~y0.struct.DSeparationJudgement` that carries
    the ``left``, ``right``, and ``conditions`` in canonical form alongside the
    ``separated`` flag.  It implements ``__bool__``, so existing code that
    treats the result as a plain boolean (``if are_m_separated(...)``,
    ``assert are_m_separated(...)``, etc.) continues to work unchanged.

    :param graph: An acyclic directed mixed graph (:class:`~y0.graph.NxMixedGraph`).
    :param x: The first variable.
    :param y: The second variable.
    :param conditions: The conditioning set (observed nodes only).  Defaults
        to the empty set.
    :returns: A :class:`~y0.struct.DSeparationJudgement` that is truthy when
        *x* and *y* are m-separated given *conditions*, falsy otherwise.
    :raises ValueError: If *x*, *y*, or any conditioning node is not a node
        of *graph*.
    """
    if conditions is None:
        conditions = frozenset()
    else:
        conditions = frozenset(conditions)

    observed_nodes: frozenset[Variable] = frozenset(graph.nodes())

    # Validate inputs
    for node in (x, y):
        if node not in observed_nodes:
            raise ValueError(f"Node {node!r} is not in the graph.")
    for node in conditions:
        if node not in observed_nodes:
            raise ValueError(
                f"Conditioning node {node!r} is not in the graph. "
                "Only observed (non-latent) nodes may be conditioned on."
            )

    # Trivial case: x == y -- by convention, a node is not separated from itself.
    if x == y:
        return DSeparationJudgement.create(left=x, right=y, conditions=conditions, separated=False)

    # Build the latent-variable DAG
    lvm, _latent_nodes = _build_lvm(graph)

    # Nodes absent from the LVM (isolated in the ADMG -- no edges at all)
    # have no connecting paths, so they are trivially separated from everything.
    lvm_nodes = frozenset(lvm.nodes())
    if x not in lvm_nodes or y not in lvm_nodes:
        return DSeparationJudgement.create(left=x, right=y, conditions=conditions, separated=True)

    # Filter conditioning set to nodes present in the LVM
    # (isolated observed nodes are inert -- conditioning on them changes nothing)
    lvm_conditions = frozenset(c for c in conditions if c in lvm_nodes)

    separated = bool(are_d_separated(lvm, x, y, conditions=lvm_conditions))
    return DSeparationJudgement.create(left=x, right=y, conditions=conditions, separated=separated)


def _resolve_candidate_pool(
    graph: NxMixedGraph,
    x: Variable,
    y: Variable,
    included: frozenset[Variable],
    restricted: Collection[Variable] | None,
    observed_nodes: frozenset[Variable],
    latent_nodes: frozenset[Variable],
) -> frozenset[Variable] | None:
    """Determine the candidate pool for the separator search.

    Returns the candidate pool as a frozenset, or ``None`` if the constraints
    are infeasible (a restricted node is not in the graph).

    :param graph: The ADMG being queried.
    :param x: Source variable.
    :param y: Target variable.
    :param included: Nodes that must be in the separator.
    :param restricted: Optional explicit pool; if ``None`` the default
        ancestor-based pool is used.
    :param observed_nodes: All observed nodes of *graph*.
    :param latent_nodes: All latent nodes introduced by the LVM construction.
    :returns: The candidate pool, or ``None`` on infeasible constraints.
    :raises ValueError: If a restricted node is not in the graph.
    """
    if restricted is not None:
        restricted_set = frozenset(restricted)
        for node in restricted_set:
            if node not in observed_nodes:
                raise ValueError(f"Restricted node {node!r} is not in the graph.")
        return restricted_set - {x, y} - latent_nodes
    # Default: ancestors of {x, y} \cup included, minus {x, y}
    seed = set(included) | {x, y}
    return _ancestors_of(graph, seed) - {x, y} - latent_nodes


def _greedy_shrink(
    z0: frozenset[Variable],
    included_valid: frozenset[Variable],
    oracle: object,
) -> frozenset[Variable]:
    r"""Greedily remove nodes from *z0* while m-separation is preserved.

    Iterates over nodes in ``z0 \\ included_valid`` in deterministic order and
    removes each one if the oracle still confirms m-separation without it.

    :param z0: The initial (full) candidate separator.
    :param included_valid: Nodes that must remain in the separator.
    :param oracle: A callable ``(frozenset[Variable]) -> bool`` that returns
        ``True`` when the given set m-separates *x* and *y*.
    :returns: A minimal subset of *z0* that still m-separates.
    """
    current = set(z0)
    shrinkable = sorted(z0 - included_valid, key=lambda v: v.name)
    for node in shrinkable:
        candidate = frozenset(current - {node})
        if oracle(candidate):  # type: ignore[operator]
            current.discard(node)
    return frozenset(current)


def _validate_separator_inputs(
    x: Variable,
    y: Variable,
    included: frozenset[Variable],
    observed_nodes: frozenset[Variable],
) -> None:
    """Validate inputs for :func:`find_minimal_m_separator`.

    :param x: Source variable.
    :param y: Target variable.
    :param included: Nodes that must be in the separator.
    :param observed_nodes: All observed nodes of the graph.
    :raises ValueError: If *x*, *y*, or any included node is not in the graph.
    """
    for node in (x, y):
        if node not in observed_nodes:
            raise ValueError(f"Node {node!r} is not in the graph.")
    for node in included:
        if node not in observed_nodes:
            raise ValueError(f"Included node {node!r} is not in the graph.")


def find_minimal_m_separator(
    graph: NxMixedGraph,
    x: Variable,
    y: Variable,
    *,
    included: Collection[Variable] | None = None,
    restricted: Collection[Variable] | None = None,
) -> frozenset[Variable] | None:
    r"""Find a minimal m-separator between *x* and *y* in *graph*.

    Returns a **minimal** set ``Z`` of observed nodes such that:

    1. ``Z`` m-separates *x* and *y* in *graph*.
    2. Every node in *included* is in ``Z``.
    3. Every node in ``Z`` is in *restricted* (if given).
    4. No proper subset of ``Z`` that still contains *included* also
       m-separates *x* and *y*.

    If no valid separator exists within the constraints, returns ``None``.

    Algorithm
    ---------
    1. **Candidate pool**: start from *restricted* (default: ancestors of
       ``{x, y} \cup included`` in the ADMG, minus ``{x, y}`` themselves).
       Exclude *x*, *y*, and any latent nodes (latents are never conditioned on).
    2. **Initial separator**: ``Z0 = candidate_pool`` (the full restricted set).
       Verify it m-separates; if not, return ``None``.
    3. **Greedy shrink**: iterate over nodes in ``Z0 \\ included``.  Remove a
       node if m-separation still holds without it.  Repeat until no more
       removals are possible.

    The result is minimal by inclusion within the constraints.

    :param graph: An acyclic directed mixed graph.
    :param x: The source variable.
    :param y: The target variable.
    :param included: Nodes that *must* be in the separator.  Defaults to the
        empty set.
    :param restricted: The pool of nodes from which the separator may be drawn.
        Defaults to the ancestors of ``{x, y} \cup included`` minus ``{x, y}``.
    :returns: A minimal m-separator as a :class:`frozenset`, or ``None`` if no
        valid separator exists within the constraints.
    :raises ValueError: If *x*, *y*, or any node in *included* / *restricted*
        is not an observed node of *graph*.
    """
    if included is None:
        included = frozenset()
    else:
        included = frozenset(included)

    observed_nodes: frozenset[Variable] = frozenset(graph.nodes())
    _validate_separator_inputs(x, y, included, observed_nodes)

    # Build LVM once -- we'll reuse it for all oracle calls
    lvm, latent_nodes = _build_lvm(graph)
    lvm_nodes = frozenset(lvm.nodes())

    def _oracle(z: frozenset[Variable]) -> bool:
        """m-separation oracle using the pre-built LVM."""
        if x not in lvm_nodes or y not in lvm_nodes:
            return True
        lvm_z = frozenset(c for c in z if c in lvm_nodes)
        return bool(are_d_separated(lvm, x, y, conditions=lvm_z))

    # Determine the candidate pool
    candidate_pool = _resolve_candidate_pool(
        graph, x, y, included, restricted, observed_nodes, latent_nodes
    )
    if candidate_pool is None:
        return None

    # included must be a subset of the candidate pool
    if not included.issubset(candidate_pool | {x, y}):
        # Some included node is outside the restricted pool -- no valid separator
        return None

    # Ensure included nodes are in the pool (exclude x and y if accidentally passed)
    included_valid = included - {x, y}

    # Step 1: start with the full candidate pool as Z0
    z0 = frozenset(candidate_pool)

    # Step 2: verify Z0 m-separates
    if not _oracle(z0):
        return None

    # Step 3: greedy shrink
    return _greedy_shrink(z0, included_valid, _oracle)
