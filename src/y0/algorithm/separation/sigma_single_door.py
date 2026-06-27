r""":math:`\sigma`-separation single-door criterion for cyclic directed graphs.

Implements the graphical admissibility test and adjustment-set finder for the
:math:`\sigma`-separation single-door criterion, a cyclic generalization of Pearl's
single-door / back-door criterion.  The admissibility primitive is
functional-form-agnostic: it depends only on the graph and the
:math:`\sigma`-separation oracle, so it is reusable for linear, additive, or
nonparametric estimators.

The module also provides an efficient *O(V + E)* per-edge implementation based on
the σ-extension trick (Rantanen et al. 2020): once the σ-extension graph is built,
σ-separation reduces to ordinary d-separation, enabling the use of the O-set
optimal adjustment criterion (Henckel, Perković & Maathuis 2022) for minimum-variance
estimation.

Complexity note
---------------
The O-set construction is O(|V| + |E|) per edge when implemented with
reachability (BFS/DFS descendants and ancestors).  An earlier version of this
module used ``nx.all_simple_paths`` to enumerate causal nodes, which is
worst-case exponential in the number of simple paths between cause and effect.
That approach caused severe timeouts on graphs with large strongly connected
components (SCCs), where the path count grows super-exponentially.  The current
implementation uses two graph-traversal sweeps instead:

    cn   = descendants(cause) ∩ ancestors(effect) − {cause}
    forb = descendants(cn) ∪ {cause}
    O-set = pa(cn) − forb

and the proper-backdoor graph is built by removing edges out of *cause* whose
targets are reachable from *cause* and can reach *effect* in the edge-removed
graph — again a reachability query, not a path enumeration.

References
----------
P. Forré, J. M. Mooij (2018). Constraint-based Causal Discovery for
Non-linear Structural Causal Models with Cycles and Latent Confounders.
[forre2018]

K. Rantanen, A. Hyttinen, M. Järvisalo (2020). Learning Optimal Cyclic Causal
Graphs from Interventional Data. *Probabilistic Graphical Models*.

L. Henckel, E. Perković, M. H. Maathuis (2022). Graphical criteria for efficient
total effect estimation via adjustment in causal linear models. *JRSS-B*, 84(2).
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from typing import Literal

import networkx as nx

from y0.algorithm.conditional_independencies import are_d_separated
from y0.dsl import Variable
from y0.graph import NxMixedGraph

from .sigma_extension import sigma_extension
from .sigma_separation import are_sigma_separated

__all__ = [
    "find_sigma_single_door_set",
    "find_sigma_single_door_sets",
    "is_sigma_single_door_admissible",
    "optimal_adjustment_set",
    "proper_backdoor_graph",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _directed_paths_from(
    graph: NxMixedGraph, source: Variable, target: Variable
) -> list[list[Variable]]:
    """Return all simple directed paths from *source* to *target* in *graph*.

    Uses :func:`networkx.all_simple_paths` on the directed subgraph only.
    Returns an empty list if either node is absent from the graph.

    :param graph: A directed mixed graph.
    :param source: Start node.
    :param target: End node.
    :returns: A list of paths, each path being a list of :class:`~y0.dsl.Variable`.
    """
    if source not in graph.directed or target not in graph.directed:
        return []
    return list(nx.all_simple_paths(graph.directed, source, target))


def _remove_directed_edge(graph: NxMixedGraph, u: Variable, v: Variable) -> NxMixedGraph:
    r"""Return a copy of *graph* with the directed edge :math:`u \to v` removed.

    :param graph: The original mixed graph.
    :param u: The source node of the edge to remove.
    :param v: The target node of the edge to remove.
    :returns: A new :class:`NxMixedGraph` without the directed edge
        :math:`u \to v`.
    :raises KeyError: If the directed edge :math:`u \to v` does not exist
        in *graph*.
    """
    reduced = graph.copy()
    reduced.directed.remove_edge(u, v)
    return reduced


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def proper_backdoor_graph(
    graph: NxMixedGraph,
    cause: Variable,
    effect: Variable,
) -> NxMixedGraph:
    r"""Return the proper back-door graph for the edge :math:`cause \to effect`.

    The proper back-door graph :math:`G^{pbd}_{XY}` is constructed from
    :math:`G \setminus \{X \to Y\}` by deleting every directed edge out of *cause*
    that lies on a **proper possible-causal path** from *cause* to *effect* — i.e.
    a directed path in the edge-removed graph that starts with a directed edge out
    of *cause*.

    This isolates the non-causal (back-door) pathways so that a d-separator in
    :math:`G^{pbd}_{XY}` blocks exactly the confounding paths without blocking
    the causal path itself.

    Implementation note: the first-hop identification uses two reachability
    queries (O(|V|+|E|) each) rather than enumerating all simple paths, which
    would be exponential in graphs with large SCCs.

    :param graph: A directed mixed graph (may contain cycles).
    :param cause: The source variable of the direct effect of interest.
    :param effect: The target variable of the direct effect of interest.
    :returns: The proper back-door graph as a new :class:`~y0.graph.NxMixedGraph`.
    :raises KeyError: If the directed edge :math:`cause \to effect` does not exist.

    Reference
    ---------
    B. van der Zander, M. Liśkiewicz (2020). Finding Minimal d-separators in
    Linear Time and Applications. *UAI 2020*.
    """
    # Step 1: remove the target edge.
    reduced = _remove_directed_edge(graph, cause, effect)

    # Step 2: identify first-hops on proper causal paths using reachability.
    # A successor `hop` of `cause` lies on a proper causal path iff:
    #   (a) cause can reach `hop` (trivially true — it's a direct successor), AND
    #   (b) `hop` can reach `effect` in the edge-removed graph.
    # This is O(|V|+|E|) via a single reverse-BFS from `effect`.
    if effect in reduced.directed:
        # Nodes that can reach `effect` in the edge-removed graph (ancestors of effect).
        can_reach_effect: set[Variable] = set(
            nx.ancestors(reduced.directed, effect)
        )
        can_reach_effect.add(effect)
    else:
        can_reach_effect = set()

    first_hops_on_causal_paths: set[Variable] = {
        hop
        for hop in reduced.directed.successors(cause)
        if hop in can_reach_effect
    }

    # Step 3: delete those directed edges out of *cause*.
    pbd = reduced.copy()
    for hop in first_hops_on_causal_paths:
        if pbd.directed.has_edge(cause, hop):
            pbd.directed.remove_edge(cause, hop)

    return pbd


def optimal_adjustment_set(
    graph: NxMixedGraph,
    cause: Variable,
    effect: Variable,
) -> frozenset[Variable] | None:
    r"""Compute the O-set (optimal adjustment set) for :math:`cause \to effect`.

    The O-set is the adjustment set that minimises asymptotic estimation variance
    among all valid adjustment sets (Henckel, Perković & Maathuis 2022).  It is
    defined as:

    .. math::

        O(X \to Y) = pa\bigl(cn(X \to Y)\bigr) \setminus forb(X \to Y)

    where:

    * :math:`cn(X \to Y)` — **causal nodes**: all nodes on directed paths from
      *cause* to *effect* in the original graph (including *cause* and *effect*).
    * :math:`forb(X \to Y)` — **forbidden nodes**: descendants of the causal nodes
      (inclusive) plus *cause* itself.

    The O-set is a valid adjustment set whenever it is non-empty and does not
    contain any descendant of *effect*.  If no valid O-set exists (e.g. the edge
    is not identifiable), ``None`` is returned.

    :param graph: A directed mixed graph (may contain cycles).
    :param cause: The source variable of the direct effect of interest.
    :param effect: The target variable of the direct effect of interest.
    :returns: The O-set as a :class:`frozenset`, or ``None`` if the O-set
        contains a descendant of *effect* (non-identifiable).

    Reference
    ---------
    L. Henckel, E. Perković, M. H. Maathuis (2022). Graphical criteria for
    efficient total effect estimation via adjustment in causal linear models.
    *Journal of the Royal Statistical Society: Series B*, 84(2), 579–599.
    """
    # Causal nodes: nodes on directed paths cause → ... → effect.
    # Equivalent to: descendants(cause) ∩ ancestors(effect) ∪ {cause, effect}
    # when both cause and effect are in the graph.
    # This is O(|V|+|E|) via two BFS sweeps — no path enumeration needed.
    if cause not in graph.directed or effect not in graph.directed:
        return frozenset()

    desc_cause: set[Variable] = set(nx.descendants(graph.directed, cause))
    desc_cause.add(cause)
    anc_effect: set[Variable] = set(nx.ancestors(graph.directed, effect))
    anc_effect.add(effect)

    causal_nodes: set[Variable] = desc_cause & anc_effect

    # If there are no directed paths at all (only cause itself is in the
    # intersection when cause == effect, or the intersection is empty), return
    # an empty O-set.
    if not causal_nodes or causal_nodes == {cause} and cause != effect:
        return frozenset()
    # If cause has no path to effect (effect not reachable from cause), empty.
    if effect not in desc_cause:
        return frozenset()

    # Forbidden nodes: descendants of causal nodes (inclusive) ∪ {cause}.
    forbidden: set[Variable] = graph.descendants_inclusive(causal_nodes)
    forbidden.add(cause)

    # O-set: parents of causal nodes, minus forbidden.
    o_set: set[Variable] = set()
    for cn in causal_nodes:
        o_set.update(graph.directed.predecessors(cn))
    o_set -= forbidden

    # Validity check: O-set must not contain any descendant of effect.
    effect_descendants = graph.descendants_inclusive(effect)
    if o_set & effect_descendants:
        return None

    return frozenset(o_set)


def find_sigma_single_door_set(
    graph: NxMixedGraph,
    cause: Variable,
    effect: Variable,
    *,
    method: Literal["optimal"] = "optimal",
    precomputed_extension: NxMixedGraph | None = None,
) -> frozenset[Variable] | None:
    r"""Find a single admissible adjustment set for :math:`cause \to effect`.

    This is the efficient *O(V + E)* per-edge implementation based on the
    σ-extension trick (Rantanen et al. 2020).  The algorithm:

    1. Build (or reuse) the σ-extension :math:`G^\sigma` — a real
       :class:`~y0.graph.NxMixedGraph` with bidirected edges added for all
       intra-SCC directed edges.
    2. Remove the edge :math:`cause \to effect` from :math:`G^\sigma`.
    3. Build the proper back-door graph :math:`G^{pbd}_{XY}` from the result.
    4. Compute the O-set :math:`O = pa(cn) \setminus forb` in :math:`G^{pbd}_{XY}`.
    5. Verify that *cause* and *effect* are d-separated by *O* in
       :math:`G^{pbd}_{XY}` (the O-set is always valid when it exists, but we
       check as a safety guard).

    :param graph: A directed mixed graph (may contain cycles).
    :param cause: The source variable of the direct effect of interest.
    :param effect: The target variable of the direct effect of interest.
    :param method: Adjustment-set strategy.  Currently only ``"optimal"``
        (O-set, minimum asymptotic variance) is supported.
    :param precomputed_extension: A pre-built σ-extension of *graph*.  Pass this
        when calling :func:`find_sigma_single_door_set` in a loop over many edges
        of the same graph to avoid recomputing the extension each time.
    :returns: An admissible :class:`frozenset` of conditioning variables, or
        ``None`` if the edge is absent or no admissible set was found.

    .. note::
        For the completeness guarantee, the returned set is always verified via
        d-separation in :math:`G^{pbd}_{XY}`.  If the O-set fails the check
        (which can happen in graphs with latent confounders), ``None`` is returned
        rather than an invalid set.
    """
    if not graph.directed.has_edge(cause, effect):
        return None

    # Build or reuse the σ-extension.
    g_sigma: NxMixedGraph = (
        precomputed_extension if precomputed_extension is not None else sigma_extension(graph)
    )

    # Build the proper back-door graph in G^σ.
    # proper_backdoor_graph handles the edge removal internally.
    g_pbd = proper_backdoor_graph(g_sigma, cause, effect)

    # Compute the O-set using the *original* G^σ (which still has the cause→effect
    # edge), so that causal nodes are identified correctly.  The O-set is then
    # verified in G^pbd (which has all causal edges from cause removed).
    z = optimal_adjustment_set(g_sigma, cause, effect)
    if z is None:
        # O-set computation returned None (e.g. O-set contains a descendant of effect).
        # Fall back to the empty set and check if it works.
        z = frozenset()

    # Safety verification: cause ⊥_d effect | Z in G^pbd.
    if not bool(are_d_separated(g_pbd, cause, effect, conditions=z)):
        return None

    return z


def is_sigma_single_door_admissible(
    graph: NxMixedGraph,
    cause: Variable,
    effect: Variable,
    conditions: Iterable[Variable],
) -> bool:
    r"""Test whether *conditions* (Z) satisfies the :math:`\sigma`-separation single-door criterion.

    A set Z is admissible for identifying the direct effect
    :math:`cause \to effect` when:

    1. Z contains no descendant of *effect* (cycle-aware, using
       :meth:`NxMixedGraph.descendants_inclusive`), and
    2. *cause* and *effect* are :math:`\sigma`-separated by Z in the graph
       with the directed edge :math:`cause \to effect` removed, i.e.

       .. math::

           cause \perp_{\sigma} effect \mid Z
           \quad \text{in } G \setminus \{cause \to effect\}.

    The criterion is functional-form-agnostic: it makes no assumption about
    the structural equations and is valid for linear, additive-noise, and
    general nonparametric SCMs that are Markov to the graph.

    :param graph: A directed mixed graph (may contain cycles).
    :param cause: The source variable of the direct effect of interest.
    :param effect: The target variable of the direct effect of interest.
    :param conditions: Candidate adjustment set Z.
    :returns: ``True`` if Z satisfies the :math:`\sigma`-separation
        single-door criterion.

    .. note::
        If the directed edge :math:`cause \to effect` is not present in
        *graph*, this function returns ``False`` (there is no direct effect
        to identify).
    """
    conditions_set = set(conditions)

    # Guard: edge must exist
    if not graph.directed.has_edge(cause, effect):
        return False

    # Condition (i): Z must not contain any descendant of effect
    effect_descendants = graph.descendants_inclusive(effect)
    if conditions_set & effect_descendants:
        return False

    # Condition (ii): sigma-separation of cause and effect given Z in G \ {cause -> effect}
    reduced = _remove_directed_edge(graph, cause, effect)
    return are_sigma_separated(reduced, cause, effect, conditions=conditions_set)


def find_sigma_single_door_sets(
    graph: NxMixedGraph,
    cause: Variable,
    effect: Variable,
    *,
    max_conditions: int | None = None,
) -> Iterable[frozenset[Variable]]:
    r"""Enumerate admissible adjustment sets for the :math:`\sigma`-separation single-door criterion.

    Yields all subsets Z of the candidate pool (non-descendants of *effect*,
    excluding *cause* and *effect* themselves) that satisfy
    :func:`is_sigma_single_door_admissible`.  Sets are enumerated in order of
    increasing size (empty set first).

    :param graph: A directed mixed graph (may contain cycles).
    :param cause: The source variable of the direct effect of interest.
    :param effect: The target variable of the direct effect of interest.
    :param max_conditions: Maximum size of Z to consider.  ``None`` (default)
        means unbounded -- all subsets of the candidate pool are tried.
        **Warning:** enumeration is :math:`O(2^k)` in the size of the
        candidate pool; set a cap for large graphs.
    :yields: Each admissible :class:`frozenset` of conditioning variables.

    .. note::
        Soundness over completeness: the underlying :math:`\sigma`-separation
        oracle uses a path-augmentation heuristic that may not be complete.
        A yielded Z is always a valid adjustment set; some valid sets may be
        missed.
    """
    if not graph.directed.has_edge(cause, effect):
        return

    # Candidate pool: all nodes except cause, effect, and descendants of effect
    effect_descendants = graph.descendants_inclusive(effect)
    candidates = sorted(
        graph.nodes() - effect_descendants - {cause, effect},
        key=lambda v: v.name,
    )

    # Remove the direct edge once; reuse the reduced graph for all Z tests
    reduced = _remove_directed_edge(graph, cause, effect)

    upper = len(candidates) if max_conditions is None else min(max_conditions, len(candidates))
    for size in range(upper + 1):
        for combo in combinations(candidates, size):
            z = frozenset(combo)
            if are_sigma_separated(reduced, cause, effect, conditions=z):
                yield z
