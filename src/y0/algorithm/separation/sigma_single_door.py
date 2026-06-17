"""σ-separation single-door criterion for cyclic directed graphs.

Implements the graphical admissibility test and adjustment-set finder for the
σ-separation single-door criterion, a cyclic generalization of Pearl's
single-door / back-door criterion.  The admissibility primitive is
functional-form-agnostic: it depends only on the graph and the σ-separation
oracle, so it is reusable for linear, additive, or nonparametric estimators.

Reference
---------
P. Forré, J. M. Mooij (2018). Constraint-based Causal Discovery for
Non-linear Structural Causal Models with Cycles and Latent Confounders.
[forre2018]
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

from y0.dsl import Variable
from y0.graph import NxMixedGraph

from .sigma_separation import are_sigma_separated

__all__ = [
    "find_sigma_single_door_sets",
    "is_sigma_single_door_admissible",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _remove_directed_edge(graph: NxMixedGraph, u: Variable, v: Variable) -> NxMixedGraph:
    """Return a copy of *graph* with the directed edge u → v removed.

    :param graph: The original mixed graph.
    :param u: The source node of the edge to remove.
    :param v: The target node of the edge to remove.
    :returns: A new :class:`NxMixedGraph` without the directed edge u → v.
    :raises KeyError: If the directed edge u → v does not exist in *graph*.
    """
    reduced = graph.copy()
    reduced.directed.remove_edge(u, v)
    return reduced


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_sigma_single_door_admissible(
    graph: NxMixedGraph,
    cause: Variable,
    effect: Variable,
    conditions: Iterable[Variable],
) -> bool:
    """Test whether *conditions* (Z) satisfies the σ-separation single-door criterion.

    A set Z is admissible for identifying the direct effect cause → effect when:

    1. Z contains no descendant of *effect* (cycle-aware, using
       :meth:`NxMixedGraph.descendants_inclusive`), and
    2. *cause* and *effect* are σ-separated by Z in the graph with the
       directed edge cause → effect removed.

    The criterion is functional-form-agnostic: it makes no assumption about
    the structural equations and is valid for linear, additive-noise, and
    general nonparametric SCMs that are Markov to the graph.

    :param graph: A directed mixed graph (may contain cycles).
    :param cause: The source variable of the direct effect of interest.
    :param effect: The target variable of the direct effect of interest.
    :param conditions: Candidate adjustment set Z.
    :returns: ``True`` if Z satisfies the σ-separation single-door criterion.

    .. note::
        If the directed edge cause → effect is not present in *graph*, this
        function returns ``False`` (there is no direct effect to identify).
    """
    conditions_set = set(conditions)

    # Guard: edge must exist
    if not graph.directed.has_edge(cause, effect):
        return False

    # Condition (i): Z must not contain any descendant of effect
    effect_descendants = graph.descendants_inclusive(effect)
    if conditions_set & effect_descendants:
        return False

    # Condition (ii): cause ⫫_σ effect | Z in G \ {cause → effect}
    reduced = _remove_directed_edge(graph, cause, effect)
    return are_sigma_separated(reduced, cause, effect, conditions=conditions_set)


def find_sigma_single_door_sets(
    graph: NxMixedGraph,
    cause: Variable,
    effect: Variable,
    *,
    max_conditions: int | None = None,
) -> Iterable[frozenset[Variable]]:
    """Enumerate admissible adjustment sets for the σ-separation single-door criterion.

    Yields all subsets Z of the candidate pool (non-descendants of *effect*,
    excluding *cause* and *effect* themselves) that satisfy
    :func:`is_sigma_single_door_admissible`.  Sets are enumerated in order of
    increasing size (empty set first).

    :param graph: A directed mixed graph (may contain cycles).
    :param cause: The source variable of the direct effect of interest.
    :param effect: The target variable of the direct effect of interest.
    :param max_conditions: Maximum size of Z to consider.  ``None`` (default)
        means unbounded — all subsets of the candidate pool are tried.
        **Warning:** enumeration is O(2^k) in the size of the candidate pool;
        set a cap for large graphs.
    :yields: Each admissible :class:`frozenset` of conditioning variables.

    .. note::
        Soundness over completeness: the underlying σ-separation oracle uses a
        path-augmentation heuristic that may not be complete.  A yielded Z is
        always a valid adjustment set; some valid sets may be missed.
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
