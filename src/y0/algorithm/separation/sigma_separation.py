"""Implementation of sigma-separation from [forre2018]_.

sigma-separation (Forre & Mooij 2017/2018) is the correct separation criterion
for directed mixed graphs that may contain directed cycles.  It generalises
d-separation and m-separation to cyclic graphs.

The canonical reduction used here (Rantanen et al. 2020, Theorem 5) is::

    sigma-sep(G, x, y | Z)  iff  d-sep(sigma_extension(G), x, y | Z)

where ``sigma_extension(G)`` adds a bidirected edge for every directed edge
within each strongly-connected component (SCC), yielding an acyclic directed
mixed graph (ADMG) with the same sigma-separation structure.  d-separation on
the ADMG is then computed via the standard algorithm.

This module implements ``are_sigma_separated`` via that reduction.  The helper
functions (``is_z_sigma_open``, ``is_collider``, etc.) are retained for
reference and for use by other modules that need the path-level predicates.
"""

from collections.abc import Iterable, Sequence
from typing import cast

from more_itertools import triplewise

from y0.algorithm.conditional_independencies import are_d_separated
from y0.algorithm.separation.sigma_extension import sigma_extension
from y0.dsl import Variable
from y0.graph import NxMixedGraph
from y0.struct import DSeparationJudgement

__all__ = [
    "are_sigma_separated",
    "get_equivalence_classes",
    "is_z_sigma_open",
]


def are_sigma_separated(
    graph: NxMixedGraph,
    left: Variable,
    right: Variable,
    *,
    conditions: Iterable[Variable] | None = None,
    cutoff: int | None = None,
) -> DSeparationJudgement:
    r"""Test if two variables are sigma-separated.

    sigma-separation is a generalisation of d-separation that works for directed
    mixed graphs containing directed cycles.  It was originally introduced in
    [forre2018]_.

    This implementation uses the canonical reduction from Forre & Mooij (2017,
    Theorem 3.5)::

        sigma-sep(G, x, y | Z)  iff  m-sep(acyclify(G), x, y | Z)

    ``acyclify(G)`` collapses each strongly-connected component (SCC) into a
    bidirected clique and broadcasts external parents to all SCC members,
    producing an acyclic directed mixed graph (ADMG).  m-separation on the ADMG
    is then computed via d-separation on the canonical latent-variable DAG
    (Drton & Richardson 2003).

    The return value is a :class:`~y0.struct.DSeparationJudgement` that carries
    ``left``, ``right``, and ``conditions`` in canonical form alongside the
    ``separated`` flag.  It implements ``__bool__``, so existing code that
    treats the result as a plain boolean continues to work unchanged.

    :param graph: A directed mixed graph (cycles allowed).
    :param left: A node in the graph.
    :param right: A node in the graph.
    :param conditions: A collection of graph nodes to condition on.
    :param cutoff: Ignored.  Retained for backward compatibility only.
        The acyclify + m-separation reduction does not use path enumeration.

    :returns: A :class:`~y0.struct.DSeparationJudgement` that is truthy when
        *left* and *right* are sigma-separated given *conditions*, falsy
        otherwise.
    """
    conditions_set: set[Variable] = set() if conditions is None else set(conditions)
    g_sigma = sigma_extension(graph)
    return are_d_separated(g_sigma, left, right, conditions=conditions_set)


def is_z_sigma_open(
    graph: NxMixedGraph,
    path: Sequence[Variable],
    *,
    sigma: dict[Variable, set[Variable]],
    conditions: set[Variable] | None = None,
) -> bool:
    r"""Check if a path is Z-sigma-open.

    :param graph: A mixed graph
    :param path: A path in the graph. Denoted as $\pi$ in the paper. The node in
        position $i$ in the path is denoted with $v_i$.
    :param conditions: A set of nodes chosen as conditions, denoted by $Z$ in the paper
    :param sigma: The set of equivalence classes. Can be calculated with
        :func:`get_equivalence_classes`, denoted by $\sigma(v)$ in the paper.

    :returns: If the path is Z-sigma-open

    A path is $Z-\sigma-\text{open}$ if:

    1. The end nodes $v_1, v_n \notin Z$
    2. Every triple of adjacent nodes in the path is of the form

       1. Collider (:func:`is_collider`)
       2. (non-collider) left chain (:func:`is_non_collider_left_chain`)
       3. (non-collider) right chain (:func:`is_non_collider_left_chain`)
       4. (non-collider) fork (:func:`is_non_collider_fork`)
       5. (non-collider) with undirected edge (:func:`is_non_collider_undirected`, not
          implemented)
    """
    if conditions is None:
        conditions = set()
    if path[0] in conditions or path[-1] in conditions:
        return False
    return all(
        _triple_has_correct_form(graph, left, middle, right, conditions, sigma)
        for left, middle, right in triplewise(path)
    )


def _triple_has_correct_form(
    graph: NxMixedGraph,
    left: Variable,
    middle: Variable,
    right: Variable,
    conditions: set[Variable],
    sigma: dict[Variable, set[Variable]],
) -> bool:
    if _triple_helper(graph, left, middle, right, conditions, sigma):
        return True
    # augment with backtracks, since you're allowed to go back (just like Season 5 of Lost).
    # this is a better solution than generating infinite paths, but might still be mathematically
    # incomplete. In this setup, v3->v4<->v6 becomes v3->v4->v5<-v4<->v6 to get some sweet backtrack paths
    # through the middle node to a neighbor and then back before going to the right node.
    neighbors = {n for n in graph.disorient().neighbors(middle) if n != middle}
    for neighbor in neighbors:
        if (
            _triple_helper(graph, left, middle, neighbor, conditions, sigma)
            and _triple_helper(graph, middle, neighbor, middle, conditions, sigma)
            and _triple_helper(graph, neighbor, middle, right, conditions, sigma)
        ):
            return True
    return False


def _triple_helper(
    graph: NxMixedGraph,
    left: Variable,
    middle: Variable,
    right: Variable,
    conditions: set[Variable],
    sigma: dict[Variable, set[Variable]],
) -> bool:
    return (
        is_collider(graph, left, middle, right, conditions)
        or is_non_collider_left_chain(graph, left, middle, right, conditions, sigma)
        or is_non_collider_right_chain(graph, left, middle, right, conditions, sigma)
        or is_non_collider_fork(graph, left, middle, right, conditions, sigma)
    )


def _has_either_edge(graph: NxMixedGraph, u: Variable, v: Variable) -> bool:
    return cast(bool, graph.directed.has_edge(u, v)) or cast(bool, graph.undirected.has_edge(u, v))


def _only_directed_edge(graph: NxMixedGraph, u: Variable, v: Variable) -> bool:
    return graph.directed.has_edge(u, v) and not graph.undirected.has_edge(u, v)


def is_collider(
    graph: NxMixedGraph,
    left: Variable,
    middle: Variable,
    right: Variable,
    conditions: set[Variable],
) -> bool:
    """Check if three nodes form a collider under the given conditions.

    :param graph: A mixed graph
    :param left: The first node in the subsequence, denoted as $v_{i-1}$ in the paper
    :param middle: The second node in the subsequence, denoted as $v_i$ in the paper
    :param right: The third node in the subsequence, denoted as $v_{i+1}$ in the paper
    :param conditions: The conditional variables, denoted as $Z$ in the paper

    :returns: If the three nodes form a collider
    """
    return (
        _has_either_edge(graph, left, middle)
        and _has_either_edge(graph, right, middle)
        and middle in conditions
    )


def is_non_collider_left_chain(
    graph: NxMixedGraph,
    left: Variable,
    middle: Variable,
    right: Variable,
    conditions: set[Variable],
    sigma: dict[Variable, set[Variable]],
) -> bool:
    r"""Check if three nodes form a non-collider (left chain) given the conditions.

    :param graph: A mixed graph
    :param left: The first node in the subsequence, denoted as $v_{i-1}$ in the paper
    :param middle: The second node in the subsequence, denoted as $v_i$ in the paper
    :param right: The third node in the subsequence, denoted as $v_{i+1}$ in the paper
    :param conditions: The conditional variables, denoted as $Z$ in the paper
    :param sigma: The set of equivalence classes. Can be calculated with
        :func:`get_equivalence_classes`, denoted by $\sigma(v)$ in the paper.

    :returns: If the three nodes form a non-collider (left chain) given the conditions.
    """
    return (
        _only_directed_edge(graph, middle, left)
        and _has_either_edge(graph, right, middle)
        and (middle not in conditions or middle in conditions.intersection(sigma[left]))
    )


def is_non_collider_right_chain(
    graph: NxMixedGraph,
    left: Variable,
    middle: Variable,
    right: Variable,
    conditions: set[Variable],
    sigma: dict[Variable, set[Variable]],
) -> bool:
    r"""Check if three nodes form a non-collider (right chain) given the conditions.

    :param graph: A mixed graph
    :param left: The first node in the subsequence, denoted as $v_{i-1}$ in the paper
    :param middle: The second node in the subsequence, denoted as $v_i$ in the paper
    :param right: The third node in the subsequence, denoted as $v_{i+1}$ in the paper
    :param conditions: The conditional variables, denoted as $Z$ in the paper
    :param sigma: The set of equivalence classes. Can be calculated with
        :func:`get_equivalence_classes`, denoted by $\sigma(v)$ in the paper.

    :returns: If the three nodes form a non-collider (right chain) given the conditions.
    """
    return (
        _has_either_edge(graph, left, middle)
        and _only_directed_edge(graph, middle, right)
        and (middle not in conditions or middle in conditions.intersection(sigma[right]))
    )


def is_non_collider_fork(
    graph: NxMixedGraph,
    left: Variable,
    middle: Variable,
    right: Variable,
    conditions: set[Variable],
    sigma: dict[Variable, set[Variable]],
) -> bool:
    r"""Check if three nodes form a non-collider (fork) given the conditions.

    :param graph: A mixed graph
    :param left: The first node in the subsequence, denoted as $v_{i-1}$ in the paper
    :param middle: The second node in the subsequence, denoted as $v_i$ in the paper
    :param right: The third node in the subsequence, denoted as $v_{i+1}$ in the paper
    :param conditions: The conditional variables, denoted as $Z$ in the paper
    :param sigma: The set of equivalence classes. Can be calculated with
        :func:`get_equivalence_classes`, denoted by $\sigma(v)$ in the paper.

    :returns: If the three nodes form a non-collider (fork) given the conditions.
    """
    a = _only_directed_edge(graph, middle, left)
    b = _only_directed_edge(graph, middle, right)
    c = middle not in conditions
    d = middle in conditions.intersection(sigma[left]).intersection(sigma[right])
    return a and b and (c or d)


def get_equivalence_classes(graph: NxMixedGraph) -> dict[Variable, set[Variable]]:
    r"""Get equivalence classes.

    :param graph: A mixed graph

    :returns: A mapping from variables to their equivalence class, defined as the second
        option from the paper (see below)

    1. The finest/trivial :math:`\sigma`-CG structure of a mixed graph G is given by
       :math:`\sigma(v) := \{v\}` for all v in V. In this way :math:`\sigma`-separation
       in G coincides with the usual notion of d-separation in a d-connection graph
       (d-CG) G (see [19]). We will take this as the definition of d-separation and
       d-CG in the following.
    2. The coarsest :math:`\sigma`-CG structure of a mixed graph G is given by
       :math:`\sigma(v) := ScG(v) := AncG(v) \cap DescG(v)` w.r.t. the underlying
       directed graph. Note that the definition of strongly connected component totally
       ignores the bi- and undirected edges of the :math:`\sigma`-CG.
    """
    return {node: _get_scc(graph, node) for node in graph.nodes()}


def _get_scc(graph: NxMixedGraph, node: Variable) -> set[Variable]:
    # TODO: It might be faster to use strongly_connected_components:
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.\
    # components.strongly_connected_components.html#networkx.algorithms.components.strongly_connected_components
    return graph.ancestors_inclusive(node).intersection(graph.descendants_inclusive(node))
