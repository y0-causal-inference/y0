"""Implementation of sigma-separation from [forre2018]_."""

from typing import Iterable, Optional, Sequence

import networkx as nx
from more_itertools import triplewise

from y0.dsl import Variable
from y0.graph import NxMixedGraph

__all__ = [
    "are_sigma_separated",
    "is_z_sigma_open",
    "get_equivalence_classes",
]


def are_sigma_separated(
    graph: NxMixedGraph,
    left: Variable,
    right: Variable,
    *,
    conditions: Optional[Iterable[Variable]] = None,
    cutoff: Optional[int] = None,
) -> bool:
    """Test if two variables are sigma-separated.

    Sigma separation is a generalization of d-separation that
    works not only for directed acyclic graphs, but also for
    directed graphs containing cycles. It was originally introduced
    in [forre2018]_.

    We say that X and Y are σ-connected by Z or not
    σ-separated by Z if there exists a path π (with some
    n ≥ 1 nodes) in G with one endnode in X and
    one endnode in Y that is Z-σ-open. σ-separated is the
    opposite of σ-connected (logical not).

    :param graph: Graph to test
    :param left: A node in the graph
    :param right: A node in the graph
    :param conditions: A collection of graph nodes
    :param cutoff: The maximum path length to check. By default, is unbounded.
    :return: If a and b are sigma-separated.
    """
    if conditions is None:
        conditions = set()
    else:
        conditions = set(conditions)

    sigma = get_equivalence_classes(graph)
    return not any(
        is_z_sigma_open(graph, path, conditions=conditions, sigma=sigma)
        # Technically, this algorithm should generate all paths, which could include
        # repeat visits to nodes and edges, but this is computationally intractable,
        # so the is_z_sigma_open() subroutine contains a novel path augmentation
        # algorithm. This might not be officially complete.
        for path in nx.all_simple_paths(graph.disorient(), left, right, cutoff=cutoff)
    )


def is_z_sigma_open(
    graph: NxMixedGraph,
    path: Sequence[Variable],
    *,
    sigma: dict[Variable, set[Variable]],
    conditions: Optional[set[Variable]] = None,
) -> bool:
    r"""Check if a path is Z-sigma-open.

    :param graph: A mixed graph
    :param path: A path in the graph. Denoted as $\pi$ in the paper. The
        node in position $i$ in the path is denoted with $v_i$.
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
       5. (non-collider) with undirected edge (:func:`is_non_collider_undirected`, not implemented)
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
    # incomplete. In this setup, 𝑣3→𝑣4↔𝑣6 becomes 𝑣3→𝑣4→𝑣5←𝑣4↔𝑣6 to get some sweet backtrack paths
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


def _has_either_edge(graph: NxMixedGraph, u, v) -> bool:
    return graph.directed.has_edge(u, v) or graph.undirected.has_edge(u, v)


def _only_directed_edge(graph, u, v) -> bool:
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
    :return: If the three nodes form a collider
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
    :return: If the three nodes form a non-collider (left chain) given the conditions.
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
    :return: If the three nodes form a non-collider (right chain) given the conditions.
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
    :return: If the three nodes form a non-collider (fork) given the conditions.
    """
    a = _only_directed_edge(graph, middle, left)
    b = _only_directed_edge(graph, middle, right)
    c = middle not in conditions
    d = middle in conditions.intersection(sigma[left]).intersection(sigma[right])
    return a and b and (c or d)


def get_equivalence_classes(graph: NxMixedGraph) -> dict[Variable, set[Variable]]:
    """Get equivalence classes.

    :param graph: A mixed graph
    :returns: A mapping from variables to their equivalence class,
        defined as the second option from the paper (see below)

    1. The finest/trivial σ-CG structure of
       a mixed graph G is given by σ(v) := {v} for all
       v ∈ V . In this way σ-separation in G coincides with
       the usual notion of d-separation in a d-connection
       graph (d-CG) G (see [19]). We will take this as the
       definition of d-separation and d-CG in the following.
    2. The coarsest σ-CG structure of a mixed graph G is
       given by σ(v) := ScG(v) := AncG(v) ∩ DescG(v)
       w.r.t. the underlying directed graph. Note that the
       definition of strongly connected component totally
       ignores the bi- and undirected edges of the σ-CG.
    """
    return {
        node: graph.ancestors_inclusive(node).intersection(graph.descendants_inclusive(node))
        for node in graph.nodes()
    }
