"""Implementation of sigma-separation."""

from typing import Iterable, Optional, Sequence

import networkx as nx
from more_itertools import triplewise

from y0.dsl import Variable
from y0.graph import NxMixedGraph

__all__ = [
    "are_sigma_separated",
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
    in https://arxiv.org/abs/1807.03024.

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
    return not any(
        is_z_sigma_open(graph, path, conditions)
        for path in nx.all_simple_paths(graph.disorient(), left, right, cutoff=cutoff)
    )


def is_z_sigma_open(
    graph: NxMixedGraph, path: Sequence[Variable], conditions: set[Variable]
) -> bool:
    r"""Check if a path is Z-sigma-open.

    :param graph: A mixed graph
    :param path: A path in the graph. Denoted as $\pi$ in the paper. The
        node in position $i$ in the path is denoted with $v_i$.
    :param conditions : A set of nodes chosen as conditions, denoted by $Z$ in the paper
    :returns: If the path is Z-sigma-open

    A path is $Z-\sigma-\text{open}$ if:

    1. The end nodes $v_1, v_n \notin Z$
    2. Every triple of adjacent nodes in the path is of the form:
       1. Collider (:func:`is_collider`)
       2. (non-collider) left chain (:func:`is_non_collider_left_chain`)
       3. (non-collider) right chain (:func:`is_non_collider_left_chain`)
       4. (non-collider) fork (:func:`is_non_collider_fork`)
       5. (non-collider) with undirected edge (:func:`is_non_collider_undirected`, not implemented)
    """
    if path[0] in conditions or path[-1] in conditions:
        return False
    return all(
        (
            is_collider(graph, left, middle, right, conditions)
            or is_non_collider_left_chain(graph, left, middle, right, conditions)
            or is_non_collider_right_chain(graph, left, middle, right, conditions)
            or is_non_collider_fork(graph, left, middle, right, conditions)
            # or is_non_collider_undirected(graph, a, b, c, z) TODO implement me!
        )
        for left, middle, right in triplewise(path)
    )


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
        graph.directed.has_edge(left, middle)
        and graph.directed.has_edge(right, middle)
        and graph.undirected.has_edge(left, middle)
        and graph.undirected.has_edge(middle, right)
    ) and middle in conditions


def is_non_collider_left_chain(
    graph: NxMixedGraph,
    left: Variable,
    middle: Variable,
    right: Variable,
    conditions: set[Variable],
) -> bool:
    """Check if three nodes form a non-collider (left chain) given the conditions.

    :param graph: A mixed graph
    :param left: The first node in the subsequence, denoted as $v_{i-1}$ in the paper
    :param middle: The second node in the subsequence, denoted as $v_i$ in the paper
    :param right: The third node in the subsequence, denoted as $v_{i+1}$ in the paper
    :param conditions: The conditional variables, denoted as $Z$ in the paper
    :return: If the three nodes form a non-collider (left chain) given the conditions.
    """
    return (
        graph.directed.has_edge(middle, left)
        and graph.directed.has_edge(right, middle)
        and graph.undirected.has_edge(middle, right)
    ) and (
        middle not in conditions
        or middle in conditions.intersection(get_sigma_equivalence_class(graph, left))
    )


def is_non_collider_right_chain(
    graph: NxMixedGraph,
    left: Variable,
    middle: Variable,
    right: Variable,
    conditions: set[Variable],
) -> bool:
    """Check if three nodes form a non-collider (right chain) given the conditions.

    :param graph: A mixed graph
    :param left: The first node in the subsequence, denoted as $v_{i-1}$ in the paper
    :param middle: The second node in the subsequence, denoted as $v_i$ in the paper
    :param right: The third node in the subsequence, denoted as $v_{i+1}$ in the paper
    :param conditions: The conditional variables, denoted as $Z$ in the paper
    :return: If the three nodes form a non-collider (right chain) given the conditions.
    """
    return (
        graph.directed.has_edge(left, middle)
        and graph.directed.has_edge(middle, right)
        and graph.undirected.has_edge(left, middle)
    ) and (
        middle not in conditions
        or middle in conditions.intersection(get_sigma_equivalence_class(graph, right))
    )


def is_non_collider_fork(
    graph: NxMixedGraph,
    left: Variable,
    middle: Variable,
    right: Variable,
    conditions: set[Variable],
) -> bool:
    """Check if three nodes form a non-collider (fork) given the conditions.

    :param graph: A mixed graph
    :param left: The first node in the subsequence, denoted as $v_{i-1}$ in the paper
    :param middle: The second node in the subsequence, denoted as $v_i$ in the paper
    :param right: The third node in the subsequence, denoted as $v_{i+1}$ in the paper
    :param conditions: The conditional variables, denoted as $Z$ in the paper
    :return: If the three nodes form a non-collider (fork) given the conditions.
    """
    return (graph.directed.has_edge(middle, left) and graph.directed.has_edge(middle, right)) and (
        middle not in conditions
        or middle
        in conditions.intersection(get_sigma_equivalence_class(graph, left)).intersection(
            get_sigma_equivalence_class(graph, right)
        )
    )


def is_non_collider_undirected(
    graph: NxMixedGraph,
    left: Variable,
    middle: Variable,
    right: Variable,
    conditions: set[Variable],
) -> bool:
    """Check if three nodes form a non-collider (with undirected) given the conditions.

    :param graph: A mixed graph
    :param left: The first node in the subsequence, denoted as $v_{i-1}$ in the paper
    :param middle: The second node in the subsequence, denoted as $v_i$ in the paper
    :param right: The third node in the subsequence, denoted as $v_{i+1}$ in the paper
    :param conditions: The conditional variables, denoted as $Z$ in the paper
    :return: If the three nodes form a non-collider (fork) given the conditions.
    """
    raise NotImplementedError("this would require additional edge types to NxMixedGraph")


def get_sigma_equivalence_class(graph: NxMixedGraph, node: Variable) -> set[Variable]:
    """Get the set of sigma equivalent nodes to the given node.

    :param graph: A mixed graph
    :param node: A node in the graph
    :returns: The set of sigma-equivalent nodes

    every equivalence class σ(v), v ∈ V , is a loop in the underlying directed
    graph structure: σ(v) ∈ L(G).
    """
    # not sure what this is yet. I think this maps to all elements in a loop that this one is in
    raise NotImplementedError
