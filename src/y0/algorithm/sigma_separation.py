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

    equivalence_classes = get_equivalence_classes(graph)
    return not any(
        is_z_sigma_open(graph, path, conditions, equivalence_classes)
        for path in nx.all_simple_paths(graph.disorient(), left, right, cutoff=cutoff)
    )


def is_z_sigma_open(
    graph: NxMixedGraph,
    path: Sequence[Variable],
    conditions: set[Variable],
    sigma: dict[Variable, set[Variable]],
) -> bool:
    r"""Check if a path is Z-sigma-open.

    :param graph: A mixed graph
    :param path: A path in the graph. Denoted as $\pi$ in the paper. The
        node in position $i$ in the path is denoted with $v_i$.
    :param conditions : A set of nodes chosen as conditions, denoted by $Z$ in the paper
    :param sigma: The set of equivalence classes. Can be calculated with
        :func:`get_equivalence_classes`, denoted by $\sigma(v)$ in the paper.
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
            or is_non_collider_left_chain(graph, left, middle, right, conditions, sigma)
            or is_non_collider_right_chain(graph, left, middle, right, conditions, sigma)
            or is_non_collider_fork(graph, left, middle, right, conditions, sigma)
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
        graph.directed.has_edge(middle, left)
        and graph.directed.has_edge(right, middle)
        and graph.undirected.has_edge(middle, right)
    ) and (middle not in conditions or middle in conditions.intersection(sigma[left]))


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
        graph.directed.has_edge(left, middle)
        and graph.directed.has_edge(middle, right)
        and graph.undirected.has_edge(left, middle)
    ) and (middle not in conditions or middle in conditions.intersection(sigma[right]))


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
    return (graph.directed.has_edge(middle, left) and graph.directed.has_edge(middle, right)) and (
        middle not in conditions
        or middle in conditions.intersection(sigma[left]).intersection(sigma[right])
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
    :raises NotImplementedError: We need to update the data model
    """
    raise NotImplementedError("this would require additional edge types to NxMixedGraph")


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


def _get_eq_classes_alt(graph):
    rv = {}
    for cycle in nx.simple_cycles(graph.directed):
        cycle = set(cycle)
        for node in cycle:
            rv[node] = cycle
    # nodes that don't appear in any cycles get their own class
    for node in graph:
        if node not in rv:
            rv[node] = {node}
    # FIXME what happens if a node appears in multiple cycles?
    #  Maybe join them together into a super-cycle?
    return rv
