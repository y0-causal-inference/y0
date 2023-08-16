# -*- coding: utf-8 -*-

"""Implement Robin Evans' simplification algorithms.

.. seealso:: https://www.fields.utoronto.ca/programs/scientific/11-12/graphicmodels/Evans.pdf slides 34-43
"""

import itertools as itt
import logging
from typing import Iterable, Mapping, NamedTuple, Optional, Set, Tuple

import networkx as nx

from ..dsl import Variable
from ..graph import DEFAULT_TAG

__all__ = [
    "simplify_latent_dag",
    "SimplifyResults",
    "remove_widow_latents",
    "transform_latents_with_parents",
    "remove_redundant_latents",
]

logger = logging.getLogger(__name__)

DEFAULT_SUFFIX = "_prime"


class SimplifyResults(NamedTuple):
    """Results from the simplification of a LV-DAG."""

    graph: nx.DiGraph
    widows: Set[Variable]
    redundant: Set[Variable]


def simplify_latent_dag(graph: nx.DiGraph, tag: Optional[str] = None):
    """Apply Robin Evans' three algorithms in succession."""
    if tag is None:
        tag = DEFAULT_TAG

    _ = transform_latents_with_parents(graph, tag=tag)
    _, widows = remove_widow_latents(graph, tag=tag)
    _, redundant = remove_redundant_latents(graph, tag=tag)

    return SimplifyResults(
        graph=graph,
        widows=widows,
        redundant=redundant,
    )


def iter_latents(graph: nx.DiGraph, *, tag: Optional[str] = None) -> Iterable[Variable]:
    """Iterate over nodes marked as latent.

    :param graph: A latent variable DAG
    :param tag: The tag for which variables are latent
    :yields: Nodes that are latent
    """
    if tag is None:
        tag = DEFAULT_TAG
    # should start with nodes the highest up (closest to source)
    for node in nx.topological_sort(graph):
        if graph.nodes[node][tag]:
            yield node


def remove_widow_latents(
    graph: nx.DiGraph, tag: Optional[str] = None
) -> Tuple[nx.DiGraph, Set[Variable]]:
    """Remove latents with no children (in-place).

    :param graph: A latent variable DAG
    :param tag: The tag for which variables are latent
    :returns: The graph, modified in place
    """
    remove = set(iter_widow_latents(graph, tag=tag))
    graph.remove_nodes_from(remove)
    return graph, remove


def iter_widow_latents(graph: nx.DiGraph, *, tag: Optional[str] = None) -> Iterable[Variable]:
    """Iterate over latent variables with no children.

    :param graph: A latent variable DAG
    :param tag: The tag for which variables are latent
    :yields: Nodes with no children
    """
    for node in iter_latents(graph, tag=tag):
        if not graph.out_edges(node):
            yield node


def transform_latents_with_parents(
    graph: nx.DiGraph,
    tag: Optional[str] = None,
    suffix: Optional[str] = None,
) -> nx.DiGraph:
    """Transform latent variables with parents into latent variables with no parents.

    :param graph: A latent variable DAG
    :param tag: The tag for which variables are latent
    :param suffix: The suffix to postpend to transformed latent variables.
    :returns: The graph, modified in place
    """
    if tag is None:
        tag = DEFAULT_TAG
    if suffix is None:
        suffix = DEFAULT_SUFFIX
    for latent_node, parents, children in iter_middle_latents(graph, tag=tag):
        graph.remove_node(latent_node)
        graph.add_edges_from(itt.product(parents, children))

        new_node = Variable(f"{latent_node}{suffix}")
        graph.add_node(new_node, **{tag: True})
        for child in children:
            graph.add_edge(new_node, child)

    return graph


def _add_modified_latent(
    graph: nx.DiGraph,
    modified_latent: Mapping[str, Iterable[str]],
    *,
    tag: Optional[str] = None,
    suffix: Optional[str] = None,
) -> None:
    if tag is None:
        tag = DEFAULT_TAG
    if suffix is None:
        suffix = DEFAULT_SUFFIX

    for old_node, children in modified_latent.items():
        new_node = f"{old_node}{suffix}"
        graph.add_node(new_node, **{tag: True})
        for child in children:
            graph.add_edge(new_node, child)


def iter_middle_latents(
    graph: nx.DiGraph, *, tag: Optional[str] = None
) -> Iterable[Tuple[Variable, Set[Variable], Set[Variable]]]:
    """Iterate over latent nodes that have both parents and children (along with them).

    :param graph: A latent variable DAG
    :param tag: The tag for which variables are latent
    :yields: Nodes with both parents and children, along with the set of parents and set of children
    """
    for node in iter_latents(graph, tag=tag):
        parents = set(graph.predecessors(node))
        if 0 == len(parents):
            continue
        children = set(graph.successors(node))
        if 0 == len(children):
            continue
        yield node, parents, children


def remove_redundant_latents(
    graph: nx.DiGraph, tag: Optional[str] = None
) -> Tuple[nx.DiGraph, Set[Variable]]:
    """Remove redundant latent variables.

    :param graph: A latent variable DAG
    :param tag: The tag for which variables are latent
    :returns: The graph, modified in place
    """
    remove = set(_iter_redundant_latents(graph, tag=tag))
    graph.remove_nodes_from(remove)
    return graph, remove


def _iter_redundant_latents(graph: nx.DiGraph, *, tag: Optional[str] = None) -> Iterable[Variable]:
    latents: Mapping[Variable, Set[Variable]] = {
        node: set(graph.successors(node)) for node in iter_latents(graph, tag=tag)
    }
    for (left, left_children), (right, right_children) in itt.product(latents.items(), repeat=2):
        if left_children == right_children and left > right:
            # if children are the same, keep the lower sort order node
            yield left
        elif left_children < right_children:
            # if left's children are a proper subset of right's children, we don't need left
            yield left
