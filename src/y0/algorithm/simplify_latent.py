# -*- coding: utf-8 -*-

"""Implement Robin Evans' simplification algorithms.

.. seealso:: https://www.fields.utoronto.ca/programs/scientific/11-12/graphicmodels/Evans.pdf slides 34-43
"""

import itertools as itt
from collections import defaultdict
from typing import DefaultDict, Iterable, List, Mapping, Optional, Set, Tuple

import networkx as nx

from ..graph import DEFAULT_TAG

__all__ = [
    'remove_widow_latents',
    'transform_latents_with_parents',
    'remove_redundant_latents',
]

DEFAULT_SUFFIX = '_prime'


def iter_latents(graph: nx.DiGraph, *, tag: Optional[str] = None) -> Iterable[str]:
    """Iterate over nodes marked as latent.

    :param graph: A latent variable DAG
    :param tag: The tag for which variables are latent
    :yields: Nodes that are latent
    """
    if tag is None:
        tag = DEFAULT_TAG
    for node, data in graph.nodes(data=True):
        if data[tag]:
            yield node


def remove_widow_latents(graph: nx.DiGraph, tag: Optional[str] = None) -> Tuple[nx.DiGraph, Set[str]]:
    """Remove latents with no children (in-place).

    :param graph: A latent variable DAG
    :param tag: The tag for which variables are latent
    :returns: The graph, modified in place
    """
    remove = set(iter_widow_latents(graph, tag=tag))
    graph.remove_nodes_from(remove)
    return graph, remove


def iter_widow_latents(graph: nx.DiGraph, *, tag: Optional[str] = None) -> Iterable[str]:
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
    remove_nodes: List[str] = []
    add_edges: List[Tuple[str, str]] = []
    modified_latents: DefaultDict[str, List[str]] = defaultdict(list)
    for latent_node, parents, children in iter_middle_latents(graph, tag=tag):
        remove_nodes.append(latent_node)
        add_edges.extend(itt.product(parents, children))
        modified_latents[latent_node].extend(children)

    graph.remove_nodes_from(remove_nodes)
    graph.add_edges_from(add_edges)
    _add_modified_latent(graph, modified_latents, tag=tag, suffix=suffix)
    # Alternatively, could only remove node-parent edges and keep the name of the original latent

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


def iter_middle_latents(graph: nx.DiGraph, *, tag: Optional[str] = None) -> Iterable[Tuple[str, Set[str], Set[str]]]:
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


def remove_redundant_latents(graph: nx.DiGraph, tag: Optional[str] = None) -> Tuple[nx.DiGraph, Set[str]]:
    """Remove redundant latent variables.

    :param graph: A latent variable DAG
    :param tag: The tag for which variables are latent
    :returns: The graph, modified in place
    """
    remove = set(_iter_redundant_latents(graph, tag=tag))
    graph.remove_nodes_from(remove)
    return graph, remove


def _iter_redundant_latents(graph, *, tag: Optional[str] = None) -> Iterable[str]:
    latents: Mapping[str, Set[str]] = {
        node: set(graph.successors(node))
        for node in iter_latents(graph, tag=tag)
    }
    for (left, left_children), (right, right_children) in itt.product(latents.items(), repeat=2):
        if left_children == right_children and left > right:
            # if children are the same, keep the lower sort order node
            yield left
        elif left_children < right_children:
            # if left's children are a proper subset of right's children, we don't need left
            yield left
