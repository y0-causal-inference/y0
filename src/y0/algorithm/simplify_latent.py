""""""

from collections import defaultdict

import itertools as itt
import networkx as nx
from typing import Iterable, Mapping, Optional, Set, Tuple

DEFAULT_TAG = 'latent'


def remove_widow_latents(graph: nx.DiGraph, tag: Optional[str] = None) -> nx.DiGraph:
    """Remove latents with no children (in-place).

    :param graph:
    :param tag:
    :return:
    """
    graph.remove_nodes_from(list(iter_widow_latents(graph, tag=tag)))
    return graph


def iter_widow_latents(graph: nx.DiGraph, *, tag: Optional[str] = None) -> Iterable[str]:
    if tag is None:
        tag = DEFAULT_TAG
    for node, data in graph.nodes.items():
        if data[tag] and 0 == len(graph.out_edges(node)):
            yield node


def transform_latents_with_parents(
    graph: nx.DiGraph,
    tag: Optional[str] = None,
    suffix: str = '_PRIME',
) -> nx.DiGraph:
    """Transform latent variables with parents into latent variables with no parents."""
    remove_nodes = []
    add_edges = []
    new_latents = defaultdict(list)
    for node, parents, children in iter_middle_latents(graph, tag=tag):
        remove_nodes.append(node)
        add_edges.extend(itt.product(parents, children))
        new_latents[node].extend(children)

    graph.remove_nodes_from(remove_nodes)
    graph.add_edges_from(add_edges)
    for latent, children in new_latents.items():
        new_node = f"{latent}{suffix}"
        graph.add_node(new_node, **{tag: True})
        for child in children:
            graph.add_edge(new_node, child)

    return graph


def iter_middle_latents(graph: nx.DiGraph, *, tag: Optional[str] = None) -> Iterable[Tuple[str, Set[str], Set[str]]]:
    if tag is None:
        tag = DEFAULT_TAG
    for node, data in graph.nodes.items():
        if not data[tag]:
            continue
        parents = {parent for parent, _ in graph.in_edges(node)}
        if 0 == len(parents):
            continue
        children = {child for _, child in graph.out_edges(node)}
        if 0 == len(children):
            continue
        yield node, parents, children


def simplification_3(
    graph: nx.DiGraph,
    tag: Optional[str] = None,
    inplace: bool = False,
) -> nx.DiGraph:
    """Transform"""
    if tag is None:
        tag = DEFAULT_TAG
    if not inplace:
        graph = graph.copy()
    latents: Mapping[str, Set[str]] = {
        node: {child for _, child in graph.out_edges(node)}
        for node, data in graph.nodes.items()
        if data[tag]
    }
    remove = []
    for (left, left_children), (right, right_children) in itt.product(latents.items(), r=2):
        if left_children == right_children:
            # if children are the same, keep the lower sort order node
            remove.append(max(left, right))
        elif left_children < right_children:
            # if left's children are a proper subset of right's children, we don't need left
            remove.append(left)
    graph.remove_nodes_from(remove)
    return graph
