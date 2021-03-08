""""""

from collections import defaultdict

import itertools as itt
import networkx as nx
from typing import Optional

DEFAULT_TAG = 'latent'


def remove_latents_with_no_children(graph: nx.DiGraph, tag: Optional[str] = None, inplace: bool = False) -> nx.DiGraph:
    """Remove latents with no children (in-place)

    :param graph:
    :param tag:
    :return:
    """
    if tag is None:
        tag = DEFAULT_TAG
    if not inplace:
        graph = graph.copy()
    remove = {
        node
        for node, data in graph.nodes.items()
        if data[tag] and 0 == len(graph.out_edges(node))
    }
    graph.remove_nodes_from(remove)
    return graph


def transform_latents_with_parents(
    graph: nx.DiGraph,
    tag: Optional[str] = None,
    inplace: bool = False,
    suffix: str = '_PRIME',
) -> nx.DiGraph:
    if tag is None:
        tag = DEFAULT_TAG
    if not inplace:
        graph = graph.copy()

    new_edges = []
    new_latents = defaultdict(list)
    remove_nodes = []
    for node, data in graph.nodes.items():
        if not data[tag]:
            continue
        in_edges = list(graph.in_edges(node))
        if 0 == len(in_edges):
            continue
        out_edges = list(graph.out_edges(node))
        parents = [parent for parent, _ in in_edges]
        children = [child for _, child in out_edges]
        new_latents[node].extend(children)
        remove_nodes.append(node)
        for parent, child in itt.product(parents, children):
            new_edges.append((parent, child))
    graph.remove_nodes_from(remove_nodes)
    graph.add_edges_from(new_edges)
    for latent, children in new_latents.items():
        new_node = f"{latent}{suffix}"
        graph.add_node(new_node, **{tag: True})
        for child in children:
            graph.add_edge(new_node, child)
    return graph
