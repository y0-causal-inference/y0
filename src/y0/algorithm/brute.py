# -*- coding: utf-8 -*-

"""An implementation of Sara Taheri's algorithm for using causal queries for experimental design.

.. seealso:: https://docs.google.com/presentation/d/1klBOjGtRkXOMSDgOCLChBTBJZ0dFxJPn9IPRLAlv_N8/edit?usp=sharing
"""

import networkx as nx
from ananke.graphs import ADMG
from more_itertools import powerset
from tabulate import tabulate
from typing import Iterable, List, NamedTuple, Optional, Set, Tuple

from y0.algorithm.simplify_latent import simplify_latent_dag
from y0.dsl import P, Variable
from y0.examples import igf_graph
from y0.graph import admg_from_latent_variable_dag, set_latent
from y0.identify import is_identifiable


class Result(NamedTuple):
    """Results from the LV-DAG check."""

    identifiable: bool
    pre_nodes: int
    pre_edges: int
    post_nodes: int
    post_edges: int
    latents: List[str]
    lvdag: nx.DiGraph
    admg: ADMG


def brute(graph: nx.DiGraph, cause: str, effect: str) -> List[Result]:
    """Run the brute force algorithm.

    Identify all latent variable configurations inducible over the given DAG that result
    in an identifiable ADMG under the causal query corresponding to the given cause/effect.

    :param graph: A regular DAG
    :param cause: The node that gets perturbed.
    :param effect: The node that we're interested in.
    """
    return [
        _get_result(lvdag, latents, cause, effect)
        for latents, lvdag in iterate_lvdags(graph, skip={cause, effect})
    ]


def _get_result(lvdag, latents, cause, effect) -> Result:
    # Book keeping
    pre_nodes, pre_edges = lvdag.number_of_nodes(), lvdag.number_of_edges()

    # Apply the robin evans algorithms
    simplify_latent_dag(lvdag)
    post_nodes, post_edges = lvdag.number_of_nodes(), lvdag.number_of_edges()

    # Convert the latent variable DAG to an ADMG
    admg = admg_from_latent_variable_dag(lvdag)

    # Check if the ADMG is identifiable under the (simple) causal query
    identifiable = is_identifiable(admg, P(Variable(effect) @ ~Variable(cause)))

    return Result(
        identifiable,
        pre_nodes,
        pre_edges,
        post_nodes,
        post_edges,
        sorted(latents),
        lvdag,
        admg,
    )


def iterate_lvdags(graph: nx.DiGraph, skip: Optional[Iterable[str]] = None) -> Iterable[Tuple[Set[str], nx.DiGraph]]:
    """Iterate over all possible latent variable configurations for the given graph.

    :param graph: A regularDAG
    :param skip: Nodes to skip in the power set of all possible latent variables. Often, the cause and effect from
        a causal query will be used here to avoid setting them as latent (since they can not be).
    :yields: latent variable DAGs for all possible latent variable configurations over the original DAG
    """
    nodes = set(graph)
    if skip:
        nodes.difference_update(skip)
    for latents in powerset(sorted(nodes)):
        yv = graph.copy()
        set_latent(yv, latents)
        yield latents, yv


def main():
    """Run the algorithm on the IGF graph with the PI3K/Erk example."""
    results = brute(igf_graph, cause='PI3K', effect='Erk')
    rows = [
        (result, post_nodes - pre_nodes, post_edges - pre_edges, len(latents), ', '.join(latents))
        for result, pre_nodes, pre_edges, post_nodes, post_edges, latents, _, _ in results

    ]
    print(tabulate(rows, headers=['ID?', 'Node Simp.', 'Edge Simp.', 'N', 'Latents']))


if __name__ == '__main__':
    main()
