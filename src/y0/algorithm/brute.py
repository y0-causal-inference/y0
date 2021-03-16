# -*- coding: utf-8 -*-

"""An implementation of Sara Taheri's algorithm for using causal queries for experimental design.

.. seealso:: https://docs.google.com/presentation/d/1klBOjGtRkXOMSDgOCLChBTBJZ0dFxJPn9IPRLAlv_N8/edit?usp=sharing
"""

import itertools as itt
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Set, Tuple, Union

import networkx as nx
from ananke.graphs import ADMG
from more_itertools import powerset
from tabulate import tabulate
from tqdm import tqdm

from y0.algorithm.simplify_latent import simplify_latent_dag
from y0.dsl import P, Variable
from y0.examples import igf_graph
from y0.graph import DEFAULT_TAG, NxMixedGraph, admg_from_latent_variable_dag, admg_to_latent_variable_dag, set_latent
from y0.identify import is_identifiable

__all__ = [
    'taheri_design_admg',
    'taheri_design_dag',
    'Result',
    'draw_results',
]


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


def taheri_design_admg(graph: ADMG, cause: str, effect: str, *, tag: Optional[str] = None) -> List[Result]:
    """Run the brute force implementation of the Taheri Design algorithm on an ADMG.

    :param graph: An ADMG
    :param cause: The node that gets perturbed.
    :param effect: The node that we're interested in.
    :return: A list of LV-DAG identifiability results. Will be length 2^(|V| - 2 - # bidirected edges)
    """
    if tag is None:
        tag = DEFAULT_TAG
    dag = admg_to_latent_variable_dag(graph, tag=tag)
    fixed_latents = {
        node
        for node, data in dag.nodes(data=True)
        if data[tag]
    }
    return _help(graph=dag, cause=cause, effect=effect, skip=fixed_latents | {cause, effect})


def taheri_design_dag(graph: nx.DiGraph, cause: str, effect: str) -> List[Result]:
    """Run the brute force implementation of the Taheri Design algorithm on a DAG.

    Identify all latent variable configurations inducible over the given DAG that result
    in an identifiable ADMG under the causal query corresponding to the given cause/effect.

    :param graph: A regular DAG
    :param cause: The node that gets perturbed.
    :param effect: The node that we're interested in.
    :return: A list of LV-DAG identifiability results. Will be length 2^(|V| - 2)
    """
    return _help(graph=graph, cause=cause, effect=effect, skip={cause, effect})


def _help(graph, cause, effect, skip):
    return [
        _get_result(lvdag, latents, cause, effect)
        for latents, lvdag in iterate_lvdags(graph, skip=skip)
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
    # TODO optimize traversal through power set space
    for latents in powerset(sorted(nodes)):
        yv = graph.copy()
        set_latent(yv, latents)
        yield latents, yv


def draw_results(
    results: Iterable[Result],
    path: Union[str, Path, Iterable[str], Iterable[Path]],
    ncols: int = 10,
    x_ratio: float = 3.5,
    y_ratio: float = 3.5,
) -> None:
    """Draw identifiable ADMGs to a file."""
    import matplotlib.pyplot as plt

    if isinstance(path, str):
        path = [path]

    identifiable_results = [result for result in results if result.identifiable]

    nrows = 1 + len(identifiable_results) // ncols
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * x_ratio, nrows * y_ratio))
    it = itt.zip_longest(axes.ravel(), tqdm(identifiable_results, desc='generating chart'))
    for i, (ax, result) in enumerate(it, start=1):
        if result is None:
            ax.axis('off')
        else:
            mixed_graph = NxMixedGraph.from_admg(result.admg)
            mixed_graph.draw(ax=ax, title=f'{i}) ' + ','.join(result.latents))

    plt.tight_layout()

    for _path in tqdm(path, desc='saving'):
        fig.savefig(_path, dpi=400)


def main():
    """Run the algorithm on the IGF graph with the PI3K/Erk example."""
    results = taheri_design_dag(igf_graph, cause='PI3K', effect='Erk')
    rows = [
        (i, result, post_nodes - pre_nodes, post_edges - pre_edges, len(latents), ', '.join(latents))
        for i, (result, pre_nodes, pre_edges, post_nodes, post_edges, latents, _, _) in enumerate(results, start=1)
    ]
    print(tabulate(rows, headers=['Row', 'ID?', 'Node Simp.', 'Edge Simp.', 'N', 'Latents']))

    draw_results(results, ['ifg_identifiable_configs.png', 'ifg_identifiable_configs.svg'])


if __name__ == '__main__':
    main()
