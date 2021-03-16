# -*- coding: utf-8 -*-

"""An implementation of Sara Taheri's algorithm for using causal queries for experimental design.

.. seealso:: https://docs.google.com/presentation/d/1klBOjGtRkXOMSDgOCLChBTBJZ0dFxJPn9IPRLAlv_N8/edit?usp=sharing
"""

import itertools as itt
import logging
import textwrap
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Set, Tuple, Union

import click
import networkx as nx
from ananke.graphs import ADMG
from more_click import verbose_option
from tabulate import tabulate
from tqdm import tqdm

from y0.algorithm.simplify_latent import simplify_latent_dag
from y0.dsl import P, Variable
from y0.graph import DEFAULT_TAG, NxMixedGraph, admg_from_latent_variable_dag, admg_to_latent_variable_dag, set_latent
from y0.identify import is_identifiable
from y0.util.combinatorics import powerset

__all__ = [
    'taheri_design_admg',
    'taheri_design_dag',
    'Result',
    'draw_results',
]

logger = logging.getLogger(__name__)


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


def taheri_design_admg(
    graph: Union[ADMG, NxMixedGraph],
    cause: str,
    effect: str,
    *,
    tag: Optional[str] = None,
    stop: Optional[int] = None,
) -> List[Result]:
    """Run the brute force implementation of the Taheri Design algorithm on an ADMG.

    :param graph: An ADMG
    :param cause: The node that gets perturbed.
    :param effect: The node that we're interested in.
    :param tag: The key for node data describing whether it is latent.
        If None, defaults to :data:`y0.graph.DEFAULT_TAG`.
    :param stop: Largest combination to get (None means length of the list and is the default)
    :return: A list of LV-DAG identifiability results. Will be length 2^(|V| - 2 - # bidirected edges)
    """
    if tag is None:
        tag = DEFAULT_TAG
    if isinstance(graph, NxMixedGraph):
        graph = graph.to_admg()
    dag = admg_to_latent_variable_dag(graph, tag=tag)
    fixed_latents = {
        node
        for node, data in dag.nodes(data=True)
        if data[tag]
    }
    return _help(graph=dag, cause=cause, effect=effect, skip=fixed_latents | {cause, effect}, tag=tag, stop=stop)


def taheri_design_dag(
    graph: nx.DiGraph,
    cause: str,
    effect: str,
    *,
    tag: Optional[str] = None,
    stop: Optional[int] = None,
) -> List[Result]:
    """Run the brute force implementation of the Taheri Design algorithm on a DAG.

    Identify all latent variable configurations inducible over the given DAG that result
    in an identifiable ADMG under the causal query corresponding to the given cause/effect.

    :param graph: A regular DAG
    :param cause: The node that gets perturbed.
    :param effect: The node that we're interested in.
    :param tag: The key for node data describing whether it is latent.
        If None, defaults to :data:`y0.graph.DEFAULT_TAG`.
    :param stop: Largest combination to get (None means length of the list and is the default)
    :return: A list of LV-DAG identifiability results. Will be length 2^(|V| - 2)
    """
    return _help(graph=graph, cause=cause, effect=effect, skip={cause, effect}, tag=tag, stop=stop)


def _help(graph, cause, effect, skip, *, tag: Optional[str] = None, stop: Optional[str] = None):
    return [
        _get_result(lvdag, latents, cause, effect, tag=tag)
        for latents, lvdag in iterate_lvdags(graph, skip=skip, tag=tag, stop=stop)
    ]


def _get_result(lvdag, latents, cause, effect, *, tag: Optional[str] = None) -> Result:
    # Book keeping
    pre_nodes, pre_edges = lvdag.number_of_nodes(), lvdag.number_of_edges()

    # Apply the robin evans algorithms
    simplify_latent_dag(lvdag, tag=tag)
    post_nodes, post_edges = lvdag.number_of_nodes(), lvdag.number_of_edges()

    # Convert the latent variable DAG to an ADMG
    admg = admg_from_latent_variable_dag(lvdag, tag=tag)

    if cause not in admg.vertices:
        raise KeyError(f'ADMG missing cause: {cause}')
    if effect not in admg.vertices:
        raise KeyError(f'ADMG missing effect: {effect}')

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


def iterate_lvdags(
    graph: nx.DiGraph,
    skip: Optional[Iterable[str]] = None,
    *,
    tag: Optional[str] = None,
    stop: Optional[str] = None,
) -> Iterable[Tuple[Set[str], nx.DiGraph]]:
    """Iterate over all possible latent variable configurations for the given graph.

    :param graph: A regularDAG
    :param skip: Nodes to skip in the power set of all possible latent variables. Often, the cause and effect from
        a causal query will be used here to avoid setting them as latent (since they can not be).
    :param tag: The key for node data describing whether it is latent.
        If None, defaults to :data:`y0.graph.DEFAULT_TAG`.
    :param stop: Largest combination to get (None means length of the list and is the default)
    :yields: latent variable DAGs for all possible latent variable configurations over the original DAG
    """
    nodes = set(graph)
    if skip:
        nodes.difference_update(skip)
    if stop is None:
        stop = len(nodes) - 1
    it = powerset(sorted(nodes), stop=stop, reverse=True, use_tqdm=True, tqdm_kwargs=dict(desc='LV powerset'))
    for latents in it:
        yv = graph.copy()
        set_latent(yv, latents, tag=tag)
        yield latents, yv


def draw_results(
    results: Iterable[Result],
    path: Union[str, Path, Iterable[str], Iterable[Path]],
    ncols: int = 10,
    x_ratio: float = 4.2,
    y_ratio: float = 4.2,
    max_size: Optional[int] = None
) -> None:
    """Draw identifiable ADMGs to a file."""
    import matplotlib.pyplot as plt

    if isinstance(path, str):
        path = [path]

    rendered_results = [result for result in results if result.identifiable]
    if max_size is not None:
        rendered_results = [
            result
            for result in results
            if len(result.admg.vertices) - len(result.latents) < max_size
        ]

    logger.debug('rendering %s identifiable queries', rendered_results)

    nrows = 1 + len(rendered_results) // ncols
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * x_ratio, nrows * y_ratio))
    it = itt.zip_longest(axes.ravel(), tqdm(rendered_results, desc='generating chart'))
    for i, (ax, result) in enumerate(it, start=1):
        if result is None:
            ax.axis('off')
        else:
            mixed_graph = NxMixedGraph.from_admg(result.admg)
            title = f'{i}) ' + ', '.join(result.latents)
            mixed_graph.draw(ax=ax, title='\n'.join(textwrap.wrap(title, width=45)))

    plt.tight_layout()

    for _path in tqdm(path, desc='saving'):
        logger.debug('saving to %s', _path)
        fig.savefig(_path, dpi=400)


def print_results(results: List[Result], file=None) -> None:
    """Print a set of results."""
    rows = [
        (i, result, post_nodes - pre_nodes, post_edges - pre_edges, len(latents), ', '.join(latents))
        for i, (result, pre_nodes, pre_edges, post_nodes, post_edges, latents, _, _) in enumerate(results, start=1)
    ]
    print(tabulate(rows, headers=['Row', 'ID?', 'Node Simp.', 'Edge Simp.', 'N', 'Latents']), file=file)


@click.command()
@verbose_option
def main():
    """Run the algorithm on the IGF graph with the PI3K/Erk example."""
    import pystow
    from y0.resources import VIRAL_PATHOGENESIS_PATH
    from y0.graph import from_causalfusion_path

    viral_pathogenesis_admg = from_causalfusion_path(VIRAL_PATHOGENESIS_PATH)

    results = taheri_design_admg(viral_pathogenesis_admg, cause='EGFR', effect='CytokineStorm', stop=3)
    draw_results(results, [
        # pystow.join('y0', 'viral_pathogenesis_egfr.png'),
        pystow.join('y0', 'viral_pathogenesis_egfr.svg'),
    ])

    results = taheri_design_admg(viral_pathogenesis_admg, cause=r'sIL6R\alpha', effect='CytokineStorm', stop=2)
    draw_results(results, [
        # pystow.join('y0', 'viral_pathogenesis_sIL6ra.png'),
        pystow.join('y0', 'viral_pathogenesis_sIL6ra.svg'),
    ])

    from y0.case_studies import igf_graph
    results = taheri_design_dag(igf_graph, cause='PI3K', effect='Erk')
    print_results(results)
    draw_results(results, [
        # pystow.join('y0', 'ifg_identifiable_configs.png'),
        pystow.join('y0', 'ifg_identifiable_configs.svg'),
    ])


if __name__ == '__main__':
    main()
