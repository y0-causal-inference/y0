# -*- coding: utf-8 -*-

"""An implementation of Sara Taheri's algorithm for using causal queries for experimental design.

.. seealso:: https://docs.google.com/presentation/d/1klBOjGtRkXOMSDgOCLChBTBJZ0dFxJPn9IPRLAlv_N8/edit?usp=sharing
"""

import itertools as itt
import logging
import textwrap
from pathlib import Path
from typing import Collection, Iterable, List, NamedTuple, Optional, Set, Tuple, Union

import click
import networkx as nx
from more_click import verbose_option
from tabulate import tabulate
from tqdm.auto import tqdm

from y0.algorithm.identify import Identification, Unidentifiable, identify
from y0.algorithm.simplify_latent import simplify_latent_dag
from y0.complexity import complexity
from y0.dsl import Expression, P, Variable
from y0.graph import DEFAULT_TAG, NxMixedGraph
from y0.mutate import canonicalize
from y0.util.combinatorics import powerset

__all__ = [
    "taheri_design_admg",
    "taheri_design_dag",
    "Result",
    "draw_results",
]

logger = logging.getLogger(__name__)


class Result(NamedTuple):
    """Results from the LV-DAG check."""

    identifiable: bool
    #: The estimand returned from the related identification algorithm. Is none if not identifiable.
    estimand: Optional[Expression]
    pre_nodes: int
    pre_edges: int
    post_nodes: int
    post_edges: int
    latents: List[Variable]
    observed: List[Variable]
    lvdag: nx.DiGraph
    admg: NxMixedGraph


def taheri_design_admg(
    graph: NxMixedGraph,
    cause: Union[str, Variable],
    effect: Union[str, Variable],
    *,
    tag: Optional[str] = None,
    stop: Optional[int] = None,
) -> List[Result]:
    r"""Run the brute force implementation of the Taheri Design algorithm on an ADMG.

    :param graph: An ADMG
    :param cause: The node that gets perturbed.
    :param effect: The node that we're interested in.
    :param tag: The key for node data describing whether it is latent.
        If None, defaults to :data:`y0.graph.DEFAULT_TAG`.
    :param stop: Largest combination to get (None means length of the list and is the default)
    :return: A list of LV-DAG identifiability results. Will be length $2^{(\|V\| - 2 - # bidirected edges)}$
    """
    if tag is None:
        tag = DEFAULT_TAG
    cause = Variable.norm(cause)
    effect = Variable.norm(effect)
    dag = graph.to_latent_variable_dag(tag=tag)
    fixed_latent = {node for node, data in dag.nodes(data=True) if data[tag]}
    return _help(
        graph=dag,
        cause=cause,
        effect=effect,
        fixed_observed={cause, effect},
        fixed_latent=fixed_latent,
        tag=tag,
        stop=stop,
    )


def taheri_design_dag(
    graph: nx.DiGraph,
    cause: Union[str, Variable],
    effect: Union[str, Variable],
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
    :return: A list of LV-DAG identifiability results. Will be length $2^(|V| - 2)$
    """
    cause = Variable.norm(cause)
    effect = Variable.norm(effect)
    return _help(
        graph=graph,
        cause=cause,
        effect=effect,
        fixed_observed={cause, effect},
        tag=tag,
        stop=stop,
    )


def _help(
    graph: nx.DiGraph,
    cause: Variable,
    effect: Variable,
    *,
    fixed_observed: Optional[Collection[Variable]] = None,
    fixed_latent: Optional[Collection[Variable]] = None,
    tag: Optional[str] = None,
    stop: Optional[int] = None,
) -> List[Result]:
    return [
        _get_result(
            lvdag=lvdag,
            latents=latents,
            observed=observed,
            cause=cause,
            effect=effect,
            tag=tag,
        )
        for latents, observed, lvdag in iterate_lvdags(
            graph,
            fixed_observed=fixed_observed,
            fixed_latents=fixed_latent,
            tag=tag,
            stop=stop,
        )
    ]


def _get_result(
    lvdag: nx.DiGraph,
    latents: Collection[Variable],
    observed: Collection[Variable],
    cause: Variable,
    effect: Variable,
    *,
    tag: Optional[str] = None,
) -> Result:
    # Book keeping
    pre_nodes, pre_edges = lvdag.number_of_nodes(), lvdag.number_of_edges()

    # Apply the robin evans algorithms
    simplify_latent_dag(lvdag, tag=tag)
    post_nodes, post_edges = lvdag.number_of_nodes(), lvdag.number_of_edges()

    # Convert the latent variable DAG to an ADMG
    admg = NxMixedGraph.from_latent_variable_dag(lvdag, tag=tag)

    if cause not in admg.nodes():
        raise KeyError(f"ADMG missing cause: {cause}")
    if effect not in admg.nodes():
        raise KeyError(f"ADMG missing effect: {effect}")

    # Check if the ADMG is identifiable under the (simple) causal query
    query = P(effect @ ~cause)
    try:
        estimand: Optional[Expression] = canonicalize(
            identify(Identification.from_expression(graph=admg, query=query))
        )
    except Unidentifiable:
        estimand = None

    return Result(
        estimand is not None,
        estimand=estimand,
        pre_nodes=pre_nodes,
        pre_edges=pre_edges,
        post_nodes=post_nodes,
        post_edges=post_edges,
        latents=sorted(latents),
        observed=sorted(observed),
        lvdag=lvdag,
        admg=admg,
    )


def iterate_lvdags(
    graph: nx.DiGraph,
    fixed_observed: Optional[Collection[Variable]] = None,
    fixed_latents: Optional[Collection[Variable]] = None,
    *,
    tag: Optional[str] = None,
    stop: Optional[int] = None,
) -> Iterable[Tuple[Set[Variable], Set[Variable], nx.DiGraph]]:
    """Iterate over all possible latent variable configurations for the given graph.

    :param graph: A regular DAG
    :param fixed_observed: Nodes to skip in the power set of all possible latent variables. Often, the cause and
        effect from a causal query will be used here to avoid setting them as latent (since they can not be).
    :param fixed_latents: Nodes to skip in the power set of all possible latent variables. Often, latent nodes
        from ADMG->LV-DAG conversion will go here.
    :param tag: The key for node data describing whether it is latent.
        If None, defaults to :data:`y0.graph.DEFAULT_TAG`.
    :param stop: Largest combination to get (None means length of the list and is the default)
    :yields: latent variable DAGs for all possible latent variable configurations over the original DAG
    """
    if tag is None:
        tag = DEFAULT_TAG

    fixed_observed = set() if not fixed_observed else set(fixed_observed)
    fixed_latents = set() if not fixed_latents else set(fixed_latents)

    inducible_nodes: Set[Variable] = set(graph)
    inducible_nodes.difference_update(fixed_observed)
    inducible_nodes.difference_update(fixed_latents)

    if stop is None:
        stop = len(inducible_nodes) - 1
    it: Iterable[Set[Variable]] = map(
        set,
        powerset(
            sorted(inducible_nodes),
            stop=stop,
            reverse=True,
            use_tqdm=True,
            tqdm_kwargs=dict(desc="LV powerset"),
        ),
    )

    graph = graph.copy()
    for node in fixed_observed:
        graph.nodes[node][tag] = False
    for node in fixed_latents:
        graph.nodes[node][tag] = True
    for induced_latents in it:
        yv = graph.copy()
        for node in inducible_nodes:
            yv.nodes[node][tag] = node in induced_latents
        yield induced_latents, inducible_nodes - induced_latents, yv  # type:ignore


def draw_results(
    results: Iterable[Result],
    path: Union[str, Path, Iterable[str], Iterable[Path]],
    ncols: int = 10,
    x_ratio: float = 4.2,
    y_ratio: float = 4.2,
    max_size: Optional[int] = None,
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
            if len(result.admg.nodes()) - len(result.latents) < max_size
        ]

    logger.debug("rendering %s identifiable queries", rendered_results)

    nrows = 1 + len(rendered_results) // ncols
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * x_ratio, nrows * y_ratio))
    it = itt.zip_longest(axes.ravel(), tqdm(rendered_results, desc="generating chart"))
    for i, (ax, result) in enumerate(it, start=1):
        if result is None:
            ax.axis("off")
        else:
            mixed_graph = result.admg
            title = f"{i}) Latent: " + ", ".join(f"${v.to_latex()}$" for v in result.latents)
            estimand_complexity = complexity(result.estimand)
            if result.estimand is not None:
                title += f"\n${result.estimand.to_latex()}$\n$C={estimand_complexity}$"
            mixed_graph.draw(ax=ax, title="\n".join(textwrap.wrap(title, width=45)))

    fig.tight_layout()

    for _path in tqdm(path, desc="saving"):
        logger.info("saving to %s", _path)
        fig.savefig(_path, dpi=400)


def print_results(results: List[Result], file=None) -> None:
    """Print a set of results."""
    rows = [
        (
            i,
            result.identifiable,
            result.post_nodes - result.pre_nodes,
            result.post_edges - result.pre_edges,
            len(result.latents),
            ", ".join(f"${v.to_latex()}$" for v in result.latents),
        )
        for i, result in enumerate(results, start=1)
    ]
    print(  # noqa:T201
        tabulate(rows, headers=["Row", "ID?", "Node Simp.", "Edge Simp.", "N", "Latents"]),
        file=file,
    )


@click.command()
@verbose_option
def main():
    """Run the algorithm on the IGF graph with the PI3K/Erk example."""
    import pystow

    from y0.examples import igf_example

    results = taheri_design_dag(igf_example.graph.directed, cause="PI3K", effect="Erk", stop=3)
    # print_results(results)
    draw_results(
        results,
        [
            pystow.join("y0", name="ifg_identifiable_configs.png"),
            pystow.join("y0", name="ifg_identifiable_configs.svg"),
        ],
        ncols=3,
    )
    import sys

    sys.exit(0)

    from y0.graph import NxMixedGraph
    from y0.resources import VIRAL_PATHOGENESIS_PATH

    viral_pathogenesis_admg = NxMixedGraph.from_causalfusion_path(VIRAL_PATHOGENESIS_PATH)

    results = taheri_design_admg(
        viral_pathogenesis_admg, cause="EGFR", effect="CytokineStorm", stop=5
    )
    draw_results(
        results,
        [
            pystow.join("y0", name="viral_pathogenesis_egfr.png"),
            pystow.join("y0", name="viral_pathogenesis_egfr.svg"),
        ],
    )

    results = taheri_design_admg(
        viral_pathogenesis_admg, cause=r"sIL6R\alpha", effect="CytokineStorm", stop=5
    )
    draw_results(
        results,
        [
            pystow.join("y0", name="viral_pathogenesis_sIL6ra.png"),
            pystow.join("y0", name="viral_pathogenesis_sIL6ra.svg"),
        ],
    )


if __name__ == "__main__":
    main()
