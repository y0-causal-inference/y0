"""Utilities for estimation."""

from __future__ import annotations

import itertools
from contextlib import redirect_stdout
from typing import Iterable, Optional, Sequence, Set

import pandas as pd

from y0.dsl import Variable
from y0.graph import NxMixedGraph


def ananke_average_causal_effect(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatment: Variable,
    outcome: Variable,
    estimator: str,
    bootstraps: int | None = None,
    alpha: float | None = None,
) -> float:
    """Estimate the average causal effect using Ananke."""
    ananke_graph = graph.to_admg()
    from ananke.estimation import CausalEffect

    with redirect_stdout(None):
        # redirect stdout gets rid of the unnecessary printing from Ananke,
        # e.g., when CausalEffect says what estimators can be used. We take
        # care of that explicitly below
        causal_effect = CausalEffect(ananke_graph, treatment.name, outcome.name)

    return causal_effect.compute_effect(
        data, estimator=estimator, n_bootstraps=bootstraps or 0, alpha=alpha or 0.05
    )


def get_district_and_predecessors(
    graph: NxMixedGraph,
    nodes: Iterable[Variable],
    topological_sort_order: Optional[Sequence[Variable]] = None,
):
    """Get the union of district, predecessors and predecessors of district for a given set of nodes.

    This code was adapted from :mod:`ananke` ananke code at:
    https://gitlab.com/causal/ananke/-/blob/dev/ananke/graphs/admg.py?ref_type=heads#L96-117

    :param graph: A NxMixedGraph
    :param nodes: List of nodes
    :param topological_sort_order: A valid topological sort order

    :return: Set corresponding to union of district, predecessors and predecessors of district of a given set of nodes
    """
    if not topological_sort_order:
        topological_sort_order = list(graph.topological_sort())

    # Get the subgraph corresponding to the nodes and nodes prior to them
    pre = graph.pre(nodes, topological_sort_order)
    sub_graph = graph.subgraph(pre + list(nodes))

    result: Set[Variable] = set()
    for node in nodes:
        result.update(sub_graph.get_district(node))
    for node in result.copy():
        result.update(sub_graph.directed.predecessors(node))
    return result - set(nodes)


def is_markov_blanket_shielded(graph: NxMixedGraph) -> bool:
    """Check if the ADMG is a Markov blanket shielded.

    Being Markov blanket (Mb) shielded means that two vertices are non-adjacent
    only when they are absent from each others' Markov blankets.

    This code was adapted from :mod:`ananke` ananke code at:
    https://gitlab.com/causal/ananke/-/blob/dev/ananke/graphs/admg.py?ref_type=heads#L381-403

    :param graph: A NxMixedGraph
    :returns: bool
    """
    # Iterate over all pairs of vertices
    for u, v in itertools.combinations(graph.nodes(), 2):
        # Check if the pair is not adjacent
        if not (
            any(
                [
                    graph.directed.has_edge(u, v),
                    graph.directed.has_edge(v, u),
                    graph.undirected.has_edge(u, v),
                ]
            )
        ):
            # If one is in the Markov blanket of the other, then it is not mb-shielded
            if _markov_blanket_overlap(graph, u, v):
                return False
    return True


def _markov_blanket_overlap(graph: NxMixedGraph, u: Variable, v: Variable) -> bool:
    return u in get_district_and_predecessors(graph, [v]) or v in get_district_and_predecessors(
        graph, [u]
    )
