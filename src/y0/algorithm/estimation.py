"""Estimation of probabilities generated from identification."""

import itertools
from contextlib import redirect_stdout
from typing import List, Literal, Optional, Union

import pandas as pd

from y0.dsl import CounterfactualVariable, P, Variable
from y0.graph import NxMixedGraph
from y0.identify import is_identifiable

__all__ = [
    "estimate_causal_effect",
    "is_a_fixable",
    "aipw",
    "is_p_fixable",
    "apipw",
]


def estimate_causal_effect(
    graph: NxMixedGraph,
    treatment: Variable,
    outcome: Variable,
    data: pd.DataFrame,
    *,
    query_type: Literal["ate", "expectation", "probability"],
    conditions: Optional[List[Variable]] = None,
) -> float:
    if query_type == "ate":
        return estimate_ate(
            graph=graph, treatment=treatment, outcome=outcome, data=data, conditions=conditions
        )
    elif query_type == "expectation":
        raise NotImplementedError
    elif query_type == "probability":
        raise NotImplementedError
    else:
        raise TypeError


def estimate_ate(
    graph: NxMixedGraph,
    treatment: Union[Variable, List[Variable]],
    outcome: Union[Variable, List[Variable]],
    data: pd.DataFrame,
    *,
    conditions: Optional[List[Variable]] = None,
    bootstraps: int = 0,
    alpha: float = 0.05,
) -> float:
    """Estimate the average treatment effect."""
    if conditions is not None:
        raise NotImplementedError("can not yet handle conditional queries")
    if isinstance(treatment, list) or isinstance(outcome, list):
        raise NotImplementedError("can not yet handle multiple treatments nor outcomes")
    if isinstance(treatment, CounterfactualVariable) or isinstance(outcome, CounterfactualVariable):
        raise NotImplementedError("can not yet handle counterfactual treatments nor outcomes")

    ananke_graph = graph.to_admg()
    from ananke.estimation import CausalEffect

    with redirect_stdout(None):
        # redirect stdout gets rid of the unnecessary printing from Ananke,
        # e.g., when CausalEffect says what estimators can be used. We take
        # care of that explicitly below
        causal_effect = CausalEffect(ananke_graph, treatment.name, outcome.name)

    # explicitly encode suggestions from Ananke
    if is_a_fixable(graph, treatment):
        if is_markov_blanket_shielded(graph):
            estimator = "eff-aipw"
        else:
            estimator = "aipw"
    elif is_p_fixable(graph, treatment):
        if is_markov_blanket_shielded(graph):
            estimator = "eff-apipw"
        else:
            estimator = "apipw"
    elif is_identifiable(graph, P(outcome @ ~treatment)):
        estimator = "anipw"
    else:
        raise RuntimeError("Effect can not be estimated")

    return causal_effect.compute_effect(
        data, estimator=estimator, n_bootstraps=bootstraps, alpha=alpha
    )


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
            if any([_markov_blanket_overlap(graph, u, v), _markov_blanket_overlap(graph, v, u)]):
                return False
    return True


def _markov_blanket_overlap(graph: NxMixedGraph, u: Variable, v: Variable) -> bool:
    return u in graph.get_markov_blanket(v) or v in graph.get_markov_blanket(u)


def is_a_fixable(graph: NxMixedGraph, treatments: Union[Variable, List[Variable]]) -> bool:
    """Check if the treatments are a-fixable.

    A treatment is said to be a-fixable if it can be fixed by removing a single directed edge from the graph.
    In other words, a treatment is a-fixable if it has exactly one descendant in its district.

    This code was adapted from :mod:`ananke` ananke code at:
    https://gitlab.com/causal/ananke/-/blob/dev/ananke/estimation/counterfactual_mean.py?ref_type=heads#L58-65

    :param graph: A NxMixedGraph
    :param treatments: A list of treatments
    :raises NotImplementedError: a-fixability on multiple treatments is an open research question
    :returns: bool
    """
    if isinstance(treatments, list):
        raise NotImplementedError(
            "a-fixability on multiple treatments is an open research question"
        )
    descendants = graph.descendants_inclusive(treatments)
    descendants_in_district = graph.get_district(treatments).intersection(descendants)
    return 1 == len(descendants_in_district)


def aipw():
    pass


def is_p_fixable(graph: NxMixedGraph, treatments: Union[Variable, List[Variable]]) -> bool:
    """Check if the treatments are p-fixable.

    This code was adapted from :mod:`ananke` ananke code at:
    https://gitlab.com/causal/ananke/-/blob/dev/ananke/estimation/counterfactual_mean.py?ref_type=heads#L85-92

    :param graph: A NxMixedGraph
    :param treatments: A list of treatments
    :raises NotImplementedError: p-fixability on multiple treatments is an open research question
    :returns: bool
    """
    if isinstance(treatments, list):
        raise NotImplementedError(
            "p-fixability on multiple treatments is an open research question"
        )
    children = graph.directed.successors(treatments)
    children_in_district = graph.get_district(treatments).intersection(children)
    return 0 == len(children_in_district)


def apipw():
    pass


def df_covers_graph(graph: NxMixedGraph, df: pd.DataFrame) -> bool:
    """Check if all variables in the graph appear as columns in the dataframe."""
    if graph.is_counterfactual():
        raise NotImplementedError
    graph_names = {node.name for node in graph.nodes()}
    data_names = set(df.columns)
    return graph_names.issubset(data_names)
