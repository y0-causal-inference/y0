"""Estimation of probabilities generated from identification."""

from __future__ import annotations

from contextlib import redirect_stdout
from typing import List, Optional, Union

import pandas as pd

from y0.algorithm.identify import identify_outcomes
from y0.dsl import CounterfactualVariable, Variable
from y0.graph import (
    NxMixedGraph,
    is_a_fixable,
    is_markov_blanket_shielded,
    is_p_fixable,
)

__all__ = [
    "estimate_ace",
]


def estimate_ace(
    graph: NxMixedGraph,
    treatments: Union[Variable, List[Variable]],
    outcomes: Union[Variable, List[Variable]],
    data: pd.DataFrame,
    *,
    conditions: Optional[List[Variable]] = None,
    bootstraps: int | None = None,
    alpha: float | None = None,
    estimator: Optional[str] = None,
) -> float:
    """Estimate the average treatment effect."""
    if conditions is not None:
        raise NotImplementedError("can not yet handle conditional queries")
    if isinstance(treatments, list):
        if len(treatments) == 1:
            treatments = treatments[0]
        else:
            raise NotImplementedError(f"can not yet handle multiple treatments: {treatments}")
    if isinstance(outcomes, list):
        if len(outcomes) == 1:
            outcomes = outcomes[0]
        else:
            raise NotImplementedError(f"can not yet handle multiple outcomes: {outcomes}")
    if isinstance(treatments, CounterfactualVariable) or isinstance(
        outcomes, CounterfactualVariable
    ):
        raise NotImplementedError("can not yet handle counterfactual treatments nor outcomes")
    if not df_covers_graph(graph=graph, data=data):
        raise ValueError

    # explicitly encode suggestions from Ananke
    if estimator is not None:
        pass
    elif is_a_fixable(graph, treatments):
        if is_markov_blanket_shielded(graph):
            estimator = "eff-aipw"
        else:
            estimator = "aipw"
    elif is_p_fixable(graph, treatments):
        if is_markov_blanket_shielded(graph):
            estimator = "eff-apipw"
        else:
            estimator = "apipw"
    elif identify_outcomes(graph=graph, treatments=treatments, outcomes=outcomes) is not None:
        estimator = "anipw"
    else:
        raise RuntimeError("Graph is not identifiable; effect can not be estimated")

    return ananke_average_causal_effect(
        graph=graph,
        data=data,
        treatment=treatments,
        outcome=outcomes,
        estimator=estimator,
        bootstraps=bootstraps,
        alpha=alpha,
    )


def df_covers_graph(graph: NxMixedGraph, data: pd.DataFrame) -> bool:
    """Check if all variables in the graph appear as columns in the dataframe."""
    if graph.is_counterfactual():
        raise NotImplementedError
    graph_names = {node.name for node in graph.nodes()}
    data_names = set(data.columns)
    return graph_names.issubset(data_names)


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
