"""Estimation of probabilities generated from identification."""

from __future__ import annotations

from contextlib import redirect_stdout
from typing import List, Literal, Optional, Union

import pandas as pd

from y0.dsl import CounterfactualVariable, P, Variable
from y0.graph import (
    NxMixedGraph,
    is_a_fixable,
    is_markov_blanket_shielded,
    is_p_fixable,
)
from y0.identify import is_identifiable

__all__ = [
    "estimate_causal_effect",
    "df_covers_graph",
    "estimate_ate",
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
    """Estimate the causal effect of a treatment on an outcome."""
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
    bootstraps: int | None = None,
    alpha: float | None = None,
    estimator: Optional[str] = None,
) -> float:
    """Estimate the average treatment effect."""
    if conditions is not None:
        raise NotImplementedError("can not yet handle conditional queries")
    if isinstance(treatment, list) or isinstance(outcome, list):
        raise NotImplementedError("can not yet handle multiple treatments nor outcomes")
    if isinstance(treatment, CounterfactualVariable) or isinstance(outcome, CounterfactualVariable):
        raise NotImplementedError("can not yet handle counterfactual treatments nor outcomes")
    if not df_covers_graph(graph=graph, data=data):
        raise ValueError

    # explicitly encode suggestions from Ananke
    if estimator is not None:
        pass
    elif is_a_fixable(graph, treatment):
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

    return ananke_average_causal_effect(
        graph=graph,
        data=data,
        treatment=treatment,
        outcome=outcome,
        estimator=estimator,
        bootstraps=bootstraps,
        alpha=alpha,
    )


def df_covers_graph(graph: NxMixedGraph, df: pd.DataFrame) -> bool:
    """Check if all variables in the graph appear as columns in the dataframe."""
    if graph.is_counterfactual():
        raise NotImplementedError
    graph_names = {node.name for node in graph.nodes()}
    data_names = set(df.columns)
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
