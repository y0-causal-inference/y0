"""Estimation of probabilities generated from identification."""

import itertools
from contextlib import redirect_stdout
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.api import GLM
from statsmodels.genmod.families import Binomial, Gaussian

from y0.dsl import CounterfactualVariable, P, Variable
from y0.graph import NxMixedGraph
from y0.identify import is_identifiable

__all__ = [
    "estimate_causal_effect",
    "is_a_fixable",
    "is_p_fixable",
    "is_markov_blanket_shielded",
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

    return _ananke_compute_effect(
        graph=graph,
        data=data,
        treatment=treatment,
        outcome=outcome,
        estimator=estimator,
        bootstraps=bootstraps,
        alpha=alpha,
    )


def _ananke_compute_effect(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatment: Variable,
    outcome: Variable,
    estimator: str,
    bootstraps: int | None = None,
    alpha: float | None = None,
) -> float:
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
    children = set(graph.directed.successors(treatments))
    children_in_district = graph.get_district(treatments).intersection(children)
    return 0 == len(children_in_district)


def _log_odds(point_estimate_t1, point_estimate_t0) -> float:
    return np.log(
        (point_estimate_t1 / (1 - point_estimate_t1))
        / (point_estimate_t0 / (1 - point_estimate_t0))
    )


def get_primal_ipw_ace(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatment: Variable,
    outcome: Variable,
    report_log_odds: bool = True,
) -> float:
    """Get ACE using the primal IPW estimator."""
    point_estimate_t1 = get_primal_ipw_point_estimate(
        graph=graph, data=data, treatment_value=1, treatment=treatment, outcome=outcome
    )
    point_estimate_t0 = get_primal_ipw_point_estimate(
        graph=graph, data=data, treatment_value=0, treatment=treatment, outcome=outcome
    )
    state_space_map = get_state_space_map(data)
    # if Y is binary report log of odds ration, if Y is continuous report ACE
    if report_log_odds and state_space_map[outcome] == "binary":
        return _log_odds(point_estimate_t1, point_estimate_t0)
    else:
        return point_estimate_t1 - point_estimate_t0


def get_primal_ipw_point_estimate(
    *,
    data: pd.DataFrame,
    graph: NxMixedGraph,
    treatment: Variable,
    treatment_value,
    outcome: Variable,
) -> float:
    """Estimate the counterfactual mean E[Y(t)] with the Primal IPW estimator.

    This can be applied to graphs that are not a-fixable but are p-fixable.
    """
    beta_primal = get_beta_primal(
        data=data,
        graph=graph,
        treatment=treatment,
        treatment_value=treatment_value,
        outcome=outcome,
    )
    return np.mean(beta_primal).item()


def fit_binary_model(data, formula, weights=None) -> GLM:
    """Fit a binary general linear model."""
    return GLM.from_formula(
        formula,
        data=data,
        family=Binomial(),
        freq_weights=weights,
    ).fit()


def fit_continuous_glm(data, formula, weights=None) -> GLM:
    """Fit a continuous general linear model."""
    return GLM.from_formula(
        formula,
        data=data,
        family=Gaussian(),
        freq_weights=weights,
    ).fit()


def get_conditional_probability_formula_for_node(graph: NxMixedGraph, node: Variable) -> str:
    """Generates the conditional probability formula for a given node based on its markov pillow."""
    markov_pillow = graph.get_markov_pillow([node])
    markov_pillow_names = [node.name for node in markov_pillow]
    formula = node.name + "~" + "+".join(markov_pillow_names)
    return formula


def get_state_space_map(data: pd.DataFrame) -> Dict[Variable, Literal["binary", "continuous"]]:
    """Get a dictionary from each variable to its type."""
    from typing import Any
    state_space_map = dict()
    binary_set = {0, 1}
    for column_name, column_values in data.items():
        if binary_set.issuperset(column_values.unique()):
            state_space_map[Variable(column_name)] = "binary"
        else:
            state_space_map[Variable(column_name)] = "continuous"
    return state_space_map


def get_beta_primal(
    *,
    data: pd.DataFrame,
    graph: NxMixedGraph,
    treatment: Variable,
    outcome: Variable,
    treatment_value,
) -> np.array:
    """Return the beta primal value for each row in the data.

    This code was adapted from :mod:`ananke` ananke code at:
    https://gitlab.com/causal/ananke/-/blob/dev/ananke/estimation/counterfactual_mean.py?ref_type=heads#L408-513

    :param data: Given data
    :param graph: A NxMixedGraph
    :param treatment: Given treatment
    :param outcome: Given outcome
    :param treatment_value: Given treatment value
    :returns: Beta primal value for each row in the data
    """
    state_space_map = get_state_space_map(data)

    # extract the outcome
    y = data[outcome.name]

    # c := pre-treatment vars and l := post-treatment vars in district of treatment
    pre_treatment_vars = graph.pre(treatment)
    post_treatment_vars = set(graph.nodes()).difference(pre_treatment_vars)
    post_treatment_vars_in_district = post_treatment_vars.intersection(
        graph.get_district(treatment)
    )

    # create copies of the data with treatment assignments t=0 and t=1
    data_t1 = data.copy()
    data_t1[treatment.name] = 1
    data_t0 = data.copy()
    data_t0[treatment.name] = 0

    indices = data[treatment.name] == treatment_value

    # prob: stores \prod_{li in l} p(li | mp(li))
    # prob_t1: stores \prod_{li in l} p(li | mp(li)) at t=1
    # prob_t0: stores \prod_{li in l} p(li | mp(li)) at t=0

    indices_t0 = data.index[data[treatment.name] == 0]
    mp_t = graph.get_markov_pillow([treatment])

    if len(mp_t) != 0:
        formula = get_conditional_probability_formula_for_node(graph=graph, node=treatment)
        model = fit_binary_model(data, formula)
        prob = model.predict(data)
        prob[indices_t0] = 1 - prob[indices_t0]
        prob_t1 = model.predict(data)
    else:
        prob = np.ones(len(data)) * np.mean(data[treatment.name])
        prob[indices_t0] = 1 - prob[indices_t0]
        prob_t1 = np.ones(len(data)) * np.mean(data[treatment.name])

    prob_t0 = 1 - prob_t1

    # iterate over vertices in l (except the treatment and outcome)
    for v in post_treatment_vars_in_district.difference([treatment, outcome]):
        # fit v | mp(v)
        formula = get_conditional_probability_formula_for_node(graph=graph, node=v)

        # p(v =v | .), p(v = v | . , t=1), p(v = v | ., t=0)
        if state_space_map[v] == "binary":
            model = fit_binary_model(data, formula)
            prob_v = model.predict(data)
            prob_v_t1 = model.predict(data_t1)
            prob_v_t0 = model.predict(data_t0)

            indices_v0 = data.index[data[v.name] == 0]

            # p(v | .), p(v | ., t=t)
            prob_v[indices_v0] = 1 - prob_v[indices_v0]
            prob_v_t1[indices_v0] = 1 - prob_v_t1[indices_v0]
            prob_v_t0[indices_v0] = 1 - prob_v_t0[indices_v0]

        else:
            model = fit_continuous_glm(data, formula)
            e_v = model.predict(data)
            e_v_t1 = model.predict(data_t1)
            e_v_t0 = model.predict(data_t0)

            std = np.std(data[v.name] - e_v)
            prob_v = norm.pdf(data[v.name], loc=e_v, scale=std)
            prob_v_t1 = norm.pdf(data[v.name], loc=e_v_t1, scale=std)
            prob_v_t0 = norm.pdf(data[v.name], loc=e_v_t0, scale=std)

        prob *= prob_v
        prob_t1 *= prob_v_t1
        prob_t0 *= prob_v_t0

    # special case when the outcome is in l
    if outcome in post_treatment_vars_in_district:
        # fit a binary/continuous model for y | mp(y)

        formula = get_conditional_probability_formula_for_node(graph=graph, node=outcome)

        if state_space_map[outcome] == "binary":
            model = fit_binary_model(data, formula)
        else:
            model = fit_continuous_glm(data, formula)

        # predict the outcome and adjust numerator of primal accordingly
        yhat_t1 = model.predict(data_t1)
        yhat_t0 = model.predict(data_t0)
        prob_sumt = prob_t1 * yhat_t1 + prob_t0 * yhat_t0
        beta_primal = indices * (prob_sumt / prob)
    else:
        prob_sumt = prob_t1 + prob_t0
        beta_primal = indices * (prob_sumt / prob) * y

    return beta_primal


def df_covers_graph(graph: NxMixedGraph, df: pd.DataFrame) -> bool:
    """Check if all variables in the graph appear as columns in the dataframe."""
    if graph.is_counterfactual():
        raise NotImplementedError
    graph_names = {node.name for node in graph.nodes()}
    data_names = set(df.columns)
    return graph_names.issubset(data_names)
