"""Implementation of ACE estimators."""

from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_linear_model import GLM

from y0.dsl import Variable
from y0.graph import NxMixedGraph, get_district_and_predecessors, is_p_fixable

__all__ = [
    "get_primal_ipw_ace",
    "get_primal_ipw_point_estimate",
    "get_beta_primal",
]

#: The list of Ananke estimators implemented in
#: :class:`ananke.estimation.counterfactual_mean.CausalEffect`
ANANKE_ESTIMATORS = Literal[
    "ipw",
    "gformula",
    "aipw",
    "eff-aipw",
    "p-ipw",
    "d-ipw",
    "apipw",
    "eff-apipw",
    "n-ipw",
    "anipw",
]


def get_primal_ipw_ace(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatment: Variable,
    outcome: Variable,
    *,
    report_log_odds: bool = False,
) -> float:
    """Get ACE using the primal IPW estimator."""
    if not is_p_fixable(graph, treatment):
        raise ValueError(f"graph must be p-fixable under treatment {treatment} to run Primal IPW")
    point_estimate_t1 = get_primal_ipw_point_estimate(
        graph=graph, data=data, treatment_value=1, treatment=treatment, outcome=outcome
    )
    point_estimate_t0 = get_primal_ipw_point_estimate(
        graph=graph, data=data, treatment_value=0, treatment=treatment, outcome=outcome
    )
    state_space_map = get_state_space_map(data)
    if not report_log_odds:
        return point_estimate_t1 - point_estimate_t0
    if state_space_map[outcome] != "binary":
        raise ValueError(f"can not report log odds ratio on non-binary outcome {outcome}")
    return _log_odd_ratio(point_estimate_t1, point_estimate_t0)


def get_primal_ipw_point_estimate(
    *,
    data: pd.DataFrame,
    graph: NxMixedGraph,
    treatment: Variable,
    treatment_value: int | float,
    outcome: Variable,
) -> float:
    """Estimate the counterfactual mean E[Y(t)] with the Primal IPW estimator on p-fixable graphs."""
    beta_primal = get_beta_primal(
        data=data,
        graph=graph,
        treatment=treatment,
        treatment_value=treatment_value,
        outcome=outcome,
    )
    return cast(float, np.mean(beta_primal).item())


def get_beta_primal(
    *,
    data: pd.DataFrame,
    graph: NxMixedGraph,
    treatment: Variable,
    outcome: Variable,
    treatment_value: int | float,
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
    mp_t = get_district_and_predecessors(graph, [treatment])

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


def fit_binary_model(data: pd.DataFrame, formula: str) -> GLM:
    """Fit a binary general linear model."""
    return GLM.from_formula(
        formula,
        data=data,
        family=Binomial(),
    ).fit()


def fit_continuous_glm(data: pd.DataFrame, formula: str) -> GLM:
    """Fit a continuous general linear model."""
    return GLM.from_formula(
        formula,
        data=data,
        family=Gaussian(),
    ).fit()


def get_conditional_probability_formula_for_node(graph: NxMixedGraph, node: Variable) -> str:
    """Generate the conditional probability formula for a given node."""
    district_and_predecessors = get_district_and_predecessors(graph, [node])
    formula = node.name + "~" + "+".join(node.name for node in district_and_predecessors)
    return formula


def get_state_space_map(data: pd.DataFrame) -> dict[Variable, Literal["binary", "continuous"]]:
    """Get a dictionary from each variable to its type."""
    binary_set = {0, 1}
    return {
        Variable(column): "binary" if binary_set.issuperset(data[column].unique()) else "continuous"
        for column in data.columns
    }


def _log_odd_ratio(point_estimate_t1: float, point_estimate_t0: float) -> float:
    return cast(
        float,
        np.log(
            (point_estimate_t1 / (1 - point_estimate_t1))
            / (point_estimate_t0 / (1 - point_estimate_t0))
        ),
    )
