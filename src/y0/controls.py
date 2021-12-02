# -*- coding: utf-8 -*-

"""Predicates for good, bad, and neutral controls."""

from .algorithm.conditional_independencies import are_d_separated
from .dsl import Probability, Variable
from .graph import NxMixedGraph

__all__ = [
    "is_good_control",
    "is_bad_control",
]


def _control_precondition(graph: NxMixedGraph, query: Probability, variable: Variable):
    if variable not in graph.nodes():
        raise ValueError(f"Test variable missing: {variable}")
    # TODO does this need to be extended to check that the
    #  query and variable aren't counterfactual?


def is_good_control(graph: NxMixedGraph, query: Probability, variable: Variable) -> bool:
    """Return if the variable is a good control.

    :param graph: An ADMG
    :param query: A query in the form of ``P(Y @ X)``
    :param variable: The variable to check
    :return: If the variable is a good control
    """
    _control_precondition(graph, query, variable)
    raise NotImplementedError


def is_bad_control(graph: NxMixedGraph, query: Probability, variable: Variable) -> bool:
    """Return if the variable is a bad control.

    A bad control is a variable that does not appear in the estimand produced
    by :func:`y0.algorithm.identify.identify` when applied to a given graph
    and query.

    :param graph: An ADMG
    :param query: A query in the form of ``P(Y @ X)``
    :param variable: The variable to check
    :return: If the variable is a bad control
    """
    _control_precondition(graph, query, variable)
    raise NotImplementedError


def is_outcome_ancestor(
    graph: NxMixedGraph, cause: Variable, effect: Variable, variable: Variable
) -> bool:
    """Check if the variable is an outcome ancestor given a causal query and graph.

    > In Model 8, Z is not a confounder nor does it block any back-door paths. Likewise,
    controlling for Z does not open any back-door paths from X to Y . Thus, in terms of
    asymptotic bias, Z is a “neutral control.” Analysis shows, however, that controlling for
    Z reduces the variation of the outcome variable Y , and helps to improve the precision
    of the ACE estimate in finite samples (Hahn, 2004; White and Lu, 2011; Henckel et al.,
    2019; Rotnitzky and Smucler, 2019).

    :param graph: An ADMG
    :param cause: The intervention in the causal query
    :param effect: The outcome of the causal query
    :param variable: The variable to check
    :return: If the variable is a bad control
    """
    if variable == cause:
        return False
    judgement = are_d_separated(graph, cause, variable)
    return judgement.separated and variable in graph.ancestors_inclusive(effect)
