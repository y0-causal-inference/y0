# -*- coding: utf-8 -*-

"""Predicates for good, bad, and neutral controls."""

from .algorithm.conditional_independencies import are_d_separated
from .dsl import Probability, Variable
from .graph import NxMixedGraph

__all__ = [
    "is_good_control",
    "is_bad_control",
    "is_outcome_ancestor",
    "is_middle_mediator",
]


def _control_precondition(graph: NxMixedGraph, cause: Variable, effect: Variable, variable: Variable):
    if variable not in graph.nodes():
        raise ValueError(f"Test variable missing: {variable}")
    # TODO does this need to be extended to check that the
    #  query and variable aren't counterfactual?


def is_good_control(graph: NxMixedGraph, cause: Variable, effect: Variable, variable: Variable) -> bool:
    """Return if the variable is a good control.

    Strategy:
    
    1. Get estimand using :func:`y0.algorithm.identify.identify`
    2. Check if ``variable`` appears in estimand

    :param graph: An ADMG
    :param cause: The intervention in the causal query
    :param effect: The outcome of the causal query
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
    :param cause: The intervention in the causal query
    :param effect: The outcome of the causal query
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


def is_middle_mediator(
    graph: NxMixedGraph, cause: Variable, effect: Variable, variable: Variable
) -> bool:
    """

    > At first look, Model 13 might seem similar to Model 12, and one may think that
    adjusting for Z would bias the effect estimate, by restricting variations of th
    mediator M. However, the key difference here is that Z is a cause, not an effect,
    of the mediator (and, consequently, also a cause of Y ). Thus, Model 13 is analogous
    to Model 8, and so controlling for Z will be neutral in terms of bias and may increase
    the precision of the ACE estimate in finite samples. Readers can find further
    discussion of this case in Pearl (2013).

    From Jeremy:
    Figure 9: Model 13: If exists M such that X is Ancestor of M and M is
    ancestor of Y and Z is an ancestor of M and Z _|_ Y | M  and Z _|_ X

    Strategy:

    1. First implement the helper function below:
        1. Use :func:`y0.algorithm.conditional_independencies.are_d_separated` with the ``given`` argument
           to check Z _|_ Y | M
        2. Use :func:`y0.algorithm.conditional_independencies.are_d_separated` to check Z _|_ X
        3. Magic
        4. Small profit
    2. Check if the helper returns true for any possible mediator M (loop over all variables, naive implementation ftw)
    3. Big Profit
    """
    raise NotImplementedError


def _middle_mediator_helper(
    graph: NxMixedGraph, cause: Variable, effect: Variable, mediator: Variable, variable: Variable
) -> bool:
    raise NotImplementedError
