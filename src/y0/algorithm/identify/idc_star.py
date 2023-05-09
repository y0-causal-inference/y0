# -*- coding: utf-8 -*-

"""Implementation of the IDC* algorithm."""

from typing import Set

from .cg import make_counterfactual_graph
from .id_star import id_star
from .utils import Unidentifiable
from ..conditional_independencies import are_d_separated
from ...dsl import Event, Expression, Variable, Zero
from ...graph import NxMixedGraph


def idc_star(graph: NxMixedGraph, outcomes: Event, conditions: Event, leonardo=0) -> Expression:
    """Run the IDC* algorithm.

    :param graph: The causal graph
    :param outcomes: The outcome variables
    :param conditions: The conditions
    :param leonardo: The number of times to apply the Leonardo rule
    :returns: An expression created by the :func:`idc_star` algorithm after simplifying the original query
    """
    # Run ID* algorithm
    try:
        if isinstance(id_star(graph, conditions), Zero):
            raise UndefinedError("The ID* algorithm returned 0, so IDC* cannot be applied.")
    except Unidentifiable:
        pass

    new_graph, new_events = make_counterfactual_graph(graph, outcomes | conditions)

    if new_events is None:
        return Zero()
    for condition in conditions:
        if rule_2_of_do_calculus_applies(new_graph, outcomes, condition):
            new_outcomes = {
                outcome.intervene(condition): value for outcome, value in outcomes.items()
            }
            new_conditions = conditions.pop(condition)
            return idc_star(graph, new_outcomes, new_conditions, leonardo + 1)
    P_prime = id_star(graph, new_events)
    return P_prime.conditional(conditions)


def rule_2_of_do_calculus_applies(
    graph: NxMixedGraph, outcomes: set[Variable], condition: Variable
) -> bool:
    r"""Check if Rule 2 of the Do-Calculus applies to the conditioned variable.

    :param identification: The identification tuple
    :param condition: The condition to check
    :returns: If rule 2 applies, see below.

    If Rule 2 of the do calculus applies to the conditioned variable, then it can be converted to a do variable.

    .. math::

        \newcommand\ci{\perp\!\!\!\perp}
        \newcommand{\ubar}[1]{\underset{\bar{}}{#1}}
        \newcommand{\obar}[1]{\overset{\bar{}}{#1}}
        \text{if } (\exists Z \in \mathbf{Z})(\mathbf{Y} \ci Z | \mathbf{X}, \mathbf{Z}
        - \{Z\})_{G_{\bar{\mathbf{X}}\ubar{Z}}} \\
        \text{then } P(\mathbf{Y}|do(\mathbf{X}),\mathbf{Z}) = P(\mathbf Y|do(\mathbf X), do(Z), \mathbf{Z} - \{Z\})
    """
    return all(
        are_d_separated(graph.remove_out_edges(condition), outcome, condition)
        for outcome in outcomes
    )
