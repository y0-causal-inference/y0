# -*- coding: utf-8 -*-

"""Implementation of the IDC algorithm."""

from typing import Optional, Set

from .id_std import identify
from .utils import Identification
from ..conditional_independencies import are_d_separated
from ...dsl import Expression, P, Sum, Variable
from ...graph import NxMixedGraph

__all__ = [
    "idc",
]


def idc(
    outcomes: Set[Variable],
    treatments: Set[Variable],
    conditions: Set[Variable],
    graph: NxMixedGraph[Variable],
    estimand: Optional[Expression] = None,
) -> Expression:
    """Run the IDC algorithm.

    :param outcomes: The outcomes in the query
    :param treatments: The treatments in the query (e.g., counterfactual variables)
    :param conditions: The conditions in the query (e.g., coming after the bar)
    :param graph: The graph
    :param estimand: If none is given, will use the joint distribution over all variables in the graph.
    :returns: An expression created by the :func:`identify` algorithm after simplifying the original query
    """
    if not estimand:
        estimand = P(graph.nodes())
    for condition in conditions:
        if rule_2_of_do_calculus_applies(
            outcomes=outcomes,
            treatments=treatments,
            conditions=conditions,
            condition=condition,
            graph=graph,
        ):
            return idc(
                outcomes=outcomes,
                treatments=treatments | {condition},
                conditions=conditions - {condition},
                graph=graph,
                estimand=estimand,
            )

    # Run ID algorithm
    new_expression = identify(
        Identification(
            outcomes=outcomes | conditions, treatments=treatments, estimand=estimand, graph=graph
        )
    )
    return new_expression / Sum.safe(expression=new_expression, ranges=outcomes)


def rule_2_of_do_calculus_applies(
    outcomes: Set[Variable],
    treatments: Set[Variable],
    conditions: Set[Variable],
    condition: Variable,
    graph: NxMixedGraph[Variable],
) -> bool:
    r"""Check if Rule 2 of the Do-Calculus applies to the conditioned variable.

    If Rule 2 of the do calculus applies to the conditioned variable, then it can be converted to a do variable.

.. math::

\newcommand\ci{\perp\!\!\!\perp}
\newcommand{\ubar}[1]{\underset{\bar{}}{#1}}
\newcommand{\obar}[1]{\overset{\bar{}}{#1}}
\text{if } (\exists Z \in \mathbf{Z})(\mathbf{Y} \ci Z | \mathbf{X}, \mathbf{Z} - \{Z\})_{G_{\bar{\mathbf{X}}\ubar{Z}}} \\
\text{then } P(\mathbf{Y}|do(\mathbf{X}),\mathbf{Z}) = P(\mathbf Y|do(\mathbf X), do(Z), \mathbf{Z} - \{Z\})


"""
    admg = graph.intervene(treatments).remove_outgoing_edges_from(frozenset([condition])).to_admg()
    judgements = [
        are_d_separated(
            admg, outcome, condition, conditions=list(treatments | (conditions - {condition}))
        )
        for outcome in outcomes
    ]
    return all([judgement.separated for judgement in judgements])
