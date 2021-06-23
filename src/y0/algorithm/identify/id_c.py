# -*- coding: utf-8 -*-

"""Implementation of the IDC algorithm."""

from typing import Optional

from .id_std import identify
from .utils import Identification
from ..conditional_independencies import are_d_separated
from ...dsl import Expression, P, Sum, Variable
from ...graph import NxMixedGraph


def idc(
    outcomes: set[Variable],
    treatments: set[Variable],
    conditions: set[Variable],
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
        if idc_condition(
            outcomes=outcomes,
            treatments=treatments,
            conditions=conditions,
            condition=condition,
            graph=graph,
        ):
            return idc(
                outcomes=outcomes,
                treatments=treatments | {condition},
                conditions=...,  # FIXME
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


def idc_condition(
    outcomes: set[Variable],
    treatments: set[Variable],
    conditions: set[Variable],
    condition: Variable,
    graph: NxMixedGraph[Variable],
) -> bool:
    """Check the IDC condition for recursion."""
    admg = graph.intervene(treatments).remove_outgoing_edges_from(condition).to_admg()
    judgement = are_d_separated(
        admg, outcomes, condition, conditions=list(treatments | (conditions - {condition}))
    )
    return judgement.separated
