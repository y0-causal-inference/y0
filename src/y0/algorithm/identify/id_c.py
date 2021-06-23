# -*- coding: utf-8 -*-

"""Implementation of the IDC algorithm."""

from .id_std import identify
from ..conditional_independencies import are_d_separated
from ...dsl import Expression
from ...graph import NxMixedGraph


def idc(outcomes, treatments, conditions, graph: NxMixedGraph[str], expression=None) -> Expression:
    """Run the IDC algorithm.

    :param outcomes:
    :param treatments:
    :param conditions:
    :param graph:
    """
    if not expression:
        expression = P(graph.nodes())
    for condition in conditions:
        if idc_condition(outcomes, treatments, conditions, condition, graph):
            return idc(outcomes, treatments | condition, graph, expression)

    # Run ID algorithm
    new_expression = identify(
        Identification(
            outcomes=outcomes | conditions, treatments=treatments, estimand=expression, graph=graph
        )
    )
    return new_expression / Sum.safe(expression=new_expression, ranges=outcomes)


def idc_condition(outcomes, treatments, conditions, condition, graph: NxMixedGraph[str]) -> bool:
    """Check the IDC condition for recursion."""
    admg = graph.intervene(treatments).remove_outgoing_edges_from(condition).to_admg()
    return are_d_separated(
        admg, outcomes, condition, conditions=list(treatments | (conditions - {condition}))
    )
