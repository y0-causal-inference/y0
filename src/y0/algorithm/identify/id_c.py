# -*- coding: utf-8 -*-

"""Implementation of the IDC algorithm."""

from .id_std import identify
from ..conditional_independencies import are_d_separated
from ...dsl import Expression
from ...graph import NxMixedGraph


def idx(outcomes, treatments, conditions, graph: NxMixedGraph[str]) -> Expression:
    """Run the IDC algorithm.

    :param outcomes:
    :param treatments:
    :param conditions:
    :param graph:
    """
    if idc_condition():
        ...
        return idx(...)

    # Run ID algorithm
    _ = identify(...)
    return ...


def idc_condition(graph: NxMixedGraph[str], conditions, outcomes) -> bool:
    """Check the IDC condition for recursion."""
    admg = graph.to_admg()
    for condition in conditions:
        if are_d_separated(outcomes, ...):
            ...
