# -*- coding: utf-8 -*-

"""Predicates for good, bad, and neutral controls."""

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
