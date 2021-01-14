# -*- coding: utf-8 -*-

"""Implementations of the identify algorithm from Shpitser and Pearl."""

from typing import List, Set, Union

from ananke.graphs import ADMG
from ananke.identification import OneLineID

from .dsl import Distribution, Probability, Variable

__all__ = [
    'is_identifiable',
]


def _get_treatments(variables: Set[Variable]) -> List[str]:
    raise NotImplementedError


def _get_outcomes(variables: Set[Variable]) -> List[str]:
    raise NotImplementedError


def is_identifiable(graph: ADMG, query: Union[Probability, Distribution]) -> bool:
    """Check if the expression is identifiable."""
    if isinstance(query, Probability):
        query = query.distribution

    if query.is_conditioned():
        raise ValueError('input distribution should not have any conditions')

    query_variables = query.get_variables()

    one_line_id = OneLineID(
        graph=graph,
        treatments=_get_treatments(query_variables),
        outcomes=_get_outcomes(query_variables),
    )
    return one_line_id.id()
