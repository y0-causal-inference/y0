# -*- coding: utf-8 -*-

"""Implementations of the identify algorithm from Shpitser and Pearl."""

from typing import List, Set

from ananke.graphs import ADMG
from ananke.identification import OneLineID

from .dsl import Expression, Variable

__all__ = [
    'is_identifiable',
]


def _get_treatments(variables: Set[Variable]) -> List[str]:
    raise NotImplementedError


def _get_outcomes(variables: Set[Variable]) -> List[str]:
    raise NotImplementedError


def is_identifiable(graph: ADMG, query: Expression) -> bool:
    """Check if the expression is identifiable."""
    query_variables = query.get_variables()

    one_line_id = OneLineID(
        graph=graph,
        treatments=_get_treatments(query_variables),
        outcomes=_get_outcomes(query_variables),
    )
    return one_line_id.id()
