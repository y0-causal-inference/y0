# -*- coding: utf-8 -*-

"""Implementations of the identify algorithm from Shpitser and Pearl."""

from ananke.graphs import ADMG

from .dsl import Expression

__all__ = [
    'is_identifiable',
]


def is_identifiable(graph: ADMG, query: Expression) -> bool:
    """Check if the expression is identifiable."""
    raise NotImplementedError
