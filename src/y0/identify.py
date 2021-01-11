# -*- coding: utf-8 -*-

"""Implementations of the identify algorithm from Shpitser and Pearl."""

from typing import Optional

from ananke.graphs import ADMG

from .dsl import Expression

__all__ = [
    'is_identifiable',
    'identify',
]


def is_identifiable(graph: ADMG, query: Expression) -> bool:
    """Check if the expression is identifiable."""
    raise NotImplementedError


def identify(graph: ADMG, query: Expression) -> Optional[Expression]:
    """Get an expression from the graph or return None."""
    raise NotImplementedError
