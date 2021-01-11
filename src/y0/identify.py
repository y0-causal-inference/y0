# -*- coding: utf-8 -*-

"""Implementations of the identify algorithm from Shpitser and Pearl."""

from typing import Optional

from .dsl import Expression
from .graph import MixedGraph

__all__ = [
    'is_identifiable',
    'identify',
]


def is_identifiable(graph: MixedGraph, query: Expression) -> bool:
    """Check if the expression is identifiable."""
    raise NotImplementedError


def identify(graph: MixedGraph, query: Expression) -> Optional[Expression]:
    """Get an expression from the graph or return None."""
    raise NotImplementedError
