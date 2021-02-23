# -*- coding: utf-8 -*-

"""An implementation to get conditional independencies of an ADMG."""

from typing import Set

from ananke.graphs import ADMG

from ..struct import ConditionalIndependency

__all__ = [
    'get_conditional_independencies',
]


def get_conditional_independencies(graph: ADMG) -> Set[ConditionalIndependency]:
    """Get the conditional independencies from the given ADMG.

    :param graph: An acyclic directed mixed graph
    :return: A set of conditional dependencies

    .. seealso:: Original issue https://github.com/y0-causal-inference/y0/issues/24
    """
    raise NotImplementedError
