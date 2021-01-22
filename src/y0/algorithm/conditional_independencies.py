# -*- coding: utf-8 -*-

"""An implementation to get conditional independencies of an ADMG."""

from __future__ import annotations

from typing import Iterable, NamedTuple, Set, Tuple

from ananke.graphs import ADMG

__all__ = [
    'get_conditional_independencies',
    'ConditionalIndependency',
]


class ConditionalIndependency(NamedTuple):
    """A conditional independency."""

    left: str
    right: str
    observations: Tuple[str, ...]

    @property
    def is_canonical(self) -> bool:
        """Return if the conditional independency is canonical."""
        return self.left < self.right and isinstance(self.observations, tuple)

    @classmethod
    def create(cls, left: str, right: str, observations: Iterable[str]) -> ConditionalIndependency:
        """Create a canonical conditional independency."""
        if left > right:
            left, right = right, left
        observations = tuple(sorted(set(observations)))
        return cls(left, right, observations)


def get_conditional_independencies(graph: ADMG) -> Set[ConditionalIndependency]:
    """Get the conditional independencies from the given ADMG.

    :param graph: An acyclic directed mixed graph
    :return: A set of conditional dependencies

    .. seealso:: Original issue https://github.com/y0-causal-inference/y0/issues/24
    """
    raise NotImplementedError
