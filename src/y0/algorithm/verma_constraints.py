# -*- coding: utf-8 -*-

"""Implementation to get Verma constraints on a graph."""

from __future__ import annotations

from typing import Iterable, NamedTuple, Set, Tuple

from ananke.graphs import ADMG

from ..dsl import Expression

__all__ = [
    'VermaConstraint',
    'get_verma_constraints',
]


class VermaConstraint(NamedTuple):
    """A Verma constraint."""

    expression: Expression
    nodes: Tuple[str, ...]

    @property
    def is_canonical(self) -> bool:
        """Return if the nodes are in a canonical order."""
        return tuple(sorted(self.nodes)) == self.nodes

    @classmethod
    def create(cls, expression, nodes: Iterable[str]) -> VermaConstraint:
        """Create a canonical Verma constraint."""
        return VermaConstraint(expression, tuple(sorted(set(nodes))))


def get_verma_constraints(graph: ADMG) -> Set[VermaConstraint]:
    """Get the Verma constraints on the graph.

    :param graph: An acyclic directed mixed graph
    :return: A set of verma constraings, which are pairs of probability expressions and set of nodes.

    .. seealso:: Original issue https://github.com/y0-causal-inference/y0/issues/25
    """
    raise NotImplementedError
