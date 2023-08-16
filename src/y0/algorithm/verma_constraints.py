# -*- coding: utf-8 -*-

"""Implementation to get Verma constraints on a graph."""

from __future__ import annotations

import logging
from typing import Iterable, List, NamedTuple, Set, Tuple

from ..dsl import Expression
from ..graph import NxMixedGraph
from ..r_utils import uses_r

__all__ = [
    "VermaConstraint",
    "get_verma_constraints",
]


logger = logging.getLogger(__name__)


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


def get_verma_constraints(graph: NxMixedGraph) -> Set[VermaConstraint]:
    """Get the Verma constraints on the graph.

    :param graph: An acyclic directed mixed graph
    :return: A set of verma constraints, which are pairs of probability expressions and set of nodes.

    .. seealso:: Original issue https://github.com/y0-causal-inference/y0/issues/25
    """
    raise NotImplementedError


@uses_r
def r_get_verma_constraints(graph: NxMixedGraph) -> List[VermaConstraint]:
    """Calculate the verma constraints on the graph using ``causaleffect``."""
    graph = graph.to_causaleffect()

    from rpy2 import robjects

    verma_constraints = robjects.r["verma.constraints"]
    return [VermaConstraint.from_element(row) for row in verma_constraints(graph)]
