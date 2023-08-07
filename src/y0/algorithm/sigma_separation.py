"""Implementation of sigma-separation."""

from typing import Iterable, Optional

from y0.dsl import Variable
from y0.graph import NxMixedGraph

__all__ = [
    "are_sigma_separated",
]


def are_sigma_separated(
    graph: NxMixedGraph,
    a: Variable,
    b: Variable,
    *,
    conditions: Optional[Iterable[Variable]] = None,
) -> bool:
    """Test if two variables are sigma-separated.

    Sigma separation is a generalization of d-separation that
    works not only for directed acyclic graphs, but also for
    directed graphs containing cycles. It was originally introduced
    in https://arxiv.org/abs/1807.03024.

    :param graph: Graph to test
    :param a: A node in the graph
    :param b: A node in the graph
    :param conditions: A collection of graph nodes
    :return: If a and b are sigma-separated.
    """
    raise NotImplementedError
