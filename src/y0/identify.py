# -*- coding: utf-8 -*-

"""Implementations of the identify algorithm from Shpitser and Pearl."""

from typing import Union

from .dsl import Distribution, Probability
from .graph import NxMixedGraph

__all__ = [
    "is_identifiable",
]


def is_identifiable(graph: NxMixedGraph, query: Union[Probability, Distribution]) -> bool:
    """Check if the expression is identifiable.

    :param graph: An ADMG
    :param query: A probability distribution with the following properties:

        1. There are no conditions
        2. All variables are counterfactuals
        3. The set of interventions on each counterfactual variable are the same
    :raises ValueError: If the given probability distribution does not meet the above properties
    :returns: Whether the graph is identifiable under the causal query.

    Example non-identifiable scenario:

    .. code-block:: python

        from y0.graph import NxMixedGraph
        from y0.dsl import P, X, Y

        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Z')
        graph.add_directed_edge('Z', 'Y')
        graph.add_undirected_edge('X', 'Z')

        assert not is_identifiable(graph, P(Y @ ~X))

    Example identifiable scenario:

    .. code-block:: python

        from y0.graph import NxMixedGraph
        from y0.dsl import P, X, Y

        graph = NxMixedGraph()
        graph.add_directed_edge('X', 'Y')
        graph.add_directed_edge('X', 'Z')
        graph.add_directed_edge('Z', 'Y')
        graph.add_undirected_edge('Y', 'Z')

        assert is_identifiable(graph, P(Y @ ~X))
    """
    if query.is_conditioned():
        raise ValueError("input distribution should not have any conditions")

    from y0.algorithm.identify import Identification, Unidentifiable, identify

    try:
        identify(
            Identification.from_expression(
                graph=graph,
                query=query,
            )
        )
    except Unidentifiable:
        return False
    else:
        return True
