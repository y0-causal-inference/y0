# -*- coding: utf-8 -*-

"""Implementations of the identify algorithm from Shpitser and Pearl."""

from typing import List, Set, Tuple, Union

from ananke.graphs import ADMG
from ananke.identification import OneLineID

from .dsl import CounterfactualVariable, Distribution, Intervention, Probability, Variable
from .graph import NxMixedGraph

__all__ = [
    'is_identifiable',
]


def _get_treatments(variables: Set[Variable]) -> List[str]:
    return list({
        variable.name
        for variable in variables
        if isinstance(variable, Intervention)
    })


def _get_outcomes(variables: Set[Variable]) -> List[str]:
    return list({
        variable.name
        for variable in variables
        if not isinstance(variable, Intervention)
    })


def _all_counterfactual(distribution: Distribution) -> bool:
    return all(
        isinstance(variable, CounterfactualVariable)
        for variable in distribution.children
    )


def _all_intervened_same(distribution: Distribution) -> bool:
    return 1 == len({
        variable.interventions  # type:ignore
        for variable in distribution.children
    })


def _get_to(query: Distribution) -> Tuple[List[str], List[str]]:
    if not _all_counterfactual(query):
        raise ValueError('all variables in input distribution should be counterfactuals')

    if not _all_intervened_same(query):
        raise ValueError('not all variables are invervened on the same')

    treatments = [
        intervention.name
        for intervention in query.children[0].interventions  # type:ignore
    ]
    outcomes = [
        variable.name
        for variable in query.children
    ]
    return treatments, outcomes


def is_identifiable(graph: Union[ADMG, NxMixedGraph], query: Union[Probability, Distribution]) -> bool:
    """Check if the expression is identifiable.

    :param graph: Either an Ananke graph or y0 NxMixedGraph that can be converted to an Ananke graph
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
    if isinstance(graph, NxMixedGraph):
        graph = graph.to_admg()

    if isinstance(query, Probability):
        query = query.distribution

    if query.is_conditioned():
        raise ValueError('input distribution should not have any conditions')

    treatments, outcomes = _get_to(query)

    one_line_id = OneLineID(
        graph=graph,
        treatments=treatments,
        outcomes=outcomes,
    )
    return one_line_id.id()
