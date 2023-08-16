# -*- coding: utf-8 -*-

"""Implementations of the identify algorithm from Shpitser and Pearl."""

from typing import List, Set, Tuple, Union

from .dsl import (
    CounterfactualVariable,
    Distribution,
    Probability,
    Variable,
    _get_outcome_variables,
    _get_treatment_variables,
)
from .graph import NxMixedGraph

__all__ = [
    "is_identifiable",
]


def _get_treatments(variables: Set[Variable]) -> List[str]:
    return [variable.name for variable in _get_treatment_variables(variables)]


def _get_outcomes(variables: Set[Variable]) -> List[str]:
    return [variable.name for variable in _get_outcome_variables(variables)]


def _all_counterfactual(distribution: Union[Probability, Distribution]) -> bool:
    return all(isinstance(variable, CounterfactualVariable) for variable in distribution.children)


def _all_intervened_same(distribution: Union[Probability, Distribution]) -> bool:
    return 1 == len(
        {
            variable.interventions  # type:ignore
            for variable in distribution.children
        }
    )


def _get_to(query: Union[Probability, Distribution]) -> Tuple[List[str], List[str]]:
    if not _all_counterfactual(query):
        raise ValueError("all variables in input distribution should be counterfactuals")

    if not _all_intervened_same(query):
        raise ValueError("not all variables are invervened on the same")

    treatments = [
        intervention.name for intervention in query.children[0].interventions  # type:ignore
    ]
    outcomes = [variable.name for variable in query.children]
    return treatments, outcomes


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

    treatments, outcomes = _get_to(query)

    try:
        from ananke.identification import OneLineID
    except ImportError:
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
    else:
        graph = graph.to_admg()
        one_line_id = OneLineID(
            graph=graph,
            treatments=treatments,
            outcomes=outcomes,
        )
        return one_line_id.id()
