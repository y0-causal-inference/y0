# -*- coding: utf-8 -*-

"""Implementations of the identify algorithm from Shpitser and Pearl."""

from typing import List, Set, Tuple, Union

from ananke.graphs import ADMG
from ananke.identification import OneLineID

from .dsl import CounterfactualVariable, Distribution, Intervention, Probability, Variable

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
        variable.interventions
        for variable in distribution.children
    })


def _get_to(query: Distribution) -> Tuple[List[str], List[str]]:
    if not _all_counterfactual(query):
        raise ValueError('all variables in input distribution should be counterfactuals')

    if not _all_intervened_same(query):
        raise ValueError('not all variables are invervened on the same')

    treatments = [
        intervention.name
        for intervention in query.children[0].interventions
    ]
    outcomes = [
        variable.name
        for variable in query.children
    ]
    return treatments, outcomes


def is_identifiable(graph: ADMG, query: Union[Probability, Distribution]) -> bool:
    """Check if the expression is identifiable."""
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
