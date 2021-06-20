# -*- coding: utf-8 -*-

"""Utilities for identification algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Hashable, Set, Tuple, TypeVar

import networkx as nx

from y0.dsl import Expression, P, Variable, _upgrade_ordering
from y0.graph import NxMixedGraph
from y0.identify import _get_outcomes, _get_treatments
from y0.mutate import canonicalize

__all__ = [
    "ancestors_and_self",
    "Identification",
    "expr_equal",
    "Fail",
    "get_outcomes_and_treatments",
    "outcomes_and_treatments_to_query",
]

Y = TypeVar("Y", bound=Hashable)


class Fail(Exception):
    """Raised on failure of the identification algorithm."""


@dataclass
class Identification:
    """A package of a query and resulting estimand from identification on a graph."""

    query: Expression
    estimand: Expression
    graph: NxMixedGraph[str]

    @classmethod
    def from_parts(
        cls,
        outcomes: Set[Variable],
        treatments: Set[Variable],
        estimand: Expression,
        graph: NxMixedGraph[str],
    ) -> Identification:
        """Instantiate an Identification from the parts of a query."""
        return cls(
            query=outcomes_and_treatments_to_query(outcomes=outcomes, treatments=treatments),
            estimand=estimand,
            graph=graph,
        )

    @property
    def outcomes(self) -> Set[Variable]:
        """Get outcomes of the query."""
        return {Variable(v) for v in _get_outcomes(self.query.get_variables())}

    @property
    def treatments(self) -> Set[Variable]:
        """Get treatments of the query."""
        return {Variable(v) for v in _get_treatments(self.query.get_variables())}

    def __eq__(self, other: Any) -> bool:
        """Check if the query, estimand, and graph are equal."""
        return (
            isinstance(other, Identification)
            and expr_equal(self.query, other.query)
            and expr_equal(self.estimand, other.estimand)
            and self.graph == other.graph
        )


def expr_equal(expected: Expression, actual: Expression) -> bool:
    """Return True if two expressions are equal after canonicalization."""
    expected_outcomes, expected_treatments = get_outcomes_and_treatments(query=expected)
    actual_outcomes, actual_treatments = get_outcomes_and_treatments(query=actual)

    if (expected_outcomes != actual_outcomes) or (expected_treatments != actual_treatments):
        return False
    ordering = tuple(expected.get_variables())  # need to impose ordering, any will do.
    expected_canonical = canonicalize(expected, ordering)
    actual_canonical = canonicalize(actual, ordering)
    return expected_canonical == actual_canonical


def get_outcomes_and_treatments(*, query: Expression) -> Tuple[Set[Variable], Set[Variable]]:
    """Get outcomes and treatments sets from the query expression."""
    query_variables = query.get_variables()
    return (
        {Variable(v) for v in _get_outcomes(query_variables)},
        {Variable(v) for v in _get_treatments(query_variables)},
    )


def outcomes_and_treatments_to_query(
    *, outcomes: Set[Variable], treatments: Set[Variable]
) -> Expression:
    """Create a query expression from a set of outcome and treatment variables."""
    if len(treatments) == 0:
        return P(outcomes)
    return P(Variable.norm(y) @ _upgrade_ordering(treatments) for y in outcomes)


def ancestors_and_self(graph: NxMixedGraph[Y], sources: Collection[Y]) -> Set[Y]:
    """Ancestors of a set include the set itself."""
    rv = set(sources)
    for source in sources:
        rv.update(nx.algorithms.dag.ancestors(graph.directed, source))
    return rv
