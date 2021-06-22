# -*- coding: utf-8 -*-

"""Utilities for identification algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Optional, Set, TypeVar

from y0.dsl import (
    Expression,
    P,
    Variable,
    outcomes_and_treatments_to_query,
)
from y0.graph import NxMixedGraph
from y0.identify import _get_outcomes, _get_treatments
from y0.mutate.canonicalize_expr import expr_equal

__all__ = [
    "Identification",
    "Fail",
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
        graph: NxMixedGraph[str],
        estimand: Optional[Expression] = None,
    ) -> Identification:
        """Instantiate an Identification from the parts of a query.

        :param outcomes: The outcomes in the query
        :param treatments: The treatments in the query (e.g., counterfactual variables)
        :param graph: The graph
        :param estimand: If none is given, will use the joint distribution
            over all variables in the graph.
        :return: An identification object
        """
        if estimand is None:
            estimand = P(graph.nodes())
        return cls(
            query=outcomes_and_treatments_to_query(outcomes=outcomes, treatments=treatments),
            estimand=estimand,
            graph=graph,
        )

    @property
    def outcome_variables(self) -> Set[Variable]:
        """Get outcomes of the query."""
        return {Variable(v) for v in _get_outcomes(self.query.get_variables())}

    @property
    def treatment_variables(self) -> Set[Variable]:
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
