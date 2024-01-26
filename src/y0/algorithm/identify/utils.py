# -*- coding: utf-8 -*-

"""Utilities for identification algorithms."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

import networkx as nx

from y0.dsl import (
    CounterfactualVariable,
    Distribution,
    Expression,
    Intervention,
    P,
    Probability,
    Variable,
)
from y0.graph import NxMixedGraph, _ensure_set
from y0.mutate.canonicalize_expr import canonical_expr_equal

__all__ = [
    "Query",
    "Identification",
    "Unidentifiable",
    "str_nodes_to_variable_nodes",
]


class Unidentifiable(Exception):  # noqa:N818
    """Raised on failure of the identification algorithm."""


class Query:
    """An identification query."""

    outcomes: set[Variable]
    treatments: set[Variable]
    conditions: set[Variable]

    def __init__(
        self,
        outcomes: Union[Variable, set[Variable]],
        treatments: Union[Variable, set[Variable]],
        conditions: Union[None, Variable, set[Variable]] = None,
    ) -> None:
        """Instantiate an identification.

        :param outcomes: The outcomes in the query
        :param treatments: The treatments in the query (e.g., counterfactual variables)
        :param conditions: The conditions in the query (e.g., coming after the bar)
        """
        self.outcomes = _ensure_set(outcomes)
        self.treatments = _ensure_set(treatments)
        self.conditions = _ensure_set(conditions or set())

    def __eq__(self, other: Any) -> bool:
        """Check if the outcomes, treatments, and conditions are equal."""
        return (
            isinstance(other, Query)
            and self.outcomes == other.outcomes
            and self.treatments == other.treatments
            and self.conditions == other.conditions
        )

    @classmethod
    def from_str(
        cls,
        outcomes: Union[str, Iterable[str]],
        treatments: Union[str, Iterable[str]],
        conditions: Optional[Iterable[str]] = None,
    ) -> Query:
        """Construct a query from text variable names."""
        return cls(
            outcomes=(
                {Variable(outcomes)}
                if isinstance(outcomes, str)
                else {Variable(n) for n in outcomes}
            ),
            treatments=(
                {Variable(treatments)}
                if isinstance(treatments, str)
                else {Variable(n) for n in treatments}
            ),
            conditions=None if conditions is None else {Variable(n) for n in conditions},
        )

    @classmethod
    def from_expression(
        cls,
        query: Union[Probability, Distribution],
    ) -> Query:
        """Instantiate an identification.

        :param query: The query probability expression
        :returns: An identification tuple
        :raises ValueError: If there are ragged counterfactual variables in the query
        """
        outcomes = {child.get_base() for child in query.children}  # clean counterfactuals
        conditions = {parent.get_base() for parent in query.parents}

        first_child = query.children[0]
        if not isinstance(first_child, CounterfactualVariable):
            if _unexp_interventions(query.children) or _unexp_interventions(query.parents):
                raise ValueError("Inconsistent usage of interventions")
            treatments = set()
        else:
            interventions = set(first_child.interventions)
            if _ragged_interventions(query.children, interventions) or _ragged_interventions(
                query.parents, interventions
            ):
                raise ValueError("Inconsistent usage of interventions")
            treatments = {intervention.get_base() for intervention in first_child.interventions}

        return Query(
            outcomes=outcomes,
            treatments=treatments,
            conditions=conditions,
        )

    def exchange_observation_with_action(
        self, variables: Union[Variable, Iterable[Variable]]
    ) -> Query:
        """Move the condition variable(s) to the treatments."""
        if isinstance(variables, Variable):
            variables = {variables}
        else:
            variables = set(variables)
        if any(v not in self.conditions for v in variables):
            raise ValueError
        return Query(
            outcomes=self.outcomes,
            treatments=self.treatments | variables,
            conditions=self.conditions - variables,
        )

    def exchange_action_with_observation(
        self, variables: Union[Variable, Iterable[Variable]]
    ) -> Query:
        """Move the treatment variable(s) to the conditions."""
        if isinstance(variables, Variable):
            variables = {variables}
        else:
            variables = set(variables)
        if any(v not in self.treatments for v in variables):
            raise ValueError
        return Query(
            outcomes=self.outcomes,
            treatments=self.treatments - variables,
            conditions=self.conditions | variables,
        )

    def with_treatments(self, extra_treatments: Iterable[Variable]) -> Query:
        """Create a new identification with additional treatments."""
        return Query(
            outcomes=self.outcomes,
            treatments=self.treatments.union(extra_treatments),
            conditions=self.conditions,
        )

    def uncondition(self) -> Query:
        """Move the conditions to outcomes."""
        return Query(
            outcomes=self.outcomes | self.conditions,
            treatments=self.treatments,
            conditions=None,
        )

    @property
    def expression(self) -> Expression:
        """Return the query as a Probabilistic expression."""
        if self.conditions and self.treatments:
            return P[self.treatments](self.outcomes | self.conditions)
        elif self.treatments:
            return P[self.treatments](self.outcomes)
        elif self.conditions:
            return P(self.outcomes | self.conditions)
        else:
            return P(self.outcomes)


def _unexp_interventions(variables: Iterable[Variable]) -> bool:
    return any(isinstance(c, CounterfactualVariable) for c in variables)


def _ragged_interventions(variables: Iterable[Variable], interventions: set[Intervention]) -> bool:
    return not all(
        isinstance(child, CounterfactualVariable) and set(child.interventions) == interventions
        for child in variables
    )


class Identification:
    """A package of a query and resulting estimand from identification on a graph."""

    query: Query
    graph: NxMixedGraph
    estimand: Expression

    def __init__(
        self,
        query: Query,
        graph: NxMixedGraph,
        estimand: Optional[Expression] = None,
    ) -> None:
        """Instantiate an identification.

        :param query: The generalized identification query (outcomes/treatments/conditions)
        :param graph: The graph
        :param estimand: If none is given, will use the joint distribution over all variables in the graph.
        """
        self.query = query
        self.graph = str_nodes_to_variable_nodes(graph)
        self.estimand = P(self.graph.nodes()) if estimand is None else estimand

    @classmethod
    def from_parts(
        cls,
        outcomes: set[Variable],
        treatments: set[Variable],
        graph: NxMixedGraph,
        estimand: Optional[Expression] = None,
        conditions: Optional[set[Variable]] = None,
    ) -> Identification:
        """Instantiate an identification.

        :param outcomes: The outcomes in the query
        :param treatments: The treatments in the query (e.g., counterfactual variables)
        :param conditions: The conditions in the query (e.g., coming after the bar)
        :param graph: The graph
        :param estimand: If none is given, will use the joint distribution over all variables in the graph.
        :returns: An identification object
        """
        return cls(
            query=Query(outcomes=outcomes, treatments=treatments, conditions=conditions),
            graph=graph,
            estimand=estimand,
        )

    @classmethod
    def from_expression(
        cls,
        *,
        query: Union[Probability, Distribution],
        graph: NxMixedGraph,
        estimand: Optional[Expression] = None,
    ) -> Identification:
        """Instantiate an identification.

        :param query: The query probability expression
        :param graph: The graph
        :param estimand: If none is given, will use the joint distribution over all variables in the graph.
        :returns: An identification object
        """
        return cls(
            query=Query.from_expression(query),
            graph=graph,
            estimand=estimand,
        )

    @property
    def outcomes(self) -> set[Variable]:
        """Return this identification object's query's outcomes."""
        return self.query.outcomes

    @property
    def treatments(self) -> set[Variable]:
        """Return this identification object's query's treatments."""
        return self.query.treatments

    @property
    def conditions(self) -> set[Variable]:
        """Return this identification object's query's conditions."""
        return self.query.conditions

    def exchange_observation_with_action(
        self, variables: Union[Variable, Iterable[Variable]]
    ) -> Identification:
        """Move the condition variable(s) to the treatments."""
        return Identification(
            query=self.query.exchange_observation_with_action(variables),
            graph=self.graph,
            estimand=self.estimand,
        )

    def exchange_action_with_observation(
        self, variables: Union[Variable, Iterable[Variable]]
    ) -> Identification:
        """Move the treatment variable(s) to the conditions."""
        return Identification(
            query=self.query.exchange_action_with_observation(variables),
            graph=self.graph,
            estimand=self.estimand,
        )

    def with_treatments(self, extra_treatments: Iterable[Variable]) -> Identification:
        """Create a new identification with additional treatments."""
        return Identification(
            query=self.query.with_treatments(extra_treatments),
            estimand=self.estimand,
            graph=self.graph,
        )

    def uncondition(self) -> Identification:
        """Move the conditions to outcomes."""
        return Identification(
            query=self.query.uncondition(),
            estimand=self.estimand,
            graph=self.graph,
        )

    def __repr__(self) -> str:
        return (
            f'Identification(outcomes="{self.outcomes}, treatments="{self.treatments}",'
            f'conditions="{self.conditions}",  graph="{self.graph!r}", estimand="{self.estimand}")'
        )

    def __eq__(self, other: Any) -> bool:
        """Check if the query, estimand, and graph are equal."""
        return (
            isinstance(other, Identification)
            and self.query == other.query
            and canonical_expr_equal(self.estimand, other.estimand)
            and self.graph == other.graph
        )


def str_nodes_to_variable_nodes(graph: NxMixedGraph) -> NxMixedGraph:
    """Generate a variable graph from this graph of strings."""
    return NxMixedGraph.from_edges(
        nodes={Variable.norm(node) for node in graph.nodes()},
        directed=_convert(graph.directed),
        undirected=_convert(graph.undirected),
    )


def _convert(graph: nx.Graph) -> list[tuple[Variable, Variable]]:
    return [(Variable.norm(u), Variable.norm(v)) for u, v in graph.edges()]
