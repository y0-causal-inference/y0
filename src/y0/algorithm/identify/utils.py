"""Utilities for identification algorithms."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from typing import Any, cast

import networkx as nx

from y0.dsl import (
    CounterfactualVariable,
    Distribution,
    Expression,
    P,
    Probability,
    Variable,
)
from y0.graph import NxMixedGraph, _ensure_set
from y0.mutate.canonicalize_expr import canonical_expr_equal

__all__ = [
    "Identification",
    "Query",
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
        outcomes: Variable | Iterable[Variable],
        treatments: Variable | Iterable[Variable],
        conditions: None | Variable | Iterable[Variable] = None,
    ) -> None:
        """Instantiate an identification.

        :param outcomes: The outcomes in the query
        :param treatments: The treatments in the query (e.g., counterfactual variables)
        :param conditions: The conditions in the query (e.g., coming after the bar)
        """
        self.outcomes = _ensure_set(outcomes)
        self.treatments = _ensure_set(treatments)
        self.conditions = _ensure_set(conditions) if conditions is not None else set()

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
        outcomes: str | Iterable[str],
        treatments: str | Iterable[str],
        conditions: Iterable[str] | None = None,
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
        query: Probability | Distribution,
    ) -> Query:
        """Instantiate an identification.

        :param query: The query probability expression

        :returns: An identification tuple

        :raises ValueError: If there are ragged counterfactual variables in the query
        """
        outcomes = {child.get_base() for child in query.children}  # clean counterfactuals
        conditions = {parent.get_base() for parent in query.parents}

        treatments: set[Variable]
        if any(isinstance(c, CounterfactualVariable) for c in chain(query.children, query.parents)):
            if not all(isinstance(c, CounterfactualVariable) for c in query.children):
                raise ValueError(
                    "if any children or parents are counterfactual variables, all children have to be"
                )
            if not all(isinstance(c, CounterfactualVariable) for c in query.parents):
                raise ValueError(
                    "if any children or parents are counterfactual variables, all parents have to be"
                )

            # todo get sets of interventions on all variables
            intervention_sets: set[frozenset[Variable]] = {
                cast(CounterfactualVariable, c).interventions
                for c in chain(query.children, query.parents)
            }
            if len(intervention_sets) != 1:
                raise ValueError("inconsistent usage of interventions")
            treatments = {x.get_base() for x in next(iter(intervention_sets))}
        else:
            treatments = set()

        return Query(outcomes=outcomes, treatments=treatments, conditions=conditions)

    def exchange_observation_with_action(self, variables: Variable | Iterable[Variable]) -> Query:
        """Move the condition variable(s) to the treatments."""
        variables = _ensure_set(variables)
        if missing := (variables - self.conditions):
            raise ValueError(f"variables don't appear in conditions: {missing}")
        return Query(
            outcomes=self.outcomes,
            treatments=self.treatments | variables,
            conditions=self.conditions - variables,
        )

    def exchange_action_with_observation(self, variables: Variable | Iterable[Variable]) -> Query:
        """Move the treatment variable(s) to the conditions."""
        variables = _ensure_set(variables)
        if missing := (variables - self.treatments):
            raise ValueError(f"variables don't appear in treatments: {missing}")
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
        distribution = Distribution.safe(self.outcomes)
        if self.conditions:
            distribution = distribution.given(self.conditions)
        elif self.treatments:
            distribution = distribution.intervene(self.treatments)
        return Probability(distribution)


class Identification:
    """A package of a query and resulting estimand from identification on a graph."""

    query: Query
    graph: NxMixedGraph
    estimand: Expression

    def __init__(
        self,
        query: Query,
        graph: NxMixedGraph,
        estimand: Expression | None = None,
    ) -> None:
        """Instantiate an identification.

        :param query: The generalized identification query
            (outcomes/treatments/conditions)
        :param graph: The graph
        :param estimand: If none is given, will use the joint distribution over all
            variables in the graph.
        """
        self.query = query
        self.graph = str_nodes_to_variable_nodes(graph)
        self.estimand = P(self.graph.nodes()) if estimand is None else estimand

    @classmethod
    def from_parts(
        cls,
        outcomes: Variable | Iterable[Variable],
        treatments: Variable | Iterable[Variable],
        graph: NxMixedGraph,
        estimand: Expression | None = None,
        conditions: Variable | Iterable[Variable] | None = None,
    ) -> Identification:
        """Instantiate an identification.

        :param outcomes: The outcomes in the query
        :param treatments: The treatments in the query (e.g., counterfactual variables)
        :param conditions: The conditions in the query (e.g., coming after the bar)
        :param graph: The graph
        :param estimand: If none is given, will use the joint distribution over all
            variables in the graph.

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
        query: Probability | Distribution,
        graph: NxMixedGraph,
        estimand: Expression | None = None,
    ) -> Identification:
        """Instantiate an identification.

        :param query: The query probability expression
        :param graph: The graph
        :param estimand: If none is given, will use the joint distribution over all
            variables in the graph.

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
        self, variables: Variable | Iterable[Variable]
    ) -> Identification:
        """Move the condition variable(s) to the treatments."""
        return Identification(
            query=self.query.exchange_observation_with_action(variables),
            graph=self.graph,
            estimand=self.estimand,
        )

    def exchange_action_with_observation(
        self, variables: Variable | Iterable[Variable]
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
