# -*- coding: utf-8 -*-

"""Utilities for identification algorithms."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

import networkx as nx
from ananke.graphs import ADMG

from y0.dsl import (
    CounterfactualVariable,
    Expression,
    Intervention,
    P,
    Probability,
    Variable,
)
from y0.graph import NxMixedGraph
from y0.mutate.canonicalize_expr import expr_equal

__all__ = [
    "Query",
    "Identification",
    "Fail",
    "str_nodes_to_variable_nodes",
]


class Fail(Exception):
    """Raised on failure of the identification algorithm."""


class Query:
    """An identification query."""

    outcomes: set[Variable]
    treatments: set[Variable]
    conditions: set[Variable]

    def __init__(
        self,
        outcomes: set[Variable],
        treatments: set[Variable],
        conditions: Optional[set[Variable]] = None,
    ) -> None:
        """Instantiate an identification.

        :param outcomes: The outcomes in the query
        :param treatments: The treatments in the query (e.g., counterfactual variables)
        :param conditions: The conditions in the query (e.g., coming after the bar)
        :raises TypeError: If any of the outcomes, treatements, or conditions are not vanilla variables
        """
        if not all(isinstance(v, Variable) for v in outcomes):
            raise TypeError
        elif any(isinstance(v, CounterfactualVariable) for v in outcomes):
            raise TypeError
        else:
            self.outcomes = outcomes

        if not all(isinstance(v, Variable) for v in treatments):
            raise TypeError
        elif any(isinstance(v, CounterfactualVariable) for v in treatments):
            raise TypeError
        else:
            self.treatments = treatments

        if conditions is None:
            self.conditions = set()
        elif not all(isinstance(v, Variable) for v in conditions):
            raise TypeError
        elif any(isinstance(v, CounterfactualVariable) for v in conditions):
            raise TypeError
        else:
            self.conditions = conditions

    def __eq__(self, other: Any) -> bool:
        """Check if the outcomes, treatments, and conditions are equal."""
        return (
            isinstance(other, Query)
            and self.outcomes == other.outcomes
            and self.treatments == other.treatments
            and self.conditions == other.conditions
        )

    @classmethod
    def from_expression(
        cls,
        query: Probability,
    ) -> Query:
        """Instantiate an identification.

        :param query: The query probability expression
        :returns: An identification tuple
        :raises ValueError: If there are ragged counterfactual variables in the query
        """
        outcomes = {Variable(v.name) for v in query.distribution.children}  # clean counterfactuals
        conditions = {Variable(v.name) for v in query.distribution.parents}

        first_child = query.distribution.children[0]
        if not isinstance(first_child, CounterfactualVariable):
            if _unexp_interventions(query.distribution.children) or _unexp_interventions(
                query.distribution.parents
            ):
                raise ValueError("Inconsistent usage of interventions")
            treatments = set()
        else:
            interventions = set(first_child.interventions)
            if _ragged_interventions(
                query.distribution.children, interventions
            ) or _ragged_interventions(query.distribution.parents, interventions):
                raise ValueError("Inconsistent usage of interventions")
            treatments = {Variable(i.name) for i in first_child.interventions}

        return Query(
            outcomes=outcomes,
            treatments=treatments,
            conditions=conditions,
        )

    def treat_condition(self, condition: Variable) -> Query:
        """Move the condition variable to the treatments."""
        if condition not in self.conditions:
            raise ValueError
        return Query(
            outcomes=self.outcomes,
            treatments=self.treatments | {condition},
            conditions=self.conditions - {condition},
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
    graph: NxMixedGraph[Variable]
    estimand: Expression

    def __init__(
        self,
        query: Query,
        graph: Union[ADMG, NxMixedGraph[Variable]],
        estimand: Optional[Expression] = None,
    ) -> None:
        """Instantiate an identification.

        :param query: The generalizd identification query (outcomes/treatments/conditions)
        :param graph: The graph
        :param estimand: If none is given, will use the joint distribution over all variables in the graph.
        """
        self.query = query
        if isinstance(graph, ADMG):
            self.graph = str_nodes_to_variable_nodes(NxMixedGraph.from_admg(graph))
        else:
            self.graph = str_nodes_to_variable_nodes(graph)
        self.estimand = P(graph.nodes()) if estimand is None else estimand

    @classmethod
    def from_parts(
        cls,
        outcomes: set[Variable],
        treatments: set[Variable],
        graph: Union[ADMG, NxMixedGraph[Variable]],
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
        query: Probability,
        graph: Union[ADMG, NxMixedGraph[str], NxMixedGraph[Variable]],
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

    def treat_condition(self, condition: Variable) -> Identification:
        """Move the condition variable to the treatments."""
        if condition not in self.conditions:
            raise ValueError
        return Identification(
            query=self.query.treat_condition(condition),
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
            and expr_equal(self.estimand, other.estimand)
            and self.graph == other.graph
        )


def str_nodes_to_variable_nodes(graph: NxMixedGraph) -> NxMixedGraph[Variable]:
    """Generate a variable graph from this graph of strings."""
    return NxMixedGraph.from_edges(
        nodes={Variable.norm(node) for node in graph.nodes()},
        directed=_convert(graph.directed),
        undirected=_convert(graph.undirected),
    )


def _convert(graph: nx.Graph) -> list[tuple[Variable, Variable]]:
    return [(Variable.norm(u), Variable.norm(v)) for u, v in graph.edges()]
