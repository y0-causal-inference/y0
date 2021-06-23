# -*- coding: utf-8 -*-

"""Utilities for identification algorithms."""

from __future__ import annotations

from typing import Any, Optional, Union

import networkx as nx
from ananke.graphs import ADMG

from y0.dsl import (
    CounterfactualVariable,
    Expression,
    P,
    Probability,
    Variable,
    get_outcomes_and_treatments,
)
from y0.graph import NxMixedGraph
from y0.mutate.canonicalize_expr import expr_equal

__all__ = [
    "Identification",
    "Fail",
    "str_nodes_to_variable_nodes",
]


class Fail(Exception):
    """Raised on failure of the identification algorithm."""


class Identification:
    """A package of a query and resulting estimand from identification on a graph."""

    outcomes: set[Variable]
    treatments: set[Variable]
    conditions: set[Variable]
    graph: NxMixedGraph[Variable]
    estimand: Expression

    def __init__(
        self,
        outcomes: set[Variable],
        treatments: set[Variable],
        graph: Union[ADMG, NxMixedGraph[Variable]],
        estimand: Optional[Expression] = None,
        conditions: Optional[set[Variable]] = None,
    ) -> None:
        """Instantiate an identification.

        :param outcomes: The outcomes in the query
        :param treatments: The treatments in the query (e.g., counterfactual variables)
        :param conditions: The conditions in the query (e.g., coming after the bar)
        :param graph: The graph
        :param estimand: If none is given, will use the joint distribution over all variables in the graph.
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

        if isinstance(graph, ADMG):
            self.graph = str_nodes_to_variable_nodes(NxMixedGraph.from_admg(graph))
        else:
            self.graph = str_nodes_to_variable_nodes(graph)
        self.estimand = P(graph.nodes()) if estimand is None else estimand

    @classmethod
    def from_query(
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
        :returns: An identification tuple
        """
        outcomes, treatments = get_outcomes_and_treatments(query=query)
        outcomes = {Variable(v.name) for v in outcomes}  # clean counterfactuals
        treatments = {Variable(v.name) for v in treatments}
        conditions = {Variable(v.name) for v in query.distribution.parents}
        return Identification(
            outcomes=outcomes,
            treatments=treatments,
            conditions=conditions,
            graph=graph,
            estimand=estimand,
        )

    def treat_condition(self, condition: Variable) -> Identification:
        """Move the condition variable to the treatments."""
        assert condition in self.conditions
        return Identification(
            outcomes=self.outcomes,
            treatments=self.treatments | {condition},
            conditions=self.conditions - {condition},
            graph=self.graph,
            estimand=self.estimand,
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
            and self.outcomes == other.outcomes
            and self.treatments == other.treatments
            and self.conditions == other.conditions
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
