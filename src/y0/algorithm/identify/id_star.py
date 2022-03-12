# -*- coding: utf-8 -*-

"""Implementation of the IDC algorithm."""

from typing import Collection, FrozenSet, Mapping, Optional, Set, Tuple

from .cg import has_same_function, make_counterfactual_graph
from .utils import Unidentifiable
from ...dsl import (
    CounterfactualEvent,
    CounterfactualVariable,
    Expression,
    Intervention,
    One,
    P,
    Probability,
    Product,
    Sum,
    Variable,
    Zero,
    _get_treatment_variables,
)
from ...graph import NxMixedGraph

__all__ = [
    "id_star",
]


def id_star(graph: NxMixedGraph, event: CounterfactualEvent) -> Expression:
    """Apply the ``ID*`` algorithm to the graph."""
    # Line 0: There's nothing in the counterfactual event
    if id_star_line_1(event):
        return One()
    # Line 2: This violates the Axiom of Effectiveness
    if id_star_line_2(event):
        return Zero()
    # Line 3: This is a tautological event and can be removed without affecting the probability
    reduced_event = id_star_line_3(event)
    if reduced_event != event:
        # if we did some reducing, recursively start over
        # FIXME this isn't *technically needed* - we could just overwrite the event with the reduced one
        return id_star(graph, reduced_event)
    # Line 4:
    new_graph, new_event = id_star_line_4(graph, event)
    # Line 5:
    if new_event == Zero():
        return new_event
    # Line 6:
    if not new_graph.is_connected():
        # FIXME missing third parameter to `id_star_line_6`
        summand, interventions_of_each_district = id_star_line_6(new_graph, new_event)
        return Sum.safe(
            Product.safe(
                id_star(graph, {element @ interventions: +element for element in district})
                for district, interventions in interventions_of_each_district.items()
            ),
            summand,
        )
    # Line 7:
    elif id_star_line_8(new_graph, reduced_event):
        raise Unidentifiable
    else:
        # Line 9
        return id_star_line_9(new_graph)


def id_star_line_1(event: CounterfactualEvent) -> bool:
    r"""Run line 1 of the ID* algorithm.

    The first line states that if :math:`\event` is an empty conjunction, then its
    probability is 1, by convention.

    :param event: a conjunction of counterfactual variables
    """
    return len(event) == 0


def id_star_line_2(event: CounterfactualEvent) -> bool:
    r"""Run line 2 of the ID* algorithm.

    The second line states that if :math:`\event` contains a counterfactual
    which violates the Axiom of Effectiveness (Pearl, 2000), then :math:`\event`
    is inconsistent, and we return probability 0.

    :param event: a conjunction of counterfactual variables
    :returns: True if violates axiom of effectiveness
    """
    return any(
        intervention.name == counterfactual.name and value.star != intervention.star
        for counterfactual, value in event.items()
        if isinstance(counterfactual, CounterfactualVariable)
        for intervention in counterfactual.interventions
    )


def id_star_line_3(event: CounterfactualEvent) -> CounterfactualEvent:
    r"""Run line 3 of the ID* algorithm.

    The third line states that if a counterfactual contains its own value in the subscript,
    then it is a tautological event, and it can be removed from :math:`\event` without
    affecting its probability.

    :param event: a conjunction of counterfactual variables
    :return: updated event or None
    """
    redundant_counterfactuals = {
        counterfactual
        for counterfactual, value in event.items()
        if isinstance(counterfactual, CounterfactualVariable) and any(
            intervention.name == counterfactual.name and value.star == intervention.star
            for intervention in counterfactual.interventions
        )
    }
    return {
        counterfactual: value
        for counterfactual, value in event.items()
        if counterfactual not in redundant_counterfactuals
    }


def id_star_line_4(
    graph: NxMixedGraph, event: CounterfactualEvent
) -> Tuple[NxMixedGraph, CounterfactualEvent]:
    r"""Run line 4 of the ID* algorithm

    Line 4 invokes make-cg to construct a counterfactual graph :math:`G'` , and the
    corresponding relabeled counterfactual event.

    :param graph: an NxMixedGraph
    :param event: a conjunction of counterfactual variables
    :return: updated graph and event
    """

    new_graph, new_event = make_counterfactual_graph(graph, event)
    return new_graph, new_event


def id_star_line_5(graph: NxMixedGraph, event: CounterfactualEvent) -> Optional[Zero]:
    r"""Run line 5 of the ID* algorithm.

    Line 5 returns probability 0 if an inconsistency was found during the construction
    of the counterfactual graph, for example, if two variables found to be the same in
    event had different value assignments.
    """
    if event == Zero():
        return event


def id_star_line_6(
    graph: NxMixedGraph, new_graph: NxMixedGraph, event: CounterfactualEvent
) -> Tuple[Collection[Variable], Mapping[FrozenSet[Variable], Set[Variable]]]:
    r"""Run line 6 of the ID* algorithm.

    Line 6 is analogous to Line 4 in the ID algorithm, it decomposes the problem into a
    set of subproblems, one for each C-component in the counterfactual graph. In the ID
    algorithm, the term corresponding to a given C-component :math:`S_i` of the causal
    diagram was the effect of all variables not in :math:`S_i` on variables in :math:`S_i` ,
    in other words :math:`P_{\mathbf{v}\backslash s_i} (s_i )`, and the outermost summation
    on line 4 was over values of variables not in :math:`\mathbf{Y},\mathbf{X}`. Here, the
    term corresponding to a given C-component :math:`S^i` of the counterfactual graph :math:`G'`
    is the conjunction of counterfactual variables where each variable contains in its
    subscript all variables not in the C-component :math:`S^i` , in other words
    :math:`\mathbf{v}(G' )\backslash s^i` , and the outermost summation is over observable
    variables not in :math:`\event'` , that is over :math:`\mathbf{v}(G' ) \backslash \event'` ,
    where we interpret :math:`\event'` as a set of counterfactuals, rather than a conjunction.
    """
    vertices = set(graph.nodes())
    summand = vertices - set(event)
    interventions_of_each_district = {
        district: vertices - district for district in graph.get_c_components()
    }
    return summand, interventions_of_each_district


def id_star_line_8(graph: NxMixedGraph, query: CounterfactualEvent) -> bool:
    r"""Run line 8 of the ID* algorithm.

    Line 8 says that if :math:`\event'` contains a "conflict," that is an inconsistent
    value assignment where at least one value is in the subscript, then we fail.

    :param graph: an NxMixedGraph
    :param query: a joint distribution over counterfactual variables
    :return: True if there is a conflict. False otherwise
    """
    return any(
        intervention.name == evidence.name and intervention.star != evidence.star
        for intervention in sub(graph)
        for evidence in ev(query)
    )


def sub(graph: NxMixedGraph) -> Collection[Intervention]:
    """sub() is the set of interventions that are in each counterfactual variable."""
    return {
        intervention
        for node in graph
        if isinstance(node, CounterfactualVariable)
        for intervention in node.interventions
    }


def ev(query: CounterfactualEvent) -> Collection[Intervention]:
    """ev(:) the set of values (either set or observed) appearing in a given counterfactual conjunction (or set of counterfactual events)"""
    return set(query.values())


def id_star_line_9(graph: NxMixedGraph) -> Probability:
    r"""Run line 9 of the ID* algorithm.

    Line 9 says if there are no conflicts, then it's safe to take the union of all
    subscripts in :math:`\event'` , and return the effect of the subscripts in :math:`\event'`
    on the variables in :math:`\event'`.

    :param graph: an NxMixedGraph
    :return: An interventional distribution.
    """
    interventions = sub(graph)
    return P[interventions](Variable(node.name) for node in graph)


# FIXME this is unused -> delete it
def get_interventions(query: CounterfactualEvent) -> Collection[Variable]:
    r"""Generate new Variables from the subscripts of counterfactual variables in the query."""
    interventions = set()
    for counterfactual in query.children:
        if isinstance(counterfactual, CounterfactualVariable):
            interventions |= set(counterfactual.interventions)
    return sorted(interventions)


# TODO update the docs: is this generally applicable, or only to graphs
#  constructed as parallel worlds? perhaps update the name?

# FIXME this is unused -> delete it
def has_same_domain_of_values(node1: Variable, node2: Variable) -> bool:
    if isinstance(node1, CounterfactualVariable) and isinstance(node2, CounterfactualVariable):
        treatment1, treatment2 = _get_treatment_variables(node1), _get_treatment_variables(node2)
    raise NotImplementedError


# FIXME this is unused -> delete it
def has_same_value(event: Collection[Variable], node1: Variable, node2: Variable) -> bool:
    n1 = None
    for node in event:
        if node == node1:
            n1 = node
    if n1 is None:
        raise ValueError

    n2 = None
    for node in event:
        if node == node2:
            n2 = node
    if n2 is None:
        raise ValueError

    # TODO not all variables have is_event().
    #  Should normal, non-counterfactual variables have this function?
    return has_same_function(n1, n2) and n1.is_event() and n2.is_event() and (n1.star == n2.star)
