# -*- coding: utf-8 -*-

"""Implementation of the ID* algorithm."""

from typing import Collection, FrozenSet, Iterable, Mapping, Optional, Set, Tuple

from .cg import make_counterfactual_graph
from .utils import Unidentifiable
from ...dsl import (
    CounterfactualVariable,
    Event,
    Expression,
    Intervention,
    One,
    P,
    Probability,
    Product,
    Sum,
    Variable,
    Zero,
)
from ...graph import NxMixedGraph

__all__ = [
    "id_star",
    "get_district_domains",
    "domain_of_counterfactual_values",
] + [f"id_star_line_{i}" for i in [4, 6, 8]]

District = FrozenSet[Variable]
DistrictInterventions = Mapping[District, Set[Variable]]


def id_star(graph: NxMixedGraph, event: Event, leonardo=0) -> Expression:
    """Apply the ``ID*`` algorithm to the graph."""
    print(f"[{leonardo}] running on event {event}")
    # Line 1: There's nothing in the counterfactual event
    if is_event_empty(event):
        return One()
    # Line 2: This violates the Axiom of Effectiveness
    if violates_axiom_of_effectiveness(event):
        return Zero()
    # Line 3: This is a tautological event and can be removed without affecting the probability
    reduced_event = remove_event_tautologies(event)
    if reduced_event != event:
        print(f"[{leonardo}] recurring on reduced event {reduced_event}")
        return id_star(graph, reduced_event, leonardo=leonardo + 1)
    # Line 4:
    cf_graph, new_event = id_star_line_4(graph, event)
    print(f"[{leonardo}] new event: {new_event}\n\tcounterfactual graph:\n\t nodes: {cf_graph.nodes()}\n\t directed: {cf_graph.directed.edges()}\n\t undirected: {cf_graph.undirected.edges()}")
    # Line 5:
    if new_event is None:
        return Zero()

    # print(cf_graph)
    # print(new_event)
    # Line 6:
    if not cf_graph.is_connected():
        summand, interventions_of_each_district = id_star_line_6(cf_graph, event)
        print(f"[{leonardo}] interventions of each district: {interventions_of_each_district}")
        district_events = get_district_events(interventions_of_each_district)
        assert 1 < len(district_events)
        return Sum.safe(
            Product.safe(
                print(f"[{leonardo}] recurring on district events: {events_of_district}")
                or id_star(graph, events_of_district, leonardo=leonardo + 1)
                for events_of_district in district_events.values()
            ),
            summand,
        )
    # Line 7 and 8:
    elif id_star_line_8(cf_graph, new_event):
        raise Unidentifiable
    else:
        # Line 9
        return id_star_line_9(cf_graph)


def get_free_variables(graph: NxMixedGraph, event: Event) -> Set[Variable]:
    """Get all nodes in the graph that don't have values fixed by the event."""
    return {variable.get_base() for variable in graph.nodes() - set(event)}


def get_district_events(
    interventions_of_each_district: DistrictInterventions,
) -> Mapping[District, Event]:
    """Takes a district and a set of interventions, and applies the set of interventions to each node in the district"""
    return {
        district: {merge_interventions(node, interventions): node.get_base() for node in district}
        for district, interventions in interventions_of_each_district.items()
    }


def merge_interventions(
    variable: Variable, interventions: Collection[Intervention]
) -> CounterfactualVariable:
    """Takes a (potentially) counterfactual variable and a set of interventions and  returns the counterfactdual
    variable augmented with the new interventions"""
    interventions = set(
        Intervention(i.name, star=False) if not isinstance(i, Intervention) else i
        for i in interventions
    )
    if isinstance(variable, CounterfactualVariable):
        interventions = interventions.union(variable.interventions)
    return CounterfactualVariable(name=variable.name, interventions=tuple(sorted(interventions)))


def is_event_empty(event: Event) -> bool:
    r"""Run line 1 of the ID* algorithm.

    The first line states that if :math:`\event` is an empty conjunction, then its
    probability is 1, by convention.

    :param event: a conjunction of counterfactual variables
    """
    return len(event) == 0


def violates_axiom_of_effectiveness(event: Event) -> bool:
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


def remove_event_tautologies(event: Event) -> Event:
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
        if is_redundant_counterfactual( counterfactual, value )
    }
    return {
        counterfactual: value
        for counterfactual, value in event.items()
        if counterfactual not in redundant_counterfactuals
    }

def is_redundant_counterfactual( counterfactual: CounterfactualVariable, value: Intervention ) -> bool:
    """Check if a counterfactual variable is intervened on itself and has the same value as the intervention"""
    return isinstance(counterfactual, CounterfactualVariable) and any(
            intervention.name == counterfactual.name and value.star == intervention.star
            for intervention in counterfactual.interventions
        )


def is_self_intervened(counterfactual: CounterfactualVariable) -> bool:
    """Check if a counterfactual variable is intervened on itself """
    return isinstance(counterfactual, CounterfactualVariable) and any(
        intervention.name == counterfactual.name
        for intervention in counterfactual.interventions
    )


def id_star_line_4(graph: NxMixedGraph, event: Event) -> Tuple[NxMixedGraph, Optional[Event]]:
    r"""Run line 4 of the ``ID*`` algorithm.

    Line 4 invokes make-cg to construct a counterfactual graph :math:`G'` , and the
    corresponding relabeled counterfactual event.

    :param graph: an NxMixedGraph
    :param event: a conjunction of counterfactual variables
    :return: updated graph and event
    """
    new_graph, new_event = make_counterfactual_graph(graph, event)
    return new_graph, new_event


# FIXME this is unused -> delete it
# def id_star_line_5(graph: NxMixedGraph, event: Optional[Event]) -> Optional[Zero]:
#     r"""Run line 5 of the ``ID*`` algorithm.
#
#     Line 5 returns probability 0 if an inconsistency was found during the construction
#     of the counterfactual graph, for example, if two variables found to be the same in
#     event had different value assignments.
#     """
#     raise NotImplementedError


def id_star_line_6(
    graph: NxMixedGraph, event: Event
) -> Tuple[Collection[Variable], DistrictInterventions]:
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
    summand = get_free_variables(graph, event)
    interventions_of_each_district = get_district_domains(graph, event)
    return summand, interventions_of_each_district



def get_district_domains(graph: NxMixedGraph, event: Event) -> DistrictInterventions:
    """for each district, intervene on the domain of each variable not in the district.
    The domain of variables in the event query are restricted to their event value"""
    nodes = set(graph.nodes())
    return {
        district: domain_of_counterfactual_values(event, nodes - district)
        for district in graph.get_c_components()
        if 1 < len(district) or not is_self_intervened(list(district)[0])
    }


def domain_of_counterfactual_values(event: Event, variables: Iterable[Variable]) -> Set[Variable]:
    """Return domain of counterfactual values.
    If a variable is part of an event, just intervene on its observed value.
    Otherwise, intervene on all values in the variable's domain.
    """
    return {event[variable] if variable in event else variable.get_base() for variable in variables}


def id_star_line_8(graph: NxMixedGraph, event: Event) -> bool:
    r"""Run line 8 of the ID* algorithm.

    Line 8 says that if :math:`\event'` contains a "conflict," that is an inconsistent
    value assignment where at least one value is in the subscript, then we fail.

    :param graph: an NxMixedGraph
    :param event: a joint distribution over counterfactual variables
    :return: True if there is a conflict. False otherwise
    """
    return any(
        intervention.name == evidence.name and intervention.star != evidence.star
        for intervention in sub(graph)
        for evidence in ev(event)
    )


def sub(graph: NxMixedGraph) -> Collection[Intervention]:
    """sub() is the set of interventions that are in each counterfactual variable."""
    return {
        intervention
        for node in graph.nodes()
        if isinstance(node, CounterfactualVariable)
        for intervention in node.interventions
    }


def ev(event: Event) -> Collection[Intervention]:
    """ev(:) the set of values (either set or observed) appearing in a given counterfactual conjunction (or set of counterfactual events)"""
    return set(event.values())


def id_star_line_9(graph: NxMixedGraph) -> Probability:
    r"""Run line 9 of the ID* algorithm.

    Line 9 says if there are no conflicts, then it's safe to take the union of all
    subscripts in :math:`\event'` , and return the effect of the subscripts in :math:`\event'`
    on the variables in :math:`\event'`.

    :param graph: an NxMixedGraph
    :return: An interventional distribution.
    """
    interventions = sub(graph)
    return P[interventions](node.get_base() for node in graph.nodes())


# FIXME this is unused -> delete it
# def get_interventions(variables: Collection[Variable]) -> Collection[Variable]:
#     r"""Generate new Variables from the subscripts of counterfactual variables in the query."""
#     interventions = set()
#     for counterfactual in variables:
#         if isinstance(counterfactual, CounterfactualVariable):
#             interventions |= set(counterfactual.interventions)
#     return sorted(interventions)


# TODO update the docs: is this generally applicable, or only to graphs
#  constructed as parallel worlds? perhaps update the name?

# FIXME this is unused -> delete it
# def has_same_domain_of_values(node1: Variable, node2: Variable) -> bool:
#     if isinstance(node1, CounterfactualVariable) and isinstance(node2, CounterfactualVariable):
#         treatment1, treatment2 = _get_treatment_variables(node1), _get_treatment_variables(node2)
#     raise NotImplementedError
#

# FIXME this is unused -> delete it
# def has_same_value(event: Collection[Variable], node1: Variable, node2: Variable) -> bool:
#     n1 = None
#     for node in event:
#         if node == node1:
#             n1 = node
#     if n1 is None:
#         raise ValueError
#
#     n2 = None
#     for node in event:
#         if node == node2:
#             n2 = node
#     if n2 is None:
#         raise ValueError
#
#     # TODO not all variables have is_event().
#     #  Should normal, non-counterfactual variables have this function?
#     return has_same_function(n1, n2) and n1.is_event() and n2.is_event() and (n1.star == n2.star)
