# -*- coding: utf-8 -*-

"""Implementation of the ID* algorithm."""

from typing import Collection, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple

from .cg import is_not_self_intervened, make_counterfactual_graph
from .utils import Unidentifiable
from ..conditional_independencies import are_d_separated
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

# __all__ = [
#     "id_star",
#     "get_district_domains",
#     "domain_of_counterfactual_values",
#     "is_self_intervened",
#     "is_event_empty",
#     "violates_axiom_of_effectiveness",
#     "remove_event_tautologies",
# ]

District = FrozenSet[Variable]
DistrictInterventions = Mapping[District, Event]


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
    print(
        f"[{leonardo}] new event: {new_event}\n\tcounterfactual graph:\n\t nodes: {cf_graph.nodes()}\n\t directed: {cf_graph.directed.edges()}\n\t undirected: {cf_graph.undirected.edges()}"
    )
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
    # Line 7:
    elif id_star_line_8(cf_graph, new_event):
        raise Unidentifiable
    else:
        # Line 9
        return id_star_line_9(cf_graph)


def get_free_variables(graph: NxMixedGraph, event: Event) -> Set[Variable]:
    """Get the possible values of the counterfactual variables in the graph that don't have values fixed by the event or a self-intervention."""
    free_variables = {
        variable for variable in graph.nodes() if is_not_self_intervened(variable)
    } - set(event)
    return {v.get_base() for v in free_variables}


def get_district_events(district_interventions: DistrictInterventions) -> Mapping[District, Event]:
    """Takes a district and a set of interventions, and applies the set of interventions to each node in the district"""
    return {
        district: get_district_event(district, interventions)  # FIXME passing interventions here is wrong
        for district, interventions in district_interventions.items()
    }


def get_district_event(district: District, interventions: Iterable[Intervention]) -> Event:
    return {
        merge_interventions(
            node, interventions
        ): _get_intervention_from_note_for_get_district_event(node)
        for node in district
    }


def _get_intervention_from_note_for_get_district_event(node) -> Intervention:
    node_base = node.get_base()
    raise NotImplementedError


def merge_interventions(
    variable: Variable, interventions: Iterable[Intervention]
) -> CounterfactualVariable:
    """Take a (potentially) counterfactual variable and a set of interventions and return the counterfactual
    variable augmented with the new interventions."""
    processed_interventions = set(
        Intervention(intervention.name, star=False)
        if not isinstance(intervention, Intervention)
        else intervention
        for intervention in interventions
    )
    if isinstance(variable, CounterfactualVariable):
        processed_interventions.update(variable.interventions)
    return CounterfactualVariable(
        name=variable.name, interventions=tuple(sorted(processed_interventions))
    )


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
        intervention.get_base() == value.get_base() and value.star != intervention.star
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
    return {
        variable: value
        for variable, value in event.items()
        if not is_redundant_counterfactual(variable, value)
    }


def is_redundant_counterfactual(variable: Variable, value: Intervention) -> bool:
    """Check if a counterfactual variable is intervened on itself and has the same value as the intervention"""
    if not isinstance(variable, CounterfactualVariable):
        return False
    else:
        return any(
            intervention.get_base() == value.get_base() and value.star == intervention.star
            for intervention in variable.interventions
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


def old_id_star_line_6(
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
    # First we get the summand
    # Then we intervene on each district
    summand = get_free_variables(graph, event)
    interventions_of_each_district = get_district_interventions(graph, event)

    return summand, interventions_of_each_district


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
    # First we get the summand
    # Then we intervene on each district
    summand = get_free_variables(graph, event)
    events_of_each_district = get_events_of_each_district(graph, event)
    return summand, events_of_each_district


def get_events_of_each_district(graph: NxMixedGraph, event: Event) -> DistrictInterventions:
    """For each district, intervene each node on the Markov pillow of the district.
    Self-interventions are not considered part of the district
    """
    nodes = set(node for node in graph.nodes() if is_not_self_intervened(node))
    new_events_of_each_district = {}
    for district in graph.subgraph(nodes).get_c_components():
        new_events_of_each_district[district] = get_events_of_district(graph, district, event)
    return new_events_of_each_district


def get_events_of_district(graph, district, event) -> Event:
    """Create new events by intervening each node on the Markov pillow of the district.
    If the node in in the original event, then the value of the new event is the same as the original event.
    :param graph: an NxMixedGraph
    :param district: a district of the graph
    :param event: a conjunction of counterfactual variables
    :return: the events of the district
    """
    markov_pillow = get_markov_pillow(graph, district)
    new_events_of_district = {}
    if len(markov_pillow) == 0:
        for node in district:
            new_events_of_district[node.get_base()] = (
                event[node] if node in event else -node.get_base()
            )
    else:
        for node in district:
            new_events_of_district[node.get_base().intervene(markov_pillow)] = (
                event[node] if node in event else -node.get_base()
            )
    return new_events_of_district


def get_district_interventions(graph: NxMixedGraph, event: Event) -> DistrictInterventions:
    """For each district, intervene on the variables not in the district.
    Self-interventions are not considered part of the district
    """
    nodes = set(node for node in graph.nodes() if is_not_self_intervened(node))
    return {
        district: intervene_on_district(district, nodes - district, event)
        for district in graph.subgraph(nodes).get_c_components()
    }


def intervene_on_district(district: District, interventions: Set[Variable], event: Event) -> Event:
    """For each district, intervene on the variables not in the district.
    The value of each variable in the district is restricted to its value in the event if it has one.
    Otherwise, we set the value to -variable.get_base()
    """
    interventions = {-i.get_base() for i in interventions}
    return dict(
        [
            (variable.intervene(interventions), event[variable])
            if variable in event
            else (variable.intervene(interventions), -variable.get_base())
            for variable in district
        ]
    )


def get_markov_pillow(graph: NxMixedGraph, district: Collection[Variable]) -> Collection[Variable]:
    """for each district, intervene on the domain of each parent not in the district."""
    parents_of_district = set()
    for node in district:
        parents_of_district |= set(graph.directed.predecessors(node))
    return parents_of_district - set(district)


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


def rule_3_applies(
    graph: NxMixedGraph, district_events: DistrictInterventions
) -> DistrictInterventions:
    """Apply rule 3 to each intervention in the district

    :param graph: A counterfactual graph
    :param district: A tuple of counterfactual variables representing the C-component (district)
    :return: The collection of counterfactual variables and the interventions that are D separated according to the graph
    """
    new_district_events: dict[District, Event] = dict()
    intervention_events = ...  # FIXME this variable does not exist!
    for district, events in district_events.items():
        new_district_events[district] = dict()
        for counterfactual in intervention_events:
            new_counterfactual = simplify_counterfactual(graph, district, counterfactual)
            new_district_events[district][new_counterfactual] = events[counterfactual]
    return new_district_events


def simplify_counterfactual(
    graph: NxMixedGraph,
    district_nodes: Set[Variable],
    counterfactual_variable: CounterfactualVariable,
) -> Variable:
    """Simplify a counterfactual variable by only including interventions that are in the Markov pillow of the district."""
    if not isinstance(counterfactual_variable, CounterfactualVariable):
        raise TypeError
    intervention_nodes = get_markov_pillow(graph, district_nodes)
    if len(intervention_nodes) == 0:
        return Variable(name=counterfactual_variable.name, star=counterfactual_variable.star)
    else:
        return Variable(
            name=counterfactual_variable.name, star=counterfactual_variable.star
        ).intervene(intervention_nodes)


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
