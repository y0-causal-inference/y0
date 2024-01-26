# -*- coding: utf-8 -*-

"""Implementation of the ID* algorithm."""

import itertools as itt
import logging
from typing import Collection, FrozenSet, Iterable, Mapping, Set, Tuple, cast

from .cg import is_not_self_intervened, make_counterfactual_graph
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
]

District = FrozenSet[Variable]
DistrictInterventions = Mapping[District, Event]

logger = logging.getLogger(__name__)


def id_star(graph: NxMixedGraph, event: Event, *, _number_recursions: int = 0) -> Expression:
    """Apply the ``ID*`` algorithm to the graph from [shpitser2012]_."""
    logger.debug(
        "[%d]: Calling ID* algorithm with graph G with\n\t nodes: %s\n"
        "\t directed: %s\n\t undirected %s\n"
        "\t outcome event: %s",
        _number_recursions,
        graph.nodes(),
        graph.directed.edges(),
        graph.undirected.edges(),
        event,
    )
    # Line 1: There's nothing in the counterfactual event (i.e., an empty conjunction),
    # then its probability is 1, by convention.
    if not event:
        return One()
    # Line 2: This violates the Axiom of Effectiveness
    if violates_axiom_of_effectiveness(event):
        return Zero()
    # Line 3: This is a tautological event and can be removed without affecting the probability
    reduced_event = remove_event_tautologies(event)
    if reduced_event != event:
        logger.debug("[%d] recurring on reduced event %s", _number_recursions, reduced_event)
        return id_star(graph, reduced_event, _number_recursions=_number_recursions + 1)
    # Line 4: invokes make-cg to construct a counterfactual graph :math:`G'` , and the
    # corresponding relabeled counterfactual event.
    cf_graph, new_event = make_counterfactual_graph(graph, event)
    logger.debug(
        "[%d] ID* Returned from make_counterfactual_graph(). New event: %s\n"
        "\tcounterfactual graph:\n"
        "\t nodes: %s\n"
        "\t directed: %s\n"
        "\t undirected: %s",
        _number_recursions,
        new_event,
        cf_graph.nodes(),
        cf_graph.directed.edges(),
        cf_graph.undirected.edges(),
    )
    # Line 5:
    if new_event is None:
        return Zero()

    # Line 6:
    nodes = set(node for node in cf_graph.nodes() if is_not_self_intervened(node))
    cf_subgraph = cf_graph.subgraph(nodes)
    if not cf_subgraph.is_connected():
        summand, events_of_each_district = id_star_line_6(cf_graph, new_event)
        logger.debug("[%d] summand: %s", _number_recursions, summand)
        assert 1 < len(events_of_each_district)
        logger.debug(
            "[%d] recurring on each district: %s ", _number_recursions, events_of_each_district
        )
        return Sum.safe(
            Product.safe(
                id_star(graph, events_of_district, _number_recursions=_number_recursions + 1)
                for events_of_district in events_of_each_district.values()
            ),
            summand,
        )

    # Line 7:
    conflicts = get_conflicts(cf_subgraph, new_event)
    if conflicts:
        raise ConflictUnidentifiable(cf_subgraph, new_event, conflicts)

    # Line 9
    return id_star_line_9(cf_subgraph)


class ConflictUnidentifiable(Unidentifiable):  # noqa:N818
    """An exception raised when line 8 of the ID* algorithm determines that the query is not identifiable.

    This happens because the event contains a conflict, i.e., an inconsistent value
    assignment where at least one value is in the subscript.
    """

    def __init__(
        self,
        cf_graph: NxMixedGraph,
        event: Event,
        conflicts: list[tuple[Intervention, Intervention]],
    ):
        """Instantiate the exception."""
        self.cf_graph = cf_graph
        self.event = event
        self.conflicts = conflicts


def get_free_variables(cf_graph: NxMixedGraph, event: Event) -> Set[Variable]:
    """Get the possible values of the counterfactual variables in the graph that are "free".

    i.e. that don't have values fixed by the event or a self-intervention.

    :param cf_graph: A counterfactual graph
    :param event: a conjunction of counterfactual variables
    :returns: The set of free variables
    """
    free_variables = {variable for variable in cf_graph.nodes() if is_not_self_intervened(variable)}
    return {v.get_base() for v in free_variables} - {e.get_base() for e in event}


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
    """Check if a counterfactual variable is intervened on itself and has the same value as the intervention."""
    if not isinstance(variable, CounterfactualVariable):
        return False
    return any(
        intervention.get_base() == value.get_base() and value.star == intervention.star
        for intervention in variable.interventions
    )


def id_star_line_6(
    cf_graph: NxMixedGraph, event: Event
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

    :param cf_graph: an NxMixedGraph
    :param event: a conjunction of counterfactual variables
    :return: a set of Variables in summand, a dictionary of districts and events
    """
    # Then we intervene on the Markov pillow of each district
    summand = get_free_variables(cf_graph, event)
    events_of_each_district = get_events_of_each_district(cf_graph, event)
    return summand, events_of_each_district


def get_events_of_each_district(graph: NxMixedGraph, event: Event) -> DistrictInterventions:
    """For each district, intervene each node in the Markov pillow of the district.

    Self-interventions are not considered part of the district

    :param graph: an NxMixedGraph
    :param event: a conjunction of counterfactual variables
    :return: a dictionary of districts and interventions of districts
    """
    nodes = {node for node in graph.nodes() if is_not_self_intervened(node)}
    subgraph = graph.subgraph(nodes)
    return {
        district: get_events_of_district(graph, district, event)
        for district in subgraph.districts()
    }


def get_events_of_district(graph, district, event) -> Event:
    """Create new events by intervening each node on the Markov pillow of the district.

    If the node in in the original event, then the value of the new event is the same as the original event.

    :param graph: an NxMixedGraph
    :param district: a district of the graph
    :param event: a conjunction of counterfactual variables
    :return: the events of the district
    """
    markov_pillow = graph.get_markov_pillow(district)
    if not markov_pillow:
        return {node.get_base(): _get_node_event(node, event) for node in district}
    return {
        node.get_base().intervene(markov_pillow): _get_node_event(node, event) for node in district
    }


def _get_node_event(node: Variable, event: Event) -> Intervention:
    if node in event:
        return event[node]
    return cast(Intervention, -node.get_base())


def get_conflicts(cf_graph: NxMixedGraph, event: Event) -> list[tuple[Intervention, Intervention]]:
    r"""Identify conflicts between interventions int the graph and value assignments in the event.

    .. note:: This is part of line 8 of the ID* algorithm

    :param cf_graph: an NxMixedGraph
    :param event: a joint distribution over counterfactual variables
    :return: A list of pairs of conflicts. The same intervention or value assignment may appear in multiple conflicts
    """
    interventions = get_cf_interventions(cf_graph.nodes())
    evidence = get_evidence(event)
    return [
        (intervention, ev)
        for intervention, ev in itt.product(interventions, evidence)
        if intervention.name == ev.name and intervention.star != ev.star
    ]


def get_cf_interventions(nodes: Iterable[Variable]) -> Set[Intervention]:
    """For the graph, get the set of interventions in each counterfactual variable (all the subscripts).

    .. note:: This was called ``sub()`` in the paper

    :param nodes: An iterable of Variables, potentially containing CounterfactualVariables
    :returns: The set of interventions over all counterfactual variables
    """
    return {
        intervention
        for node in nodes
        if isinstance(node, CounterfactualVariable)
        for intervention in node.interventions
    }


def get_evidence(event: Event) -> Set[Intervention]:
    """Get the evidence (interventions and values) of the counterfactual conjunction.

    The evidence (either set or observed) appearing in a given counterfactual conjunction
    (or set of counterfactual events)

    .. note:: This was called ``ev()`` in the paper

    :param event: a conjunction of counterfactual variables
    :returns: The set of values and interventions in the given counterfactual conjuction
    """
    return set(event.values()) | get_cf_interventions(event)


def id_star_line_9(cf_graph: NxMixedGraph) -> Probability:
    r"""Run line 9 of the ID* algorithm.

    Line 9 says if there are no conflicts, then it's safe to take the union of all
    subscripts in :math:`\event'` , and return the effect of the subscripts in :math:`\event'`
    on the variables in :math:`\event'`.

    :param cf_graph: A counterfactual graph
    :return: An interventional distribution.
    """
    interventions = get_cf_interventions(cf_graph.nodes())
    bases = [node.get_base() for node in cf_graph.nodes()]
    if len(interventions) > 0:
        return P[interventions](bases)
    else:
        return P(bases)
