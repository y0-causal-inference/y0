"""This file contains code that implements the original ID* algorithm.

The algorithm is described in the paper "Complete Identification Methods for the Causal Hierarchy"
but the algorithm itself is not complete, as there are identifiable queries that cannot be
identified with this algorithm.
"""

import logging
from typing import Collection, Set, Tuple, cast

from y0.algorithm.identify.cg import is_not_self_intervened, make_counterfactual_graph
from y0.algorithm.identify.id_star import (
    ConflictUnidentifiable,
    District,
    DistrictInterventions,
    get_conflicts,
    get_free_variables,
    id_star_line_9,
    remove_event_tautologies,
    violates_axiom_of_effectiveness,
)
from y0.dsl import Event, Expression, Intervention, One, Product, Sum, Variable, Zero
from y0.graph import NxMixedGraph

logger = logging.getLogger(__name__)


def original_id_star_line_6(
    graph: NxMixedGraph, event: Event
) -> Tuple[Collection[Variable], DistrictInterventions]:
    r"""Run line 6 of the ID* algorithm.

    Line 6 is analogous to Line 4 in the ID algorithm, it decomposes
    the problem into a set of subproblems, one for each C-component in
    the counterfactual graph. In the ID algorithm, the term
    corresponding to a given C-component :math:`S_i` of the causal
    diagram was the effect of all variables not in :math:`S_i` on
    variables in :math:`S_i` , in other words
    :math:`P_{\mathbf{v}\backslash s_i} (s_i )`, and the outermost
    summation on line 4 was over values of variables not in
    :math:`\mathbf{Y},\mathbf{X}`. Here, the term corresponding to a
    given C-component :math:`S^i` of the counterfactual graph
    :math:`G'` is the conjunction of counterfactual variables where
    each variable contains in its subscript all variables not in the
    C-component :math:`S^i` , in other words :math:`\mathbf{v}(G'
    )\backslash s^i` , and the outermost summation is over observable
    variables not in :math:`\event'` , that is over
    :math:`\mathbf{v}(G' ) \backslash \event'` , where we interpret
    :math:`\event'` as a set of counterfactuals, rather than a
    conjunction.

    Unfortunately, the conjunction of all counterfactual variables in
    the C-component where each variable contains its subscript over
    all variables not in the C-component results in a problem.  The
    problem is that the query contains redundant interventions that
    must be removed for the query to be identifiable.  The removal of
    redundant interventions is not specified as part of the algorithm
    itself.  Instead, it must be specified outside the algorithm.  The
    algorithm is not complete without this step.  Tikka et al. (2022)
    address this problem by removing the rudundant interventions in
    step 7 of the algorithm after running ID* on each C-component.
    However, it is simpler to not include redundant interventions in
    the first place.  Instead of intervening on every variable not in
    the C-component, we intervene on every variable that is a parent
    of the C-component.  This is also known as the Markov pillow. Zucker
    suggested this approach in an email to Shpitser on May 20, 2022. In
    this function we demonstrate what happens when we intervene on
    every variable not in the C-component.

    :param graph: an NxMixedGraph
    :param event: a conjunction of counterfactual variables
    :return: a set of Variables in summand, a dictionary of districts and events
    """
    # First we get the summand, then we intervene on each district
    summand = get_free_variables(graph, event)
    interventions_of_each_district = get_district_interventions(graph, event)
    return summand, interventions_of_each_district


def get_district_interventions(graph: NxMixedGraph, event: Event) -> DistrictInterventions:
    """For each district, intervene on the variables not in the district.

    Self-interventions are not considered part of the district

    :param graph: an NxMixedGraph
    :param event: a conjunction of counterfactual variables
    :return: a dictionary of districts and interventions of districts

    """
    nodes = set(node for node in graph.nodes() if is_not_self_intervened(node))
    return {
        district: intervene_on_district(district, nodes - district, event)
        for district in graph.subgraph(nodes).districts()
    }


def intervene_on_district(district: District, interventions: Set[Variable], event: Event) -> Event:
    """Intervene on the variables not in the district.

    Each variable in the district takes its value in the event if it has one.
    Otherwise, we set the value to ``-variable.get_base()``.

    :param district: a district of the graph
    :param event: a conjunction of counterfactual values
    :param interventions: a set of Variables to act as interventions
    :returns: A new event
    """
    interventions = {-i.get_base() for i in interventions}
    return {
        variable.intervene(interventions): (
            event[variable] if variable in event else cast(Intervention, -variable.get_base())
        )
        for variable in district
    }


def original_id_star(
    graph: NxMixedGraph, event: Event, *, _number_recursions: int = 0
) -> Expression:
    """Apply the ``ID*`` algorithm to the graph."""
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
        return original_id_star(graph, reduced_event, _number_recursions=_number_recursions + 1)
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
        summand, events_of_each_district = original_id_star_line_6(cf_graph, new_event)
        logger.debug("[%d] summand: %s", _number_recursions, summand)
        assert 1 < len(events_of_each_district)
        logger.debug(
            "[%d] recurring on each district: %s ", _number_recursions, events_of_each_district
        )
        return Sum.safe(
            Product.safe(
                original_id_star(
                    graph, events_of_district, _number_recursions=_number_recursions + 1
                )
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
