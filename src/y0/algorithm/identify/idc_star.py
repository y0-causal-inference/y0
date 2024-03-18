# -*- coding: utf-8 -*-

"""Implementation of the IDC* algorithm."""

import logging
from typing import Iterable, Tuple

from .cg import is_not_self_intervened, make_counterfactual_graph
from .id_star import id_star
from .utils import Unidentifiable
from ..conditional_independencies import are_d_separated
from ...dsl import Event, Expression, Variable, Zero
from ...graph import NxMixedGraph

__all__ = [
    "idc_star",
]

logger = logging.getLogger(__name__)


def get_new_outcomes_and_conditions(
    new_event: Event, outcomes: Event, conditions: Event
) -> Tuple[Event, Event]:
    """Get the new outcomes and conditions."""
    remaining_outcomes, missing_outcomes = get_remaining_and_missing_events(new_event, outcomes)
    remaining_conditions, missing_conditions = get_remaining_and_missing_events(
        new_event, conditions
    )
    new_event_keys = set(new_event) - set(outcomes) - set(conditions)
    if len(missing_outcomes) > 0 and len(missing_conditions) > 0:
        for outcome in new_event_keys:
            if outcome.get_base() in {missing.get_base() for missing in missing_outcomes}:
                remaining_outcomes[outcome] = new_event[outcome]
        for condition in new_event_keys:
            if condition.get_base() in {missing.get_base() for missing in missing_conditions}:
                remaining_conditions[condition] = new_event[condition]
        return remaining_outcomes, remaining_conditions
    elif len(missing_outcomes) > 0:
        for outcome in new_event_keys:
            remaining_outcomes[outcome] = new_event[outcome]
        return remaining_outcomes, remaining_conditions
    elif len(missing_conditions) > 0:
        for condition in new_event_keys:
            remaining_conditions[condition] = new_event[condition]
        return remaining_outcomes, remaining_conditions
    else:
        return remaining_outcomes, remaining_conditions


def get_remaining_and_missing_events(new_event: Event, old_event: Event) -> Tuple[Event, Event]:
    """Get the outcome from the event."""
    remaining = {k: v for k, v in old_event.items() if k in new_event}
    missing = {k: v for k, v in old_event.items() if k not in new_event}
    return remaining, missing


def idc_star(
    graph: NxMixedGraph, outcomes: Event, conditions: Event, *, _number_recursions: int = 0
) -> Expression:
    r"""Run the IDC* algorithm from [shpitser2012]_.

    :param graph: The causal graph
    :param outcomes: The outcome events corresponds to :math:`\gamma`
    :param conditions: The condition events corresponds to :math:`\delta`
    :param _number_recursions: The number of times the algorithm has recurred
    :returns: An expression created by the :func:`idc_star` algorithm after simplifying the original query
    :raises ValueError: If ID* returns zero
    """
    logger.debug(
        f"[{_number_recursions}]: Calling IDC* algorithm with graph G with\n\t nodes: {graph.nodes()}\n\t directed: "
        f"{graph.directed.edges()}\n\t undirected {graph.undirected.edges()}\n\t outcomes: {outcomes}\n\t "
        f"and conditions: {conditions}"
    )
    try:
        logger.debug(
            "[%d]: line 1 IDC* algorithm: call ID* algorithm with events %s",
            _number_recursions,
            conditions,
        )
        if isinstance(id_star(graph, conditions), Zero):
            raise ValueError("The ID* algorithm returned 0, so IDC* cannot be applied.")
        else:
            logger.debug(
                "[%d]: line 1 IDC* algorithm: ID* algorithm is not inconsistent", _number_recursions
            )
    except Unidentifiable:
        logger.debug(
            "[%d]: line 1 IDC* algorithm: ID* algorithm is not identifiable, but is also not inconsistent.",
            _number_recursions,
        )

    _events = outcomes | conditions
    logger.debug(
        "[%d]: line 2 IDC* algorithm: make counterfactual graph. Events: %s",
        _number_recursions,
        _events,
    )
    cf_graph, new_events = make_counterfactual_graph(graph, _events)
    logger.debug(
        f"[{_number_recursions}]: IDC* returned from make_counterfactual_graph with New events: {new_events}\n"
        "\tcounterfactual graph:\n"
        f"\t nodes: {cf_graph.nodes()}\n"
        f"\t directed: {cf_graph.directed.edges()}\n"
        f"\t undirected: {cf_graph.undirected.edges()}"
    )
    if new_events is None:
        logger.debug(
            f"[{_number_recursions}]: line 3 IDC* algorithm: make_counterfactual_graph is inconsistent. Returning Zero."
        )
        return Zero()

    new_outcomes, new_conditions = get_new_outcomes_and_conditions(new_events, outcomes, conditions)
    logger.debug(
        f"[{_number_recursions}]: line 3 IDC* algorithm: make_counterfactual_graph is not inconsistent.\n"
        f"\tNew outcomes: {outcomes}\tNew conditions: {conditions}\n"
    )
    for condition in new_conditions:
        logger.debug(
            f"[{_number_recursions}]: line 4 IDC* algorithm: for each condition, check if rule 2 of do calculus applies"
        )
        if cf_rule_2_of_do_calculus_applies(cf_graph, new_outcomes, condition):
            logger.debug(
                f"\t[{_number_recursions}]: line 4 IDC* algorithm: rule 2 of do calculus applies:\n\t\t{outcomes} "
                f"""is D-separated from {condition} in G{"'"*(_number_recursions + 1)} ({condition}_bar)"""
            )
            new_outcomes = {
                (
                    outcome.intervene(condition)
                    if condition in cf_graph.ancestors_inclusive(outcome)
                    else outcome
                ): value
                for outcome, value in new_outcomes.items()
            }
            new_conditions = {k: v for k, v in new_conditions.items() if k != condition}
            logger.debug(
                f"\t[{_number_recursions}]: line 4 IDC* algorithm: call IDC* algorithm on new outcomes {new_outcomes} "
                f"and new conditions {new_conditions}"
            )
            return idc_star(
                graph, new_outcomes, new_conditions, _number_recursions=_number_recursions + 1
            )
        else:
            logger.debug(
                f"\t[{_number_recursions}]: line 4 IDC* algorithm: rule 2 of do calculus "
                f"does not apply to condition {condition}"
            )
    logger.debug(
        f"[{_number_recursions}]: line 5 IDC* algorithm: Call ID* algorithm on new events {new_outcomes} "
        f"| {new_conditions} and old graph:\n"
        f"\t nodes: {graph.nodes()}\n"
        f"\t directed: {graph.directed.edges()}\n"
        f"\t undirected: {graph.undirected.edges()}"
    )
    id_star_estimand = id_star(
        graph, new_outcomes | new_conditions, _number_recursions=_number_recursions + 1
    )
    logger.debug(f"[{_number_recursions}]: Returned from ID* with estimand {id_star_estimand}")
    if len(conditions) == 0:
        return id_star_estimand

    idc_star_estimand = id_star_estimand.conditional([c.get_base() for c in conditions])
    logger.debug(f"[{_number_recursions}]: Returning from IDC* with estimand {idc_star_estimand}")
    return idc_star_estimand


def cf_rule_2_of_do_calculus_applies(
    cf_graph: NxMixedGraph, outcomes: Iterable[Variable], condition: Variable
) -> bool:
    r"""Check if Rule 2 of the Do-Calculus applies to the conditioned variable.

    Rule 2 is about Observations that are interchangeable with Interventions.
    This occurs when there is no backdoor path from the conditioned variable to the outcomes.
    Rule 2 operationalizes this criterion to say that if we remove the outgoing edge from
    the conditioned variable, then if the intervened node is D-separated from all outcomes,
    then the backdoor criterion has been satisfied.

    :param cf_graph: an NxMixedGraph
    :param outcomes: The outcomes to check
    :param condition: The condition to check
    :returns: If rule 2 applies, see below.

    If Rule 2 of the do calculus applies to the conditioned variable, then it can be converted to a do variable.

    .. math::

        \newcommand\ci{\perp\!\!\!\perp}
        \newcommand{\ubar}[1]{\underset{\bar{}}{#1}}
        \newcommand{\obar}[1]{\overset{\bar{}}{#1}}
        \text{if } (\exists Z \in \mathbf{Z})(\mathbf{Y} \ci Z | \mathbf{X}, \mathbf{Z}
        - \{Z\})_{G_{\bar{\mathbf{X}}\ubar{Z}}} \\
        \text{then } P(\mathbf{Y}|do(\mathbf{X}),\mathbf{Z}) = P(\mathbf Y|do(\mathbf X), do(Z), \mathbf{Z} - \{Z\})
    """
    #: also called "blocked nodes"
    conditions = {n for n in cf_graph.nodes() if not is_not_self_intervened(n)}
    graph_mod = cf_graph.remove_out_edges(condition)
    return all(
        are_d_separated(graph_mod, outcome, condition, conditions=conditions)
        for outcome in outcomes
    )
