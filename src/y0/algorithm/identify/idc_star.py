# -*- coding: utf-8 -*-

"""Implementation of the IDC* algorithm."""

from .cg import is_not_self_intervened, make_counterfactual_graph
from .id_star import id_star
from .utils import Unidentifiable
from ..conditional_independencies import are_d_separated
from ...dsl import Event, Expression, Variable, Zero
from ...graph import NxMixedGraph


def idc_star(graph: NxMixedGraph, outcomes: Event, conditions: Event, leonardo=0) -> Expression:
    r"""Run the IDC* algorithm.

    :param graph: The causal graph
    :param outcomes: The outcome events corresponds to :math:`\gamma`
    :param conditions: The condition events corresponds to :math:`\delta`
    :param leonardo: The number of times to apply the Leonardo rule
    :returns: An expression created by the :func:`idc_star` algorithm after simplifying the original query
    """
    # Run ID* algorithm
    try:
        print(f"[{leonardo}]: line 1 IDC* algorithm: call ID* algorithm with events {conditions}")
        if isinstance(id_star(graph, conditions, leonardo + 1), Zero):
            raise UndefinedError("The ID* algorithm returned 0, so IDC* cannot be applied.")
        else:
            print(f"[{leonardo}]: line 1 IDC* algorithm: ID* algorithm is not inconsistent")
    except Unidentifiable:
        print(
            f"[{leonardo}]: line 1 IDC* algorithm: ID* algorithm is not identifiable, but is also not  inconsistent."
        )
    print(
        f"[{leonardo}]: line 2 IDC* algorithm: make counterfactual graph. Events: {outcomes | conditions}"
    )
    new_graph, new_events = make_counterfactual_graph(graph, outcomes | conditions)

    print(
        f"[{leonardo}]: Returned from make_counterfactual_graph with New events: {new_events}\n\tcounterfactual graph:\n\t nodes: {new_graph.nodes()}\n\t directed: {new_graph.directed.edges()}\n\t undirected: {new_graph.undirected.edges()}"
    )
    if new_events is None:
        print(
            f"[{leonardo}]: line 3 IDC* algorithm: make_counterfactual_graph is inconsistent. Returning Zero."
        )
        return Zero()
    else:
        print(
            f"[{leonardo}]: line 3 IDC* algorithm: make_counterfactual_graph is not inconsistent. Continuing."
        )
    for condition in conditions:
        print(
            f"[{leonardo}]: line 4 IDC* algorithm: for each condition, check if rule 2 of do calculus applies"
        )
        if rule_2_of_do_calculus_applies(new_graph, outcomes, condition):
            print(
                f"\t[{leonardo}]: line 4 IDC* algorithm: rule 2 of do calculus applies {outcomes} is D-separated from {condition} in G'_({condition}_bar)"
            )
            new_outcomes = {
                outcome.intervene(condition) if condition in new_graph.ancestors_inclusive(outcome) else outcome: value
                for outcome, value in outcomes.items()
            }
            new_conditions = {k: v for k, v in conditions.items() if k != condition}
            print(
                f"\t[{leonardo}]: line 4 IDC* algorithm: call IDC* algorithm on new outcomes {new_outcomes} and new conditions {new_conditions}"
            )
            return idc_star(graph, new_outcomes, new_conditions, leonardo + 1)
        else:
            print(
                f"\t[{leonardo}]: line 4 IDC* algorithm: rule 2 of do calculus does not apply to condition {condition}"
            )
    print(
        f"[{leonardo}]: line 5 IDC* algorithm: Call ID* algorithm on new events {new_events} and old graph:\n\t nodes: {graph.nodes()}\n\t directed: {graph.directed.edges()}\n\t undirected: {graph.undirected.edges()}"
    )
    
    P_prime = id_star(graph, new_events)
    if len(conditions) == 0:
        return P_prime
    else:
        return P_prime.conditional(conditions.values())


def rule_2_of_do_calculus_applies(
    graph: NxMixedGraph, outcomes: set[Variable], condition: Variable
) -> bool:
    r"""Check if Rule 2 of the Do-Calculus applies to the conditioned variable.

    :param identification: The identification tuple
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
    blocked_nodes = {n for n in graph.nodes() if not is_not_self_intervened(n)}
    return all(
        are_d_separated(
            graph.remove_out_edges(condition), outcome, condition, conditions=blocked_nodes
        )
        for outcome in outcomes
    )
