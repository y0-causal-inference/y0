# -*- coding: utf-8 -*-

"""Implementation of the ``IDC*`` algorithm."""

from .cg import make_counterfactual_graph
from .id_star import id_star
from ..conditional_independencies import are_d_separated
from ...dsl import CounterfactualEvent, Expression, P, Variable, Zero
from ...graph import NxMixedGraph

__all__ = [
    "idc_star",
]


# FIXME replace with :class:`y0.algorithms.identify.utils.Unidentifiable`
class Inconsistent(ValueError):
    pass


def idc_star(graph: NxMixedGraph, query: CounterfactualEvent) -> Expression:
    r"""Run the ``IDC*`` algorithm.

    INPUT:
        G a causal diagram,
        :math:`\event` a conjunction of counterfactual outcomes,
        :math:`\delta` a conjunction of counterfactual observations
    :returns: an expression for :math:`P(\event | \delta)` in terms of P, FAIL, or UNDEFINED
    """
    delta = set(query.parents)
    if not delta:
        raise ValueError(f"Query {query} must be conditional")
    # Line 1:
    if not id_star(graph, P(delta)):
        raise ValueError(f"Query {query} is undefined")
    event = set(query.children)
    # Line 2:
    try:
        new_graph, new_query = make_counterfactual_graph(graph, P(event.union(delta)))
        new_event = {g for g in event if g in new_query.children}
        new_delta = {d for d in delta if d in new_query.children}
        vertices = set(new_graph.nodes())
    # Line 3:
    except Inconsistent:
        # (f"query {event.union(delta)} is inconsistent")
        return Zero()
    # Line 4:
    for counterfactual in new_delta:
        # TODO do we need to extend the notion of d-separation from 1-1 to 1-many?
        if are_d_separated(new_graph.remove_out_edges(counterfactual), counterfactual, new_event):
            counterfactual_value = Variable(counterfactual.name)
            parents = new_delta - {counterfactual}
            children = {g.remove_in_edges(counterfactual_value) for g in new_event}
            return idc_star(graph, P(children | parents))
    # Line 5:
    estimand = id_star(graph, new_query)
    return estimand.marginalize(vertices - delta)


def idc_star_line_2(graph: NxMixedGraph, query: CounterfactualEvent) -> Expression:
    r"""Run line 2 of the ``IDC*`` algorithm.

    The second line states that if :math:`\event` contains a counterfactual which violates
    the Axiom of Effectiveness (Pearl, 2000), then :math:`\event` is inconsistent, and we
    return probability 0.
    """
    delta = query.parents
    event_and_delta = query.uncondition()
    return make_counterfactual_graph(graph, event_and_delta)


def idc_star_line_4(graph: NxMixedGraph, query: CounterfactualEvent) -> bool:
    r"""Run line 4 of the ``IDC*`` algorithm.

    Line 4 of IDC* is the central line of the algorithm and is
    analogous to line 1 of IDC. In IDC, we moved a value
    assignment :math:`Z = z` from being observed to being fixed if
    there were no back-door paths from :math:`Z` to the outcome
    variables :math:`Y` given the context of the effect of
    :math:`do(\mathbf{x})`. Here in IDC*, we move a counterfactual
    value assignment :math:`Y_\mathbf{x} = y` from being observed (that is being a
    part of :math:`\delta`), to being fixed (that is appearing in every
    subscript of :math:`\event'` ) if there are no back-door paths from :math:`Y_\mathbf{x}` to
    the counterfactual of interest :math:`\event'` .
    """
    event = set(query.children)
    raise NotImplementedError
