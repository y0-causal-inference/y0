# -*- coding: utf-8 -*-

"""Implementation of the IDC algorithm."""

from typing import Collection, Mapping, Optional, Tuple

from .cg import has_same_function, make_counterfactual_graph
from .utils import Unidentifiable
from ..conditional_independencies import are_d_separated
from ...dsl import (
    CounterfactualVariable,
    CounterfactualEvent,
    Expression,
    One,
    P,
    Product,
    Sum,
    Variable,
    Zero,
    _get_treatment_variables,
)
from ...graph import NxMixedGraph

__all__ = [
    "idc_star_line_2",
    "id_star_line_1",
    "id_star_line_2",
    "id_star_line_3",
    "id_star_line_4",
    "id_star_line_5",
    "id_star_line_6",
    "id_star_line_7",
    "id_star_line_8",
    "id_star_line_9",
]


class Inconsistent(ValueError):
    pass


def id_star(graph: NxMixedGraph, event: CounterfactualEvent) -> Expression:
    # Line 0
    if id_star_line_1(graph, event) is not None:
        return One()
    # Line 2: This violates the Axiom of Effectiveness
    if id_star_line_2(graph, event) is not None:
        return Zero()
    # Line 3: This is a tautological event and can be removed without affecting the probability
    new_query = id_star_line_3(graph, event)
    if new_query is not None:
        return id_star(graph, new_query)

    # Line 4:
    try:
        new_graph, new_query = id_star_line_4(graph, event)
        vertices = set(new_graph.nodes())
        new_event = set(new_query.children)
    # Line 5:
    except Inconsistent:
        return Zero()
    # Line 6:
    if not new_graph.is_connected():
        return Sum.safe(
            Product.safe(
                id_star(new_graph, P[vertices - district](district))
                for district in new_graph.get_c_components()
            ),
            vertices - new_event,
        )
    # Line 7:
    else:
        # Line 8 is syntactically impossible with the dsl
        if id_star_line_8(new_graph, new_query):
            raise Unidentifiable
        else:
            # Line 9
            return id_star_line_9(new_graph, new_query)


# def get_val(counterfactual: CounterfactualVariable, graph: NxMixedGraph) -> Intervention:
#     var = Variable(counterfactual.name)
#     for intervention in counterfactual.interventions:
#         if Variable(intervention.name) in graph.ancestors_inclusive(var):
#             if intervention.star:
#                 return ~var
#     return -var


def id_star_line_1(graph: NxMixedGraph, event: CounterfactualEvent) -> Optional[One]:
    r"""Run line 1 of the ID* algorithm.

    The first line states that if :math:`\event` is an empty conjunction, then its
    probability is 1, by convention.

    :param graph: an NxMixedGraph
    :param event: a conjunction of counterfactual variables
    :return: One() or None
    """
    if len(event) == 0:
        return One()
    else:
        return None


def id_star_line_2(graph: NxMixedGraph, event: CounterfactualEvent) -> Optional[Zero]:
    r"""Run line 2 of the ID* algorithm.

    The second line states that if :math:`\event` contains a counterfactual
    which violates the Axiom of Effectiveness (Pearl, 2000), then :math:`\event`
    is inconsistent, and we return probability 0.

    :param graph: an NxMixedGraph
    :param event: a conjunction of counterfactual variables
    :return: Zero() or None
    """
    for counterfactual in event:
        if isinstance(counterfactual, CounterfactualVariable):
            for intervention in counterfactual.interventions:
                if (
                    (intervention.name == counterfactual.name)
                    and (event[counterfactual].star != intervention.star)
                ):
                    return Zero()
    return None


def id_star_line_3(
    graph: NxMixedGraph, event: CounterfactualEvent
) -> CounterfactualEvent:
    r"""Run line 3 of the ID* algorithm.

    The third line states that if a counterfactual contains its own value in the subscript,
    then it is a tautological event, and it can be removed from :math:`\event` without
    affecting its probability.

    :param graph: an NxMixedGraph
    :param event: a conjunction of counterfactual variables
    :return: updated event or None
    """
    for counterfactual in event:
        if isinstance(counterfactual, CounterfactualVariable):
           for intervention in counterfactual.interventions:
               if (
                  (intervention.name == counterfactual.name)
                  and (event[counterfactual].star == intervention.star)
                  ):
                  event.pop(counterfactual, None)
    return event

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


def id_star_line_6(graph: NxMixedGraph, event: CounterfactualEvent) -> Tuple[Collection[Variable], Collection[CounterfactualEvent]]:
    r"""Run line 6 of the ID* algorithm.

    Line 6 is analogous to Line 4 in the ID algorithm, it decomposes the problem into a
    set of subproblems, one for each C-component in the counterfactual graph. In the ID
    algorithm, the term corresponding to a given C-component :math:`S_i` of the causal
    diagram was the effect of all variables not in :math:`S_i` on variables in :math:`S_i` ,
    in other words :math:`P_{\mathbf{v}\backslash s_i (s_i )`, and the outermost summation
    on line 4 was over values of variables not in :math:`\mathbf{Y},\mathbf{X}`. Here, the
    term corresponding to a given C-component :math:`S^i` of the counterfactual graph :math:`G'`
    is the conjunction of counterfactual variables where each variable contains in its
    subscript all variables not in the C-component :math:`S^i` , in other words
    :math:`\mathbf{v}(G' )\backslash s^i` , and the outermost summation is over observable
    variables not in :math:`\event'` , that is over :math:`\mathbf{v}(G' ) \backslash \event'` ,
    where we interpret :math:`\event'` as a set of counterfactuals, rather than a conjunction.
    """
    vertices = set(graph.nodes())
    return #x [vertices - event], [vertices - district], rict in graph.get_c_components()]


def id_star_line_7(graph: NxMixedGraph, query: CounterfactualEvent) -> Collection[Expression]:
    r"""Run line 7 of the ID* algorithm.

    Line 7 is the base case, where our counterfactual graph has a single C-component
    """
    raise NotImplementedError


def id_star_line_8(graph: NxMixedGraph, query: CounterfactualEvent) -> bool:
    r"""Run line 8 of the ID* algorithm.

    Line 8 says that if :math:`\event'` contains a "conflict," that is an inconsistent
    value assignment where at least one value is in the subscript, then we fail.

    :param graph: an NxMixedGraph
    :param query: a joint distribution over counterfactual variables
    :return: True if there is a conflict. False otherwise
    """
    interventions = get_interventions(query)
    evidence = get_varnames(query)
    return any(
        (intervention.name in evidence) and evidence[intervention.name] != intervention.star
        for intervention in interventions
    )


def id_star_line_9(graph: NxMixedGraph, query: CounterfactualEvent) -> CounterfactualEvent:
    r"""Run line 9 of the ID* algorithm.

    Line 9 says if there are no conflicts, then it's safe to take the union of all
    subscripts in :math:`\event'` , and return the effect of the subscripts in :math:`\event'`
    on the variables in :math:`\event'`.

    :param graph: an NxMixedGraph
    :param query: a joint distribution over counterfactual variables
    :return: An interventional distribution.
    """
    interventions = get_interventions(query)
    variables = get_varnames(query)
    return P[interventions](Variable(name=name, star=star) for name, star in variables.items())


def idc_star_line_2(graph: NxMixedGraph, query: CounterfactualEvent) -> Expression:
    r"""Run line 2 of the IDC* algorithm.

    The second line states that if :math:`\event` contains a counterfactual which violates
    the Axiom of Effectiveness (Pearl, 2000), then :math:`\event` is inconsistent, and we
    return probability 0.
    """
    delta = query.parents
    event_and_delta = query.uncondition()
    return make_counterfactual_graph(graph, event_and_delta)


def idc_star_line_4(graph: NxMixedGraph, query: CounterfactualEvent) -> bool:
    r"""Run line 4 of the IDC* algorithm.

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


def idc_star(graph: NxMixedGraph, query: CounterfactualEvent) -> Expression:
    r"""Run the IDC* algorithm.

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


def get_varnames(query: CounterfactualEvent) -> Mapping[str, Optional[bool]]:
    """Return new Variables generated from the names of the outcome variables in the query."""
    return {child.name: child.star for child in query.children}


def get_interventions(query: CounterfactualEvent) -> Collection[Variable]:
    r"""Generate new Variables from the subscripts of counterfactual variables in the query."""
    interventions = set()
    for counterfactual in query.children:
        if isinstance(counterfactual, CounterfactualVariable):
            interventions |= set(counterfactual.interventions)
    return sorted(interventions)


# TODO update the docs: is this generally applicable, or only to graphs
#  constructed as parallel worlds? perhaps update the name?


def has_same_domain_of_values(node1: Variable, node2: Variable) -> bool:
    if isinstance(node1, CounterfactualVariable) and isinstance(node2, CounterfactualVariable):
        treatment1, treatment2 = _get_treatment_variables(node1), _get_treatment_variables(node2)
    raise NotImplementedError


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


# TODO unused, isn't this already implemented in NxMixedGraph?
def to_adj(
    graph: NxMixedGraph,
) -> Tuple[
    Collection[Variable],
    Mapping[Variable, Collection[Variable]],
    Mapping[Variable, Collection[Variable]],
]:
    nodes: list[Variable] = list(graph.nodes())
    directed: dict[Variable, list[Variable]] = {u: [] for u in nodes}
    undirected: dict[Variable, list[Variable]] = {u: [] for u in nodes}
    for u, v in graph.directed.edges():
        directed[u].append(v)
    for u, v in graph.undirected.edges():
        undirected[u].append(v)
    return nodes, directed, undirected
