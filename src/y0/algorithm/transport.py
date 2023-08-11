"""Implement of surrogate outcomes and transportability from https://arxiv.org/abs/1806.07172."""

import logging
from copy import deepcopy
from dataclasses import dataclass
from operator import attrgetter
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Union, cast

from y0.algorithm.conditional_independencies import are_d_separated
from y0.dsl import (
    CounterfactualVariable,
    Expression,
    Intervention,
    One,
    Population,
    Product,
    Sum,
    Variable,
)
from y0.graph import NxMixedGraph

__all__ = [
    "transport",
    "TransportQuery",
]

logger = logging.getLogger(__name__)

TARGET_DOMAIN = Population("pi*")


def get_nodes_to_transport(
    *,
    surrogate_interventions: Union[Set[Variable], Variable],
    surrogate_outcomes: Union[Set[Variable], Variable],
    graph: NxMixedGraph,
) -> Set[Variable]:
    """Identify which nodes the transport nodes should point to.

    :param surrogate_interventions: The interventions performed in an experiment.
    :param surrogate_outcomes: The outcomes observed in an experiment.
    :param graph: The graph of the target domain.
    :returns: A set of variables representing target domain nodes where transportability nodes should be added.
    """
    if isinstance(surrogate_interventions, Variable):
        surrogate_interventions = {surrogate_interventions}
    if isinstance(surrogate_outcomes, Variable):
        surrogate_outcomes = {surrogate_outcomes}

    # Find the c_component with surrogate_outcomes
    c_component_surrogate_outcomes: Set[Variable] = set()
    for component in graph.districts():
        # Check if surrogate_outcomes is present in the current set
        if surrogate_outcomes.intersection(component):
            c_component_surrogate_outcomes.update(component)

    # subgraph where interventions in edges are removed
    interventions_overbar = graph.remove_in_edges(surrogate_interventions)
    # Ancestors of surrogate_outcomes in interventions_overbar
    Ancestors_surrogate_outcomes = interventions_overbar.ancestors_inclusive(surrogate_outcomes)

    # Descendants of interventions in graph
    Descendants_interventions = graph.descendants_inclusive(surrogate_interventions)

    return (Descendants_interventions - surrogate_outcomes).union(
        c_component_surrogate_outcomes - Ancestors_surrogate_outcomes
    )


_TRANSPORT_PREFIX = "T_"


def transport_variable(variable: Variable) -> Variable:
    if isinstance(variable, (CounterfactualVariable, Intervention)):
        raise TypeError
    return Variable(_TRANSPORT_PREFIX + variable.name)


def is_transport_node(node: Variable) -> bool:
    return not isinstance(node, (CounterfactualVariable, Intervention)) and node.name.startswith(
        _TRANSPORT_PREFIX
    )


def get_transport_nodes(graph: NxMixedGraph) -> Set[Variable]:
    return {node for node in graph.nodes() if is_transport_node(node)}


def create_transport_diagram(
    *,
    nodes_to_transport: Iterable[Variable],
    graph: NxMixedGraph,
) -> NxMixedGraph:
    """Create a NxMixedGraph identical to graph but with transport nodes added.

    :param nodes_to_transport: nodes which have transport nodes pointing to them.
    :param graph: The graph of the target domain.
    :returns: graph with transport nodes added
    """
    # TODO we discussed the possibility of using a dictionary with needed nodes
    #  instead of creating a graph for each diagram.
    rv = NxMixedGraph()
    for node in graph.nodes():
        rv.add_node(node)
    for u, v in graph.directed.edges():
        rv.add_directed_edge(u, v)
    for u, v in graph.undirected.edges():
        rv.add_undirected_edge(u, v)
    for node in nodes_to_transport:
        transport_node = transport_variable(node)
        rv.add_directed_edge(transport_node, node)
    return rv


@dataclass
class TransportQuery:
    target_interventions: Set[Variable]
    target_outcomes: Set[Variable]
    graphs: Dict[Population, NxMixedGraph]
    domains: Set[Population]
    surrogate_interventions: Dict[Population, Set[Variable]]
    target_experiments: Set[Variable]


@dataclass
class TRSOQuery:
    target_interventions: Set[Variable]
    target_outcomes: Set[Variable]
    expression: Expression
    active_interventions: Set[Variable]
    domain: Population
    domains: Set[Population]
    graphs: Dict[Population, NxMixedGraph]
    surrogate_interventions: Dict[Population, Set[Variable]]


def surrogate_to_transport(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    graph: NxMixedGraph,
    surrogate_outcomes: Dict[Population, Set[Variable]],
    surrogate_interventions: Dict[Population, Set[Variable]],
) -> TransportQuery:
    """Create transportability diagrams and query from a surrogate outcome problem.

    :param target_outcomes: A set of target variables for causal effects.
    :param target_interventions: A set of interventions for the target domain.
    :param graph: The graph of the target domain.
    :param surrogate_outcomes:
    :param surrogate_interventions:
    :returns: An octuple representing the query transformation of a surrogate outcome query.
    """
    if set(surrogate_outcomes) != set(surrogate_interventions):
        raise ValueError("Inconsistent surrogate outcome and intervention domains")

    graphs = {
        domain: create_transport_diagram(
            graph=graph,
            nodes_to_transport=get_nodes_to_transport(
                surrogate_interventions=surrogate_interventions[domain],
                surrogate_outcomes=domain_outcomes,
                graph=graph,
            ),
        )
        for domain, domain_outcomes in surrogate_outcomes.items()
    }
    graphs[TARGET_DOMAIN] = graph

    return TransportQuery(
        target_interventions=target_interventions,
        target_outcomes=target_outcomes,
        graphs=graphs,
        domains=set(surrogate_outcomes),
        surrogate_interventions=surrogate_interventions,
        target_experiments=set(),
    )


def trso_line1(
    target_outcomes: Set[Variable],
    expression: Expression,
    graph: NxMixedGraph,
) -> Expression:
    """Return the probability in the case where no interventions are present.

    :param target_outcomes: A set of nodes that comprise our target outcomes.
    :param expression: The distribution in the current domain.
    :param graph: The graph with transport nodes in this domain.
    :returns: Sum over the probabilities of nodes other than target outcomes.
    """
    return Sum.safe(expression, graph.nodes() - target_outcomes)


def trso_line2(
    query: TRSOQuery,
    outcomes_ancestors: Set[Variable],
) -> TRSOQuery:
    """Restrict the interventions and diagram to only include ancestors of target variables.

    :param query: A transport query
    :param outcomes_ancestors: the ancestors of target variables in transportability_diagram
    :returns: Dictionary of modified trso inputs.
    """
    new_query = deepcopy(query)
    new_query.target_interventions.intersection_update(outcomes_ancestors)
    new_query.graphs[new_query.domain] = query.graphs[query.domain].subgraph(outcomes_ancestors)
    new_query.expression = Sum.safe(
        query.expression,
        query.graphs[query.domain].nodes() - outcomes_ancestors,
    )
    return new_query


def trso_line3(query: TRSOQuery, additional_interventions: Set[Variable]) -> TRSOQuery:
    """

    :param query: A transport query
    :param additional_interventions: interventions to be added to target_interventions
    :returns: dictionary of modified trso inputs.
    """
    new_query = deepcopy(query)
    new_query.target_interventions.update(additional_interventions)
    return new_query


def trso_line4(
    query: TRSOQuery,
    components: Iterable[FrozenSet[Variable]],
) -> Dict[FrozenSet[Variable], TRSOQuery]:
    """Find the trso inputs for each C-component.

    :param query: A transport query
    :param components: Set of c_components of transportability_diagram without target_interventions
    :returns: Dictionary with components as keys and dictionary of modified trso inputs as values
    """
    graph = query.graphs[query.domain]
    rv = {}
    for component in components:
        new_query = deepcopy(query)
        new_query.target_outcomes = set(component)
        new_query.target_interventions = graph.nodes() - component
        rv[component] = new_query
    return rv


def trso_line6(query: TRSOQuery) -> Dict[Population, TRSOQuery]:
    """Find the active interventions in each diagram, run trso with active interventions."""
    expressions = {}
    for domain, graph in query.graphs.items():
        if domain == TARGET_DOMAIN:
            continue

        surrogate_interventions = query.surrogate_interventions[domain]
        surrogate_intersect_target = surrogate_interventions.intersection(
            query.target_interventions
        )
        if not surrogate_intersect_target:
            continue

        if not all_transports_d_separated(
            graph,
            target_interventions=query.target_interventions,
            target_outcomes=query.target_outcomes,
        ):
            continue

        new_query = deepcopy(query)
        new_query.target_interventions = query.target_interventions - surrogate_interventions
        new_query.domain = domain
        new_query.graphs[new_query.domain] = graph.remove_nodes_from(surrogate_intersect_target)
        new_query.active_interventions = surrogate_intersect_target
        expressions[domain] = new_query

    return expressions


def all_transports_d_separated(graph, target_interventions, target_outcomes) -> bool:
    transportability_nodes = get_transport_nodes(graph)
    graph_without_interventions = graph.remove_in_edges(target_interventions)
    return all(
        are_d_separated(
            graph_without_interventions,
            transportability_node,
            outcome,
            conditions=target_interventions,
        )
        for transportability_node in transportability_nodes
        if transportability_node in graph  # FIXME check if this is okay to exclude
        for outcome in target_outcomes
    )


def trso_line9(query: TRSOQuery, district: set[Variable]) -> Expression:
    sorted_district_nodes = sorted(district, key=attrgetter("name"))
    # Will this always return the same order?
    my_product: Expression = One()
    for i in range(len(sorted_district_nodes)):
        # TODO I am not convinced this is correct, still trying to decipher the paper
        subset_including_node = set(sorted_district_nodes[: i + 1])
        subset_up_to_node = set(sorted_district_nodes[:i])
        numerator = Sum.safe(query.expression, district - subset_including_node)
        denominator = Sum.safe(query.expression, district - subset_up_to_node)
        if denominator == 0:
            pass
            # TODO is this possible to be zero, should we have a check?
        my_product *= numerator / denominator
    return Sum.safe(my_product, district - query.target_outcomes)


def trso_line10(
    query: TRSOQuery, district: set[Variable], new_surrogate_interventions
) -> TRSOQuery:
    sorted_district_nodes = sorted(district, key=attrgetter("name"))
    my_product = One()
    for i, node in enumerate(sorted_district_nodes):
        # TODO I am not convinced this is correct, still trying to decipher the paper
        subset_up_to_node = set(sorted_district_nodes[:i])
        previous_node_without_district = set([sorted_district_nodes[i - 1]]) - district
        # FIXME calling this doesn't make sense - expression is an object, not a func
        my_product *= query.expression(
            node.given(
                subset_up_to_node.intersection(district).union(previous_node_without_district)
            )
        )

    new_query = deepcopy(query)
    new_query.target_interventions = query.target_interventions.intersection(district)
    new_query.expression = my_product
    new_query.graphs[query.domain] = query.graphs[query.domain].subgraph(district)
    new_query.surrogate_interventions = new_surrogate_interventions
    return new_query


# TODO Tikka paper says that topological ordering is available globaly
def trso(query: TRSOQuery) -> Optional[Expression]:
    # Check that domain is in query.domains
    # check that query.surrogate_interventions keys are equals to domains
    # check that query.graphs keys are equal to domains
    graph = query.graphs[query.domain]
    # line 1
    if not query.target_interventions:
        return trso_line1(query.target_outcomes, query.expression, graph)

    # line 2
    outcome_ancestors = graph.ancestors_inclusive(query.target_outcomes)
    if graph.nodes() - outcome_ancestors:
        new_query = trso_line2(query, outcome_ancestors)
        return trso(new_query)

    # line 3
    # TODO give meaningful name to this variable
    target_interventions_overbar = graph.remove_in_edges(query.target_interventions)
    additional_interventions = (
        cast(set[Variable], graph.nodes())
        - query.target_interventions
        - target_interventions_overbar.ancestors_inclusive(query.target_outcomes)
    )
    if additional_interventions:
        new_query = trso_line3(query, additional_interventions)
        return trso(new_query)

    # line 4
    districts_without_interventions: set[frozenset[Variable]] = graph.remove_nodes_from(
        query.target_interventions
    ).districts()
    if len(districts_without_interventions) > 1:
        subqueries = trso_line4(
            query,
            districts_without_interventions,
        )
        terms = []
        for subquery in subqueries.values():
            term = trso(subquery)
            if term is None:
                raise NotImplementedError
            terms.append(term)

        return Sum.safe(
            Product.safe(terms),
            graph.nodes() - query.target_interventions.union(query.target_outcomes),
        )

    # line 6
    if not query.active_interventions:
        expressions: Dict[Population, Expression] = {}
        for domain, subquery in trso_line6(query).items():
            expression = trso(subquery)
            if expression is not None:  # line7
                expressions[domain] = expression
        if len(expressions) == 1:
            return list(expressions.values())[0]
        elif len(expressions) > 1:
            logger.warning("more than one expression were non-none")
            # What if more than 1 expression doesn't fail?
            # Is it non-deterministic or can we prove it will be length 1?
            return list(expressions.values())[0]
        else:
            pass

    # line8
    districts = graph.districts()
    # line 11, return fail. keep explict tests for 0 and 1 to ensure adequate testing
    if len(districts) == 0:
        return None
    if len(districts) == 1:
        return None
    # line 8, i.e. len(districts)>1

    # line9
    # TODO check which of the raises below should be passthroughs, annotate explicitly
    if len(districts_without_interventions) == 1:
        district_without_interventions = districts_without_interventions.pop()
        if districts_without_interventions in districts:
            return trso_line9(query, set(district_without_interventions))
        # raise NotImplementedError(
        #     "single district without interventions found, but it's not in the districts"
        # )
        # This case is covered by line 10.
        # FIXME ^ doesn't seem quite right since this is exact checking while line 10 is subsets
    elif len(districts_without_interventions) == 0:
        raise NotImplementedError("no districts without interventions found")
    else:  # multiple districts
        raise NotImplementedError("multiple districts without interventions found")

    # line10
    target_districts = []
    for district in districts:
        if district_without_interventions.issubset(district):
            target_districts.append(district)
    if len(target_districts) != 1:
        logger.warning("Incorrect number of districts found on line 10")
        # TODO This shouldn't be possible, should we remove this check?
    target_district = target_districts.pop()
    # district is C' districts should be D[C'], but we chose to return set of nodes instead of subgraph
    if len(query.active_interventions) == 0:
        # FIXME is this even possible? doesn't line 6 check this and return something else?
        new_surrogate_interventions = dict()
    elif _pillow_has_transport(graph, target_district):
        return None
    else:
        new_surrogate_interventions = query.surrogate_interventions

    return trso_line10(
        query,
        set(target_district),
        new_surrogate_interventions,
    )


def _pillow_has_transport(graph, district) -> bool:
    return any(is_transport_node(node) for node in graph.get_markov_pillow(district))


def transport(
    graph: NxMixedGraph,
    transports: Mapping[Variable, List[Population]],
    treatments: List[Variable],
    outcomes: List[Variable],
    conditions: Optional[List[Variable]] = None,
):
    """Transport algorithm from https://arxiv.org/abs/1806.07172."""
    if conditions is not None:
        raise NotImplementedError
    raise NotImplementedError
