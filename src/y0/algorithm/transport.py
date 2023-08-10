"""Implement of surrogate outcomes and transportability from https://arxiv.org/abs/1806.07172."""

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

from y0.algorithm.conditional_independencies import are_d_separated
from y0.dsl import (
    PP,
    CounterfactualVariable,
    Expression,
    Intervention,
    Population,
    Product,
    Sum,
    Variable,
)
from y0.graph import NxMixedGraph

__all__ = [
    "transport",
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
    c_component_surrogate_outcomes = set()
    for component in graph.get_c_components():
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


TRANSPORT_PREFIX = "T_"


def transport_variable(variable: Variable) -> Variable:
    if isinstance(variable, (CounterfactualVariable, Intervention)):
        raise TypeError
    return Variable(TRANSPORT_PREFIX + variable.name)


def is_transport_node(node: Variable) -> bool:
    return not isinstance(node, (CounterfactualVariable, Intervention)) and node.name.startswith(
        TRANSPORT_PREFIX
    )


def get_transport_nodes(graph: NxMixedGraph) -> Set[Variable]:
    return {node for node in graph if is_transport_node(node)}


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
    transportability_diagrams: Dict[Population, NxMixedGraph]
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
    transportability_diagrams: Dict[Population, NxMixedGraph]
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

    transportability_diagrams = {
        domain: create_transport_diagram(
            graph=graph,
            nodes_to_transport=get_nodes_to_transport(
                surrogate_interventions=surrogate_interventions[domain],
                surrogate_outcomes=surrogate_outcomes[domain],
                graph=graph,
            ),
        )
        for domain in surrogate_outcomes
    }
    transportability_diagrams[TARGET_DOMAIN] = graph

    return TransportQuery(
        target_interventions=target_interventions,
        target_outcomes=target_outcomes,
        transportability_diagrams=transportability_diagrams,
        domains=set(surrogate_outcomes),
        surrogate_interventions=surrogate_interventions,
        target_experiments=set(),
    )


def trso_line1(
    target_outcomes: Set[Variable],
    expression: Expression,
    transportability_diagram: NxMixedGraph,
) -> Expression:
    """Return the probability in the case where no interventions are present.

    :param target_outcomes: A set of nodes that comprise our target outcomes.
    :param expression: The distribution in the current domain.
    :param transportability_diagram: The graph with transport nodes in this domain.
    :returns: Sum over the probabilities of nodes other than target outcomes.
    """
    return Sum.safe(expression, transportability_diagram.nodes() - target_outcomes)


def trso_line2(
    query: TRSOQuery,
    outcomes_ancestors: Set[Variable],
) -> TRSOQuery:
    """Restrict the interventions and diagram to only include ancestors of target variables.

    :param query: A transport query
    :param probability: The distribution in the current domain.
    :param domain: current domain
    :param outcomes_ancestors: the ancestors of target variables in transportability_diagram
    :returns: Dictionary of modified trso inputs.
    """
    new_query = deepcopy(query)
    new_query.target_interventions.intersection_update(outcomes_ancestors)
    new_query.transportability_diagrams[new_query.domain] = query.transportability_diagrams[
        query.domain
    ].subgraph(outcomes_ancestors)
    new_query.expression = Sum.safe(
        query.expression,
        query.transportability_diagrams[query.domain].nodes() - outcomes_ancestors,
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
    components: Set[FrozenSet[Variable]],
) -> Dict[FrozenSet[Variable], TRSOQuery]:
    """Find the trso inputs for each C-component.

    :param query: A transport query
    :param domain: current domain
    :param components: Set of c_components of transportability_diagram without target_interventions
    :returns: Dictionary with components as keys and dictionary of modified trso inputs as values
    """
    transportability_diagram = query.transportability_diagrams[query.domain]
    rv = {}
    for component in components:
        new_query = deepcopy(query)
        new_query.target_outcomes = component
        new_query.target_interventions = transportability_diagram.nodes() - component
        rv[component] = new_query
    return rv


def trso_line6(
    query: TRSOQuery,
) -> Dict[Population, TRSOQuery]:
    """Find the active interventions in each diagram, run trso with active interventions.

    :param query: A transport query
    :param domain: current domain
    :returns:
    """
    transportability_diagram = query.transportability_diagrams[query.domain]
    expressions = {}
    for loop_domain, loop_transportability_diagram in query.transportability_diagrams.items():
        if not query.available_interventions[loop_domain].intersection(query.target_interventions):
            continue

        transportability_nodes = get_transport_nodes(loop_transportability_diagram)
        diagram_without_interventions = loop_transportability_diagram.remove_in_edges(
            query.target_interventions
        )
        if not all(
            are_d_separated(
                diagram_without_interventions,
                transportability_node,
                outcome,
                conditions=query.target_interventions,
            )
            for transportability_node in transportability_nodes
            for outcome in query.target_outcomes
        ):
            continue

        new_query = deepcopy(query)
        new_query.target_interventions = (
            query.target_interventions - query.available_interventions[loop_domain]
        )
        new_query.domain = loop_domain
        new_query.transportability_diagrams[query.domain] = transportability_diagram.subgraph(
            transportability_diagram.nodes()
            - query.available_interventions[loop_domain].intersection(query.target_interventions)
        )
        new_query.active_interventions =query.available_interventions[loop_domain].intersection(
            query.target_interventions
        )
        expressions[loop_domain] = (new_query)

    return expressions


def trso_line9(query, expression, active_interventions, domain) -> Expression:
    pass


def trso_line10(
    query, expression, active_interventions, domain, district, new_available_interventions
) -> Expression:
    pass


# TODO Tikka paper says that topological ordering is available globaly
def trso(
    query: TRSOQuery,
) -> Optional[Expression]:
    # Check that domain is in query.domains
    # check that query.surrogate_interventions keys are equals to domains
    # check that query.transportability_diagrams keys are equal to domains
    transportability_diagram = query.transportability_diagrams[query.domain]
    # line 1
    if not query.target_interventions:
        return trso_line1(query.target_outcomes, query.expression, transportability_diagram)

    # line 2
    outcome_ancestors = transportability_diagram.ancestors_inclusive(query.target_outcomes)
    if transportability_diagram.nodes() - outcome_ancestors:
        new_query = trso_line2(
            query,
            outcome_ancestors,
        )
        return trso(
            query=new_query,
        )

    # line 3
    # TODO give meaningful name to this variable
    target_interventions_overbar = transportability_diagram.remove_in_edges(
        query.target_interventions
    )
    additional_interventions = (
        cast(set[Variable], transportability_diagram.nodes())
        - query.target_interventions
        - target_interventions_overbar.ancestors_inclusive(query.target_outcomes)
    )
    if additional_interventions:
        new_query = trso_line3(query, additional_interventions)
        return trso(
            query=new_query,
        )

    # line 4
    districts_without_interventions = transportability_diagram.subgraph(
        transportability_diagram.nodes() - query.target_interventions
    ).get_c_components()
    if len(districts_without_interventions) > 1:
        trso_line4inputs = trso_line4(
            query,
            districts_without_interventions,
        )

        return Sum.safe(
            Product.safe(
                trso(
                    query=trso_line4input,
                )
                for trso_line4input in trso_line4inputs.values()
            ),
            transportability_diagram.nodes()
            - query.target_interventions.union(query.target_outcomes),
        )

    # line 6
    if not query.active_interventions:
        subqueries = trso_line6(query)
        expressions = {}
        for loop_domain, loop_query in subqueries.items():
            loop_expression = trso(
                query=loop_query,
            )
            # line7
            if loop_expression is not None:
                expressions[loop_domain] = loop_expression
        if len(expressions) == 1:
            return list(expressions.values())[0]
        elif len(expressions) > 1:
            logger.warning("more than one expression were non-none")
            # What if more than 1 expression doesn't fail?
            # Is it non-deterministic or can we prove it will be length 1?
            return list(expressions.values())[0]

    # line8
    districts = transportability_diagram.get_c_components()
    # line 11, return fail
    if len(districts) <= 1:
        return None
    # line 8, i.e. len(districts)>1

    # line9
    if districts_without_interventions in districts:
        return trso_line9(
            query,

        )

    # line10
    # FIXME why aren't results collated over all districts? then pick which one to return?
    for district in districts:
        if not districts_without_interventions.issubset(district):
            continue
        # district is C' districts should be D[C'], but we chose to return set of nodes instead of subgraph
        if len(query.active_interventions) == 0:
            # FIXME is this even possible? doesn't line 6 check this and return something else?
            new_available_interventions = dict()
        elif any(
            is_transport_node(node) for node in transportability_diagram.get_markov_pillow(district)
        ):
            return None
        else:
            new_available_interventions = query.available_interventions

        return trso_line10(
            query,
            district,
            new_available_interventions,
        )


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
