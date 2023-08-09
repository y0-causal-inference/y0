"""Implement of surrogate outcomes and transportability.

..seealso:: https://arxiv.org/abs/1806.07172
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Tuple, Union

from y0.algorithm.conditional_independencies import are_d_separated
from y0.dsl import (
    Expression,
    Population,
    PopulationProbability,
    Product,
    Sum,
    Transport,
    Variable,
)
from y0.graph import NxMixedGraph

__all__ = [
    "transport",
]
TARGET_DOMAIN = Population("pi*")


def find_transport_vertices(
    interventions: Union[Set[Variable], Variable],
    surrogate_outcomes: Union[Set[Variable], Variable],
    graph: NxMixedGraph,
) -> Set[Variable]:
    """
    Identify which vertices the transport vertices should point to.
    :param interventions: The interventions performed in an experiment.
    :param surrogate_outcomes: The outcomes observed in an experiment.
    :param graph: The graph of the target domain.
    :returns: A set of variables representing target domain nodes where transportability nodes should be added.
    """
    if isinstance(interventions, Variable):
        interventions = {interventions}
    if isinstance(surrogate_outcomes, Variable):
        surrogate_outcomes = {surrogate_outcomes}

    # Find the c_component with surrogate_outcomes
    c_components = graph.get_c_components()
    c_component_surrogate_outcomes = set()
    for index, component in enumerate(c_components):
        # Check if surrogate_outcomes is present in the current set
        if surrogate_outcomes.intersection(component):
            c_component_surrogate_outcomes = c_component_surrogate_outcomes.union(component)

    # subgraph where interventions in edges are removed
    interventions_overbar = graph.remove_in_edges(interventions)
    # Ancestors of surrogate_outcomes in interventions_overbar
    Ancestors_surrogate_outcomes = interventions_overbar.ancestors_inclusive(surrogate_outcomes)

    # Descendants of interventions in graph
    Descendants_interventions = graph.descendants_inclusive(interventions)

    return (Descendants_interventions - surrogate_outcomes).union(
        c_component_surrogate_outcomes - Ancestors_surrogate_outcomes
    )


def create_transport_diagram(
    transport_vertices: Union[Set[Variable], Variable],
    graph: NxMixedGraph,
) -> NxMixedGraph:
    """
    Create a NxMixedGraph identical to graph but with transport vertices added.
    :param transport_vertices: Vertices which have transport nodes pointing to them.
    :param graph: The graph of the target domain.
    :returns: graph with transport vertices added
    """

    # TODO we discussed the possibility of using a dictionary with needed nodes
    # instead of creating a graph for each diagram.
    transportability_diagram = NxMixedGraph()
    for node in graph.nodes():
        transportability_diagram.add_node(node)
    for u, v in graph.directed.edges():
        transportability_diagram.add_directed_edge(u, v)
    for u, v in graph.undirected.edges():
        transportability_diagram.add_undirected_edge(u, v)

    for vertex in transport_vertices:
        # TODO Make this a true Transport instead of a Variable
        # T_vertex = Transport(vertex)
        T_vertex = Variable("T" + vertex.to_text())
        transportability_diagram.add_node(T_vertex)
        transportability_diagram.add_directed_edge(T_vertex, vertex)

    return transportability_diagram


class TransportQuery:
    target_interventions: Set[Variable]
    target_outcomes: Set[Variable]
    transportability_diagrams: Dict[Population, NxMixedGraph]
    graph: NxMixedGraph
    domains: Set[Population]
    target_domain: Population
    surrogate_interventions: Dict[Population, Set[Variable]]
    target_experiments: Set[Variable]


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
    :param intervention_outcome_pairs : A set of Experiments available in each domain.
    :returns: An octuple representing the query transformation of a surrogate outcome query.
    """
    if set(surrogate_outcomes) != set(surrogate_interventions):
        raise ValueError("Inconsistent surrogate outcome and intervention domains")
    transportability_diagrams = {
        domain: create_transport_diagram(
            graph,
            find_transport_vertices(
                surrogate_interventions[domain], surrogate_outcomes[domain], graph
            ),
        )
        for domain in surrogate_outcomes
    }

    return TransportQuery(
        target_interventions=target_interventions,
        target_outcomes=target_outcomes,
        transportability_diagrams=transportability_diagrams,
        graph=graph,
        domains=set(surrogate_outcomes),
        target_domain=TARGET_DOMAIN,
        surrogate_interventions=surrogate_interventions,
        target_experiments=set(),
    )


def trso_line1(
    target_outcomes: Set[Variable],
    probability: PopulationProbability,
    transportability_diagram: NxMixedGraph,
) -> Sum:
    """
    Return the probability in the case where no interventions are present.
    :param target_outcomes: A set of nodes that comprise our target outcomes.
    :param probability : The distribution in the current domain.
    :param transportability_diagram : The graph with transport nodes in this domain.
    :returns: Sum over the probabilities of nodes other than target outcomes.

    """
    return Sum.safe(probability, transportability_diagram.nodes() - target_outcomes)


def trso_line2(
    query: TransportQuery,
    probability: Expression,
    domain: Variable,
    outcomes_anc: Set[Variable],
) -> Tuple[TransportQuery, Expression]:
    """
    Restrict the interventions and diagram to only include ancestors of target variables.
    :param target_outcomes: A set of target variables for causal effects.
    :param target_interventions: A set of interventions for the target domain.
    :param probability : The distribution in the current domain.
    :param active_interventions : which interventions are currently active
    :param domain : current domain
    :param transportability_diagrams : Dictionary of all available transportability diagrams
    :param available_interventions : A dictionary of sets of Experiments available in each domain.
    :param outcomes_anc : the ancestors of target variables in transportability_diagram
    :returns: Dictionary of modified trso inputs.

    """
    new_query = deepcopy(query)
    new_query.target_interventions.intersection_update(outcomes_anc)
    new_query.transportability_diagrams[domain] = new_query.transportability_diagrams[
        domain
    ].subgraph(outcomes_anc)
    new_expression = Sum.safe(
        probability, new_query.transportability_diagrams[domain].nodes() - outcomes_anc
    )
    return (new_query, new_expression)


def trso_line3(
    query: TransportQuery,
    additional_interventions: Set[Variable],
) -> TransportQuery:
    """

    :param target_outcomes: A set of target variables for causal effects.
    :param target_interventions: A set of interventions for the target domain.
    :param probability : The distribution in the current domain.
    :param active_interventions : which interventions are currently active
    :param domain : current domain
    :param transportability_diagrams : Dictionary of all available transportability diagrams
    :param available_interventions : A dictionary of sets of Experiments available in each domain.
    :param additional_interventions : interventions to be added to target_interventions
    :returns: dictionary of modified trso inputs.

    """
    new_query = deepcopy(query)
    new_query.target_interventions.update(additional_interventions)
    return new_query


def trso_line4(
    query: TransportQuery,
    domain: Variable,
    components: Set[FrozenSet[Variable]],
) -> Dict[FrozenSet[Variable], TransportQuery]:
    """Find the trso inputs for each c-component.

    :param target_outcomes: A set of target variables for causal effects.
    :param target_interventions: A set of interventions for the target domain.
    :param probability : The distribution in the current domain.
    :param active_interventions : which interventions are currently active
    :param domain : current domain
    :param transportability_diagrams : Dictionary of all available transportability diagrams
    :param available_interventions : A dictionary of sets of Experiments available in each domain.
    :param components : Set of c_components of transportability_diagram without target_interventions
    :returns: Dictionary with components as keys and dictionary of modified trso inputs as values

    """
    transportability_diagram = query.transportability_diagrams[domain]
    rv = {}
    for component in components:
        new_query = deepcopy(query)
        new_query.target_outcomes = component
        new_query.target_interventions = transportability_diagram.nodes() - component
        rv[component] = new_query
    return rv


def trso_line6(
    query: TransportQuery,
    active_interventions: Set[Variable],
    domain: Variable,
) -> Dict[Population, Tuple[TransportQuery, Set[Variable]]]:
    """Find the active interventions in each diagram, run trso with active interventions.

    :param target_outcomes: A set of target variables for causal effects.
    :param target_interventions: A set of interventions for the target domain.
    :param probability : The distribution in the current domain.
    :param active_interventions : which interventions are currently active
    :param domain : current domain
    :param transportability_diagrams : Dictionary of all available transportability diagrams
    :param available_interventions : A dictionary of sets of Experiments available in each domain.
    :return List of Dictionary of modified trso inputs

    """
    transportability_diagram = query.transportability_diagrams[domain]
    expressions = {}
    for loop_domain, loop_transportability_diagram in query.transportability_diagrams.items():
        if not query.available_interventions[loop_domain].intersection(query.target_interventions):
            continue
        transportability_nodes = loop_transportability_diagram.get_transport_nodes()
        diagram_without_interventions = loop_transportability_diagram.remove_in_edges(
            query.target_interventions
        )

        if not all(
            are_d_separated(
                diagram_without_interventions,
                node,
                outcome,
                conditions=query.target_interventions,
            )
            for node in transportability_nodes
            for outcome in query.target_outcomes
        ):
            continue
        new_query = deepcopy(query)
        new_query.target_interventions = (
            query.target_interventions - query.available_interventions[loop_domain]
        )

        new_query.domain = loop_domain
        new_query.transportability_diagrams[domain] = transportability_diagram.subgraph(
            transportability_diagram.nodes()
            - query.available_interventions[loop_domain].intersection(query.target_interventions)
        )
        new_active_interventions = query.available_interventions[loop_domain].intersection(
            query.target_interventions
        )
        expressions[loop_domain] = (new_query, new_active_interventions)

    return expressions


# TODO Tikka paper says that topological ordering is available globaly
# TODO some functions need transportability_diagrams while others need transportability_diagram
def trso(
    query: TransportQuery,
    active_interventions: Set[Variable],
    domain: Population,
    probability: Expression,
) -> Expression:
    # Check that domain is in query.domains
    # check that query.surrogate_interventions keys are equals to domains
    # check that query.transportability_diagrams keys are equal to domains
    transportability_diagram = query.transportability_diagrams[domain]
    # line 1
    if not query.target_interventions:
        return trso_line1(query.target_outcomes, probability, transportability_diagram)

    # line 2
    outcomes_anc = transportability_diagram.ancestors_inclusive(query.target_outcomes)
    if transportability_diagram.nodes() - outcomes_anc:
        new_query, new_probability = trso_line2(
            query,
            probability,
            domain,
            outcomes_anc,
        )
        return trso(
            query=new_query,
            active_interventions=active_interventions,
            domain=domain,
            probability=new_probability,
        )

    # line 3

    target_interventions_overbar = transportability_diagram.remove_in_edges(
        query.target_interventions
    )
    additional_interventions = (
        transportability_diagram.nodes()
        - query.target_interventions
        - target_interventions_overbar.ancestors_inclusive(query.target_outcomes)
    )
    if additional_interventions:
        new_query = trso_line3(
            query,
            additional_interventions,
        )
        return trso(
            query=new_query,
            active_interventions=active_interventions,
            domain=domain,
            probability=probability,
        )

    # line 4
    districts_without_interventions = transportability_diagram.subgraph(
        transportability_diagram.nodes() - query.target_interventions
    ).get_c_components()
    if len(districts_without_interventions) > 1:
        trso_line4inputs = trso_line4(
            query,
            domain,
            districts_without_interventions,
        )

        return Sum.safe(
            Product.safe(
                [
                    trso(
                        query=trso_line4input,
                        active_interventions=active_interventions,
                        domain=domain,
                        probability=probability,
                    )
                    for trso_line4input in trso_line4inputs.values()
                ],
            ),
            transportability_diagram.nodes()
            - query.target_interventions.union(query.target_outcomes),
        )

    # line 5
    else:
        # line 6
        if not active_interventions:
            trso_inputs = trso_line6(
                query,
                active_interventions,
                domain,
            )
            expressions = {}
            for loop_domain, (loop_query, loop_active_interventions) in trso_inputs.items():
                expressionk = trso(
                    query=loop_query,
                    active_interventions=loop_active_interventions,
                    domain=loop_domain,
                    probability=probability,
                )
                # line7
                if expressionk:
                    expressions[loop_domain] = expressionk
                    # return expressionk
            if len(expressions) == 1:
                return list(expressions.values())[0]
            elif len(expressions) > 1:
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
                probability,
                active_interventions,
                domain,
            )
        # line10
        for district in districts:
            if not districts_without_interventions.issubset(district):
                continue
            # district is C' districts should be D[C'], but we chose to return set of nodes instead of subgraph
            if len(active_interventions) == 0:
                new_available_interventions = dict()
            elif any(
                isinstance(
                    t, Transport
                )  # TODO this doesn't match how we create transportability_diagram
                for t in transportability_diagram.get_markov_pillow(district)
            ):
                return None
            else:
                new_available_interventions = query.available_interventions

            return trso_line10(
                query,
                probability,
                active_interventions,
                domain,
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
