"""Implement of surrogate outcomes and transportability.

..seealso:: https://arxiv.org/abs/1806.07172
"""

from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Tuple, Union

from y0.algorithm.conditional_independencies import are_d_separated
from y0.dsl import Population, PopulationProbability, Product, Sum, Transport, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "transport",
]


def find_transport_vertices(
    interventions: Union[Set[Variable], Variable],
    surrogate_outcomes: Union[Set[Variable], Variable],
    graph: NxMixedGraph,
) -> Set[Variable]:
    """

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


def surrogate_to_transport(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    graph: NxMixedGraph,
    available_interventions: List[Tuple[Set[Variable]]],
) -> tuple[Variable]:
    """

    :param target_outcomes: A set of target variables for causal effects.
    :param target_interventions: A set of interventions for the target domain.
    :param graph: The graph of the target domain.
    :param available_interventions : A set of Experiments available in each domain.
    :returns: An octuple representing the query transformation of a surrogate outcome query.

    """
    # TODO what structure do we want for available_interventions?
    # We said dictionary, but it should be keyed by domains which doesn't exist yet.

    transportability_diagrams = {}
    domains = [Variable(f"pi{i+1}") for i in range(len(available_interventions))]

    surrogate_interventions = {
        domain: intervention for (intervention, _), domain in zip(available_interventions, domains)
    }
    surrogate_outcomes = {
        domain: outcome for (_, outcome), domain in zip(available_interventions, domains)
    }

    for domain in domains:
        transport_vertices = find_transport_vertices(
            surrogate_interventions[domain], surrogate_outcomes[domain], graph
        )
        transportability_diagram = create_transport_diagram(graph, transport_vertices)
        transportability_diagrams[domain] = transportability_diagram

    target_domain = Variable("pi*")
    experiments_in_target_domain = set()

    return (
        target_interventions,
        target_outcomes,
        transportability_diagrams,
        graph,
        domains,
        target_domain,
        surrogate_interventions,  # TODO this is now a dictionary. Need to make changes throughout the code.
        experiments_in_target_domain,
    )


def trso_line1(
    target_outcomes: Set[Variable],
    probability: PopulationProbability,
    transportability_diagram: NxMixedGraph,
) -> Sum:
    """
    Parameters
    ----------
    target_outcomes: A set of nodes that comprise our target outcomes.
    probability : The distribution in the current domain.
    transportability_diagram : The graph with transport nodes in this domain.

    Returns
    -------
    Sum
        Sum over the probabilities of nodes other than target outcomes.

    """
    return Sum.safe(probability, transportability_diagram.nodes() - target_outcomes)


def trso_line2(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    probability: PopulationProbability,
    active_interventions: Set[Variable],
    domain: Variable,
    transportability_diagram: NxMixedGraph,
    available_interventions: List[Set[Variable]],
    outcomes_anc: Set[Variable],
) -> Dict:
    """


    Parameters
    ----------
    target_outcomes: A set of target variables for causal effects.
    target_interventions: A set of interventions for the target domain.
    probability : The distribution in the current domain.
    active_interventions : which interventions are currently active
    domain : current domain
    transportability_diagram : graph with transport nodes in this domain
    available_interventions : A set of Experiments available in each domain.
    outcomes_anc : the ancestors of target variables in transportability_diagram

    Returns
    -------
    Dict
        Dictionary of modified trso inputs.

    """
    return dict(
        target_outcomes=target_outcomes,
        target_interventions=target_interventions.intersection(outcomes_anc),
        probability=Sum.safe(probability, transportability_diagram.nodes() - outcomes_anc),
        active_interventions=active_interventions,
        domain=domain,
        transportability_diagram=transportability_diagram.subgraph(outcomes_anc),
        available_interventions=available_interventions,
    )


def trso_line3(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    probability: PopulationProbability,
    active_interventions: Set[Variable],
    domain: Variable,
    transportability_diagram: NxMixedGraph,
    available_interventions: List[Set[Variable]],
    additional_interventions: Set[Variable],
) -> Dict:
    """


    Parameters
    ----------
    target_outcomes: A set of target variables for causal effects.
    target_interventions: A set of interventions for the target domain.
    probability : The distribution in the current domain.
    active_interventions : which interventions are currently active
    domain : current domain
    transportability_diagram : graph with transport nodes in this domain
    available_interventions : A set of Experiments available in each domain.
    additional_interventions : interventions to be added to target_interventions

    Returns
    -------
    Dict
        Dictionary of modified trso inputs.

    """
    return dict(
        target_outcomes=target_outcomes,
        target_interventions=target_interventions.union(additional_interventions),
        probability=probability,
        active_interventions=active_interventions,
        domain=domain,
        transportability_diagram=transportability_diagram,
        available_interventions=available_interventions,
    )


def trso_line4(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    probability: PopulationProbability,
    active_interventions: Set[Variable],
    domain: Variable,
    transportability_diagram: NxMixedGraph,
    available_interventions: List[Set[Variable]],
    components: Set[FrozenSet[Variable]],
) -> Dict[FrozenSet, Dict]:
    """


    Parameters
    ----------
    target_outcomes: A set of target variables for causal effects.
    target_interventions: A set of interventions for the target domain.
    probability : The distribution in the current domain.
    active_interventions : which interventions are currently active
    domain : current domain
    transportability_diagram : graph with transport nodes in this domain
    available_interventions : A set of Experiments available in each domain.
    components : Set of c_components of transportability_diagram without target_interventions

    Returns
    -------
    Dict
        Dictionary with components as keys and dictionary of modified trso inputs as values

    """
    return {
        component: dict(
            target_outcomes=component,
            target_interventions=transportability_diagram.nodes() - component,
            probability=probability,
            active_interventions=active_interventions,
            domain=domain,
            transportability_diagram=transportability_diagram,
            available_interventions=available_interventions,
        )
        for component in components
    }


def trso_line6(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    probability: PopulationProbability,
    active_interventions: Set[Variable],
    domain: Variable,
    transportability_diagram: NxMixedGraph,
    available_interventions: List[Set[Variable]],
    transportability_diagrams: Dict[Variable, NxMixedGraph],
) -> List[Dict]:
    """
    Parameters
    ----------
    target_outcomes: A set of target variables for causal effects.
    target_interventions: A set of interventions for the target domain.
    probability : The distribution in the current domain.
    active_interventions : which interventions are currently active
    domain : current domain
    transportability_diagrams : Dictionary of all available transportability diagrams
    available_interventions : A set of Experiments available in each domain.
    transportability_diagrams : Dictionary of all available transportability diagrams

    Returns
    -------
    Dict
        List of Dictionary of modified trso inputs

    """
    if not active_interventions:
        expressions = []
        for i in range(len(transportability_diagrams)):
            if available_interventions[i].intersection(target_interventions):
                transportability_nodes = transportability_diagrams[i].get_transport_nodes()
                diagram_without_interventions = transportability_diagrams[i].remove_in_edges(
                    target_interventions
                )

                if all(
                    are_d_separated(
                        diagram_without_interventions,
                        node,
                        outcome,
                        conditions=target_interventions,
                    )
                    for node in transportability_nodes
                    for outcome in target_outcomes
                ):
                    expressions.append(
                        dict(
                            target_outcomes=target_outcomes,
                            target_interventions=target_interventions - available_interventions[i],
                            probability=probability,
                            active_interventions=available_interventions[i].intersection(
                                target_interventions
                            ),
                            domain=i,
                            transportability_diagram=transportability_diagram.subgraph(
                                transportability_diagram.nodes()
                                - available_interventions[i].intersection(target_interventions)
                            ),
                            available_interventions=available_interventions,
                        )
                    )
    else:
        raise NotImplementedError
    return expressions


# TODO Tikka paper says that topological ordering is available globaly
# TODO some functions need transportability_diagrams while others need transportability_diagram
def trso(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    probability: PopulationProbability,
    active_interventions: Set[Variable],
    domain: Variable,
    transportability_diagrams: Dict[Variable, NxMixedGraph],
    available_interventions: List[Set[Variable]],
):
    transportability_diagram = transportability_diagrams[domain]
    # line 1
    if not target_interventions:
        return trso_line1(target_outcomes, probability, transportability_diagram)

    # line 2
    outcomes_anc = transportability_diagram.ancestors_inclusive(target_outcomes)
    if transportability_diagram.nodes() - outcomes_anc:
        return trso(
            **trso_line2(
                target_outcomes,
                target_interventions,
                probability,
                active_interventions,
                domain,
                transportability_diagram,
                available_interventions,
                outcomes_anc,
            )
        )

    # line 3

    target_interventions_overbar = transportability_diagram.remove_in_edges(target_interventions)
    additional_interventions = (
        transportability_diagram.nodes()
        - target_interventions
        - target_interventions_overbar.ancestors_inclusive(target_outcomes)
    )
    if additional_interventions:
        return trso(
            **trso_line3(
                target_outcomes,
                target_interventions,
                probability,
                active_interventions,
                domain,
                transportability_diagram,
                available_interventions,
                additional_interventions,
            )
        )

    # line 4
    districts_without_interventions = transportability_diagram.subgraph(
        transportability_diagram.nodes() - target_interventions
    ).get_c_components()
    if len(districts_without_interventions) > 1:
        trso_line4inputs = trso_line4(
            target_outcomes,
            target_interventions,
            probability,
            active_interventions,
            domain,
            transportability_diagram,
            available_interventions,
            districts_without_interventions,
        )

        return Sum.safe(
            Product.safe(
                [trso(**trso_line4input) for trso_line4input in trso_line4inputs.values()],
            ),
            transportability_diagram.nodes() - target_interventions.union(target_outcomes),
        )

    # line 5
    else:
        # line 6
        trso_inputs = trso_line6(
            target_outcomes,
            target_interventions,
            probability,
            active_interventions,
            domain,
            transportability_diagram,
            available_interventions,
        )

        for trso_input in trso_inputs:
            expressionk = trso(**trso_input)  # FIXME this is super opaque, write out explicitly
            # line7
            if expressionk:
                return expressionk

        # line8
        districts = transportability_diagram.get_c_components()
        if len(districts) > 1:
            # line9
            if districts_without_interventions in districts:
                return trso_line9(
                    target_outcomes,
                    target_interventions,
                    probability,
                    active_interventions,
                    domain,
                    transportability_diagram,
                    available_interventions,
                )
            # line10
            for district in districts:
                if districts_without_interventions.issubset(district):
                    # district is C' districts should be D[C'], but we chose to return set of nodes instead of subgraph
                    if len(active_interventions) == 0:
                        new_available_interventions = set()
                    elif any(
                        isinstance(t, Transport)
                        for t in transportability_diagram.get_markov_pillow(district)
                    ):
                        return None
                    else:
                        new_available_interventions = available_interventions

                    return trso_line10(
                        target_outcomes,
                        target_interventions,
                        probability,
                        active_interventions,
                        domain,
                        transportability_diagram,
                        available_interventions,
                        district,
                        new_available_interventions,
                    )

        # line11
        else:
            # use guard clauses, return early
            return None


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
