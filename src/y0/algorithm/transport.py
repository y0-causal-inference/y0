"""Implement of surrogate outcomes and transportability.

..seealso:: https://arxiv.org/abs/1806.07172
"""

from typing import Dict, List, Mapping, Optional, Set, Union

from y0.algorithm.conditional_independencies import are_d_separated
from y0.dsl import Population, Product, Sum, Transport, Variable
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
    Parameters
    ----------
    interventions: The interventions performed in an experiment.
    surrogate_outcomes: The outcomes observed in an experiment.
    graph: The graph of the target domain.

    Returns
    -------
        Set of variables representing target domain nodes where transportability nodes should be added.
    """

    if isinstance(interventions, Variable):
        interventions = {interventions}

    if isinstance(surrogate_outcomes, Variable):
        surrogate_outcomes = {surrogate_outcomes}
    # Find the c_component with Wi
    c_components = graph.get_c_components()
    c_component_surrogate_outcomes = set()
    for index, component in enumerate(c_components):
        # Check if Wi is present in the current set
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


def add_transportability_nodes(
    interventions: Set[Variable], surrogate_outcomes: Set[Variable], graph: NxMixedGraph
) -> NxMixedGraph:
    """
    Parameters
    ----------
    interventions : The interventions performed in an experiment.
    surrogate_outcomes : The outcomes observed in an experiment.
    graph : The graph of the target domain.

    Returns
    -------
    NxMixedGraph
        A graph with transportability nodes.

    """
    raise NotImplementedError
    vertices = find_transport_vertices(interventions, surrogate_outcomes, graph)
    for vertex in vertices:
        T_vertex = Transport(vertex)
        graph.add_node(T_vertex)
        graph.add_directed_edge(T_vertex, vertex)
    return graph


def surrogate_to_transport(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    graph: NxMixedGraph,
    available_experiments: List[List[List[Variable]]],
):
    """


    Parameters
    ----------
    target_outcomes: A set of target variables for causal effects.
    target_interventions: A set of interventions for the target domain.
    graph: The graph of the target domain.
    available_experiments : A set of Experiments available in each domain.


    Returns
    -------
    TransportabilityQuery
        An octuple representing the query transformation of a surrogate outcome query.

    """
    raise NotImplementedError
    # Create a dictionary of transportability diagrams for each domain.
    experiment_interventions, experiment_surrogate_outcomes = zip(*available_experiments)

    transportability_diagrams = {
        i: add_transportability_nodes(
            experiment_interventions[i], experiment_surrogate_outcomes[i], graph
        )
        for i in range(len(available_experiments))
    }
    domains = [Variable(f"Pi{i}") for i in range(1, len(available_experiments) + 1)]
    target_domains = Variable("Pi*")
    experiments_in_target_domain = set()

    return (
        target_interventions,
        target_outcomes,
        transportability_diagrams,
        graph,
        domains,
        target_domains,
        experiment_interventions,
        experiments_in_target_domain,
    )


def trso_line1(
    target_outcomes: Set[Variable],
    probability,
    transportability_diagram: NxMixedGraph,
):
    return Sum.safe(probability, transportability_diagram.nodes() - target_outcomes)


def trso_line2(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    probability,
    active_interventions: Set[Variable],
    domain: Variable,
    transportability_diagram: NxMixedGraph,
    available_interventions: List[Set[Variable]],
    outcomes_anc: Set[Variable],
):
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
    probability,
    active_interventions: Set[Variable],
    domain: Variable,
    transportability_diagram: NxMixedGraph,
    available_interventions: List[Set[Variable]],
    additional_interventions: Set[Variable],
):
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
    probability,
    active_interventions: Set[Variable],
    domain: Variable,
    transportability_diagram: NxMixedGraph,
    available_interventions: List[Set[Variable]],
    component: Set[Variable],
):
    return dict(
        target_outcomes=component,
        target_interventions=transportability_diagram.nodes() - component,
        probability=probability,
        active_interventions=active_interventions,
        domain=domain,
        transportability_diagram=transportability_diagram,
        available_interventions=available_interventions,
    )


def trso_line6(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    probability,
    active_interventions: Set[Variable],
    domain: Variable,
    transportability_diagram: NxMixedGraph,
    available_interventions: List[Set[Variable]],
    transportability_diagrams: Dict[int, NxMixedGraph],
):
    raise NotImplementedError

    if not active_interventions:
        expressions = []
        for i in range(len(transportability_diagrams)):
            if available_interventions[i].intersection(target_interventions):
                transportability_nodes = transportability_diagrams.get_transport_nodes()
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

    return expressions


# TODO Tikka paper says that topological ordering is available globaly
def trso(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    probability,
    active_interventions: Set[Variable],
    domain: Variable,
    transportability_diagrams: Dict[int, NxMixedGraph],
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
        return Sum.safe(
            Product.safe(
                [
                    trso(
                        **trso_line4(
                            target_outcomes,
                            target_interventions,
                            probability,
                            active_interventions,
                            domain,
                            transportability_diagram,
                            available_interventions,
                            component,
                        )
                    )
                    for component in districts_without_interventions
                ],
                len(districts_without_interventions),
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
            expressionk = trso(**trso_input)
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
