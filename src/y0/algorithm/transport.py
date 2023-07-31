"""Implement of surrogate outcomes and transportability.

..seealso:: https://arxiv.org/abs/1806.07172
"""

from typing import List, Mapping, Optional, Set

from y0.algorithm.conditional_independencies import are_d_separated
from y0.dsl import Population, Product, Sum, Transport, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "transport",
]


def find_transport_vertices(
    interventions: List[Variable], surrogate_outcomes: List[Variable], graph: NxMixedGraph
) -> Set[Variable]:
    """
    Parameters
    ----------
    interventions : List[Variables]
        The interventions performed in an experiment.
    surrogate_outcomes : List[Variables]
        The outcomes observed in an experiment.
    graph : NxMixedGraph
        The graph of the target domain.

    Returns
    -------
    Set[Variable]
        Set of variables representing target domain nodes where transportability nodes should be added.
    """
    # Find the c_component with Wi
    c_components = graph.get_c_components()
    c_component_surrogate_outcomes = set()
    for index, component in enumerate(c_components):
        # Check if Wi is present in the current set
        if set(surrogate_outcomes).intersection(component):
            c_component_surrogate_outcomes = c_component_surrogate_outcomes.union(component)

    # subgraph where interventions in edges are removed
    interventions_overbar = graph.remove_in_edges(interventions)
    # Ancestors of surrogate_outcomes in interventions_overbar
    Ancestors_surrogate_outcomes = interventions_overbar.ancestors_inclusive(surrogate_outcomes)

    # Descendants of interventions in graph
    Descendants_interventions = graph.descendants_inclusive(interventions)

    return (Descendants_interventions - set(surrogate_outcomes)).union(
        c_component_surrogate_outcomes - Ancestors_surrogate_outcomes
    )


def add_transportability_nodes(
    interventions: List[Variable], surrogate_outcomes: List[Variable], graph: NxMixedGraph
) -> NxMixedGraph:
    """
    Parameters
    ----------
    interventions : List[Variable]
        The interventions performed in an experiment.
    surrogate_outcomes : List[Variable]
        The outcomes observed in an experiment.
    graph : NxMixedGraph
        The graph of the target domain.

    Returns
    -------
    NxMixedGraph
        A graph with transportability nodes.

    """
    vertices = find_transport_vertices(interventions, surrogate_outcomes, graph)
    for vertex in vertices:
        T_vertex = Transport(vertex)
        # graph.add_node(T_vertex)
        # graph.add_directed_edge(T_vertex, vertex)
    return graph


def surrogate_to_transport(
    target_outcomes: List[Variable],
    target_interventions: List[Variable],
    graph: NxMixedGraph,
    available_experiments: List[List[List[Variable]]],
):
    """


    Parameters
    ----------
    target_outcomes : List[Variable]
        A list of target variables for causal effects.
    target_interventions : List[Variable]
        A list of interventions for the target domain.
    graph : NxMixedGraph
        The graph of the target domain.
    available_experiments : List[List[List[Variable]]]
        A list of Experiments available in each domain.


    Returns
    -------
    TransportabilityQuery
        An octuple representing the query transformation of a surrogate outcome query.

    """
    raise NotImplementedError
    # Create a list of transportability diagrams for each domain.
    experiment_interventions, experiment_surrogate_outcomes = zip(*available_experiments)

    transportability_diagrams = [
        add_transportability_nodes(
            experiment_interventions[i], experiment_surrogate_outcomes[i], graph
        )
        for i in range(len(available_experiments))
    ]
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


# TODO Set of all tranportability diagrams and topological ordering are available globaly trso
def trso(
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    probability,
    active_experiments: Set[Variable],
    domain: Variable,
    diagram: NxMixedGraph,
    available_experiment_interventions: List[Set[Variable]],
):
    # line 1
    if not target_interventions:
        return Sum.safe(probability, diagram.nodes() - target_outcomes)

    # line 2
    outcomes_anc = diagram.ancestors_inclusive(target_outcomes)
    if diagram.nodes() - outcomes_anc:
        return trso(
            target_outcomes,
            target_interventions.intersection(outcomes_anc),
            Sum.safe(probability, diagram.nodes() - outcomes_anc),
            active_experiments,
            domain,
            diagram.subgraph(outcomes_anc),
            available_experiment_interventions,
        )

    # line 3

    target_interventions_overbar = diagram.remove_in_edges(target_interventions)
    additional_interventions = (
        diagram.nodes()
        - target_interventions
        - target_interventions_overbar.ancestors_inclusive(target_outcomes)
    )
    if additional_interventions:
        return trso(
            target_outcomes,
            target_interventions.union(additional_interventions),
            probability,
            active_experiments,
            domain,
            diagram,
            available_experiment_interventions,
        )

    # line 4
    C_components = diagram.subgraph(diagram.nodes() - target_interventions).get_c_components()
    if len(C_components) > 1:
        return Sum.safe(
            Product.safe(
                [
                    trso(
                        component,
                        diagram.nodes() - component,
                        probability,
                        active_experiments,
                        domain,
                        diagram,
                        available_experiment_interventions,
                    )
                    for component in C_components
                ],
                len(C_components),
            ),
            diagram.nodes() - target_interventions.union(target_outcomes),
        )

    # line 5
    else:
        # line 6
        if not active_experiments:
            # TODO this implementation of E doesn't allow lookup of ith iteration, does that matter?
            E = []
            for i in range(len(available_experiment_interventions)):
                if available_experiment_interventions[i].intersection(target_interventions):
                    raise NotImplementedError
                    # #transportability_diagrams, transportability_nodes don't exist
                    # transportability_diagrams[i].remove_in_edges(target_interventions)
                    # if are_d_separated(transportability_nodes[i],target_outcomes, ):
                    #     Ei = trso(target_outcomes,target_interventions - available_experiment_interventions[i],probability,available_experiment_interventions[i].intersection(target_interventions),i,diagram.subgraph(diagram.nodes() - available_experiment_interventions[i].intersection(target_interventions)),available_experiment_interventions)
                    #     E.append(Ei)

                    # TODO why is line 7 originally outside this loop?
                    # line 7
                    # if Ei != FAIL:
                    #     return Ei
        raise NotImplementedError

        # line8


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
