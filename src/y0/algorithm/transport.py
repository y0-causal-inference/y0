"""Implement of surrogate outcomes and transportability.

..seealso:: https://arxiv.org/abs/1806.07172
"""

from typing import List, Mapping, Optional, Set

from y0.dsl import Population, Transport, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "transport",
]


def find_transport_vertices(
    Zi: List[Variable], Wi: List[Variable], graph: NxMixedGraph
) -> Set[Variable]:
    """
    Parameters
    ----------
    Zi : List[Variables]
        The interventions performed in an experiment.
    Wi : List[Variables]
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
    c_component_Wi = set()
    for index, component in enumerate(c_components):
        # Check if Wi is present in the current set
        if set(Wi).intersection(component):
            c_component_Wi = c_component_Wi.union(component)

    # subgraph where Zi in edges are removed
    Zi_Bar = graph.remove_in_edges(Zi)
    # Ancestors of Wi in Zi_Bar
    Ancestors_Wi = Zi_Bar.ancestors_inclusive(Wi)

    # Descendants of Zi in graph
    Descendants_Zi = graph.descendants_inclusive(Zi)

    return (Descendants_Zi - set(Wi)).union(c_component_Wi - Ancestors_Wi)


def add_transportability_nodes(
    Zi: List[Variable], Wi: List[Variable], graph: NxMixedGraph
) -> NxMixedGraph:
    """
    Parameters
    ----------
    Zi : List[Variable]
        The interventions performed in an experiment.
    Wi : List[Variable]
        The outcomes observed in an experiment.
    graph : NxMixedGraph
        The graph of the target domain.

    Returns
    -------
    NxMixedGraph
        A graph with transportability nodes.

    """
    vertices = find_transport_vertices(Zi, Wi, graph)

    for vertex in vertices:
        raise NotImplementedError

    return graph


def surrogate_to_transport(
    Y: List[Variable],
    X: List[Variable],
    S: List[List[List[Variable]]],
    W: List[Variable],
    graph: NxMixedGraph,
):
    """


    Parameters
    ----------
    Y : List[Variable]
        A list of target variables for causal effects.
    X : List[Variable]
        A list of interventions for the target domain.
    S : List[List[List[Variable]]]
        A list of Experiments available in each domain.
    Z : List[Variable]
        A list of interventions performed in available experiments
    W : List[Variable]
        A list of outcomes observed in available experiments.
    graph : NxMixedGraph
        The graph of the target domain.

    Returns
    -------
    TransportabilityQuery
        An octuple representing the query transformation of a surrogate outcome query.

    """
    raise NotImplementedError
    # Create a list of transportability diagrams for each domain.
    D = [add_transportability_nodes(S[i][0], S[i][1], graph) for i in range(len(S))]

    Z = None
    PI = None
    pi_star = None
    D = None
    O = None
    G = graph
    return (X, Y, D, G, PI, pi_star, Z, O)


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
