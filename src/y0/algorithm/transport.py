"""Implement of surrogate outcomes and transportability.

..seealso:: https://arxiv.org/abs/1806.07172
"""

from typing import List, Mapping, Optional, Set

from y0.dsl import Population, Transport, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "transport",
]


def find_transport_vertices(Zi: Variable, Wi: Variable, graph: NxMixedGraph) -> Set[Variable]:
    """
    Parameters
    ----------
    Zi : Variable
        The intervention performed in an experiment.
    Wi : Variable
        The outcome observed in an experiment.
    graph : NxMixedGraph
        The graph of the target domain.

    Returns
    -------
    Set[Variable]
        Set of variables representing target domain nodes where transportability nodes should be added.
    """
    # Find the c_component with Wi
    c_components = graph.get_c_components()
    for index, component in enumerate(c_components):
        # Check if Wi is present in the current set
        if Wi in component:
            c_component_Wi = component
            break
    # subgraph where Zi in edges are removed
    Zi_Bar = graph.remove_in_edges(Zi)
    # Ancestors of Wi in Zi_Bar
    Ancestors_Wi = Zi_Bar.ancestors_inclusive(Wi)

    # Descendants of Zi in graph
    Descendants_Zi = graph.descendants_inclusive(Zi)

    return (Descendants_Zi - {Wi}).union(c_component_Wi - Ancestors_Wi)


def add_transportability_nodes(Zi: Variable, Wi: Variable, graph: NxMixedGraph) -> NxMixedGraph:
    """
    Parameters
    ----------
    Zi : Variable
        The intervention performed in an experiment.
    Wi : Variable
        The outcome observed in an experiment.
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
    Y: List[Variable], X: List[Variable], Z: List[Variable], W: List[Variable], graph: NxMixedGraph
):
    """


    Parameters
    ----------
    Y : List[Variable]
        A list of target variables for causal effects.
    X : List[Variable]
        A list of interventions for the target domain.
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
    number_of_diagrams = len(Z)
    PI = list(range(1, number_of_diagrams))
    # D = emptylist of size len(Z)
    for i in range(len(Z)):
        Di = add_transportability_nodes(Z[i], W[i], graph)

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
