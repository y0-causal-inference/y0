"""Cyclic ID algorithm for identifying causal effects in cyclic graphs."""

from typing import Annotated

from ...dsl import Expression, Variable
from ...graph import NxMixedGraph
from ...util import InPaperAs

___all__ = [
    "initialize_district_distribution",
    "cyclic_id",
]


def initialize_district_distribution(
    graph: NxMixedGraph,
    district: set[Variable],
    apt_order: list[Variable],
) -> Expression:
    """Initialize the probability distribution for a given district before identification.

    This implements Proposition 9.8(3): each district's initial distribution is built by
    finding its strongly connected components (feedback loops), computing each
    component's distribution conditioned on its variables (predecessors) that are in
    apt-order, and taking the product over these components.

    :param graph: The mixed graph representing the causal structure.
    :param district: The set of variables in the district to initialize.
    :param apt_order: Apt-order of variables in the graph.

    :returns: Initial distribution for the district
    """
    raise NotImplementedError("initialize_district_distribution is not yet implemented.")


def cyclic_id(
    graph: Annotated[NxMixedGraph, InPaperAs("G")],
    outcomes: Annotated[set[Variable], InPaperAs("Y")],
    interventions: Annotated[set[Variable], InPaperAs("W")],
) -> Annotated[Expression, InPaperAs(r"P(Y \mid do(W))")]:
    """Identify causal effects in cyclic graphs.

    :param graph: Causal graph
    :param outcomes: Target variables Y
    :param interventions: Intervention variables W

    :returns: Identified causal effect P(Y | do(W))

    :raises Unidentifiable: If the causal effect cannot be identified based on the query
        and graph.
    """
    raise NotImplementedError("cyclic_id not yet implemented")
