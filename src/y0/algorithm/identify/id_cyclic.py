"""Cyclic ID algorithm for identifying causal effects in cyclic graphs."""

from typing import Annotated

from y0.algorithm.identify import Unidentifiable
from y0.algorithm.identify.idcd import _apt_order_predecessors, idcd
from y0.algorithm.ioscm.utils import (
    get_apt_order,
    get_consolidated_district,
    get_graph_consolidated_districts,
    get_strongly_connected_components,
)
from y0.dsl import Probability, Product

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
    # find all SCCs (feedback loops) within the district
    district_subgraph = graph.subgraph(district)
    sccs = get_strongly_connected_components(district_subgraph)

    loop_distributions = []

    for scc in sccs:
        predecessors = _apt_order_predecessors(scc, apt_order)
        # compute P(SCC | predecessors) or P(SCC) if no predecessors
        distribution = Probability.safe(scc.union(predecessors)).conditional(predecessors)
        loop_distributions.append(distribution)

    # even if there's only one loop distribution, this doe sthe right thing
    return Product.safe(loop_distributions)


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
    :raises ValueError: If preconditions are violated.
    :raises Unidentifiable: If the causal effect cannot be identified based on the query
        and graph.
    """
    # input validation
    if not isinstance(outcomes, set):
        raise TypeError("Outcomes must be a set.")

    if not isinstance(interventions, set):
        raise TypeError("Interventions must be a set.")

    # line 2: validate preconditions
    # require: Y ⊆ V, W ⊆ V, Y ∩ W = ∅
    all_nodes = set(graph.nodes())

    if not outcomes:
        raise ValueError("Outcomes set cannot be empty.")

    if not outcomes.issubset(all_nodes):
        raise ValueError("Outcomes must be a subset of the graph's nodes.")

    if not interventions.issubset(all_nodes):
        raise ValueError("Interventions must be a subset of the graph's nodes.")

    if outcomes & interventions:
        raise ValueError("Outcomes and interventions must be disjoint sets.")

    # line 3: compute ancestral closure H in the mutilated graph G \ W
    graph_minus_interventions = graph.remove_nodes_from(interventions)
    ancestral_closure = graph_minus_interventions.ancestors_inclusive(outcomes)

    # line 4: get consolidated districts of H
    h_subgraph = graph_minus_interventions.subgraph(ancestral_closure)
    consolidated_districts = get_graph_consolidated_districts(h_subgraph)

    # get apt-order for the full graph
    apt_order_full = get_apt_order(graph)

    # line 5: for each district, identify Q[C]
    district_distributions = {}

    for district_c in consolidated_districts:
        # get consolidated district of C in full graph G
        consolidated_district_of_c = get_consolidated_district(graph, district_c)

        # initialize the district distribution using Proposition 9.8(3) from the paper
        initial_distribution = initialize_district_distribution(
            graph=graph,
            district=consolidated_district_of_c,
            apt_order=apt_order_full,
        )

        try:
            # call IDCD with that distribution
            result = idcd(
                graph=graph,
                outcomes=set(district_c),
                district=consolidated_district_of_c,
                distribution=initial_distribution,
            )
            district_distributions[district_c] = result

        except Unidentifiable as e:
            # lines 6-8: if that fails, then return FAIL or raise Unidentifiable
            raise Unidentifiable(
                f"Cannot identify P{outcomes} | do{interventions})). "
                f"District {district_c} failed identification."
            ) from e

    # line 10: compute the tensor product Q[H] = ⨂ Q[C]
    q_h = Product.safe(district_distributions.values())

    # line 11: marginalize to get P(Y | do(W))
    marginalize_out = ancestral_closure - outcomes

    if marginalize_out:
        result = q_h.marginalize(marginalize_out)
    else:
        result = q_h

    return result
