"""IDCD Algorithm Implementation (Algorithm 1, Lines 13-28) from Forré & Mooij (2019).

.. [forré20a] http://proceedings.mlr.press/v115/forre20a/forre20a.pdf.
"""

import logging
from collections.abc import Iterable, Sequence
from typing import Annotated

from .api import identify_outcomes
from .utils import Unidentifiable
from ..ioscm.utils import (
    get_apt_order,
    get_consolidated_district,
    get_graph_consolidated_districts,
    get_strongly_connected_components,
)
from ...dsl import Expression, Probability, Product, Variable
from ...graph import NxMixedGraph, _ensure_set
from ...util import InPaperAs

__all__ = [
    "compute_scc_distributions",
    "cyclic_id",
    "get_apt_order_predecessors",
    "idcd",
    "identify_through_scc_decomposition",
    "initialize_component_distribution",
    "initialize_district_distribution",
    "marginalize_to_ancestors",
    "validate_preconditions",
]

logger = logging.getLogger(__name__)


def cyclic_id(  # noqa:C901
    graph: Annotated[NxMixedGraph, InPaperAs("G")],
    outcomes: Annotated[Variable | Iterable[Variable], InPaperAs("Y")],
    interventions: Annotated[Variable | Iterable[Variable], InPaperAs("W")],
    *,
    ordering: Sequence[Variable] | None = None,
) -> Annotated[Expression, InPaperAs(r"P(Y \mid do(W))")]:
    """Identify causal effects in cyclic graphs.

    :param graph: Causal graph
    :param outcomes: Target variables $Y$
    :param interventions: Intervention variables $W$
    :param ordering: Ordering of variables in the graph. If not given, an apt-order is
        calculated with :func:`get_apt_order`

    :returns: Identified causal effect $P(Y | do(W))$

    :raises ValueError: If preconditions are violated.
    :raises Unidentifiable: If the causal effect cannot be identified based on the query
        and graph.
    """
    outcomes = _ensure_set(outcomes)
    interventions = _ensure_set(interventions)

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
    
    if ordering is None:
        ordering = get_apt_order(graph)

    # line 5: for each district, identify Q[C]
    district_distributions = {}

    for district_c in consolidated_districts:        
        
        # get consolidated district of C in full graph G
        consolidated_district_of_c = get_consolidated_district(graph, district_c)
        
        if not district_c.issubset(consolidated_district_of_c):
            consolidated_district_of_c = district_c


        # initialize the district distribution using Proposition 9.8(3) from the paper
        initial_distribution = initialize_district_distribution(
            graph=graph,
            district=consolidated_district_of_c,
            ordering=ordering,
        )

        try:
            district_distributions[district_c] = idcd(
                graph=graph,
                outcomes=set(district_c),
                district=consolidated_district_of_c,
                distribution=initial_distribution,
            )
        except Unidentifiable as e:
            # lines 6-8: if that fails, then return FAIL or raise Unidentifiable
            raise Unidentifiable(
                f"Cannot identify P{outcomes} | do{interventions})). "
                f"District {district_c} failed identification."
            ) from e

    # line 10: compute the tensor product Q[H] = ⨂ Q[C]
    q_h = Product.safe(district_distributions.values()).simplify()

    # line 11: marginalize to get P(Y | do(W))
    marginalize_out = ancestral_closure - outcomes
    if marginalize_out:
        result = q_h.marginalize(marginalize_out)
    else:
        result = q_h

    return result


def idcd(
    graph: Annotated[NxMixedGraph, InPaperAs("G")],
    outcomes: Annotated[set[Variable], InPaperAs("C")],
    district: Annotated[set[Variable], InPaperAs("D")],
    *,
    distribution: Annotated[Expression, InPaperAs("Q[D]")] | None = None,
    _recursion_level: int = 0,
) -> Annotated[Expression, InPaperAs("Q[C]")]:
    r"""Identify causal effects within consolidated districts of cyclic graphs.

    This implements Algorithm 1, Lines 13-28 from [forré20a]_. IDCD is a helper function
    called by the main ID algorithm at Line 5.

    Let $G$ be a directed mixed graph (DMG), $C \subseteq D \subseteq V$ where $D$ is a
    consolidated district with $\text{CD}(G_D) = \{D\}$, and $Q[D]$ is a probability
    distribution over $D$. IDCD identifies the causal effect $Q[C]$ by recursively
    shrinking the district through ancestral closure and SCC decomposition. The
    algorithm identifies whether a causal effect $P(\text{outcomes} |
    do(\text{interventions}))$ can be computed from observational data, and if so,
    returns the symbolic expression.

    :param graph: The causal directed mixed graph (may contain cycles).
    :param outcomes: Variables whose causal effect we want to identify. (denoted as $C$
        in paper)
    :param district: A consolidated district containing the outcome variables. (denoted
        as $D$ in paper)
    :param distribution: Initial probability distribution over district variables.
        (Denoted as $Q[D]$ in paper). If none, will be set to the joint distribution
        over the district
    :param _recursion_level: Recursion depth tracker for logging.

    :returns: Identified causal effect as a symbolic probability expression. (Denoted as
        $Q[C]$ in paper)

    :raises ValueError: If preconditions are violated (If preconditions violated
        (outcomes not subset of district, or district not subset of graph nodes.)
    :raises Unidentifiable: If causal effect cannot be identified from observational
        data.
    """
    # line 14 -
    validate_preconditions(graph, outcomes, district, _recursion_level=_recursion_level)

    if distribution is None:
        distribution = Probability.safe(district)

    # line 15: A <- An^G[D](C)
    ancestral_closure: Annotated[set[Variable], InPaperAs("A")] = (
        graph.subgraph(district).ancestors_inclusive(outcomes) & district
    )
    logger.debug(f"[{_recursion_level}]: Line 15 {ancestral_closure=}, {len(ancestral_closure)=}")

    # line 16
    distribution_a = marginalize_to_ancestors(
        distribution, district, ancestral_closure, _recursion_level=_recursion_level
    )

    # lines 17-18
    if ancestral_closure == outcomes:
        logger.debug(f"[{_recursion_level}]: Lines 17-18 - SUCCESS (Ancestral closure = outcomes)")
        return distribution_a

    # lines 19-20
    if ancestral_closure == district:
        logger.debug(f"[{_recursion_level}]: Lines 19-20 - FAILURE (Ancestral closure = district)")
        raise Unidentifiable(
            f"Cannot identify causal effect on {outcomes=}."
            f"Reason: Ancestral closure equals district: "
            f"{outcomes=}, "
            f"{district=}, "
            f"{ancestral_closure=}"
        )

    # checking recursive case (must have outcomes ⊊ ancestral_closure ⊊ district)
    # strict subsets: outcomes and ancestral_closure must be strictly smaller
    if not (outcomes < ancestral_closure < district):
        raise ValueError(
            f"Unexpected state: expected outcomes ⊊ ancestral_closure ⊊ district, but got "
            f"{outcomes=}, {ancestral_closure=}, {district=}"
        )

    # lines 21-26
    return identify_through_scc_decomposition(
        graph, outcomes, ancestral_closure, _recursion_level=_recursion_level
    )


def validate_preconditions(
    graph: NxMixedGraph,
    outcomes: Annotated[set[Variable], InPaperAs("C")],
    district: Annotated[set[Variable], InPaperAs("D")],
    *,
    _recursion_level: int = 0,
) -> None:
    r"""Validate IDCD algorithm preconditions.

    Checks that the inputs satisfy the required subset relationships: $C \subseteq D
    \subseteq V$ This corresponds to Line 14 of the IDCD algorithm in the paper. Ensures
    that:

    1. Outcome set $C$ is non-empty and contained within district $D$.
    2. District $D$ is non-empty and contained within graph nodes $V$
    3. $D$ forms a single consolidated district in subgraph $G[D]$.

    :param graph: The causal graph.
    :param outcomes: Outcome variable set that must be non-empty.
    :param district: Consolidated district containing outcomes.
    :param _recursion_level: Current recursion depth for logging.

    :raises ValueError: If any precondition is violated:

        - outcomes is empty
        - district is empty
        - outcomes is not subset of district
        - district is not subset of graph nodes
    """
    nodes: Annotated[set[Variable], InPaperAs("V")] = set(graph.nodes())

    # check outcomes is non-empty
    if not outcomes:
        raise ValueError("no outcomes given")

    # check district is non-empty
    if not district:
        raise ValueError("District D cannot be empty")

    # check outcomes ⊆ district
    outcomes_minus_district = outcomes - district
    if outcomes_minus_district:
        raise ValueError(
            f"Outcomes must be subset of district. "
            f"{outcomes=}, {district=}, {outcomes_minus_district=}"
        )

    # check district ⊆ graph nodes
    if not district.issubset(nodes):
        raise ValueError(
            f"District must be subset of graph nodes. "
            f"D={district}, V={nodes}, D\\V={district - nodes}"
        )
    logger.debug(
        f"[{_recursion_level}]: Line 14 - Preconditions satisfied: "
        f"{outcomes=}, {district=}, {len(nodes)=}"
    )


def marginalize_to_ancestors(
    distribution: Expression,
    district: set[Variable],
    ancestral_closure: set[Variable],
    *,
    _recursion_level: int = 0,
) -> Expression:
    r"""Marginalize distribution to ancestral closure.

    Reduces the distribution to only include variables in the ancestral closure by
    marginalizing out (summing over) all other variables in the district. Implements
    Algorithm 1, Line 16 in the paper: In the paper notation, this is denoted as: $Q[A]
    \gets \int Q[D] d(x_{D \setminus A})$. Mathematically, if we have
    $P(\text{district})$ and want $P(\text{ancestral closure})$, we marginalize out
    (district = ancestral_closure).

    :param distribution: Probability distribution over district variables.
    :param district: Set of all variables in current district.
    :param ancestral_closure: Subset of district variables to keep (ancestors of
        outcomes).
    :param _recursion_level: Current recursion depth for logging.

    :returns: Distribution over ancestral closure variables. If ancestral_closure ==
        district, returns the original distribution.
    """
    marginalize_out = district - ancestral_closure

    if not marginalize_out:
        logger.debug(
            f"[{_recursion_level}]: Line 16 - No marginalization (district = ancestral_closure)"
        )
        return distribution

    logger.debug(f"[{_recursion_level}]: Line 16 - Marginalizing out {len(marginalize_out)} vars")
    return distribution.marginalize(marginalize_out)


def identify_through_scc_decomposition(
    graph: NxMixedGraph,
    outcomes: set[Variable],
    ancestral_closure: set[Variable],
    *,
    _recursion_level: int = 0,
) -> Expression:
    r"""Identify causal effect through SCC decomposition.

    Implements the recursive case of IDCD by decomposing the ancestral closure into
    strongly connected components (SCCs) and computing a distribution for each, and then
    recursively calling IDCD on the consolidated district. Strategy:

    1. Find all SCCs in the subgraph induced by ancestral closure.
    2. Filter to SCCs that are in the consolidated district of outcomes.
    3. For each relevant SCC, construct its conditional distribution. (Line 23)
    4. Take the product of all SCC distributions (Line 25).
    5. Recursively call IDCD on the consolidated district. (Line 26)

    This corresponds to Lines 21-26 of the algorithm in the paper in full notation:

    - Line 22: Loop over SCCs in $G[A]$ where $S \subseteq Cd^G[A](C)$
    - Line 23: Construct $R_A[S] ← P(S | Pred^G_<(S) ∩ A, do(J U V - A))$
    - Line 25: $Q[Cd^G[A](C)] ← ⊗ R_A[S]$ (product of SCC distributions)
    - Line 26: Recursive IDCD call

    :param graph: The full causal graph.
    :param outcomes: Variables whose causal effect we want to identify.
    :param ancestral_closure: Ancestral closure of outcomes within current district.
    :param _recursion_level: Current recursion depth.

    :returns: Result of recursive IDCD call.
    """
    logger.debug(
        f"[{_recursion_level}]: Lines 21-26 - Recursive case (outcomes ⊂ ancestral_closure ⊂ district)"
    )

    ancestral_closure_subgraph = graph.subgraph(ancestral_closure)
    sccs = get_strongly_connected_components(ancestral_closure_subgraph)
    consolidated_district = get_consolidated_district(ancestral_closure_subgraph, outcomes)

    # line 22 - Filter to SCCs within consolidated district
    relevant_sccs = [scc for scc in sccs if scc.issubset(consolidated_district)]

    if not relevant_sccs:  # pragma: no cover
        raise Unidentifiable(f"No SCCs in {consolidated_district=}")

    logger.debug(f"[{_recursion_level}]: Found - {len(relevant_sccs)} relevant SCCs")

    nodes: Annotated[set[Variable], InPaperAs("V")] = set(graph.nodes())
    intervention_set: Annotated[set[Variable], InPaperAs("J")] = nodes - ancestral_closure

    # line 23 - Construct distributions for each SCC
    scc_distributions: Annotated[dict[frozenset[Variable], Expression], InPaperAs("R_A")] = (
        compute_scc_distributions(
            graph=graph,
            subgraph_a=ancestral_closure_subgraph,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
            intervention_set=intervention_set,
        )
    )

    # line 25 - Take product of SCC distributions
    logger.debug(f"[{_recursion_level}]: Line 25 - Product over {len(scc_distributions)} SCCs")
    district_distribution: Annotated[Expression, InPaperAs("⊗ R_A[S]")] = Product.safe(
        scc_distributions.values()
    ).simplify()

    # line 26 - Recursive IDCD call
    logger.debug(f"[{_recursion_level}]: Line 26 - Recursive call")
    return idcd(
        graph=graph,
        outcomes=outcomes,
        district=consolidated_district,
        distribution=district_distribution,
        _recursion_level=_recursion_level + 1,
    )


def compute_scc_distributions(
    graph: Annotated[NxMixedGraph, InPaperAs("G")],
    subgraph_a: Annotated[NxMixedGraph, InPaperAs("G[A]")],
    relevant_sccs: list[frozenset[Variable]],
    ancestral_closure: Annotated[set[Variable], InPaperAs("A")],
    intervention_set: Annotated[set[Variable], InPaperAs("J")],
) -> dict[frozenset[Variable], Expression]:
    r"""Compute distributions for each strongly connected component (SCC).

    For each SCC, compute its conditional distribution by calling the ID algorithm
    (:func:`identify_outcomes`) with apt-order substituted for topological order. This
    implements Line 23 of Algorithm 1 from the paper: In paper notation: $R_A[S] ← P(S |
    Pred^G_<(S) ∩ A, do(J U V A))$ where:

    - $S$ is a strongly connected component (SCC)
    - $Pred^G_<(S)$ are S's predecessors in apt-order
    - $A$ is the ancestral closure
    - $J$ is the intervention set
    - $V$ is all graph nodes

    :param graph: The full causal graph.
    :param subgraph_a: Subgraph $G[A]$.
    :param relevant_sccs: SCCs to process.
    :param ancestral_closure: Ancestral closure.
    :param intervention_set: Set of variables under intervention.

    :returns: Dictionary mapping each SCC to its conditional distribution. Keys are
        frozensets of Variables (the SCCs), and values are symbolic Expressions.
    """
    apt_order_a = get_apt_order(subgraph_a)
    scc_distributions = {
        scc: identify_outcomes(
            graph=graph,
            outcomes=scc,
            treatments=intervention_set,
            conditions=get_apt_order_predecessors(scc, apt_order_a, ancestral_closure) or None,
            strict=True,
            ordering=apt_order_a,
        )
        for scc in relevant_sccs
    }
    return scc_distributions


def get_apt_order_predecessors(
    scc: frozenset[Variable],
    apt_order: list[Variable],
    ancestral_closure: set[Variable],
) -> set[Variable]:
    r"""Get predecessors of SCC in apt-order (helper for line 23).

    Returns all variables that:

    1. Come before the SCC in the apt-order.
    2. Are within the ancestral closure.

    For multi-node SCCs, we use the earliest position of any SCC member in the apt-order
    to determine predecessors. This ensures all SCC members are treated as a single unit
    in ordering. Where:

    - Pred^G_<(S) means "predecessors of S in the apt-order <"
    - A is the ancestral closure
    - ∩ is set intersection

    :param scc: The strongly connected component.
    :param apt_order: Apt-order for the graph.
    :param ancestral_closure: Set to intersect with.

    :returns: Set of variables that are both: - Before the SCC in apt-order. - In the
        ancestral closure.
    """
    return ancestral_closure.intersection(_get_predecessors(scc, apt_order))


def _get_predecessors(variables: Iterable[Variable], ordering: Sequence[Variable]) -> set[Variable]:
    """Get predecessors in the apt-order."""
    positions = [ordering.index(variable) for variable in variables if variable in ordering]
    if not positions:
        return set()
    min_position = min(positions)
    return set(ordering[:min_position])


def initialize_district_distribution(
    graph: NxMixedGraph,
    district: set[Variable],
    ordering: Sequence[Variable],
) -> Expression:
    """Initialize the probability distribution for a given district before identification.

    This implements Proposition 9.8(3): each district's initial distribution is built by
    finding its strongly connected components (feedback loops), computing each
    component's distribution conditioned on its variables (predecessors) that are in
    apt-order, and taking the product over these components.

    :param graph: The mixed graph representing the causal structure.
    :param district: The set of variables in the district to initialize.
    :param ordering: Apt-order of variables in the graph.

    :returns: Initial distribution for the district
    """
    # find all SCCs (feedback loops) within the district
    district_subgraph = graph.subgraph(district)
    sccs = get_strongly_connected_components(district_subgraph)
    return Product.safe(
        initialize_component_distribution(set(scc), _get_predecessors(scc, ordering))
        for scc in sccs
    ).simplify()


def initialize_component_distribution(
    nodes: set[Variable], predecessors: set[Variable]
) -> Expression:
    """Initialize the probability distribution for a component given its predecessors.

    Implements Proposition 9.8(3): For a component $S$ with predecessors $P$ in
    apt-order, compute $P(S | P) = P(S U P) / P(P)$.

    :param nodes: The component nodes (SCC)
    :param predecessors: The predecessor nodes in apt-order.

    :returns: Conditional probability P(nodes | predecessors)
    """
    if not predecessors:
        return Probability.safe(nodes)

    return Probability.safe(nodes | predecessors).conditional(predecessors)
