"""IDCD Algorithm Implementation (Algorithm 1, Lines 13-28) from Forré & Mooij (2019)."""

import logging

from ..identify import Unidentifiable, identify_outcomes
from ..ioscm.utils import (
    get_apt_order,
    get_consolidated_district,
    get_strongly_connected_components,
)
from ...dsl import Expression, Product, Variable
from ...graph import NxMixedGraph

__all__ = ["idcd"]

logger = logging.getLogger(__name__)


def idcd(
    graph: NxMixedGraph,
    targets: set[Variable],
    district: set[Variable],
    distribution: Expression,
    *,
    _number_recursions: int = 0,
) -> Expression:
    r"""Identify causal effects within consolidated districts of cyclic graphs.

    This implements Algorithm 1, Lines 13-28 from [forré20a]_. IDCD is a helper function
    called by the main ID algorithm at Line 5.

    Let $G$ be a directed mixed graph (DMG), $C \subseteq D \subseteq V$ where $D$ is a
    consolidated district with $\text{CD}(G_D) = \{D\}$, and $Q[D]$ is a probability
    distribution over $D$. IDCD identifies the causal effect $Q[C]$ by recursively
    shrinking the district through ancestral closure and SCC decomposition.

    :param graph: The causal directed mixed graph (may contain cycles).
    :param targets: Target set of variables to identify within district D.
    :param district: Consolidated district containing C.
    :param distribution: Probability distribution over district D.
    :param _number_recursions: Recursion depth tracker for logging.

    :returns: Identified causal effect Q[C].

    :raises ValueError: If preconditions are violated (C ⊆ D ⊆ V).
    :raises Unidentifiable: If causal effect cannot be identified.
    """
    # line 14
    validate_preconditions(graph, targets, district, _number_recursions)

    # line 15: A <- An^G[D](C)
    subgraph_d = graph.subgraph(district)
    ancestral_closure = subgraph_d.ancestors_inclusive(targets) & district
    logger.debug(
        f"[{_number_recursions}]: Line 15 - A = {sorted(ancestral_closure)}, |A| = {len(ancestral_closure)}"
    )

    # line 16
    distribution_a = marginalize_to_ancestors(
        distribution, district, ancestral_closure, _number_recursions
    )

    # lines 17-18
    if ancestral_closure == targets:
        logger.debug(f"[{_number_recursions}]: Lines 17-18 - SUCCESS (Ancestral closure = targets)")
        return distribution_a

    # lines 19-20
    if ancestral_closure == district:
        logger.debug(
            f"[{_number_recursions}]: Lines 19-20 - FAILURE (Ancestral closure = district)"
        )
        raise Unidentifiable(
            f"Causal effect Q[{sorted(targets)}] is unidentifiable within district D = {sorted(ancestral_closure)}"
        )

    # TODO - add test for this error case here
    # checking recursive case (must have targets ⊊ ancestral_closure ⊊ district)
    # strict subsets: targets and ancestral_closure must be strictly smaller
    if not (targets < ancestral_closure and ancestral_closure < district):
        raise ValueError(
            f"Unexpected state: expected targets ⊊ ancestral_closure ⊊ district, but got "
            f"targets={sorted(targets)}, ancestral_closure={sorted(ancestral_closure)}, district={sorted(district)}"
        )

    # TODO needs end-to-end test where this gets called
    # lines 21-26
    return identify_through_scc_decomposition(
        graph,
        targets,
        ancestral_closure,
    )


# ------------------------------------------------------------
def validate_preconditions(
    graph: NxMixedGraph,
    targets: set[Variable],
    district: set[Variable],  # district containing C
    recursion_level: int = 0,
) -> None:
    """Validate IDCD algorithm preconditions.

    Line 14: require C ⊆ D ⊆ V, CD(G_D) = {D}

    Ensures that:

    1. Target set C is non-empty and contained within district D.
    2. District D is non-empty and contained within graph nodes V
    3. D forms a single consolidated district in subgraph G[D].

    :param graph: The causal graph.
    :param targets: Target variable set.
    :param district: Consolidated district containing targets.
    :param recursion_level: Current recursion depth for logging.

    :raises ValueError: If any precondition is violated.

    References: Algorithm 1, Line 14 from Forré & Mooij (2019)
    """
    nodes = set(graph.nodes())

    # check C is non-empty
    if not targets:
        raise ValueError("Target set C cannot be empty")

    # check D is non-empty
    if not district:
        raise ValueError("District D cannot be empty")

    # check C ⊆ D
    if not targets.issubset(district):
        raise ValueError(
            f"Target must be subset of district. "
            f"C={sorted(targets)}, D={sorted(district)}, C\\D={sorted(targets - district)}"
        )

    if not district.issubset(nodes):
        raise ValueError(
            f"District must be subset of graph nodes. "
            f"D={district}, V={nodes}, D\\V={district - nodes}"
        )

    # check CD(G_D) = {D}
    subgraph_d = graph.subgraph(district)
    consolidated_district = get_consolidated_district(subgraph_d, district)

    # TODO: Testing - verify error is raised when CD(G_D) != {D} (unit test for this case)
    if consolidated_district != district:
        raise ValueError(
            f"D must be a single consolidated district in G[D]."
            f"Expected CD(G_D) = {{D}}, but got CD(G_D) = {{{sorted(consolidated_district)}}}"
        )

    logger.debug(
        f"[{recursion_level}]: Line 14 - Preconditions satisfied: "
        f"C={sorted(targets)}, D={sorted(district)}, |V|={len(nodes)}"
    )


# ---------------------------------------------------------


def marginalize_to_ancestors(
    distribution: Expression,
    district: set[Variable],
    ancestral_closure: set[Variable],
    recursion_level: int = 0,
) -> Expression:
    r"""Marginalize distribution to ancestral closure.

    Implements Algorithm 1, Line 16: Q[A] ← ∫ Q[D] d(x_{D\A})

    Marginalizes Q[D] over variables D\A using apt-order for deterministic ordering.

    :param distribution: Distribution over D.
    :param district: District.
    :param ancestral_closure: Ancestral closure.
    :param recursion_level: Current recursion depth for logging.

    :returns: Distribution Q[A].
    """
    marginalize_out = district - ancestral_closure

    if not marginalize_out:
        logger.debug(
            f"[{recursion_level}]: Line 16 - No marginalization (district = ancestral_closure)"
        )
        return distribution

    logger.debug(f"[{recursion_level}]: Line 16 - Marginalizing out {len(marginalize_out)} vars")
    return distribution.marginalize(marginalize_out)


# ---------------------------------------------------------
def identify_through_scc_decomposition(
    graph: NxMixedGraph,
    targets: set[Variable],
    ancestral_closure: set[Variable],
    recursion_level: int = 0,
) -> Expression:
    r"""Identify causal effect through SCC decomposition.

    - Line 22: Loop over SCCs in G[A] where S ⊆ Cd^G[A](C)
    - Line 23: Compute R_A[S] ← P(S | Pred^G_<(S) ∩ A, do(J U V - A))
    - Line 25: Q[Cd^G[A](C)] ← ⊗ R_A[S]
    - Line 26: Recursive IDCD call

    :param graph: The full causal graph.
    :param targets: Target variable set.
    :param ancestral_closure: Ancestral closure.
    :param district: Original district.
    :param recursion_level: Current recursion depth.

    :returns: Result of recursive IDCD call.
    """
    logger.debug(f"[{recursion_level}]: Lines 21-26 - Recursive case (C ⊂ A ⊂ D)")

    subgraph_a = graph.subgraph(ancestral_closure)
    sccs = get_strongly_connected_components(subgraph_a)
    consolidated_district_a = get_consolidated_district(subgraph_a, targets)

    # line 22
    # TODO - add test for no relevant SCCs found
    relevant_sccs = [scc for scc in sccs if scc.issubset(consolidated_district_a)]

    if not relevant_sccs:
        raise Unidentifiable(f"No SCCs in Cd^G[A](C) = {sorted(consolidated_district_a)}")

    logger.debug(f"[{recursion_level}]: Line 22 - {len(relevant_sccs)} relevant SCCs")

    # line 23
    scc_distributions = compute_scc_distributions(
        graph, subgraph_a, relevant_sccs, ancestral_closure
    )

    # line 25
    logger.debug(f"[{recursion_level}]: Line 25 - Product over {len(scc_distributions)} SCCs")
    district_distribution = Product.safe(scc_distributions.values())

    # line 26
    logger.debug(f"[{recursion_level}]: Line 26 - Recursive call")
    return idcd(
        graph=graph,
        targets=targets,
        district=consolidated_district_a,
        distribution=district_distribution,
        _number_recursions=recursion_level + 1,
    )


# ---------------------------------------------------------


def compute_scc_distributions(
    graph: NxMixedGraph,
    subgraph_a: NxMixedGraph,
    relevant_sccs: list[frozenset[Variable]],
    ancestral_closure: set[Variable],
) -> dict[frozenset[Variable], Expression]:
    r"""Compute distributions for each strongly connected component (SCC).

    Implements Algorithm 1, Line 23: R_A[S] ← P(S | Pred^G_<(S) ∩ A, do

    For each SCC S, calls the main ID algorithm to identify the conditional causal
    effect.

    :param graph: The full causal graph.
    :param subgraph_a: Subgraph G[A].
    :param relevant_sccs: SCCs to process.
    :param ancestral_closure: Ancestral closure.
    :param recursion_level: Current recursion depth.

    :returns: Dictionary mapping each SCC to its distribution R_A[S].
    """
    nodes = set(graph.nodes())
    apt_order_a = get_apt_order(subgraph_a)

    intervention_set = nodes - ancestral_closure

    scc_distributions = {
        # Call main ID algorithm to identify R_A[S]
        scc: identify_outcomes(
            outcomes=scc,
            treatments=intervention_set,
            # Compute Pred^G_<(S) ∩ A (predecessors in apt-order)
            conditions=_get_apt_order_predecessors(scc, apt_order_a, ancestral_closure),
            graph=graph,
            strict=True,
            ordering=apt_order_a,
        )
        for scc in relevant_sccs
    }
    return scc_distributions


# ---------------------------------------------------------


def _get_apt_order_predecessors(
    scc: frozenset[Variable],
    apt_order: list[Variable],
    ancestral_closure: set[Variable],
) -> set[Variable]:
    r"""Get predecessors of SCC in apt-order (helper for line 23).

    Computes Pred^G_<(S) ∩ A, which is the set of all variables that come before S in
    the apt-order and are within A.

    :param scc: The strongly connected component.
    :param apt_order: Apt-order for the graph.
    :param ancestral_closure: Set to intersect with.

    :returns: Set of predecessors Pred^G_<(S) ∩ A.
    """
    positions = [apt_order.index(v) for v in scc if v in apt_order]

    if not positions:
        return set()

    min_position = min(positions)
    return ancestral_closure.intersection(apt_order[:min_position])
