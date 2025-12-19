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
    The algorithm identifies whether a causal effect P(targets | do(interventions)) can
    be computed from observational data, and if so, returns the symbolic expression.

    :param graph: The causal directed mixed graph (may contain cycles).
    :param targets: Variables whose causal effect we want to identify. (denoted as C in paper)
    :param district: A consolidated district containing the target variables. (denoted as D in paper)
    :param distribution: Probability distribution over district variables. (Denoted as Q[D] in paper)
    :param _number_recursions: Recursion depth tracker for logging.

    :returns: Identified causal effect as a symbolic probability expression. (Denoted as Q[C] in paper)

    :raises ValueError: If preconditions are violated (If preconditions violated (targets not subset of district, or
    district not subset of graph nodes.)
    :raises Unidentifiable: If causal effect cannot be identified from observational data.
    """
    # line 14 -
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
            f"Cannot identify causal effect on targets {sorted(targets)}."
            f"Reason: Ancestral closure equals district: "
            f"Targets: {sorted(targets)}, "
            f"District: {sorted(district)}, "
            f"Ancestral Closure: {sorted(ancestral_closure)}"
        )

    # TODO implement test for this in test_invalid_subsets_raise
    # checking recursive case (must have targets ⊊ ancestral_closure ⊊ district)
    # strict subsets: targets and ancestral_closure must be strictly smaller
    if not (targets < ancestral_closure and ancestral_closure < district):
        raise ValueError(
            f"Unexpected state: expected targets ⊊ ancestral_closure ⊊ district, but got "
            f"targets={sorted(targets)}, ancestral_closure={sorted(ancestral_closure)}, district={sorted(district)}"
        )

    # lines 21-26
    return identify_through_scc_decomposition(
        graph, targets, ancestral_closure, original_distribution=distribution
    )


# ------------------------------------------------------------
def validate_preconditions(
    graph: NxMixedGraph,
    targets: set[Variable],
    district: set[Variable],
    recursion_level: int = 0,
) -> None:
    """Validate IDCD algorithm preconditions.

    Checks that the inputs satisfy the required subset relationships:
    targets ⊆ district ⊆ graph.nodes()
    This corresponds to Line 14 of the IDCD algorithm in the paper.
    Ensures that:

    1. Target set C is non-empty and contained within district D.
    2. District D is non-empty and contained within graph nodes V
    3. D forms a single consolidated district in subgraph G[D].

    :param graph: The causal graph.
    :param targets: Target variable set that must be non-empty.
    :param district: Consolidated district containing targets.
    :param recursion_level: Current recursion depth for logging.

    :raises ValueError: If any precondition is violated:
        - targets is empty
        - district is empty
        - targets is not subset of district
        - district is not subset of graph nodes

    References: Algorithm 1, Line 14 from Forré & Mooij (2019)
    """
    nodes = set(graph.nodes())

    # check targets is non-empty
    if not targets:
        raise ValueError("Target set C cannot be empty")

    # check district is non-empty
    if not district:
        raise ValueError("District D cannot be empty")

    # check targets ⊆ district
    if not targets.issubset(district):
        raise ValueError(
            f"Target must be subset of district. "
            f"C={sorted(targets)}, D={sorted(district)}, C\\D={sorted(targets - district)}"
        )

    # check district ⊆ graph nodes
    if not district.issubset(nodes):
        raise ValueError(
            f"District must be subset of graph nodes. "
            f"D={district}, V={nodes}, D\\V={district - nodes}"
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

    Reduces the distribution to only include variables in the ancestral closure  # noqa: D205
    by marginalizing out (summing over) all other variables in the district.
    Implements Algorithm 1, Line 16 in the paper:
    In the paper notation, this is denoted as: Q[A] ← ∫ Q[D] d(x_{D\A})
    Mathematically if we have P(district) and want P(ancestral_closure), we marginalize out (district = ancestral_closure).

    :param distribution: Probability distribution over district variables.
    :param district: Set of all variables in current district.
    :param ancestral_closure: Subset of district variables to keep (ancestors of targets).
    :param recursion_level: Current recursion depth for logging.

    :returns: Distribution over ancestral closure variables. If ancestral_closure == district, returns the original distribution.
    Reference:
        Algorithm 1, Line 16 from Forré & Mooij (2019)

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
    original_distribution: Expression,
    recursion_level: int = 0,
) -> Expression:
    r"""Identify causal effect through SCC decomposition.

    Implements the recursive case of IDCD by decomposing the ancestral closure
    into strongly connected components (SCCs) and computing a distribution for each,
    and then recursively calling IDCD on the consolidated district.
    Strategy:
    1. Find all SCCs in the subgraph induced by ancestral closure.
    2. Filter to SCCs that are in the consolidated district of targets.
    3. For each relevant SCC, construct its conditional distribution. (Line 23)
    4. Take the product of all SCC distributions (Line 25).
    5. Recursively call IDCD on the consolidated district. (Line 26)

    This corresponds to Lines 21-26 of the algorithm in the paper in full notation:
    - Line 22: Loop over SCCs in G[A] where S ⊆ Cd^G[A](C)
    - Line 23: Construct R_A[S] ← P(S | Pred^G_<(S) ∩ A, do(J U V - A))
    - Line 25: Q[Cd^G[A](C)] ← ⊗ R_A[S] (product of SCC distributions)
    - Line 26: Recursive IDCD call

    Args:
        :param graph: The full causal graph.
        :param targets: Variables whose causal effect we want to identify.
        :param ancestral_closure: Ancestral closure of targets within current district.
        :param district: Probability distribution over current district.
        :param recursion_level: Current recursion depth.

    :returns:
        Result of recursive IDCD call.
    """
    logger.debug(
        f"[{recursion_level}]: Lines 21-26 - Recursive case (targets ⊂ ancestral_closure ⊂ district)"
    )

    subgraph_a = graph.subgraph(ancestral_closure)
    sccs = get_strongly_connected_components(subgraph_a)
    consolidated_district_a = get_consolidated_district(subgraph_a, targets)

    # line 22 - Filter to SCCs within consolidated district
    relevant_sccs = [scc for scc in sccs if scc.issubset(consolidated_district_a)]

    if not relevant_sccs:
        raise Unidentifiable(
            f"No SCCs in consolidated district = {sorted(consolidated_district_a)}"
        )

    logger.debug(f"[{recursion_level}]: Found - {len(relevant_sccs)} relevant SCCs")

    nodes = set(graph.nodes())
    intervention_set = nodes - ancestral_closure

    # line 23 - Construct distributions for each SCC
    scc_distributions = compute_scc_distributions(
        graph=graph,
        subgraph_a=subgraph_a,
        relevant_sccs=relevant_sccs,
        ancestral_closure=ancestral_closure,
        # original_distribution=original_distribution,
        intervention_set=intervention_set,
    )

    # line 25 - Take product of SCC distributions
    logger.debug(f"[{recursion_level}]: Line 25 - Product over {len(scc_distributions)} SCCs")
    district_distribution = Product.safe(scc_distributions.values())

    # line 26 - Recursive IDCD call
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
    intervention_set: set[Variable],
) -> dict[frozenset[Variable], Expression]:
    r"""Compute distributions for each strongly connected component (SCC).

    For each SCC, compute its conditional distribution by calling the ID algorithm (identify_outcomes) with
    apt-order substituted for topological order.
    This implements Line 23 of Algorithm 1 from the paper:
    In paper notation: R_A[S] ← P(S | Pred^G_<(S) ∩ A, do(J U V \ A))
    Where:
    - S is an SCC (strongly connected component)
    - Pred^G_<(S) are S's predecessors in apt-order
    - A is the ancestral closure
    - J is the intervention set
    - V is all graph nodes


    Args:
        :param graph: The full causal graph.
        :param subgraph_a: Subgraph G[A].
        :param relevant_sccs: SCCs to process.
        :param ancestral_closure: Ancestral closure.
        :param original_distribution: Original distribution from IDCD call.
        :param intervention_set: Set of variables under intervention.
        :param recursion_level: Current recursion depth.
    :returns:
        Dictionary mapping each SCC to its conditional distribution.
        Keys are frozensets of Variables (the SCCs), and values are symbolic Expressions.
    """
    apt_order_a = get_apt_order(subgraph_a)
    scc_distributions = {
        scc: identify_outcomes(
            graph=graph,
            outcomes=scc,
            treatments=intervention_set,
            conditions=predecessors if predecessors else None,
            strict=True,
            ordering=apt_order_a,
        )
        for scc in relevant_sccs
        for predecessors in [_get_apt_order_predecessors(scc, apt_order_a, ancestral_closure)]
    }
    return scc_distributions


# ---------------------------------------------------------


def _get_apt_order_predecessors(
    scc: frozenset[Variable],
    apt_order: list[Variable],
    ancestral_closure: set[Variable],
) -> set[Variable]:
    r"""Get predecessors of SCC in apt-order (helper for line 23).

    Returns all variables that:
    1. Come before the SCC in the apt-order.
    2. Are within the ancestral closure.
    For multi-node SCCs, we use the earliest position of any SCC member in the apt-order to determine
    predecessors. This ensures all SCC members are treated as a single unit in ordering.

    Where:
    - Pred^G_<(S) means "predecessors of S in the apt-order <"
    - A is the ancestral closure
    - ∩ is set intersection

    Args:
        :param scc: The strongly connected component.
        :param apt_order: Apt-order for the graph.
        :param ancestral_closure: Set to intersect with.

    :returns:
        Set of variables that are both:
        - Before the SCC in apt-order.
        - In the ancestral closure.
    """
    positions = [apt_order.index(v) for v in scc if v in apt_order]

    if not positions:
        return set()

    min_position = min(positions)
    return ancestral_closure.intersection(apt_order[:min_position])
