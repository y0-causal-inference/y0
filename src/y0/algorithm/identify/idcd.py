"""IDCD Algorithm Implementation (Algorithm 1, Lines 13-28) from Forré & Mooij (2019)."""

import logging

from ..identify.id_std import identify
from ..identify.utils import Identification, Unidentifiable
from ..identify.id_std import identify
from ..identify.utils import Identification, Unidentifiable
from ..ioscm.utils import (
    get_apt_order,
    get_consolidated_district,
    get_strongly_connected_components,
)
from ...dsl import Expression, P, Product, Variable
from ...graph import NxMixedGraph

__all__ = ["idcd"]

logger = logging.getLogger(__name__)


def idcd(
    graph: NxMixedGraph,
    C: set[Variable],
    D: set[Variable],
    Q_D: Expression,
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
    :param C: Target set of variables to identify within district D.
    :param D: Consolidated district containing C.
    :param Q_D: Probability distribution over district D.
    :param _number_recursions: Recursion depth tracker for logging.

    :returns: Identified causal effect Q[C].

    :raises ValueError: If preconditions are violated (C ⊆ D ⊆ V).
    :raises Unidentifiable: If causal effect cannot be identified.
    """
    # line 14
    validate_preconditions(graph, C, D, _number_recursions)

    # line 15
    subgraph_D = graph.subgraph(D)
    A = line_15(subgraph_D, C, D, _number_recursions)

    # line 16
    Q_A = line_16(subgraph_D, Q_D, D, A, _number_recursions)

    # lines 17-18
    if A == C:
        logger.debug(f"[{_number_recursions}]: Lines 17-18 - SUCCESS (A = C)")
        return Q_A

    # lines 19-20
    if A == D:
        logger.debug(f"[{_number_recursions}]: Lines 19-20 - FAILURE (A = D)")
        raise Unidentifiable(
            f"Cannot identify: ancestral closure equals district (A = D = {sorted(A)})"
        )

    # lines 21-26
    return lines_21_26(graph, C, A, D, _number_recursions)


def validate_preconditions(
    graph: NxMixedGraph,
    targets: set[Variable],
    district: set[Variable],  # district containing C
    recursion_level: int,
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
     References:
        Algorithm 1, Line 14 from Forré & Mooij (2019)
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
            f"C must be subset of D. "
            f"C={sorted(targets)}, D={sorted(district)}, C\\D={sorted(targets - district)}"
        )

    if not district.issubset(nodes):
        raise ValueError(
            f"D must be subset of V. "
            f"D={sorted(district)}, V={sorted(nodes)}, D\\V={sorted(district - nodes)}"
        )

    # check CD(G_D) = {D}
    subgraph_d = graph.subgraph(district)
    consolidated_district = get_consolidated_district(subgraph_d, district)

    if consolidated_district != district:
        raise ValueError(
            f"D must be a single consolidated district in G[D]."
            f"Expected CD(G_D) = {{D}}, but got CD(G_D) = {{{sorted(consolidated_district)}}}"
        )

    logger.debug(
        f"[{recursion_level}]: Line 14 - Preconditions satisfied: "
        f"C={sorted(targets)}, D={sorted(district)}, |V|={len(nodes)}"
    )


# ------------------------------------------------------------


def line_15(
    subgraph_D: NxMixedGraph,
    C: set[Variable],
    D: set[Variable],
    recursion_level: int,
) -> set[Variable]:
    r"""Run line 15 of IDCD algorithm.

    Line 15: A ← Anc^G[D](C) ∩ D

    Computes the ancestral closure of C within subgraph G[D].

    :param subgraph_D: Subgraph G[D] induced by district D.
    :param C: Target variable set.
    :param D: District containing C.
    :param recursion_level: Current recursion depth for logging.

    :returns: Ancestral closure A.
    """
    A = set().union(*(subgraph_D.ancestors_inclusive(c) for c in C)) & D
    logger.debug(f"[{recursion_level}]: Line 15 - A = {sorted(A)}, |A| = {len(A)}")
    return A


def line_16(
    subgraph_D: NxMixedGraph,
    Q_D: Expression,
    D: set[Variable],
    A: set[Variable],
    recursion_level: int,
) -> Expression:
    r"""Run line 16 of IDCD algorithm.

    Line 16: Q[A] ← ∫ Q[D] d(x_{D\A})

    Marginalizes Q[D] over variables D A using apt-order for deterministic ordering.

    :param subgraph_D: Subgraph G[D].
    :param Q_D: Distribution over D.
    :param D: District.
    :param A: Ancestral closure.
    :param recursion_level: Current recursion depth for logging.

    :returns: Distribution Q[A].
    """
    marginalize_out = D - A

    if not marginalize_out:
        logger.debug(f"[{recursion_level}]: Line 16 - No marginalization (D = A)")
        return Q_D

    apt_order_D = get_apt_order(subgraph_D)
    apt_order_marginalize = [v for v in apt_order_D if v in marginalize_out]

    logger.debug(f"[{recursion_level}]: Line 16 - Marginalizing {len(apt_order_marginalize)} vars")
    logger.debug(f"[{recursion_level}]: Line 16 - Marginalizing {len(apt_order_marginalize)} vars")

    return Q_D.marginalize(apt_order_marginalize)


def lines_21_26(
    graph: NxMixedGraph,
    C: set[Variable],
    A: set[Variable],
    D: set[Variable],
    recursion_level: int,
) -> Expression:
    r"""Run lines 21-26 of IDCD algorithm.

    Lines 21-26: Recursive case when C ⊂ A ⊂ D

    - Line 22: Loop over SCCs in G[A] where S ⊆ Cd^G[A](C)
    - Line 23: Compute R_A[S] ← P(S | Pred^G_<(S) ∩ A, do(J ∪ V - A))
    - Line 25: Q[Cd^G[A](C)] ← ⊗ R_A[S]
    - Line 26: Recursive IDCD call

    :param graph: The full causal graph.
    :param C: Target variable set.
    :param A: Ancestral closure.
    :param D: Original district.
    :param recursion_level: Current recursion depth.

    :returns: Result of recursive IDCD call.
    """
    logger.debug(f"[{recursion_level}]: Lines 21-26 - Recursive case (C ⊂ A ⊂ D)")

    subgraph_A = graph.subgraph(A)
    sccs = get_strongly_connected_components(subgraph_A)
    CdG_A_C = get_consolidated_district(subgraph_A, C)

    # line 22
    relevant_sccs = [S for S in sccs if S.issubset(CdG_A_C)]

    if not relevant_sccs:
        raise Unidentifiable(f"No SCCs in Cd^G[A](C) = {sorted(CdG_A_C)}")

    logger.debug(f"[{recursion_level}]: Line 22 - {len(relevant_sccs)} relevant SCCs")

    # line 23
    RA = line_23(graph, subgraph_A, relevant_sccs, A, recursion_level)

    # line 25
    Q_Cd = line_25(RA, relevant_sccs, recursion_level)

    # line 26
    logger.debug(f"[{recursion_level}]: Line 26 - Recursive call")
    return idcd(
        graph=graph,
        C=C,
        D=CdG_A_C,
        Q_D=Q_Cd,
        _number_recursions=recursion_level + 1,
    )


def line_23(
    graph: NxMixedGraph,
    subgraph_A: NxMixedGraph,
    relevant_sccs: list[frozenset[Variable]],
    A: set[Variable],
    recursion_level: int,
) -> dict[frozenset[Variable], Expression]:
    r"""Run line 23 of IDCD algorithm.

    Line 23: R_A[S] ← P(S | Pred^G_<(S) ∩ A, do(J ∪ V - A))

    For each SCC S, calls the main ID algorithm to identify the conditional causal
    effect.

    :param graph: The full causal graph.
    :param subgraph_A: Subgraph G[A].
    :param relevant_sccs: SCCs to process.
    :param A: Ancestral closure.
    :param recursion_level: Current recursion depth.

    :returns: Dictionary mapping each SCC to its distribution R_A[S].
    """
    V = set(graph.nodes())
    J = set()  # Assuming no background interventions (J = ∅)
    apt_order_A = get_apt_order(subgraph_A)

    RA: dict[frozenset[Variable], Expression] = {}

    for S in relevant_sccs:
        # Compute Pred^G_<(S) ∩ A (predecessors in apt-order)
        conditioning_set = _get_apt_order_predecessors(S, apt_order_A, A)

        intervention_set = J | (V - A)

        logger.debug(
            f"[{recursion_level}]: Line 23 - R_A[{sorted(S)}]: "
            f"P({sorted(S)} | {sorted(conditioning_set)}, do({sorted(intervention_set)}))"
        )

        # Build identification query for main ID algorithm
        identification_obj = Identification.from_parts(
            outcomes=S,
            treatments=intervention_set,
            conditions=conditioning_set,
            graph=graph,
            estimand=P(graph.nodes()),
        )

        # Call main ID algorithm to identify R_A[S]
        RA[S] = identify(identification_obj)

    return RA


def line_25(
    RA: dict[frozenset[Variable], Expression],
    relevant_sccs: list[frozenset[Variable]],
    recursion_level: int,
) -> Expression:
    r"""Run line 25 of IDCD algorithm.

    Line 25: Q[Cd^G[A](C)] ← ⊗ R_A[S]

    Computes the tensor product (⊗) over all R_A[S] distributions.

    :param RA: Dictionary of SCC distributions.
    :param relevant_sccs: List of SCCs.
    :param recursion_level: Current recursion depth.

    :returns: Tensor product expression Q[Cd^G[A](C)].
    """
    # Sort SCCs for deterministic ordering
    sorted_sccs = sorted(relevant_sccs, key=lambda s: min(s) if s else Variable(""))

    logger.debug(f"[{recursion_level}]: Line 25 - Product over {len(RA)} SCCs")

    return Product.safe([RA[S] for S in sorted_sccs])


def _get_apt_order_predecessors(
    S: frozenset[Variable],
    apt_order: list[Variable],
    A: set[Variable],
) -> set[Variable]:
    r"""Get predecessors of SCC in apt-order (helper for line 23).

    Computes Pred^G_<(S) ∩ A, which is the set of all variables that come before S in
    the apt-order and are within A.

    :param S: The SCC.
    :param apt_order: Apt-order for the graph.
    :param A: Set to intersect with.

    :returns: Set of predecessors Pred^G_<(S) ∩ A.
    """
    s_positions = [apt_order.index(v) for v in S if v in apt_order]

    if not s_positions:
        raise ValueError(f"SCC {sorted(S)} not found in apt-order")

    min_position = min(s_positions)

    return {apt_order[i] for i in range(min_position) if apt_order[i] in A}

