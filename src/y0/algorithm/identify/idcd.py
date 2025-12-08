"""
IDCD Algorithm Implementation (Algorithm 1, Lines 13-28) from Forré & Mooij (2019).

This module implements the IDCD (IDentification within Consolidated Districts) algorithm,
which is a helper function for the main cyclic ID algorithm. IDCD handles identification
of causal effects within consolidated districts of directed mixed graphs that may contain
cycles and latent confounders.
"""

import logging

from ..ioscm.utils import (
    get_apt_order,
    get_consolidated_district,
    get_strongly_connected_components,
)
from ..identify.utils import Identification, Unidentifiable
from ..identify.id_std import identify
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
    """
    Run IDCD algorithm from Forré & Mooij (2019), Algorithm 1, Lines 13-28.
    
    Identifies causal effects within consolidated districts of cyclic graphs.
    This is a helper function called by the main ID algorithm at Line 5.
    
    :param graph: The causal directed mixed graph (may contain cycles)
    :param C: Target set of variables to identify within district D
    :param D: Consolidated district containing C
        Precondition: C ⊆ D ⊆ V and CD(G_D) = {D}
    :param Q_D: Probability distribution over district D
    :param _number_recursions: Recursion depth tracker for logging
    :returns: Identified causal effect Q[C]
    :raises ValueError: If preconditions are violated
    :raises Unidentifiable: If causal effect cannot be identified
    
    References:
        Forré, P., & Mooij, J. M. (2019). Causal Calculus in the Presence of
        Cycles, Latent Confounders and Selection Bias. arXiv:1901.00433
        Algorithm 1, Lines 13-28.
    """
    
    # ================================================================
    # Line 14: PRECONDITION VALIDATION
    # require: C ⊆ D ⊆ V
    # ================================================================
    
    V = set(graph.nodes())
    
    # Precondition 1: C must not be empty
    if not C:
        raise ValueError("Target set C cannot be empty")
    
    # Precondition 2: D must not be empty
    if not D:
        raise ValueError("District D cannot be empty")
    
    # Precondition 3: C ⊆ D (target variables must be within district)
    if not C.issubset(D):
        raise ValueError(
            f"C must be a subset of D. "
            f"C = {sorted(C)}, D = {sorted(D)}, C \\ D = {sorted(C - D)}"
        )
    
    # Precondition 4: D ⊆ V (district must be within graph nodes)
    if not D.issubset(V):
        raise ValueError(
            f"D must be a subset of graph nodes V. "
            f"D = {sorted(D)}, V = {sorted(V)}, D \\ V = {sorted(D - V)}"
        )
    
    logger.debug(
        f"[{_number_recursions}]: Calling IDCD\n"
        f"\t C (target): {sorted(C)}\n"
        f"\t D (district): {sorted(D)}\n"
        f"\t |V| = {len(V)}"
    )
    
    # ================================================================
    # Line 15: A ← Anc^G[D](C) ∩ D
    # Compute ancestral closure of C within subgraph G[D]
    # ================================================================
    
    logger.debug(f"[{_number_recursions}]: Line 15 - Computing A = Anc^G[D](C) ∩ D")
    
    # Create subgraph G[D] induced by nodes in D
    subgraph_D = graph.subgraph(D)
    
    # Get ancestors of all variables in C within G[D], intersected with D
    A = set().union(*(subgraph_D.ancestors_inclusive(c) for c in C)) & D
    
    logger.debug(
        f"[{_number_recursions}]: Computed ancestral closure\n"
        f"\t A = {sorted(A)}\n"
        f"\t |A| = {len(A)}"
    )
    
    # ================================================================
    # Line 16: Q[A] ← ∫ Q[D] d(x_{D\A})
    # Marginalize Q[D] over variables in D \ A
    # ================================================================
    
    logger.debug(f"[{_number_recursions}]: Line 16 - Marginalizing Q[D] to Q[A]")
    
    marginalize_out = D - A
    
    if marginalize_out:
        # Get apt-order for subgraph G[D]
        apt_order_D = get_apt_order(subgraph_D)
        
        # Filter to variables being marginalized out, maintaining apt-order
        apt_order_marginalize = [v for v in apt_order_D if v in marginalize_out]
        
        logger.debug(
            f"[{_number_recursions}]: Marginalizing over {len(apt_order_marginalize)} variable(s) (in apt-order):\n"
            f"\t {apt_order_marginalize}"
        )
        Q_A = Q_D.marginalize(apt_order_marginalize)
    else:
        logger.debug(f"[{_number_recursions}]: No marginalization needed (D = A)")
        Q_A = Q_D
    
    logger.debug(f"[{_number_recursions}]: Q[A] computed")
    
    # ================================================================
    # Lines 17-18: Terminal case - SUCCESS
    # if A = C then return Q[A]
    # ================================================================
    
    if A == C:
        logger.debug(
            f"[{_number_recursions}]: Lines 17-18 - SUCCESS (Terminal case)\n"
            f"\t A = C = {sorted(C)}\n"
            f"\t Returning Q[A]"
        )
        return Q_A
    
    # ================================================================
    # Lines 19-20: Terminal case - FAILURE
    # else if A = D then return FAIL
    # ================================================================
    
    if A == D:
        logger.debug(
            f"[{_number_recursions}]: Lines 19-20 - FAILURE (Terminal case)\n"
            f"\t A = D = {sorted(D)}\n"
            f"\t Cannot make progress (ancestral closure equals district)"
        )
        raise Unidentifiable(
            f"Cannot identify causal effect: ancestral closure equals district "
            f"(A = D = {sorted(A)}). This means the algorithm cannot shrink the "
            f"district further and cannot identify the target variables C = {sorted(C)}."
        )
    
    # ================================================================
    # Lines 21-26: Recursive case (C ⊂ A ⊂ D)
    # ================================================================
    
    # At this point, we know A ≠ C and A ≠ D, so this is C ⊂ A ⊂ D
    logger.debug(
        f"[{_number_recursions}]: Lines 21-26 - Recursive case: C ⊂ A ⊂ D\n"
        f"\t C: {sorted(C)}\n"
        f"\t A: {sorted(A)}\n"
        f"\t D: {sorted(D)}"
    )
    
    # Get structure of subgraph G[A]
    subgraph_A = graph.subgraph(A)
    sccs = get_strongly_connected_components(subgraph_A)
    CdG_A_C = get_consolidated_district(subgraph_A, C)
    
    logger.debug(
        f"[{_number_recursions}]: Line 22 - Analyzing G[A] structure\n"
        f"\t Total SCCs in G[A]: {len(sccs)}\n"
        f"\t Cd^G[A](C): {sorted(CdG_A_C)}"
    )
    
    # Line 22: for S ∈ S(G[A]) s.t. S ⊆ Cd^G[A](C)
    relevant_sccs = [S for S in sccs if S.issubset(CdG_A_C)]
    
    if not relevant_sccs:
        raise Unidentifiable(
            f"No SCCs found in consolidated district Cd^G[A](C) = {sorted(CdG_A_C)}"
        )
    
    logger.debug(f"[{_number_recursions}]: Processing {len(relevant_sccs)} relevant SCC(s)")
    
    # Get apt-order for computing predecessors (Line 23)
    apt_order_A = get_apt_order(subgraph_A)
    
    RA = {}
    J = set()  # Assuming no background interventions (J = ∅)
    
    for S in relevant_sccs:
        # Line 23: R_A[S] ← P(S | Pred^G_<(S) ∩ A, do(J ∪ V \ A))
        
        # Find position of S in apt-order
        s_positions = [apt_order_A.index(v) for v in S if v in apt_order_A]
        
        if not s_positions:
            raise ValueError(f"SCC {sorted(S)} not found in apt-order")
        
        min_position = min(s_positions)
        
        # Pred^G_<(S) ∩ A = predecessors of S in apt-order, within A
        conditioning_set = {apt_order_A[i] for i in range(min_position) if apt_order_A[i] in A}
        
        # Intervention set: J ∪ V \ A
        intervention_set = J | (V - A)
        
        logger.debug(
            f"[{_number_recursions}]: Line 23 - Computing R_A[{sorted(S)}]\n"
            f"\t Outcomes: {sorted(S)}\n"
            f"\t Conditions (Pred^G_<(S) ∩ A): {sorted(conditioning_set)}\n"
            f"\t Interventions (J ∪ V \\ A): {sorted(intervention_set)}"
        )
        
        # Build identification query for main ID algorithm
        identification_obj = Identification.from_parts(
            outcomes=S,
            treatments=intervention_set,
            conditions=conditioning_set,
            graph=graph,  # Use full graph, not subgraph
            estimand=P(graph.nodes())  # Observational distribution over all nodes
        )
        
        # Call main ID algorithm
        RA[S] = identify(identification_obj)
        
        logger.debug(f"[{_number_recursions}]: R_A[{sorted(S)}] computed")
    
    # Line 25
    logger.debug(
        f"[{_number_recursions}]: Line 25 - Computing product over {len(RA)} SCC(s)"
    )
    
    # Sort SCCs for deterministic ordering
    sorted_sccs = sorted(relevant_sccs, key=lambda s: min(s) if s else Variable(""))
    Q_Cd = Product.safe([RA[S] for S in sorted_sccs])
    
    logger.debug(
        f"[{_number_recursions}]: Product computed over SCCs: "
        f"{[sorted(S) for S in sorted_sccs]}"
    )
    
    # Line 26: Recursive call with the new consolidated district
    logger.debug(
        f"[{_number_recursions}]: Line 26 - Recursive IDCD call\n"
        f"\t C: {sorted(C)}\n"
        f"\t New D = Cd^G[A](C): {sorted(CdG_A_C)}"
    )
    
    return idcd(
        graph=graph,
        C=C,
        D=CdG_A_C,
        Q_D=Q_Cd,
        _number_recursions=_number_recursions + 1
    )