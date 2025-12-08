"""
IDCD Algorith Implementation (Algorith 1, Lines 13-28) from Forré & Mooij (2019).

This module implements the IDCD (IDentification within Consolidated Districts) algorithm,
which is a helper function for the main cyclic ID algorithm. IDCD handles identification
of causal effects within consolidated districts of directed mixed graphs that may contain
cycles and latent confounders.
"""

# adding imports from y0

import logging

# importing utility functions
from ..ioscm.utils import (
    get_apt_order,
    get_consolidated_district,
    get_strongly_connected_components
)

from ..identify.utils import Unidentifiable
from ..identify.utils import Identification
from ..identify.id_std import identify
from ...dsl import Expression, P, Variable
from ...graph import NxMixedGraph


__all__ = ["idcd"]

logger = logging.getLogger(__name__)

# defining the idcd function signature
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
    :raises Unidentifiable: If causal effect cannot be identified
    """
    
    # line 14 - precondition check
    # require: C ⊆ D ⊆ V and CD(G_D) = {D}
    
    V = set(graph.nodes())
    
    # precondition 1: C must not be empty
    if not C:
        raise ValueError("Target set C must not be empty.")
    
    # precondition 2: D cannot be empty
    if not D:
        raise ValueError("District D must not be empty.")
    
    # precondition 3: C must be a subset of D - target variables have to be within district
    if not C.issubset(D):
        raise ValueError(
            f"C must be a subset of D."
            f"C = {sorted(C)}, D = {sorted(D)}, C \\ D = {sorted(C - D)}" 
        )   
        
    # precondition 4: D must be a subset of V - district has to be within graph nodes
    
    if not D.issubset(V):
        raise ValueError(
            f"D must be a subset of V."
            f"D = {sorted(D)}, V = {sorted(V)}, D \\ V = {sorted(D - V)}"
        )
     
    logger.debug(
        f"[{_number_recursions}]: Calling IDCD\n"
        f"\t C (target): {sorted(C)}\n"
        f"\t D (district): {sorted(D)}\n"
        f"\t |V| = {len(V)}"
    )

    # line 15 - computing the ancestral closure of C within the subgraph G[D]
    
    logger.debug(f"[{_number_recursions}]: line 15 IDCD: compute An(C)_G[D]")
    
    # create subgraph G[D]
    subgraph_D = graph.subgraph(D)
    
    # Get ancestors of all variables in C within G[D], intersected with D
    A = set().union(*(subgraph_D.ancestors_inclusive(c) for c in C)) & D   
        
    
    # line 16 - marginalize Q[D] over variables in D \ A
    
    logger.debug(f"[{_number_recursions}]: line 16 IDCD: marginalize Q[D] over D \\ A")
    
    marginalize_out = D - A
    
    if marginalize_out:
        # getting the apt order in the context of the district subgraph
        apt_order = get_apt_order(subgraph_D, list(marginalize_out))
        logger.debug(f"[{_number_recursions}]: Marginalizing out variables: (APT order) {apt_order}")
        Q_A = Q_D.marginalize(apt_order)
    else:
        Q_A = Q_D
    
    logger.debug(f"[{_number_recursions}]: Q[A] computed")
    
    
    # line 17 - checking if A == C: return Q[A]
    if A == C:
        logger.debug(f"[{_number_recursions}]: line 17 IDCD: A == C, returning Q[A]")
        # line 18 - return Q[A]
        return Q_A
    
    # line 19 - if A == D: return FAIL/Unidentifiable
    if A == D:
        logger.debug(f"[{_number_recursions}]: line 19 IDCD: A == D, raising Unidentifiable")
        # line 20 - return FAIL
        raise Unidentifiable("Causal effect is unidentifiable within district D.")
    
    # line 21
    elif C < A < D:
        logger.debug(f"[{_number_recursions}]: line 21 IDCD: C ⊂ A ⊂ D, decomposing further"
    )
    
    # get consolidated district containing C in G[A]
    subgraph_A = graph.subgraph(A)
    CdG_A_C = get_consolidated_district(subgraph_A, C)
    
    # compute SCCS as sets of variables
    sccs = get_strongly_connected_components(subgraph_A)
    
    # apt order SCCs 
    apt_ordered_sccs = get_apt_order(subgraph_A, sccs)
    
    # filter SCCs to those within consolidated district containing C
    relevant_sccs = [S for S in apt_ordered_sccs if S.issubset(CdG_A_C)]
    
    # store the conditional interventional distributions
    RA = {}
    
    V = set(graph.nodes())
    J = set() # replace or pass J 
    
    for idx, S in enumerate(relevant_sccs):
        # earlier SCCs in apt-order 
        predecessors_in_order = set().union(*relevant_sccs[:idx])
        # conditioning set: restricted to actual variables in A
        conditioning_set = predecessors_in_order & A
        
        # intervention: original interventions + variables outside A
        intervention_set = J | (V - A)
        
        # build the identification query for standard ID alg. call
        identification_obj = Identification.from_parts(
            outcomes=S,
            treatments=intervention_set,
            conditions=conditioning_set,
            graph=subgraph_A,
            estimand=Q_A,
        )
        
        RA[S] = identify(identification_obj)
        
        logger.debug(
            f"[{_number_recursions}]: SCC {sorted(S)}: "
            f"Condition on {sorted(conditioning_set)}; "
            f"do on {sorted(intervention_set)} -> {RA[S]}"
        )
        
        

        
