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
    
    """
    