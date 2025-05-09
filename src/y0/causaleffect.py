"""Interface to the R causaleffect package via :mod:`rpy2`."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .graph import CausalEffectGraph, NxMixedGraph
from .r_utils import uses_r
from .struct import VermaConstraint

__all__ = [
    "r_get_verma_constraints",
]

logger = logging.getLogger(__name__)


@uses_r
def r_get_verma_constraints(graph: NxMixedGraph | CausalEffectGraph) -> Sequence[VermaConstraint]:
    """Calculate the verma constraints on the graph using ``causaleffect``."""
    from rpy2 import robjects

    if isinstance(graph, NxMixedGraph):
        graph = graph.to_causaleffect()
    verma_constraints = robjects.r["verma.constraints"]
    return [VermaConstraint.from_element(row) for row in verma_constraints(graph)]
