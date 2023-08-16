# -*- coding: utf-8 -*-

"""Interface to the R causaleffect package via :mod:`rpy2`."""

from __future__ import annotations

import logging
from typing import Sequence, Union

from rpy2 import robjects

from .graph import CausalEffectGraph, NxMixedGraph
from .r_utils import uses_r
from .struct import VermaConstraint

logger = logging.getLogger(__name__)


@uses_r
def r_get_verma_constraints(
    graph: Union[NxMixedGraph, CausalEffectGraph]
) -> Sequence[VermaConstraint]:
    """Calculate the verma constraints on the graph using ``causaleffect``."""
    if isinstance(graph, NxMixedGraph):
        graph = graph.to_causaleffect()
    verma_constraints = robjects.r["verma.constraints"]
    return [VermaConstraint.from_element(row) for row in verma_constraints(graph)]
