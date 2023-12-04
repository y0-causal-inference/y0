"""High-level API for identification algorithms."""

from typing import Union

from .id_c import idc
from .id_std import identify
from .utils import Identification, Query, Unidentifiable
from ...dsl import Expression, Variable
from ...graph import NxMixedGraph, _ensure_set

__all__ = [
    "identify_outcomes",
]


def identify_outcomes(
    graph: NxMixedGraph,
    treatments: Union[Variable, set[Variable]],
    outcomes: Union[Variable, set[Variable]],
    conditions: Union[None, Variable, set[Variable]] = None,
) -> Expression | None:
    """Calculate the estimand for the treatment(s)m outcome(s), and optional condition(s).

    :param graph: An acyclic directed mixed graph
    :param treatments: The node or nodes that are treated
    :param outcomes: The node or nodes that are outcomes
    :param conditions: Optional condition or condition nodes.
        If given, uses the IDC algorithm via :func:`y0.algorithm.identify.idc`.
        Otherwise, uses the ID algorithm via :func:`y0.algorithm.identify.identify`.
    :returns:
        An expression representing the estimand if the query is identifiable.
        If the query is not identifiable, returns none.
    """
    treatments = _ensure_set(treatments)
    outcomes = _ensure_set(outcomes)

    query = Query(treatments=treatments, outcomes=outcomes, conditions=conditions)
    identification = Identification(graph=graph, query=query)

    try:
        if conditions is None:
            rv = identify(identification)
        else:
            rv = idc(identification)
    except Unidentifiable:
        return None
    return rv
