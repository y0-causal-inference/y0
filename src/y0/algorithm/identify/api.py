"""High-level API for identification algorithms."""

from collections.abc import Iterable, Sequence
from typing import Literal, overload

from .id_c import idc
from .id_std import identify
from .utils import Identification, Unidentifiable
from ...dsl import Expression, Variable
from ...graph import NxMixedGraph

__all__ = [
    "identify_outcomes",
]


# docstr-coverage:excused `overload`
@overload
def identify_outcomes(
    graph: NxMixedGraph,
    treatments: Variable | Iterable[Variable],
    outcomes: Variable | Iterable[Variable],
    conditions: None | Variable | Iterable[Variable] = ...,
    *,
    strict: Literal[True] = ...,
    ordering: Sequence[Variable] | None = ...,
) -> Expression: ...


# docstr-coverage:excused `overload`
@overload
def identify_outcomes(
    graph: NxMixedGraph,
    treatments: Variable | Iterable[Variable],
    outcomes: Variable | Iterable[Variable],
    conditions: None | Variable | Iterable[Variable] = ...,
    *,
    strict: Literal[False] = ...,
    ordering: Sequence[Variable] | None = ...,
) -> Expression | None: ...


def identify_outcomes(
    graph: NxMixedGraph,
    treatments: Variable | Iterable[Variable],
    outcomes: Variable | Iterable[Variable],
    conditions: None | Variable | Iterable[Variable] = None,
    *,
    strict: bool = False,
    ordering: Sequence[Variable] | None = None,
) -> Expression | None:
    """Calculate the estimand for the treatment(s)m outcome(s), and optional condition(s).

    :param graph: An acyclic directed mixed graph
    :param treatments: The node or nodes that are treated
    :param outcomes: The node or nodes that are outcomes
    :param conditions: Optional condition or condition nodes. If given, uses the IDC
        algorithm via :func:`y0.algorithm.identify.idc`. Otherwise, uses the ID
        algorithm via :func:`y0.algorithm.identify.identify`.
    :param strict: If true, raises an error identification fails rather than returning
        None
    :param ordering: A topological ordering of the variables. If not passed, is
        calculated from the directed component of the mixed graph.

    :returns: An expression representing the estimand if the query is identifiable. If
        the query is not identifiable, returns none.
    """
    identification = Identification.from_parts(
        graph=graph, treatments=treatments, outcomes=outcomes, conditions=conditions
    )

    try:
        if not identification.conditions:
            rv = identify(identification, ordering=ordering)
        else:
            rv = idc(identification, ordering=ordering)
    except Unidentifiable:
        if strict:
            raise
        return None
    return rv
