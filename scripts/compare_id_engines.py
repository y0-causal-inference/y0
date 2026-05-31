#!/usr/bin/env python
"""Compare handwritten and generated ID engines on a small benchmark set."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from y0.algorithm.identify import Identification
from y0.algorithm.identify.id_dispatch import identify_with_engine
from y0.algorithm.identify.id_extracted_bridge import (
    identify_line1_from_extracted,
    supports_query_line1,
)
from y0.dsl import Distribution, P, Probability, Variable
from y0.graph import NxMixedGraph
from y0.mutate.canonicalize_expr import canonical_expr_equal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Case:
    """A small benchmark case for engine comparison."""

    case_id: str
    graph: NxMixedGraph
    query: Probability | Distribution


def _cases() -> list[Case]:
    """Return the benchmark case list used for engine comparison."""
    x = Variable("X")
    y = Variable("Y")
    z = Variable("Z")
    return [
        Case(
            case_id="id.line1.extracted",
            graph=NxMixedGraph.from_edges(directed=[(x, y)]),
            query=P(y),
        ),
        Case(
            case_id="id.simple.identifiable",
            graph=NxMixedGraph.from_edges(directed=[(x, y)]),
            query=P(y @ ~x),
        ),
        Case(
            case_id="id.frontdoor_like.identifiable",
            graph=NxMixedGraph.from_edges(directed=[(x, z), (z, y)]),
            query=P(y @ ~x),
        ),
        Case(
            case_id="id.simple.hedge_like",
            graph=NxMixedGraph.from_edges(directed=[(x, y)], undirected=[(x, y)]),
            query=P(y @ ~x),
        ),
    ]


def _detect_execution_route(identification: Identification) -> str:
    if not supports_query_line1(identification):
        return "fallback"
    try:
        identify_line1_from_extracted(identification)
    except Exception as error:
        return f"fallback({type(error).__name__})"
    return "extracted"


def _compare_case(case: Case) -> bool:
    identification = Identification.from_expression(graph=case.graph, query=case.query)
    execution_route = _detect_execution_route(identification)

    handwritten_error: Exception | None = None
    generated_error: Exception | None = None
    handwritten_expr = None
    generated_expr = None

    try:
        handwritten_expr = identify_with_engine(identification, engine="handwritten")
    except Exception as error:
        handwritten_error = error

    try:
        generated_expr = identify_with_engine(identification, engine="generated")
    except Exception as error:
        generated_error = error

    if handwritten_error or generated_error:
        same_error_class = (
            handwritten_error is not None
            and generated_error is not None
            and type(handwritten_error) is type(generated_error)
        )
        if same_error_class:
            logger.info(
                "[OK ] %s: both raised %s (route=%s)",
                case.case_id,
                type(handwritten_error).__name__,
                execution_route,
            )
            return True
        logger.error(
            "[ERR] %s: handwritten=%s generated=%s (route=%s)",
            case.case_id,
            type(handwritten_error).__name__ if handwritten_error else "none",
            type(generated_error).__name__ if generated_error else "none",
            execution_route,
        )
        return False

    if handwritten_expr is None or generated_expr is None:
        logger.error("[ERR] %s: missing expression output", case.case_id)
        return False

    if canonical_expr_equal(handwritten_expr, generated_expr):
        logger.info(
            "[OK ] %s: canonical expressions match (route=%s)", case.case_id, execution_route
        )
        return True

    logger.error(
        "[ERR] %s: expression mismatch handwritten=%s generated=%s (route=%s)",
        case.case_id,
        handwritten_expr.to_text(),
        generated_expr.to_text(),
        execution_route,
    )
    return False


def main() -> None:
    """Run comparison and print a plain-text summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mismatches = sum(0 if _compare_case(case) else 1 for case in _cases())

    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
