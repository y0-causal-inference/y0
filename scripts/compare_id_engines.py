#!/usr/bin/env python
"""Compare handwritten and generated ID engines on a small benchmark set."""

from __future__ import annotations

from dataclasses import dataclass

from y0.algorithm.identify import Identification, Unidentifiable
from y0.algorithm.identify.id_dispatch import identify_with_engine
from y0.dsl import P, Variable
from y0.graph import NxMixedGraph
from y0.mutate.canonicalize_expr import canonical_expr_equal


@dataclass(frozen=True)
class Case:
    """A small benchmark case for engine comparison."""

    case_id: str
    graph: NxMixedGraph
    query: object


def _cases() -> list[Case]:
    x = Variable("X")
    y = Variable("Y")
    z = Variable("Z")
    return [
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


def main() -> None:
    """Run comparison and print a plain-text summary."""
    mismatches = 0
    for case in _cases():
        identification = Identification.from_expression(graph=case.graph, query=case.query)
        handwritten_error: Exception | None = None
        generated_error: Exception | None = None
        handwritten_expr = None
        generated_expr = None

        try:
            handwritten_expr = identify_with_engine(identification, engine="handwritten")
        except Exception as error:  # noqa: BLE001
            handwritten_error = error

        try:
            generated_expr = identify_with_engine(identification, engine="generated")
        except Exception as error:  # noqa: BLE001
            generated_error = error

        if handwritten_error or generated_error:
            same_error_class = (
                handwritten_error is not None
                and generated_error is not None
                and type(handwritten_error) is type(generated_error)
            )
            if same_error_class:
                print(f"[OK ] {case.case_id}: both raised {type(handwritten_error).__name__}")
            else:
                mismatches += 1
                print(
                    f"[ERR] {case.case_id}: handwritten={type(handwritten_error).__name__ if handwritten_error else 'none'} "
                    f"generated={type(generated_error).__name__ if generated_error else 'none'}"
                )
            continue

        if handwritten_expr is None or generated_expr is None:
            mismatches += 1
            print(f"[ERR] {case.case_id}: missing expression output")
            continue

        if canonical_expr_equal(handwritten_expr, generated_expr):
            print(f"[OK ] {case.case_id}: canonical expressions match")
        else:
            mismatches += 1
            print(
                f"[ERR] {case.case_id}: expression mismatch "
                f"handwritten={handwritten_expr.to_text()} generated={generated_expr.to_text()}"
            )

    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
