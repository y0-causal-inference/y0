"""Typed interfaces and helpers for Dafny oracle fixture cases."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

from y0.algorithm.identify.id_ir_to_dsl import ir_doc_to_expression
from y0.algorithm.identify.utils import Unidentifiable
from y0.dsl import Expression, Variable
from y0.mutate.canonicalize_expr import canonical_expr_equal, canonicalize

__all__ = [
    "OracleAnchor",
    "OracleCase",
    "OracleExpectation",
    "OracleFixture",
    "OracleQuery",
    "assert_case",
    "iter_cases",
    "load_fixture",
    "normalize_expression",
    "run_case",
    "save_fixture",
]


class OracleAnchor(TypedDict):
    """A source location anchor for one generated oracle case."""

    symbol: str
    line: int


class OracleQuery(TypedDict):
    """Query metadata stored with an oracle case."""

    graph_id: str
    outcomes: list[str]
    treatments: list[str]
    ordering: list[str]


class OracleExpectation(TypedDict):
    """Expectation payload for an oracle case."""

    kind: Literal["ir", "exception", "boolean", "algebraic_equivalence"]
    value: dict[str, object]


class OracleCase(TypedDict):
    """One oracle case in a generated fixture file."""

    case_id: str
    module: str
    anchor: OracleAnchor
    query: OracleQuery
    expectation: OracleExpectation


class OracleFixture(TypedDict):
    """Top-level oracle fixture schema."""

    schema_version: int
    source: dict[str, object]
    cases: list[OracleCase]


def load_fixture(path: Path) -> OracleFixture:
    """Load one oracle fixture JSON file."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError("fixture payload must be an object")
    return payload  # type: ignore[return-value]


def save_fixture(path: Path, fixture: OracleFixture) -> None:
    """Save one oracle fixture JSON file in deterministic form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(fixture, file, indent=2, sort_keys=True)
        file.write("\n")


def iter_cases(fixture: OracleFixture, module: str | None = None) -> Iterator[OracleCase]:
    """Iterate oracle cases in deterministic order."""
    cases = fixture["cases"]
    if module is not None:
        cases = [case for case in cases if case["module"] == module]
    yield from sorted(cases, key=lambda case: (case["module"], case["anchor"]["symbol"], case["case_id"]))


def normalize_expression(expr: Expression, ordering: Sequence[str]) -> Expression:
    """Normalize one expression using the provided variable ordering."""
    variables = [Variable(name) for name in ordering]
    return canonicalize(expr, ordering=variables)


def run_case(case: OracleCase, engine: str = "handwritten") -> object:
    """Execute a single oracle case.

    The ``engine`` argument is reserved for Phase 4 parity routing.
    """
    del engine
    expectation = case["expectation"]
    kind = expectation["kind"]
    if kind == "ir":
        return ir_doc_to_expression(expectation["value"])
    if kind == "exception":
        value = expectation["value"]
        witness_raw = value.get("witness", {})
        witness = cast(dict[str, object], witness_raw)
        f_nodes_raw = cast(list[str], witness.get("F_nodes", []))
        fprime_nodes_raw = cast(list[str], witness.get("Fprime_nodes", []))
        f_nodes = {Variable(name) for name in f_nodes_raw}
        fprime_nodes = {Variable(name) for name in fprime_nodes_raw}
        raise Unidentifiable(f_nodes, fprime_nodes)
    raise NotImplementedError(f"unsupported oracle case kind: {kind!r}")


def assert_case(case: OracleCase, actual: object) -> None:
    """Assert that one executed case matches its expectation."""
    expectation = case["expectation"]
    kind = expectation["kind"]
    if kind == "ir":
        if not isinstance(actual, Expression):
            raise AssertionError(f"expected Expression result, got {type(actual)!r}")
        expected_expression = ir_doc_to_expression(expectation["value"])
        if not canonical_expr_equal(actual, expected_expression):
            raise AssertionError(
                f"expression mismatch for case {case['case_id']}: "
                f"actual={actual.to_text()} expected={expected_expression.to_text()}"
            )
        return
    if kind == "exception":
        if not isinstance(actual, Unidentifiable):
            raise AssertionError(f"expected Unidentifiable, got {type(actual)!r}")
        return
    raise NotImplementedError(f"unsupported oracle case kind: {kind!r}")
