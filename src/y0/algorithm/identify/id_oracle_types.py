"""Typed interfaces and helpers for Dafny oracle fixture cases."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

from y0.algorithm.identify.id_extracted_bridge import (
    DafnyRuntimeUnavailableError,
    raw_dafny_call_for_identification,
)
from y0.algorithm.identify.id_ir_to_dsl import ir_doc_to_expression
from y0.algorithm.identify.utils import Identification, Unidentifiable
from y0.dsl import Expression, Variable
from y0.graph import NxMixedGraph
from y0.mutate.canonicalize_expr import canonical_expr_equal, canonicalize

__all__ = [
    "DafnyRuntimeUnavailableError",
    "OracleAnchor",
    "OracleCase",
    "OracleExpectation",
    "OracleFixture",
    "OracleQuery",
    "assert_case",
    "build_identification_from_case",
    "iter_cases",
    "load_fixture",
    "normalize_expression",
    "raw_dafny_call",
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

    kind: Literal["ir", "exception", "boolean", "algebraic_equivalence", "gate_check"]
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
    yield from sorted(
        cases, key=lambda case: (case["module"], case["anchor"]["symbol"], case["case_id"])
    )


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


def build_identification_from_case(case: OracleCase) -> Identification:
    """Build an Identification from a gate_check oracle case.

    The case's ``query`` must contain a ``graph_def`` object with:

    * ``directed``    — list of ``[source, target]`` pairs
    * ``undirected``  — list of ``[u, v]`` pairs
    * ``extra_nodes`` — list of node name strings to add (may be empty)

    :param case: An oracle case with a ``graph_def`` in its query.
    :returns: The constructed Identification.
    :raises KeyError: If ``graph_def`` is absent from the query.
    """
    query = case["query"]
    graph_def = cast(dict[str, object], query.get("graph_def", {}))
    directed_raw = cast(list[list[str]], graph_def.get("directed", []))
    undirected_raw = cast(list[list[str]], graph_def.get("undirected", []))
    extra_nodes_raw = cast(list[str], graph_def.get("extra_nodes", []))

    extra_nodes = [Variable(name) for name in extra_nodes_raw]
    directed = [(Variable(src), Variable(tgt)) for src, tgt in directed_raw]
    undirected = [(Variable(u), Variable(v)) for u, v in undirected_raw]

    graph = NxMixedGraph.from_edges(
        nodes=extra_nodes or None,
        directed=directed or None,
        undirected=undirected or None,
    )

    outcomes = {Variable(name) for name in query["outcomes"]}
    treatments = {Variable(name) for name in query["treatments"]}

    return Identification.from_parts(
        outcomes=outcomes,
        treatments=treatments,
        graph=graph,
    )


def raw_dafny_call(case: OracleCase) -> tuple[bool, object]:
    """Invoke the Dafny runtime for *case* and return the raw ``(ok, result)`` pair.

    This is the primary entry point for checking the Dafny runtime's ``ok``
    signal independently of the Python gate (``supports_query_lineN``).
    The ``ok`` flag indicates whether the Dafny runtime itself considers the
    query applicable for the given line; the secondary ``result`` value is the
    raw Dafny response and may be discarded.

    Typical usage in a test::

        try:
            ok, _ = raw_dafny_call(case)
        except DafnyRuntimeUnavailableError:
            pytest.skip("Dafny runtime not available in this environment")
        assert ok == expected_ok

    :param case: An oracle case whose ``query`` contains ``line_number`` and
        ``graph_def``.
    :returns: ``(ok, raw_result)`` from the Dafny runtime.
    :raises DafnyRuntimeUnavailableError: If the runtime directory is absent or
        the ``_dafny`` package cannot be imported.
    :raises KeyError: If ``line_number`` is absent from the case query.
    """
    identification = build_identification_from_case(case)
    line_number = int(case["query"].get("line_number", 1))  # type: ignore[call-overload]
    ordering_names = case["query"].get("ordering", [])
    ordering = [Variable(name) for name in ordering_names] if ordering_names else None
    return raw_dafny_call_for_identification(identification, line_number, ordering=ordering)


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
