"""Tests for ID intermediate representation schema and normalization."""

import json
from pathlib import Path

import pytest

from y0.algorithm.identify.id_ir import (
    IRValidationError,
    canonicalize_ir_document,
    validate_ir_document,
)
from y0.algorithm.identify.id_ir_to_dsl import ir_doc_to_expression
from y0.algorithm.identify.utils import Unidentifiable
from y0.dsl import Expression
from y0.mutate.canonicalize_expr import canonicalize


def _fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "id_ir_cases.json"


def test_invalid_ir_rejected() -> None:
    """IR should reject overlapping conditioning and intervention sets."""
    bad = {
        "version": "1",
        "engine": "id",
        "query": {
            "graph_id": "bad",
            "outcomes": ["Y"],
            "treatments": ["X"],
            "ordering": ["X", "Y"],
        },
        "result": {
            "tag": "prob",
            "vars": ["Y"],
            "given": ["X"],
            "intervened": ["X"],
        },
    }
    with pytest.raises(IRValidationError):
        validate_ir_document(bad)


def test_valid_ir_cases_round_trip() -> None:
    """Seed fixtures should validate and round-trip deterministically."""
    payload = json.loads(_fixture_path().read_text())
    cases = payload["cases"]
    if not cases:
        pytest.fail("expected non-empty IR fixture cases")

    canonical_docs = [canonicalize_ir_document(case) for case in cases]

    reparsed = [json.loads(json.dumps(doc, sort_keys=True)) for doc in canonical_docs]
    if reparsed != canonical_docs:
        pytest.fail("canonical IR documents did not round-trip deterministically")


def test_canonical_ordering_stable() -> None:
    """Canonicalization should sort product factors and merge adjacent sums."""
    doc = {
        "version": "1",
        "engine": "id",
        "query": {
            "graph_id": "canon",
            "outcomes": ["Y"],
            "treatments": ["X"],
            "ordering": ["X", "Z", "Y"],
        },
        "result": {
            "tag": "sum",
            "over": ["Z"],
            "body": {
                "tag": "sum",
                "over": ["W"],
                "body": {
                    "tag": "product",
                    "factors": [
                        {"tag": "prob", "vars": ["Y"], "given": ["X", "Z"], "intervened": []},
                        {"tag": "prob", "vars": ["Z"], "given": ["X"], "intervened": []},
                    ],
                },
            },
        },
    }

    canonical = canonicalize_ir_document(doc)
    reversed_doc = json.loads(json.dumps(doc))
    reversed_doc["result"]["body"]["body"]["factors"].reverse()
    canonical_reversed = canonicalize_ir_document(reversed_doc)

    if canonical["result"]["tag"] != "sum":
        pytest.fail("expected canonical result tag to be 'sum'")
    if canonical["result"]["over"] != ["W", "Z"]:
        pytest.fail("expected merged sum ordering to be ['W', 'Z']")
    if canonical_reversed != canonical:
        pytest.fail("expected canonicalization to be stable under factor order changes")


def test_ir_to_dsl_translation_for_identifiable_cases() -> None:
    """Each identifiable IR fixture should translate to a canonicalizable DSL expression."""
    payload = json.loads(_fixture_path().read_text())
    cases = payload["cases"]
    identifiable_cases = [case for case in cases if case["result"]["tag"] != "fail"]

    if not identifiable_cases:
        pytest.fail("expected at least one identifiable IR fixture")

    for case in identifiable_cases:
        expression = ir_doc_to_expression(case)
        if not isinstance(expression, Expression):
            pytest.fail(f"expected Expression output for case {case['query']['graph_id']}")

        ordering = case["query"]["ordering"]
        canonicalized_once = canonicalize(expression, ordering=ordering)
        canonicalized_twice = canonicalize(canonicalized_once, ordering=ordering)
        if canonicalized_once != canonicalized_twice:
            pytest.fail(f"expected stable canonicalization for case {case['query']['graph_id']}")


def test_ir_fail_case_raises_unidentifiable() -> None:
    """A hedge fail node should map to Unidentifiable during translation."""
    payload = json.loads(_fixture_path().read_text())
    fail_case = next((case for case in payload["cases"] if case["result"]["tag"] == "fail"), None)
    if fail_case is None:
        pytest.fail("expected at least one fail fixture case")

    with pytest.raises(Unidentifiable):
        ir_doc_to_expression(fail_case)
