"""Minimal oracle runner checks for generated ID fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from y0.algorithm.identify.id_oracle_types import (
    assert_case,
    iter_cases,
    load_fixture,
    run_case,
)
from y0.algorithm.identify.utils import Unidentifiable
from y0.dsl import Expression


def _fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "generated" / "dafny_oracle" / "id_cases.v1.json"


def test_fixture_loads() -> None:
    """Fixture file should deserialize and contain exactly two seed cases."""
    fixture = load_fixture(_fixture_path())
    if fixture["schema_version"] != 1:
        pytest.fail("expected schema_version=1")
    if len(fixture["cases"]) != 2:
        pytest.fail("expected exactly two seed cases")


def test_case_iteration_order_stable() -> None:
    """Case iteration must be deterministic across repeated calls."""
    fixture = load_fixture(_fixture_path())
    first = [case["case_id"] for case in iter_cases(fixture, module="identification")]
    second = [case["case_id"] for case in iter_cases(fixture, module="identification")]
    if first != second:
        pytest.fail("expected deterministic case iteration order")


def test_identifiable_case_executes() -> None:
    """The identifiable seed case should execute to a valid expression."""
    fixture = load_fixture(_fixture_path())
    case = next(case for case in iter_cases(fixture) if case["expectation"]["kind"] == "ir")
    result = run_case(case)
    if not isinstance(result, Expression):
        pytest.fail("expected an Expression result for identifiable case")
    assert_case(case, result)


def test_fail_case_raises_unidentifiable() -> None:
    """The hedge seed case should raise Unidentifiable."""
    fixture = load_fixture(_fixture_path())
    case = next(case for case in iter_cases(fixture) if case["expectation"]["kind"] == "exception")
    with pytest.raises(Unidentifiable):
        run_case(case)
