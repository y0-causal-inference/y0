"""ID correspondence tests driven by generated Dafny oracle fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from y0.algorithm.identify.id_extracted_bridge import (
    supports_query_line1,
    supports_query_line2,
    supports_query_line3,
    supports_query_line4,
    supports_query_line5,
    supports_query_line6,
    supports_query_line7,
)
from y0.algorithm.identify.id_oracle_types import (
    assert_case,
    build_identification_from_case,
    iter_cases,
    load_fixture,
    run_case,
)
from y0.algorithm.identify.utils import Unidentifiable

_GATE_BY_LINE: dict[int, object] = {
    1: supports_query_line1,
    2: supports_query_line2,
    3: supports_query_line3,
    4: supports_query_line4,
    5: supports_query_line5,
    6: supports_query_line6,
    7: supports_query_line7,
}


def _fixture_path() -> Path:
    return (
        Path(__file__).resolve().parent / "data" / "generated" / "dafny_oracle" / "id_cases.v1.json"
    )


@pytest.mark.parametrize(
    "case_id",
    [
        "id.line1.extracted.identifiable",
        # TODO: Line 2 cases require IR schema extension for "recursive" node type
        # "id.line2.chain_with_isolated.reduction",
        # "id.line2.multipath_with_irrelevant.reduction",
        "id.line4.frontdoor_small.identifiable",
        "id.line5.figure1a.hedge",
        "id.full.line3.recursive_like",
        "id.full.line7.recursive_like",
    ],
)
def test_dafny_id_case(case_id: str) -> None:
    """Generated ID oracle cases should match expected behavior."""
    fixture = load_fixture(_fixture_path())
    case = next(
        case for case in iter_cases(fixture, module="identification") if case["case_id"] == case_id
    )

    if case["expectation"]["kind"] == "exception":
        with pytest.raises(Unidentifiable):
            run_case(case)
        return

    actual = run_case(case)
    assert_case(case, actual)


@pytest.mark.parametrize(
    "case_id",
    [
        "id.line1.boundary.nonempty_treatments_no_path",
        "id.line1.boundary.treatments_with_path",
    ],
)
def test_dafny_gate_check_case(case_id: str) -> None:
    """Boundary oracle cases verify Python gate decisions against expected ok values.

    These tests check that ``supports_query_lineN`` returns the expected boolean
    for boundary inputs — cases near the edge of each line's applicability
    condition.  No Dafny runtime is required; the gate functions are pure Python.
    """
    fixture = load_fixture(_fixture_path())
    case = next(
        c for c in iter_cases(fixture, module="identification") if c["case_id"] == case_id
    )
    assert case["expectation"]["kind"] == "gate_check", (
        f"expected gate_check kind for {case_id}, got {case['expectation']['kind']!r}"
    )

    expectation_value = cast(dict[str, object], case["expectation"]["value"])
    ok_expected = bool(expectation_value["ok_expected"])
    line_number = int(cast(int, case["query"].get("line_number", 1)))  # type: ignore[arg-type]

    gate_fn = _GATE_BY_LINE[line_number]
    identification = build_identification_from_case(case)

    ok_actual = gate_fn(identification)  # type: ignore[operator]
    assert ok_actual == ok_expected, (
        f"Gate for Line {line_number} returned {ok_actual!r} "
        f"but expected {ok_expected!r} for case {case_id!r}."
    )
