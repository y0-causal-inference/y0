"""ID correspondence tests driven by generated Dafny oracle fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from y0.algorithm.identify.id_oracle_types import assert_case, iter_cases, load_fixture, run_case
from y0.algorithm.identify.utils import Unidentifiable


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
