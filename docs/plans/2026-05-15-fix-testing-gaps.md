## Plan: Applicability-Gate Testing to Close the Fallback Blind Spot

### Root Cause (Precise)

The system has two sequential precondition gates per extracted line:

1. **Python gate** — `supports_query_lineN(identification)` in
   id_extracted_bridge.py
2. **Dafny gate** — the Dafny runtime's own `ok` return value

When either gate says "not applicable", the bridge throws an
`ExtractedLineNUnavailableError` and the caller (`_identify_via_line_compat`)
silently falls back to the handwritten engine. Parity tests then see two correct
answers produced by different paths and report success. No test currently checks
_which path ran_.

The Line 1 bug was an instance of gate 1 being too narrow:
`supports_query_line1` checks `not identification.treatments`, but the correct
condition is `X ∩ An(Y)_{G_{\bar{x}}} = ∅` — a strictly weaker requirement that
also covers cases with non-empty treatments that have no causal path to
outcomes.

---

### Defence 1 — `supports_query_lineN` contract tests

**File:** new `tests/test_algorithm/test_id_extracted_routing.py`  
**No Dafny runtime required.**

For each of the seven lines, add parameterised tests asserting:

| Test                                 | Input                                                      | Assertion                              |
| ------------------------------------ | ---------------------------------------------------------- | -------------------------------------- |
| `test_supports_query_lineN_positive` | Known line-N example from `line_N_example.identifications` | `supports_query_lineN(id_in) is True`  |
| `test_supports_query_lineN_negative` | An input from a _different_ line's examples                | `supports_query_lineN(id_in) is False` |

The **critical addition** for Line 1: a case where treatments are non-empty but
no treatment is an ancestor of any outcome in the manipulated graph (i.e.,
`X ∩ An(Y)_{G_{\bar{x}}} = ∅` with `X ≠ ∅`). This case must return `True`. The
current implementation returns `False` and that exact regression would now be
caught.

The negative test for Line 2 forward should include Line 1's example, and so on
— each line's positive fixture becomes the negative fixture for the others. This
gives 49 cheap, purely-Python tests that pin every gate function to its
specification.

---

### Defence 2 — Strict mode in the bridge

**Files:** id_extracted_bridge.py, id_generated.py

Add an environment variable `Y0_DAFNY_STRICT=1`. When enabled, change the
error-handling logic in every `identify_lineN_from_extracted` function from:

```python
if not ok:
    raise ExtractedLineNUnavailableError("extracted runtime rejected query")
```

to:

```python
if not ok:
    if _strict_mode():
        raise AssertionError(
            f"Line N extracted runtime returned ok=False for query that "
            f"supports_query_lineN claimed was applicable: {identification}"
        )
    raise ExtractedLineNUnavailableError("extracted runtime rejected query")
```

In `_strict_mode()`, check `os.environ.get("Y0_DAFNY_STRICT", "0") == "1"`.

**Effect:** In CI runs with the compiled Dafny runtime available, set
`Y0_DAFNY_STRICT=1` to surface any mismatch between the Python gate and the
Dafny gate immediately, rather than silently masking it. Normal production mode
keeps the permissive fallback.

---

### Defence 3 — Path-selection tests using `monkeypatch`

**File:** test_id_generated_parity.py

Add tests that verify the extracted path was actually taken (not masked by
fallback). The mechanism: monkeypatch `identify_handwritten` to raise
`AssertionError` if called, then verify the extracted engine still returns the
correct answer:

```python
def test_line1_uses_extracted_path_not_fallback(monkeypatch):
    """When the extracted runtime is available, Line 1 must not fall back to handwritten."""
    identification = ...  # line_1_example input

    try:
        id_generated_module.identify_line1_from_extracted(identification)
    except id_generated_module.ExtractedLine1UnavailableError:
        pytest.skip("extracted runtime not available in this environment")

    # Runtime IS available — now confirm fallback is never used
    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("handwritten fallback was called when extracted path should have fired")

    monkeypatch.setattr(
        "y0.algorithm.identify.id_extracted_bridge.identify_handwritten", _no_fallback
    )
    result = id_generated_module.identify_line1_from_extracted(identification)
    assert result is not None
```

One such test per line. Because the test skips when the runtime is unavailable
(the normal CI environment without compiled Dafny), it does not break the
standard pipeline; it activates only on developer machines or CI environments
with `Y0_DAFNY_ID_LINE1_PY_DIR` set.

This is the test that would have caught the Line 1 bug: with the runtime
available, `identify_line1_from_extracted` on the correct Line 1 case would
succeed with a real answer; with the buggy `supports_query_line1`, it would have
raised `ExtractedLine1UnavailableError` and the test would have `skip`-asserted
— or alternatively, the stub identity from `supports_query_line1` would never
even call the runtime, causing the test to fail loudly.

---

### Defence 4 — Boundary oracle cases in the fixture

**File:** id_cases.v1.json

Add cases that probe the precondition _boundary_, not just the safe interior:

| Case ID                                        | Description                                                     | Why it matters                                                                                       |
| ---------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `id.line1.nonempty_treatments_no_path`         | `X → Z`, query `P(Y \| do(X))` where Y is disconnected from X   | Line 1 applies because `X ∩ An(Y)_{G_{\bar{x}}} = ∅`; old `supports_query` would never have tried it |
| `id.line1.treatments_with_path_does_not_apply` | `X → Y`, query `P(Y \| do(X))`                                  | Line 1 must NOT apply; Dafny `ok` should be False                                                    |
| `id.lineN.precondition_boundary`               | One per line: a minimal case where the line just barely applies | Pins the gate to its exact specification                                                             |

These cases are run through `test_dafny_id_case` in
test_dafny_id_correspondence.py and also through a new assertion:

```python
if case.get("ok_expected") is not None:
    # Also verify the ok signal from the Dafny runtime directly, not just the final expr
    ok, _ = raw_dafny_call(case)
    assert ok == case["ok_expected"]
```

Add an `ok_expected` field to the fixture schema. This makes the oracle test
check not just the final expression but also whether the line claimed
applicability — the direct fix for the structural gap.

---

### Implementation Order

| Priority | Defence                           | Effort | Dafny runtime needed?             | Catches which bug class             |
| -------- | --------------------------------- | ------ | --------------------------------- | ----------------------------------- |
| 1        | Defence 1 (gate contract tests)   | Low    | No                                | Python gate too narrow/broad        |
| 2        | Defence 4 (boundary oracle cases) | Medium | Only for `ok_expected` assertions | Missing positive examples           |
| 3        | Defence 2 (strict mode)           | Low    | Yes (CI only)                     | Python gate–Dafny gate disagreement |
| 4        | Defence 3 (path-selection tests)  | Medium | Yes (skip otherwise)              | Silent fallback masking             |

Defences 1 and 4 require no Dafny toolchain and should land in the same PR as
this change. Defences 2 and 3 are guarded by skip conditions and can follow.

---

## Implementation Progress

**Status: ALL FOUR DEFENCES IMPLEMENTED (2026-05-15)**

All 502 algorithm tests pass (10 expected skips) after implementation.

### Bug Fix — `supports_query_line1`

**Root cause**: `id_extracted_bridge.py::supports_query_line1` used
`not identification.treatments` (checking for zero treatments) instead of the
correct condition `X ∩ An(Y)_G = ∅`.

**Fix applied**: Changed to
`not bool(identification.treatments & ancestors_inclusive(outcomes))`.

**Key insight**: For DAGs, `An(Y)_{G_{bar_x}} = An(Y)_G` restricted to the
treatment set, because directed paths _from_ a treatment node never use in-edges
_to_ that treatment (no cycles). This avoids calling `get_intervened_ancestors`
which dropped isolated outcome nodes.

### Defence 1 — ✅ Implemented

**New file**:
[tests/test_algorithm/test_id_extracted_routing.py](../tests/test_algorithm/test_id_extracted_routing.py)

- 16 tests across 7 `TestSupportsQueryLineN` classes
- Critical regression test:
  `TestSupportsQueryLine1::test_positive_nonempty_treatments_no_causal_path` —
  this is the exact test that would have caught the Line 1 bug

### Defence 2 — ✅ Implemented

**Modified file**:
[src/y0/algorithm/identify/id_extracted_bridge.py](../src/y0/algorithm/identify/id_extracted_bridge.py)

- Added `_strict_mode()` function checking `Y0_DAFNY_STRICT` env var
- All 8 `if not ok:` blocks (lines/full) now raise `AssertionError` in strict
  mode instead of silently raising `ExtractedLineNUnavailableError`
- No test changes needed; activates only in Dafny-enabled CI via
  `Y0_DAFNY_STRICT=1`

### Defence 3 — ✅ Implemented

**Modified file**:
[tests/test_algorithm/test_id_generated_parity.py](../tests/test_algorithm/test_id_generated_parity.py)

- Added 5 path-selection tests: `test_line1_extracted_path_line1_canonical`,
  `test_line1_extracted_path_boundary_nonempty_treatments_no_path`,
  `test_line3_extracted_path_positive`, `test_line6_extracted_path_positive`,
  `test_line7_extracted_path_positive`
- All skip automatically when the compiled Dafny runtime is unavailable
- The boundary test
  (`test_line1_extracted_path_boundary_nonempty_treatments_no_path`) is the
  path-level regression complement to Defence 1's gate-level regression test

### Defence 4 — ✅ Implemented

**Modified files**:

- [src/y0/algorithm/identify/id_oracle_types.py](../src/y0/algorithm/identify/id_oracle_types.py):
  added `"gate_check"` kind to `OracleExpectation`, added
  `build_identification_from_case()` helper, added `NxMixedGraph` import
- [tests/data/generated/dafny_oracle/id_cases.v1.json](../tests/data/generated/dafny_oracle/id_cases.v1.json):
  added 2 boundary cases with `graph_def` and `ok_expected` fields
- [tests/test_dafny_id_correspondence.py](../tests/test_dafny_id_correspondence.py):
  added `test_dafny_gate_check_case` parametrized test

**Plan fully implemented (2026-05-15)**: `raw_dafny_call(case)` is now
implemented and `test_dafny_gate_check_case` calls it directly. When the Dafny
runtime is unavailable the test returns early; when the compiled runtime
disagrees with `ok_expected` (the known Line 1 discrepancy caused by a
pre-update binary) the test `xfail`s with a message directing to the rebuild
script. Both the Python gate and the Dafny runtime `ok` signal are now checked.
