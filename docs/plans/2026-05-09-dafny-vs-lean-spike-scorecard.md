# Dafny vs Lean Spike: Benchmark Cases and Scoring Template

## Purpose

Provide a repeatable, evidence-based evaluation template for deciding whether y0 should continue with Dafny as the primary formal method stack, or move core work to Lean.

## Scope of the spike

Evaluate one representative slice end-to-end:

1. ID Line 4 decomposition path
2. One non-identifiable hedge/failure path
3. One generated artifact consumed by Python conformance tests
4. One post-proof refactor to measure maintainability cost

## Benchmark case suite

Use the same case IDs and expected outcomes in both tracks.

### Case group A: Identifiable path cases

1. case_id: id.a1.line1.trivial-no-treatment
   - Query: P(Y) with empty treatment set
   - Expected: success, simple marginal expression
   - Stress area: base-case handling and expression normalization

2. case_id: id.a2.line2.ancestral-reduction
   - Query: includes non-ancestors of Y
   - Expected: success, equivalent reduced ancestral graph query
   - Stress area: graph restriction correctness

3. case_id: id.a3.line4.multi-component
   - Query: C(G\\X) has more than one district
   - Expected: success, sum(product(subcalls)) shape
   - Stress area: decomposition correctness and deterministic ordering

4. case_id: id.a4.line6.direct-q-evaluation
   - Query: single S where S in C(G)
   - Expected: success, sum of local factorization term
   - Stress area: line-6 branch correctness

5. case_id: id.a5.frontdoor-like
   - Query: frontdoor structure
   - Expected: success, known frontdoor estimand class
   - Stress area: realistic nontrivial identifiable formula

### Case group B: Non-identifiable path cases

1. case_id: id.b1.line5.hedge-basic
   - Query: simple hedge witness
   - Expected: failure classified as hedge with witness metadata
   - Stress area: correct failure semantics, not false success

2. case_id: id.b2.figure1-family
   - Query: one Figure 1 style non-identifiable graph
   - Expected: failure classified as unidentifiable
   - Stress area: parity with existing y0 behavior on known hard examples

### Case group C: Determinism and stability cases

1. case_id: id.c1.repeated-run-stability
   - Input: run same query 20 times
   - Expected: identical artifact and normalized expression output
   - Stress area: ordering determinism, no nondeterministic traversal

2. case_id: id.c2.refactor-resilience
   - Input: after mechanical refactor (rename helpers / move modules)
   - Expected: no semantic output change and minimal proof repair cost
   - Stress area: maintainability

### Case group D: Integration cases

1. case_id: id.d1.fixture-runner-integration
   - Input: generated artifact consumed by common Python runner
   - Expected: green conformance run in both tracks
   - Stress area: pipeline ergonomics

2. case_id: id.d2.ci-reproducibility
   - Input: clean clone, clean env, one-shot run
   - Expected: deterministic outputs and stable CI pass
   - Stress area: operational reliability

### Case group E: Concrete code extraction cases

Applies to Dafny only. Lean extraction via Lean 4 lake build + FFI is scored separately in Category 6.

1. case_id: id.e1.translate-line1
   - Input: rewrite ID Line 1 branch as a concrete Dafny method (no ghost, no axiom)
   - Run: dafny translate py on that method
   - Expected: runnable Python emitted that handles the empty-treatment case
   - Stress area: can Dafny produce extractable code for even the simplest ID branch?

2. case_id: id.e2.translate-line4
   - Input: rewrite ID Line 4 branch as a concrete Dafny method
   - Run: dafny translate py
   - Expected: runnable Python with deterministic district iteration
   - Stress area: does set-iteration and C-component decomposition extract cleanly?

3. case_id: id.e3.translate-hedge-failure
   - Input: rewrite ID Line 5 hedge branch as a concrete Dafny method
   - Run: dafny translate py
   - Expected: runnable Python that raises an appropriate exception with witness data
   - Stress area: does the failure path survive extraction without stripping?

4. case_id: id.e4.extraction-parity
   - Input: run extracted Python against benchmark cases A-B
   - Expected: extracted code produces same result class as handwritten identify()
   - Stress area: end-to-end extraction-to-parity for a realistic ID slice

Note: each extraction case requires removing `ghost`, `{:axiom}`, and `assume`
statements from the target branch before `dafny translate py` can succeed. Record
the rewrite cost (LOC changed, axioms eliminated) as part of the required output.

---

## Required outputs per case

For each case and each track (Dafny, Lean), record:

1. status: pass/fail
2. wall_clock_seconds
3. proof_loc (approximate)
4. glue_code_loc (parser/mapper/runner integration)
5. artifact_hash (or snapshot digest)
6. notes (brief root-cause if fail)

For case group E (extraction), also record:

7. ghost_constructs_removed (count of ghost/axiom/assume eliminated per branch)
8. extraction_loc (LOC of extracted Python per branch)
9. extraction_status: success | partial | failed
10. parity_with_handwritten: pass | fail | not-run

## Scoring rubric (100 points)

### Category 1: Semantic confidence and failure precision (30)

1. Correct success/failure class across all cases (15)
2. Correct failure witness quality for non-identifiable cases (10)
3. Expression/IR parity after normalization (5)

### Category 2: Engineering throughput (25)

1. Time to first green on benchmark suite (10)
2. Total implementation churn (proof + glue LOC) (10)
3. Debug/triage burden during failures (5)

### Category 3: Integration with y0 pipeline (20)

1. Ease of producing fixture artifacts (8)
2. Ease of consuming artifacts in Python tests (7)
3. Compatibility with existing test workflows and CI (5)

### Category 4: Maintainability and onboarding (15)

1. Refactor resilience cost (7)
2. Team readability and reviewability (5)
3. Onboarding effort estimate for new contributor (3)

### Category 5: CI reliability and determinism (10)

1. Reproducibility across clean runs (6)
2. Flake rate / nondeterministic output risk (4)

### Category 6: Concrete code extraction feasibility (15 — Dafny only; N/A for Lean)

This axis scores how much of the ID algorithm can be expressed as extractable
concrete Dafny code (no ghost, no axiom, no assume) and how close that extracted
code is to production-ready Python.

1. At least one ID branch extracts successfully via `dafny translate py` (6)
2. Extracted Python passes at least the A-B benchmark cases without modification (5)
3. Rewrite cost per branch is acceptable (< 50 LOC changed per branch on average) (4)

If this axis scores 0, it means Dafny is viable only as an oracle/verifier, not as
a direct code generator for this problem. That is a meaningful result, not a
disqualifier.

## Pass/fail gate before scoring

A track is only score-eligible if all hard gates pass:

1. Passes case groups A-D in shared runner
2. Produces deterministic fixture artifact for the same input corpus
3. Can be rerun in a clean environment with same results
4. Supports one post-proof refactor without semantic regression

For Dafny only, also record extraction gate outcome (does not block scoring):

5. At least one ID branch attempted for concrete extraction via `dafny translate py`
   - Result: extractable | partially-extractable | not-extractable
   - This gates Category 6 scoring but not overall eligibility

If a track fails any hard gate, mark as Not Ready and skip weighted scoring.

## Fill-in score sheet

### Hard gates

| Gate | Dafny | Lean | Notes |
|---|---|---|---|
| A-D cases pass |  |  |  |
| Deterministic artifacts |  |  |  |
| Clean env reproducibility |  |  |  |
| Post-proof refactor passes |  |  |  |

### Extraction gate (Dafny only, does not block overall eligibility)

| Branch | ghost_constructs_removed | extraction_status | parity_with_handwritten |
|---|---:|---|---|
| Line 1 |  |  |  |
| Line 4 |  |  |  |
| Line 5 (hedge) |  |  |  |

### Weighted scores

| Category | Max | Dafny | Lean | Notes |
|---|---:|---:|---:|---|
| Semantic confidence | 30 |  |  |  |
| Engineering throughput | 25 |  |  |  |
| y0 integration | 20 |  |  |  |
| Maintainability/onboarding | 15 |  |  |  |
| CI reliability | 10 |  |  |  |
| Extraction feasibility (Dafny only) | 15 | | N/A |  |
| Total (Dafny) | 115 |  | — |  |
| Total (Lean) | 100 | — |  |  |

## Decision rule

1. If one track fails hard gates and the other passes: choose passing track.
2. If both pass hard gates:
   - choose higher total score if delta is >= 10 points
   - if delta is < 10 points, prefer the track with lower maintenance burden and better team fit
3. If both fail hard gates: do not migrate; reduce scope and rerun spike.

## Suggested command checklist (per track)

1. Generate artifacts for benchmark corpus
2. Run shared oracle runner
3. Run determinism snapshot check
4. Run clean-environment repro check
5. Apply one planned refactor and repeat 1-4

### Additional steps for Dafny extraction axis

6. For each target branch, strip ghost/axiom/assume and convert to concrete method
7. dafny translate py src/dafny/identification.dfy --output /tmp/id_extracted/
8. Run /tmp/id_extracted/ against benchmark cases A-B
9. Record extraction_status and parity_with_handwritten in the extraction gate table

## Reporting template

Use this concise format in a final memo:

1. Summary recommendation
2. Hard-gate outcomes
3. Weighted score table
4. Top 3 risks for chosen track
5. Mitigation plan for first month after decision
