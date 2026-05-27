# Lean 4 Port of the Dafny Do-Calculus Formalization — Plan and Progress Tracker

## Purpose

Port the Dafny do-calculus formalization in `src/dafny/` to Lean 4 + Mathlib.
The Dafny codebase remains the conformance oracle for the Python `y0` engine.
The Lean port's goal is to **prove the theorems that Dafny must axiomize** —
specifically the probability layer and the do-calculus rules — by grounding them
in Mathlib's existing measure theory and kernel infrastructure.

## Toolchain

- Lean 4.30.0 + Lake 5.0.0 (installed via elan at `~/.elan/bin/`)
- Mathlib4 (probability theory, `Finset`, `PMF`, `Kernel`, conditional
  independence)
- Lake project root: `src/lean/` within this repository

---

## Test Scaffolding Plan

Three distinct layers are required. All three should be green before any phase
is considered done.

### Layer 1: `lake build` (typecheck gate)

`lake build` is the Lean equivalent of `dafny verify`. Every `sorry`-free
definition and theorem is machine-checked automatically. This is the primary CI
gate.

A `tox -e lean` environment will run `cd src/lean && lake build` and fail on
non-zero exit. During early phases `sorry`-backed stubs are allowed;
`set_option warningAsError true` can be enabled per-module once a module is
intended to be `sorry`-free.

### Layer 2: `#guard` conformance tests (algorithm correctness)

The existing fixture `tests/data/generated/dafny_oracle/id_cases.v1.json`
encodes ground-truth inputs/outputs for every ID algorithm case. The file
`src/lean/Y0Lean/Tests/Conformance.lean` will contain `#guard` blocks that:

1. Run `IDImpl` on each fixture case using a shared JSON-decoded input
2. Assert the output matches the expected `IDResult`

This gives the same correctness guarantee as `test_dafny_id_correspondence.py`
without an external Python harness. Coverage grows to include `Ancestors`,
`Descendants`, `DSeparation`, and `CComponentCompiled` as those are ported.

### Layer 3: `sorry` count tracking (proof progress metric)

Analogous to the `{:axiom}` count in the Dafny de-axiomitization tracker:

```bash
rg -c '\bsorry\b' src/lean --include='*.lean' | awk -F: '{sum += $2} END {print sum}'
```

The script `scripts/count_lean_sorrys.py` wraps this and records the count. The
tracker table below records it at each significant commit. The project succeeds
when the count reaches zero (or only explicit interface `sorry`s remain, labelled
analogously to `Axiom_NonNegativity` in Dafny).

---

## Phased Work Plan

### Phase L0: Project Skeleton

**Goal:** `lake build` passes on an empty project with Mathlib as a dependency.
The sorry-count script runs. A `tox -e lean` environment exists.

| ID | Item | Status |
| --- | --- | --- |
| L0-001 | Create `src/lean/` with `lakefile.lean`, `lean-toolchain`, `Y0Lean.lean` | Done |
| L0-002 | Pin Mathlib4 version in `lake-manifest.json` | Done (Mathlib `c5ea00351c`, Lean 4.30.0) |
| L0-003 | Add `tox -e lean` environment (`cd src/lean && lake build`) to `tox.ini` | Done |
| L0-004 | Add `scripts/count_lean_sorrys.py` | Done |
| L0-005 | Commit skeleton: `lake build` green, 0 definitions, 0 sorrys | Done |

### Phase L1: Type Layer

**Goal:** All datatypes, type aliases, and record types ported. No proof
obligations — these are `structure`/`inductive` definitions only. Should be
`sorry`-free from day one.

| ID | Dafny source | Lean target | Status |
| --- | --- | --- | --- |
| L1-001 | `Node = nat`, `Graph = map<Node,set<Node>>` | `abbrev Node := ℕ`; `abbrev Graph := Finmap Node (Finset Node)` | Not started |
| L1-002 | `Option<T>`, `EdgeDir`, `TrailStep` | `Option` (stdlib); `inductive EdgeDir`; `structure TrailStep` | Not started |
| L1-003 | `SMGraph`, `BiEdge` | `structure SMGraph`; `structure BiEdge` | Not started |
| L1-004 | `CausalQuery`, `IDResult` | `structure CausalQuery`; `inductive IDResult` | Not started |
| L1-005 | `IRNode`, `IRQuery`, `IRDoc`, `Edge` (conformance IR) | `inductive IRNode`; `structure IRQuery` etc. | Not started |
| L1-006 | `InterventionalKernel` | `structure InterventionalKernel` | Not started |
| L1-007 | `type PMF = map<Outcome, real>` | Use `Mathlib.Probability.PMF` directly | Not started |

### Phase L2: Computable Algorithms

**Goal:** ID algorithm and supporting algorithms compile and pass `#guard`
conformance tests against `id_cases.v1.json`. Termination proofs required.

| ID | Dafny source | Notes | Status |
| --- | --- | --- | --- |
| L2-001 | `KahnSort` (topological order) | `termination_by` on remaining node count | Not started |
| L2-002 | `Ancestors`, `Descendants` (compiled BFS) | `termination_by fuel` | Not started |
| L2-003 | `BidirectedBFSLoop`, `CComponentCompiled` | `termination_by fuel`; port proved in DA-P3-004 | Not started |
| L2-004 | `DSeparation` (trail-based) | `termination_by` on path length bound | Not started |
| L2-005 | `IDImpl` (the ID algorithm) | `termination_by` on recursion measure | Not started |
| L2-006 | JSON decoder for `id_cases.v1.json` cases | Lean has `Json` in Mathlib/Std | Not started |
| L2-007 | `#guard` conformance tests for all active `id_cases.v1.json` cases | Reuse existing fixture; parity with `test_dafny_id_correspondence.py` | Not started |

### Phase L3: Probability Layer via Mathlib

**Goal:** Replace the ~15–20 Dafny probability axioms with references to
existing Mathlib theorems. Most items require only a one-line `exact` or `apply`.

| ID | Dafny axiom | Mathlib replacement | Status |
| --- | --- | --- | --- |
| L3-001 | `Axiom_NonNegativity` | `PMF.apply_nonneg` | Not started |
| L3-002 | `Axiom_Additivity` | `MeasureTheory.measure_union_disjoint` | Not started |
| L3-003 | `ProductPMF`, `ProductPMF_IsDistribution` | `PMF.prod` / `MeasureTheory.Measure.prod` | Not started |
| L3-004 | `MarkovFactorization` predicate | Markov kernel product `Kernel.const ×ₖ Kernel.id` | Not started |
| L3-005 | `TruncatePMF`, `TruncatePMF_IsDistribution` | `PMF.normalize`, `MeasureTheory.cond` | Not started |
| L3-006 | `SumTruncatedAssignmentMasses_Normalized` | Sum-of-normalize = 1 from `PMF` API | Not started |
| L3-007 | `SetToSequence`, `SetOfSetsToSeq` | `Finset.sort`, `Finset.toList` | Not started |
| L3-008 | `ConditionalFactor`, `ConditionalLocalPMF` | `MeasureTheory.cond` | Not started |
| L3-009 | `MarkovFactorization_IsDistribution` | Follows from Markov kernel normalization | Not started |
| L3-010 | `TruncatePMF_Markov` | Truncated measure still satisfies Markov | Not started |

### Phase L4: Graph Theory Lemmas

**Goal:** Port the key proved lemmas from `dag.dfy` and `semi_markovian.dfy`.
Many will be shorter than the Dafny originals because `simp` + `omega` + `aesop`
cover arithmetic and `Finset` reasoning more directly than Z3.

| ID | Dafny source | Notes | Status |
| --- | --- | --- | --- |
| L4-001 | `KahnSort_Correct` (topological order correctness) | Lean: follow from `Finset.sort` properties | Not started |
| L4-002 | `AncestorsCompiled_Correct`, `DescendantsCompiled_Correct` | BFS completeness; use `Finset` induction | Not started |
| L4-003 | d-separation decomposition, symmetry, weak union | Proved in DA-P4 series; retranslate trail definition | Not started |
| L4-004 | `CComponentCompiled_Correct` (BFS ↔ ghost C-component) | Path-following lemma; proved in DA-P3-004 | Not started |
| L4-005 | `CComponents_Partition` | Coverage + disjointness; proved in DA-P3-003 | Not started |
| L4-006 | `RemoveNodesSM_PreservesWellFormedness` | DAG node deletion; proved in DA-P3-001 | Not started |
| L4-007 | Local Markov property | Depends on L3 Markov factorization | Not started |

### Phase L5: ID Algorithm Correctness (soundness)

**Goal:** Prove `ID_Line1`–`ID_Line7` correctness lemmas and
`Theorem2_Soundness`. Requires L3 + L4 substantially complete. Largest single
proof effort.

| ID | Dafny axiom | Notes | Status |
| --- | --- | --- | --- |
| L5-001 | `ID_Line1` base case | Trivial; `n ≥ 1 → n * n ≥ 1`; proved in Dafny | Not started |
| L5-002 | `ID_Line2` | Set inclusion reduction | Not started |
| L5-003 | `ID_Line3` | Marginalisation correctness | Not started |
| L5-004 | `ID_Line4` | Topological order + ancestors | Not started |
| L5-005 | `ID_Line5` | Hedge detection (non-identifiability) | Not started |
| L5-006 | `ID_Line6` | C-component factorization | Not started |
| L5-007 | `ID_Line7` | Recursive correctness | Not started |
| L5-008 | `Theorem2_Soundness` | Full induction over ID recursion | Not started |
| L5-009 | `IdentifiedIsDistribution` | Soundness corollary | Not started |

### Phase L6: Do-Calculus Rules (open research)

**Goal:** Prove Rule 1/2/3 and derive Backdoor/Frontdoor as theorems. No
mechanized proof of these exists in any proof assistant. This is the novel
research contribution.

#### Proof strategy for L6

The proof chain runs:

```
TruncatePMF (concrete product formula)
  → Global Markov Property for interventional distributions
      → d-sep in mutilated graph ⟹ conditional independence in P_x
          → Rule 1 / Rule 2 / Rule 3  (each is one instantiation)
```

**Source material:**

- **Pearl, _Causality_ (2000/2009):** Use for authoritative theorem _statements_
  only. The book proofs are semantic ("by definition of do") and rely on
  "by inspection of the graph" steps that are not machine-friendly.
- **Shpitser & Pearl (2006), _Identification of Joint Interventional
  Distributions_ (the paper in `docs/plans/Shpitser-ID.md`):** Use Lemmas 4–8
  as the primary proof roadmap. These give explicit do-calculus rule
  applications for each line of the ID algorithm and are the closest thing to a
  proof-in-steps format in the literature.
- **Global Markov Property (GMP) for truncated distributions:** This is the
  critical unlocking lemma — not cleanly separated in Pearl's book. Proves that
  if `(Y ⊥⊥ Z | X, W)` holds in `G_{X̄}` (d-separation in the manipulated
  graph) then `P_x(Y | Z, W) = P_x(Y | W)`. Once this is established, each
  rule is one instantiation at a specific graph surgery. This lemma should be
  the first target of L6.

**Key dependency:** L6-001 (GMP) must land before L6-002 onward. L6-003
through L6-005 can proceed in parallel once GMP is available.

| ID | Dafny axiom | Notes | Status |
| --- | --- | --- | --- |
| L6-001 | `GlobalMarkov` (GMP) | **Key unlocking lemma.** Prove GMP for truncated factorization: d-sep in mutilated graph → cond. indep. in `P_x`. Uses Mathlib `MeasureTheory.IndepSets` or `PMF` conditional independence API. | Not started |
| L6-002 | `InterventionSemantics` | Define `do(X=x)` formally via `TruncatePMF`; show mutilated graph induces truncated factorization. Depends on L6-001. | Not started |
| L6-003 | `Rule1_InsertDeleteObservation` | Instantiate GMP at `G_{X̄}`. Shpitser–Pearl Lemma 5 is the template. | Not started |
| L6-004 | `Rule2_ActionObservationExchange` | Instantiate GMP at `G_{X̄Z̲}`. Shpitser–Pearl Lemma 6 / Lemma 8 are the template. | Not started |
| L6-005 | `Rule3_InsertDeleteAction` | Instantiate GMP at `G_{X̄,Z̄(W)}`. Hardest — requires `Z(W) = Z \ An(W)_{G_{X̄}}` graph surgery. | Not started |
| L6-006 | `BackdoorAdjustment` | Follows from Rule 2 + d-separation of back-door set. | Not started |
| L6-007 | `FrontdoorCriterion` | Composition of Rule 1 and Rule 3. Shpitser–Pearl Section 4 gives the derivation. | Not started |
| L6-008 | `Theorem3_Completeness` | Hardest overall; requires hedge witness lift (Theorem 6 + Corollary 2 of Shpitser–Pearl). | Not started |

---

## Sorry Count Snapshots

Analogous to the `{:axiom}` count table in the Dafny tracker. Run:

```bash
uv run python scripts/count_lean_sorrys.py
```

| Date | Commit | Total `sorry` | Delta | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-26 | (L0 skeleton) | 0 | — | Skeleton only; no definitions |

---

## Verification Log

| Date | Commit | `lake build` result | Conformance tests | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-26 | (L0 skeleton) | ✅ 0 jobs | n/a | Skeleton + Mathlib cached |

---

## Commit Queue

Items ready to commit but not yet committed. Move to ledger after committing.

| ID | Proposed message | Phase items | Verification |
| --- | --- | --- | --- |
| — | — | — | — |

---

## Commit Ledger

| Date | Commit | Message | Phase items | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-26 | (pending) | feat(lean): L0 skeleton — lakefile, lean-toolchain, Mathlib, tox env, sorry counter | L0-001–L0-005 | `lake build` green, 0 sorrys |

---

## Decisions and Blockers

| Date | Decision / Blocker | Resolution |
| --- | --- | --- |
| 2026-05-26 | Lean 4.30.0 + Lake 5.0.0 confirmed installed via elan | No install step needed; `which lake` → `~/.elan/bin/lake` |
| 2026-05-26 | No existing do-calculus formalization in Mathlib or any Lean repo | Must build from scratch; Mathlib covers probability layer only (via `PMF`, `Kernel`, `MeasureTheory`) |
| 2026-05-26 | Only parallel effort: CausalQIF (1 week old, stops at d-separation) | No foundation to build on; independent effort targeting the same theorems |
| 2026-05-26 | Conformance test strategy | Reuse `tests/data/generated/dafny_oracle/id_cases.v1.json`; `#guard` tests in `src/lean/Y0Lean/Tests/Conformance.lean` |
| 2026-05-26 | Project location | `src/lean/` within this repo; `tox -e lean` calls `lake build` from that subdirectory |
| 2026-05-26 | PMF representation | Use `Mathlib.Probability.PMF` directly rather than porting `type PMF = map<Outcome, real>`; this collapses L3-001/L3-002 |
| 2026-05-26 | Dafny codebase role | Remains as-is; serves as the Python conformance oracle. Lean port is additive, not a replacement. |
| 2026-05-26 | L6 proof source material | Use Pearl _Causality_ for theorem statements only (book proofs are semantic/informal). Use Shpitser–Pearl (2006) Lemmas 4–8 as the mechanizable proof roadmap. Global Markov Property for truncated distributions (L6-001) is the critical unlocking lemma that is not cleanly isolated in the book; prove it first. |
| 2026-05-26 | L6 proof chain | `TruncatePMF` definition → GMP (L6-001) → Rule 1/2/3 as instantiations → Backdoor/Frontdoor as derivations → Completeness (Corollary 2 of Shpitser–Pearl) |
