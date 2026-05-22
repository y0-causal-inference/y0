# Dafny De-Axiomitization Progress Tracker

Date opened: 2026-05-19

Source plan: `docs/plans/2026-05-19-dafny-de-axiomitization-plan.md`

This is the working tracker for implementing the de-axiomitization plan. Keep
the plan stable as the strategy document; update this file after each batch of
work with status, verification results, axiom-count movement, and the commit
message used or intended.

## How To Use This Tracker

1. Before a work batch, move the relevant item to `In progress` and add the
   intended commit message to the commit queue.
2. After edits, record verification commands and results.
3. After commit, move the queued message to the commit ledger with the commit
   SHA and a short note about what changed.
4. If a task is blocked, write the blocker and the next proof obligation needed.

Status values: `Planned`, `In progress`, `Done`, `Blocked`, `Deferred`.

## Baseline

| Date | Commit | Total `{:axiom}` | Local `assume {:axiom}` | Declaration axioms | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-05-19 | `ece2091` | 103 | 23 | 80 | Baseline from the de-axiomitization plan. |

Current repository note: `.github/copilot-instructions.md` is untracked and not
part of this tracker unless intentionally added in a later batch.

## Active Sprint Board

| ID | Plan phase | Work item | Status | Owner notes | Verification target | Commit message |
| --- | --- | --- | --- | --- | --- | --- |
| DA-P0-001 | P0 | Build an axiom ledger with category and reason for each remaining declaration axiom. | Done | Ledger created in `docs/plans/2026-05-19-dafny-axiom-ledger.md`. | Documentation check only unless code comments change. | `docs(dafny): add de-axiomitization axiom ledger` |
| DA-P0-002 | P0 | Identify documentation-only or no-postcondition axioms and decide whether to add contracts or demote them. | Done | Phase 0 found 12 contract-gap axioms; demotion/restatement decisions recorded in the axiom ledger. | Documentation check only for Phase 0 inventory. | `docs(dafny): add de-axiomitization axiom ledger` |
| DA-P1-001 | P1 | Prove `KahnsAlgorithm_Correct` as currently stated. | Done | Replaced the axiom with a direct existential-witness proof from the definition of `IsDAG`. | Full Dafny stack verify. | `proof(dafny): prove trivial Kahn correctness wrapper` |
| DA-P1-002 | P1 | Prove `RemoveIncomingCompiled_Correct` and `RemoveOutgoingCompiled_Correct`. | Done | Both lemmas now have proof bodies in `dag.dfy`; full stack verify passed. | Full Dafny stack verify. | `proof(dafny): prove graph surgery compiled equivalence` |
| DA-P1-003 | P1 | Prove `BidirectedBFS_ContainsSelf`. | Done | Added a general `BidirectedBFS_FrontierSubset` helper in `semi_markovian.dfy`; `BidirectedBFS_ContainsSelf` is now a direct corollary. | Full Dafny stack verify. | `proof(dafny): prove bidirected BFS contains start node` |
| DA-P1-004 | P1 | Remove the local assume from `CComponent_Connected`. | Done | Direct use of the `CComponents` ghost set-comprehension was sufficient; no new helper lemma was needed. | Full Dafny stack verify. | `proof(dafny): prove component membership implies bidirected connectivity` |
| DA-P2-001 | P2 | Add generic one-node and two-node C-component helper lemmas. | Done | Added `SingletonNode_SingleCComponent` and the two-node `TwoNodeBidirected_*` helper trio in `semi_markovian.dfy`. | Full Dafny stack verify. | `proof(dafny): add small C-component helper lemmas` |
| DA-P2-002 | P2 | Remove local C-component assumes from `BowArc_NotIdentifiable`. | Done | Replaced the two `|CComponents(...)| == 1` assumptions in `BowArc_NotIdentifiable` with calls to the new helpers; shipped in the same batch as `DA-P2-001`. | Full Dafny stack verify. | `proof(dafny): remove bow-arc hedge component assumes` |
| DA-P2-003 | P2 | Prove `Figure1b_NotIdentifiable` with an explicit hedge. | Done | Reused the bow-arc witness pattern on the `{2,3}` slice of `Figure1bGraph`, with `{3}` as the shared root-set witness. | Full Dafny stack verify. | `proof(dafny): prove Figure 1b non-identifiability` |
| DA-P3-001 | P3 | Prove `RemoveNodesSM` preserves well-formedness. | Done | Added DAG-level node deletion support and a filtered-topological-order proof so `RemoveNodesSM` now preserves `WellFormedSM`. | Full Dafny stack verify. | `proof(dafny): prove SM node removal preserves well-formedness` |
| DA-P3-002 | P3 | Remove local `WellFormedSM(smX)` assume from `CComponentsWithout_Partition`. | Done | Completed in the same batch as `DA-P3-001` by replacing the local assume with `RemoveNodesSM_PreservesWellFormedness(sm, X)`. | Full Dafny stack verify. | `proof(dafny): prove CComponentsWithout partition facts` |
| DA-P4-001 | P4 | Prove `DSep_Decomposition`. | Done | Replaced the axiom with the direct monotonicity proof that `z in Z` implies `z in Z + Z'` inside the `DSep` quantifier. | Full Dafny stack verify. | `proof(dafny): prove d-separation decomposition` |
| DA-P4-002 | P4 | Design trail reversal helpers for `DSep_Symmetry`. | Done | Added `ReverseDir`, `ReverseStep`, `ReverseTrail`, and verified helper lemmas that reversal preserves per-step validity and trail endpoints/connectivity. | Full Dafny stack verify. | `proof(dafny): add trail reversal helpers` |
| DA-P4-003 | P4 | Prove `DSep_Symmetry` using the new reversal helpers. | Done | Added reversal-index and mirrored-collider helpers, factored `TrailBlockedAtPos`, and proved blocking witnesses transfer across reversed connected trails. | Full Dafny stack verify. | `proof(dafny): prove d-separation symmetry` |
| DA-P4-004 | P4 | Prove `DSep_WeakUnion`. | Done | Revised `TrailBlocked` so only internal nodes can block a trail, then used the first blocked collider plus a `Z'` descendant witness to build the contradiction trail needed for weak union. | Full Dafny stack verify. | `proof(dafny): prove d-separation weak union` |
| DA-P4-005 | P4 | Prove `DSep_Contraction`. | Done | Used the new first-blocked-position machinery to show any new blocker under `W + Z'` must be a non-collider in `Z'`, then contradicted `DSep(G, Y, Z', W)` with the corresponding unblocked prefix trail. | Full Dafny stack verify. | `proof(dafny): prove d-separation contraction` |
| DA-P4-006 | P4 | Review or restate `DSep_Intersection` under positivity assumptions. | Done | Removed the misplaced positivity/faithfulness caveat from the graph-level `DSep_Intersection` comment and kept the current pure graph statement axiomatic pending a direct graph proof. | Focused `dag.dfy` verify. | `docs(dafny): review d-separation intersection assumptions` |
| DA-P4-007 | P4 | Reassess `LocalMarkov` proof path. | Done | `LocalMarkov` now looks like a dedicated graph-theoretic helper batch rather than a proof-boundary issue: split on the trail's first step, use all-forward prefixes to force descendant endpoints, and isolate the first forward-to-backward pivot to show no descendant of `v` can open through `Parents(v)` in a DAG. | Documentation check. | `docs(dafny): assess local Markov proof path` |
| DA-P4-008 | P4 | If continuing P4, split `LocalMarkov` into pivot and descendant-parent helper lemmas. | Done | Completed the full LocalMarkov proof: added topological-order/ancestry bridge lemmas, proved the descendant-parent disjointness fact and the first forward-to-backward pivot blocker, then replaced the `LocalMarkov` axiom with a proof body over `NonDescendants(G, v) - Parents(G, v)`. | Focused `dag.dfy` verify plus full Dafny stack verify. | `proof(dafny): prove local Markov property` |
| DA-P4-009 | P4 | Prove `DSep_Intersection` as a direct graph theorem. | Done | Replaced the axiom with an alternating first-blocked-prefix descent proof: a `W`-unblocked trail to `Z` or `Z'` must acquire its first new blocker from the opposite set under added conditioning, which yields a strictly shorter `W`-unblocked prefix into that opposite endpoint set. | Focused `dag.dfy` verify plus full Dafny stack verify. | `proof(dafny): prove d-separation intersection` |
| DA-P5-001 | P5 | Decide concrete finite-sum design for `ProbEvent` or document why it remains abstract. | Done | Decision recorded in `docs/plans/2026-05-21-dafny-probability-finite-sum-decision.md`: make `ProbEvent` concrete as a finite sum over `A * p.Keys`, keep the Kolmogorov axioms explicit in the first slice, and defer the assignment/sample-space representation issues behind `Marginalize`, `TruncatePMF`, `ProductPMF`, and `IntProbConcrete`. | Documentation check. | `docs(dafny): decide probability finite-sum proof path` |
| DA-P5-002 | P5 | Add a reusable finite-support summation helper, define concrete `ProbEvent`, and prove `ComplementRule`, `InclusionExclusion`, and `TotalProbability`. | Done | Added the axiomatic `SetToSequence` bridge plus `SumOutcomeMasses` and `FiniteSupportSum` in `probability.dfy`, made `ProbEvent` concrete over `A * p.Keys`, and replaced the three derived probability axioms with proof bodies. | Focused `probability.dfy` verify plus full Dafny stack verify. | `proof(dafny): define finite-sum event probability` |
| DA-P5-003 | P5 | Resolve the PMF outcome versus node-assignment representation mismatch before constructor-level probability semantics. | Planned | Design note recorded in `docs/plans/2026-05-21-dafny-probability-assignment-alignment.md`, the first translation-layer prototype landed in `DA-P5-004`, and `DA-P5-005` added reusable assignment-event algebra plus sharper `TruncatePMF` consequences; the remaining constructor-level work is to extend the same boundary to `ProductPMF`, `Marginalize`, and a more concrete `TruncatePMF`. | Documentation or focused Dafny verify. | `docs(dafny): plan probability assignment alignment` |
| DA-P5-004 | P5 | Add an explicit outcome-to-assignment bridge and ground `IntProbConcrete` with assignment events. | Done | Added `Value`, `OutcomeToAssignment`, and support-bounded `AssignmentEvent` in `interventional.dfy`, then replaced the `IntProbConcrete` axiom with a concrete conditional-probability definition and gave `IntProb_Grounded` a checked equality contract. | Focused `interventional.dfy` verify plus full Dafny stack verify. | `proof(dafny): ground interventional assignment events` |
| DA-P5-005 | P5 | Add assignment-event algebra and use it to sharpen `TruncatePMF` support semantics. | Done | Added assignment compatibility, conflict, extension, and merge helpers together with compatible/intersection, incompatible/intersection, and strengthening lemmas for `AssignmentEvent`; then sharpened `TruncatePMF` with support-matching contracts and proved intervention-probability-one plus conflicting-assignment-zero consequences. | Focused `interventional.dfy` verify plus full Dafny stack verify. | `proof(dafny): add assignment-event algebra` |
| DA-P6-001 | P6 | Decide whether `DoCalculus.IntProb` remains abstract or wraps concrete semantics. | Planned | Required before rule proofs are meaningful. | Documentation or targeted Dafny verify. | `docs(dafny): decide do-calculus IntProb proof boundary` |
| DA-P7-001 | P7 | Prove low-level recursive `IDImpl` subquery validity obligations. | Deferred | Wait for P3 and P5 support. | Full Dafny stack verify. | `proof(dafny): prove ID recursive subquery validity` |

## Axiom Count Snapshots

Use these commands from the repository root:

```bash
rg -o "\{:axiom\}" src/dafny/*.dfy | wc -l
rg -o "assume\s+\{:axiom\}" src/dafny/*.dfy | wc -l
```

| Date | Commit or working tree | Total `{:axiom}` | Local `assume {:axiom}` | Declaration axioms | Delta | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 2026-05-19 | `ece2091` | 103 | 23 | 80 | Baseline | Plan committed. |
| 2026-05-19 | `0e6fa23` | 101 | 23 | 78 | -2 declaration axioms | `RemoveIncomingCompiled_Correct` and `RemoveOutgoingCompiled_Correct` are now proved and committed. |
| 2026-05-19 | `1e46df7` | 100 | 23 | 77 | -1 declaration axiom | `KahnsAlgorithm_Correct` is now proved and committed. |
| 2026-05-19 | `b8b2c03` | 99 | 23 | 76 | -1 declaration axiom | `BidirectedBFS_ContainsSelf` is now proved and committed. |
| 2026-05-19 | `d04e297` | 98 | 22 | 76 | -1 local assume | `CComponent_Connected` no longer uses a local `assume {:axiom}`. |
| 2026-05-19 | `f3d459d` | 96 | 20 | 76 | -2 local assumes | Added small one-node/two-node C-component helpers and removed the two local C-component assumes from `BowArc_NotIdentifiable`. |
| 2026-05-19 | `6a89013` | 95 | 20 | 75 | -1 declaration axiom | `Figure1b_NotIdentifiable` is now proved and committed. |
| 2026-05-19 | `c2dca87` | 94 | 19 | 75 | -1 local assume | `RemoveNodesSM` now preserves well-formedness, and `CComponentsWithout_Partition` no longer uses the local `WellFormedSM(smX)` assumption. |
| 2026-05-19 | `5a4c783` | 93 | 19 | 74 | -1 declaration axiom | `DSep_Decomposition` is now proved and committed. |
| 2026-05-19 | `2157e42` | 93 | 19 | 74 | No axiom-count change | Added trail reversal helpers for future `DSep_Symmetry` work. |
| 2026-05-19 | `29c5895` | 92 | 19 | 73 | -1 declaration axiom | `DSep_Symmetry` is now proved and committed; `PureDSepErasesObservation` now states the Rule 1 precondition rewrite explicitly. |
| 2026-05-19 | `4c1fbd6` | 91 | 19 | 72 | -1 declaration axiom | `DSep_WeakUnion` is now proved and committed after revising single-edge trails to be unblocked. |
| 2026-05-19 | `f255893` | 90 | 19 | 71 | -1 declaration axiom | `DSep_Contraction` is now proved and committed. |
| 2026-05-19 | `1774828` | 90 | 19 | 71 | No axiom-count change | Reviewed `DSep_Intersection` and removed the misplaced positivity caveat from the graph-level d-separation layer. |
| 2026-05-19 | `80dcddd` | 90 | 19 | 71 | No axiom-count change | `DA-P4-007` reassessed the `LocalMarkov` proof path and concluded it should be split into explicit DAG/trail helper lemmas rather than treated as a proof-boundary problem. |
| 2026-05-19 | `5b8559c` | 90 | 19 | 71 | No axiom-count change | Started `DA-P4-008`: restated `LocalMarkov` to exclude parents and added the first entry helpers. |
| 2026-05-19 | `17ac9a5` | 89 | 19 | 70 | -1 declaration axiom | Completed `DA-P4-008`: `LocalMarkov` is now proved and committed. |
| 2026-05-21 | `31d624b` | 88 | 19 | 69 | -1 declaration axiom | `DSep_Intersection` is now proved and committed. |
| 2026-05-21 | Working tree | 88 | 19 | 69 | No axiom-count change | `DA-P5-001` recorded the concrete finite-sum `ProbEvent` decision without code changes. |
| 2026-05-21 | Working tree | 86 | 19 | 67 | -2 declaration axioms | Added `SetToSequence` as a small enumeration bridge axiom, but removed the three derived probability axioms by defining concrete `ProbEvent` and proving `ComplementRule`, `InclusionExclusion`, and `TotalProbability`. |
| 2026-05-21 | `c914de5` | 86 | 19 | 67 | No axiom-count change | Committed `DA-P5-001` and `DA-P5-002`: the probability finite-sum decision note, the finite-support event-sum substrate, concrete `ProbEvent`, and the first derived-law proof batch. |
| 2026-05-21 | `87143dd` | 86 | 19 | 67 | No axiom-count change | Committed the `DA-P5-003` alignment note that separates node values from joint outcomes and rejects an in-place reinterpretation of `Prob.PMF`. |
| 2026-05-21 | `c148a9c` | 85 | 19 | 66 | -1 declaration axiom | Committed `DA-P5-004`: added `OutcomeToAssignment` plus support-bounded `AssignmentEvent`, made `IntProbConcrete` concrete, and replaced the narrative `IntProb_Grounded` axiom with a checked equality over truncated-PMF assignment events. |
| 2026-05-21 | Working tree | 85 | 19 | 66 | No axiom-count change | Added assignment-event algebra (`CompatibleAssignments`, `MergeAssignments`, compatible/incompatible intersection, strengthening) and sharpened `TruncatePMF` with support, probability-one, and conflicting-assignment-zero facts. |

## Verification Log

| Date | Work item | Command | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-19 | Tracker creation | `git diff --check -- docs/plans/2026-05-19-dafny-de-axiomitization-progress.md` | Passed | Documentation-only check. |
| 2026-05-19 | Phase 0 inventory | `rg -n "^\s*(ghost\s+)?(lemma|function|predicate)\s+\{:axiom\}" src/dafny/*.dfy` | Passed | Counted 80 declaration axioms for the ledger. |
| 2026-05-19 | Phase 0 contract-gap scan | Best-effort script over `src/dafny/*.dfy` | Passed | Identified 12 axiom lemmas lacking `ensures` before the next top-level declaration. |
| 2026-05-19 | Phase 0 docs validation | `git diff --check -- docs/plans/2026-05-19-dafny-axiom-ledger.md docs/plans/2026-05-19-dafny-de-axiomitization-progress.md` | Passed | Both Phase 0 docs are clean. |
| 2026-05-19 | DA-P1-002 focused verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 29 verified, 0 errors after removing the two axiom annotations. |
| 2026-05-19 | DA-P1-002 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 97 verified, 0 errors. |
| 2026-05-19 | DA-P1-001 focused verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 30 verified, 0 errors after replacing `KahnsAlgorithm_Correct`. |
| 2026-05-19 | DA-P1-001 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 98 verified, 0 errors. |
| 2026-05-19 | DA-P1-003 focused verify | `/opt/homebrew/bin/dafny verify semi_markovian.dfy` | Passed | `semi_markovian.dfy` verified with 16 verified, 0 errors after replacing `BidirectedBFS_ContainsSelf`. |
| 2026-05-19 | DA-P1-003 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 100 verified, 0 errors. |
| 2026-05-19 | DA-P1-004 focused verify | `/opt/homebrew/bin/dafny verify semi_markovian.dfy` | Passed | `semi_markovian.dfy` verified with 16 verified, 0 errors after removing the local assume from `CComponent_Connected`. |
| 2026-05-19 | DA-P1-004 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 100 verified, 0 errors. |
| 2026-05-19 | DA-P2-001 / DA-P2-002 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 104 verified, 0 errors after adding the small C-component helper lemmas and replacing the two bow-arc local assumptions. |
| 2026-05-19 | DA-P2-003 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 105 verified, 0 errors after replacing `Figure1b_NotIdentifiable` with an explicit hedge proof. |
| 2026-05-19 | DA-P3-001 focused verify | `/opt/homebrew/bin/dafny verify semi_markovian.dfy` | Passed | `semi_markovian.dfy` verified with 21 verified, 0 errors after proving `RemoveNodesSM_PreservesWellFormedness` and removing the local `WellFormedSM(smX)` assumption. |
| 2026-05-19 | DA-P3-001 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 107 verified, 0 errors. |
| 2026-05-19 | DA-P4-001 focused verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 32 verified, 0 errors after replacing `DSep_Decomposition`. |
| 2026-05-19 | DA-P4-001 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 108 verified, 0 errors. |
| 2026-05-19 | DA-P4-002 focused verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 41 verified, 0 errors after adding trail reversal helpers for symmetry proofs. |
| 2026-05-19 | DA-P4-002 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 117 verified, 0 errors. |
| 2026-05-19 | DA-P4-003 focused verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 50 verified, 0 errors after proving `DSep_Symmetry` and factoring `TrailBlockedAtPos`. |
| 2026-05-19 | DA-P4-003 downstream focused verify | `/opt/homebrew/bin/dafny verify do_calculus.dfy` | Passed | `do_calculus.dfy` verified with 1 verified, 0 errors after making the `Rule1_InsertDeleteObservation` precondition rewrite explicit in `PureDSepErasesObservation`. |
| 2026-05-19 | DA-P4-003 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 126 verified, 0 errors. |
| 2026-05-19 | DA-P4-004 helper-substrate verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 77 verified, 0 errors after adding constructive ancestry/trail helper lemmas while probing `DSep_WeakUnion`. |
| 2026-05-19 | DA-P4-004 focused verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 82 verified, 0 errors after revising single-edge blocking semantics and replacing `DSep_WeakUnion` with a proof body. |
| 2026-05-19 | DA-P4-004 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 158 verified, 0 errors. |
| 2026-05-19 | DA-P4-005 focused verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 85 verified, 0 errors after replacing `DSep_Contraction` with a proof body. |
| 2026-05-19 | DA-P4-005 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 161 verified, 0 errors. |
| 2026-05-19 | DA-P4-006 focused verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 85 verified, 0 errors after restating `DSep_Intersection` as a graph-level axiom and removing the misplaced positivity caveat. |
| 2026-05-19 | DA-P4-007 docs validation | `git diff --check -- docs/plans/2026-05-19-dafny-de-axiomitization-plan.md docs/plans/2026-05-19-dafny-de-axiomitization-progress.md docs/plans/2026-05-19-dafny-axiom-ledger.md` | Passed | The plan, tracker, and axiom-ledger updates for the `LocalMarkov` reassessment are clean. |
| 2026-05-19 | DA-P4-008 focused verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 89 verified, 0 errors after restating `LocalMarkov` to exclude parents and adding `ForwardTrail_EndInDescendants` plus `BackwardFirstStep_BlockedByParents`. |
| 2026-05-19 | DA-P4-008 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 165 verified, 0 errors. |
| 2026-05-19 | DA-P4-008 LocalMarkov proof verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 106 verified, 0 errors after adding the forward-to-backward pivot blocker, the descendant-parent disjointness lemma, the topological-order bridge lemmas, and the `LocalMarkov` proof body. |
| 2026-05-19 | DA-P4-008 LocalMarkov full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 182 verified, 0 errors after replacing the `LocalMarkov` axiom with a proof body. |
| 2026-05-20 | DA-P4-009 focused verify | `/opt/homebrew/bin/dafny verify dag.dfy` | Passed | `dag.dfy` verified with 109 verified, 0 errors after replacing `DSep_Intersection` with the alternating-prefix descent proof. |
| 2026-05-20 | DA-P4-009 full-stack verify | `/opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy` | Passed | Full stack verified with 185 verified, 0 errors; axiom counts are now 88 total / 19 local assumes / 69 declaration axioms. |
| 2026-05-21 | DA-P5-001 docs validation | `git diff --check -- docs/plans/2026-05-19-dafny-de-axiomitization-progress.md docs/plans/2026-05-21-dafny-probability-finite-sum-decision.md` | Passed | The P5 probability decision note and tracker updates are clean. |
| 2026-05-21 | DA-P5-002 focused verify | `/opt/homebrew/bin/dafny verify src/dafny/probability.dfy` | Passed | Iterated focused verification stayed green through the finite-support helper, concrete `ProbEvent`, and the `ComplementRule`, `InclusionExclusion`, and `TotalProbability` proofs; the final focused run reported 22 verified, 0 errors. |
| 2026-05-21 | DA-P5-002 full-stack verify | `/opt/homebrew/bin/dafny verify src/dafny/probability.dfy src/dafny/dag.dfy src/dafny/interventional.dfy src/dafny/do_calculus.dfy src/dafny/semi_markovian.dfy src/dafny/identification.dfy` | Passed | Full stack verified with 192 verified, 0 errors; axiom counts are now 86 total / 19 local assumes / 67 declaration axioms. |
| 2026-05-21 | DA-P5-004 focused verify | `/opt/homebrew/bin/dafny verify src/dafny/probability.dfy src/dafny/dag.dfy src/dafny/interventional.dfy` | Passed | The new outcome-to-assignment bridge, support-bounded assignment events, concrete `IntProbConcrete`, and formal `IntProb_Grounded` contract verified together with 139 verified, 0 errors. |
| 2026-05-21 | DA-P5-004 full-stack verify | `/opt/homebrew/bin/dafny verify src/dafny/probability.dfy src/dafny/dag.dfy src/dafny/interventional.dfy src/dafny/do_calculus.dfy src/dafny/semi_markovian.dfy src/dafny/identification.dfy` | Passed | Full stack verified with 197 verified, 0 errors; axiom counts are now 85 total / 19 local assumes / 66 declaration axioms. |
| 2026-05-21 | DA-P5-005 focused verify | `/opt/homebrew/bin/dafny verify src/dafny/probability.dfy src/dafny/dag.dfy src/dafny/interventional.dfy` | Passed | The assignment-event algebra, new `MatchesAssignment_Extension` helper, and sharpened `TruncatePMF` support/probability lemmas verified cleanly with 163 verified, 0 errors. |
| 2026-05-21 | DA-P5-005 full-stack verify | `/opt/homebrew/bin/dafny verify src/dafny/probability.dfy src/dafny/dag.dfy src/dafny/interventional.dfy src/dafny/do_calculus.dfy src/dafny/semi_markovian.dfy src/dafny/identification.dfy` | Passed | Full stack verified with 221 verified, 0 errors; axiom counts remain 85 total / 19 local assumes / 66 declaration axioms. |

## Commit Queue

Move rows from here to the commit ledger after committing.

| Work item | Proposed commit message | Included changes | Verification before commit | Status |
| --- | --- | --- | --- | --- |
| Tracker setup | `docs(dafny): add de-axiomitization progress tracker` | Add this working tracker. | Markdown diff check. | Ready |
| DA-P0-001 | `docs(dafny): add de-axiomitization axiom ledger` | Add category/reason ledger for remaining declaration axioms and update Phase 0 progress. | Markdown diff check. | Ready |
| DA-P5-005 | `proof(dafny): add assignment-event algebra` | Add assignment compatibility/merge/event algebra in `interventional.dfy`, sharpen `TruncatePMF` support contracts, and update the tracker and ledger. | Focused `interventional.dfy` verify plus full-stack verify. | Ready |

## Commit Ledger

| Date | Commit | Message | Work items | Verification | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-05-19 | `ece2091` | `added de-axiomitization plan with GPT-5.5-xhigh. Will implement with GPT-5.4-xhigh` | Plan baseline | Not recorded here | Adds source plan for this tracker. |
| 2026-05-19 | `0e6fa23` | `proof(dafny): prove graph surgery compiled equivalence` | `DA-P1-002` | Focused `dag.dfy` verify plus full-stack verify | Commits the first Phase 1 proof batch. |
| 2026-05-19 | `1e46df7` | `proof(dafny): prove trivial Kahn correctness wrapper` | `DA-P1-001` | Focused `dag.dfy` verify plus full-stack verify | Commits the weak-contract `KahnsAlgorithm_Correct` proof. |
| 2026-05-19 | `b8b2c03` | `proof(dafny): prove bidirected BFS contains start node` | `DA-P1-003` | Focused `semi_markovian.dfy` verify plus full-stack verify | Commits the `BidirectedBFS_FrontierSubset` helper and the `BidirectedBFS_ContainsSelf` proof. |
| 2026-05-19 | `d04e297` | `proof(dafny): prove component membership implies bidirected connectivity` | `DA-P1-004` | Focused `semi_markovian.dfy` verify plus full-stack verify | Commits the direct proof of `CComponent_Connected` from the `CComponents` ghost body. |
| 2026-05-19 | `f3d459d` | `proof(dafny): add small C-component helper lemmas` | `DA-P2-001`, `DA-P2-002` | Full-stack verify | Commits the singleton/two-node C-component helpers and uses them to remove the two local C-component assumptions in `BowArc_NotIdentifiable`. |
| 2026-05-19 | `6a89013` | `proof(dafny): prove Figure 1b non-identifiability` | `DA-P2-003` | Full-stack verify | Commits the explicit hedge witness proof for `Figure1b_NotIdentifiable`. |
| 2026-05-19 | `c2dca87` | `proof(dafny): prove SM node removal preserves well-formedness` | `DA-P3-001`, `DA-P3-002` | Focused `semi_markovian.dfy` verify plus full-stack verify | Commits the DAG node-deletion preservation proof and uses it to remove the local `WellFormedSM(smX)` assumption from `CComponentsWithout_Partition`. |
| 2026-05-19 | `5a4c783` | `proof(dafny): prove d-separation decomposition` | `DA-P4-001` | Focused `dag.dfy` verify plus full-stack verify | Commits the direct `DSep` subset proof for decomposition. |
| 2026-05-19 | `2157e42` | `proof(dafny): add trail reversal helpers` | `DA-P4-002` | Focused `dag.dfy` verify plus full-stack verify | Commits the `ReverseTrail` helper surface and the basic validity/connectivity lemmas needed before attempting `DSep_Symmetry`. |
| 2026-05-19 | `29c5895` | `proof(dafny): prove d-separation symmetry` | `DA-P4-003` | Focused `dag.dfy` verify, focused `do_calculus.dfy` verify, plus full-stack verify | Commits the reversal-based proof of `DSep_Symmetry` and the downstream `Rule1` precondition rewrite in `PureDSepErasesObservation`. |
| 2026-05-19 | `4c1fbd6` | `proof(dafny): prove d-separation weak union` | `DA-P4-004` | Focused `dag.dfy` verify plus full-stack verify | Commits the single-edge blocking semantics fix, the constructive ancestry/trail substrate, and the contradiction proof for `DSep_WeakUnion`. |
| 2026-05-19 | `f255893` | `proof(dafny): prove d-separation contraction` | `DA-P4-005` | Focused `dag.dfy` verify plus full-stack verify | Commits the contraction proof based on identifying the first blocker under `W + Z'` and reducing it to an unblocked prefix trail into `Z'`. |
| 2026-05-19 | `1774828` | `docs(dafny): review d-separation intersection assumptions` | `DA-P4-006` | Focused `dag.dfy` verify | Commits the `DSep_Intersection` statement review: positivity caveats were removed from the graph-level `DSep` layer, while the axiom itself remains in place pending a direct proof. |
| 2026-05-19 | `80dcddd` | `docs(dafny): assess local Markov proof path` | `DA-P4-007` | Documentation diff check | Records that `LocalMarkov` is still axiomatic, but now has a concrete helper-lemma proof plan instead of a vague boundary note. |
| 2026-05-19 | `5b8559c` | `proof(dafny): add local Markov entry helpers` | `DA-P4-008` (partial) | Focused `dag.dfy` verify plus full-stack verify | Commits the first LocalMarkov helper slice: restated the target to exclude parents and added the initial forward/parent-entry helper lemmas. |
| 2026-05-19 | `17ac9a5` | `proof(dafny): prove local Markov property` | `DA-P4-008` | Focused `dag.dfy` verify plus full-stack verify | Commits the completed LocalMarkov proof, including the first forward-to-backward pivot blocker, descendant-parent disjointness, topological-order bridge lemmas, and the non-axiomatic theorem body. |
| 2026-05-21 | `31d624b` | `Dsep_Intersection passes` | `DA-P4-009` | Focused `dag.dfy` verify plus full-stack verify | Commits the direct graph proof of `DSep_Intersection`; the tracker row previously recorded as a working-tree result is now closed out as an actual commit. |
| 2026-05-21 | `c914de5` | `proof(dafny): define finite-sum event probability` | `DA-P5-001`, `DA-P5-002` | Focused `probability.dfy` verify plus full-stack verify | Commits the probability finite-sum decision note, the `SetToSequence` / `FiniteSupportSum` substrate, concrete `ProbEvent`, and the proofs of `ComplementRule`, `InclusionExclusion`, and `TotalProbability`. |
| 2026-05-21 | `87143dd` | `docs(dafny): scope probability assignment alignment` | `DA-P5-003` (planning) | Documentation diff check | Commits the design note that separates node values from joint outcomes, rejects an in-place reinterpretation of `Prob.PMF`, and scopes the translation-layer prototype path. |
| 2026-05-21 | `c148a9c` | `proof(dafny): ground interventional assignment events` | `DA-P5-004` | Focused `interventional.dfy` verify plus full-stack verify | Commits the `OutcomeToAssignment` bridge, support-bounded `AssignmentEvent`, concrete `IntProbConcrete`, and the explicit `IntProb_Grounded` equality over truncated-PMF assignment events. |

## Decisions And Blockers

| Date | Item | Decision or blocker | Follow-up |
| --- | --- | --- | --- |
| 2026-05-19 | Burn-down metric | Track local assumes separately from declaration axioms. | Prioritize local assumes and derivable wrappers before deep foundations. |
| 2026-05-19 | Foundational axioms | Do not count Kolmogorov-style foundational axioms as debt unless replacing the abstraction. | Record them in the ledger with reason `Foundational`. |
| 2026-05-19 | Phase 0 contract gaps | Twelve declaration axioms currently have no formal `ensures`; treat them as contract gaps, not successful theorem statements. | Demote or restate them before claiming proof progress on those symbols. |
| 2026-05-19 | Single-edge trail semantics | Revised `TrailBlocked` so only internal nodes can block a trail; this removes the fork counterexample and aligns the formalization with the weak-union proof. | Recheck later d-separation theorems against the updated semantics. |
| 2026-05-19 | `DSep_Intersection` review | The positivity/faithfulness caveat does not belong at the graph-level `DSep` layer. Keep the current pure graph statement and leave it axiomatic until a direct graph proof is available. | If we revisit the theorem, prove it as a graph property or defer it explicitly rather than attaching distributional assumptions to `DSep`. |
| 2026-05-19 | `LocalMarkov` reassessment | `LocalMarkov` is no longer best viewed as a proof-boundary issue. The current substrate already covers forward-prefix descendant reasoning; the missing pieces are a first-backward-step pivot lemma and a DAG fact that descendants of `v` cannot open through `Parents(v)`. | If continuing P4, split `LocalMarkov` into those helper batches instead of attempting the theorem in one jump. |
| 2026-05-19 | `LocalMarkov` target set | Under the revised single-edge trail semantics, `LocalMarkov` must exclude parents from the target set; otherwise one-edge backward trails to parents remain unblocked. | Keep the theorem target as `NonDescendants(G, v) - Parents(G, v)` in future helper and proof work. |
| 2026-05-19 | `LocalMarkov` proof shape | The completed proof rests on three cases: immediate backward entry through a parent, all-forward trails into descendants, and the first `Forward -> Backward` pivot, which is blocked because descendants of a descendant cannot intersect `Parents(v)` in a DAG. | Remaining Phase 4 graph work is now the direct proof or explicit deferral of `DSep_Intersection`. |
| 2026-05-20 | `DSep_Intersection` proof shape | The direct graph proof works by alternating the two premises along the first blocker introduced by extra conditioning: each newly blocked non-collider from the opposite set yields a strictly shorter `W`-unblocked prefix into that set, contradicting finite trail length. | Phase 4 is now closed; move to `DA-P5-001`. |
| 2026-05-21 | P5 probability boundary | Choose a concrete finite-support `ProbEvent` over `A * p.Keys` rather than widening the probability trust boundary with more algebraic axioms. Keep the Kolmogorov axioms explicit for the first slice, and treat the PMF outcome vs node-assignment alignment as a separate blocker for `Marginalize`, `TruncatePMF`, `ProductPMF`, and `IntProbConcrete`. | Start the finite-support summation helper plus concrete `ProbEvent` batch before revisiting constructor-level probability semantics. |
| 2026-05-21 | Finite-support sum bridge | `Outcome` now carries the `!new` characteristic and event sums are concrete, but abstract unordered outcomes still need an enumeration bridge. The batch therefore uses an axiomatic `SetToSequence` helper plus a recursive sequence fold to define `FiniteSupportSum` and `ProbEvent`. | Revisit whether `SetToSequence` should remain the long-term bridge or be replaced by a stronger canonical-support representation before constructor-level PMF work. |
| 2026-05-21 | PMF versus assignment alignment | The next P5 blocker is the mismatch between abstract `Prob.Outcome`-indexed PMFs and `Assignment = map<Node, Prob.Outcome>` semantics in `interventional.dfy`. Constructor-level probability semantics need either a shared sample space or a verified translation layer before `ProductPMF`, `Marginalize`, `TruncatePMF`, or `IntProbConcrete` can become concrete. | Start `DA-P5-003` with a design note and a narrow prototype around the probability/interventional boundary. |
| 2026-05-21 | Immediate alignment path | The first concrete fix takes the translation-layer path: keep `Prob.PMF` outcome-keyed for now, add `OutcomeToAssignment` and support-bounded `AssignmentEvent` in `interventional.dfy`, and use them to ground `IntProbConcrete` without a broad in-place PMF refactor. | Extend the same bridge to `TruncatePMF`, `Marginalize`, and `ProductPMF`, or revisit a joint-assignment-keyed PMF model once the boundary work is complete. |
| 2026-05-21 | Assignment-event algebra | The translation-layer path now has reusable event algebra: compatible assignments intersect to their merge, conflicting assignments intersect to the empty event, and assignment extension gives event inclusion. `TruncatePMF` now also guarantees retained support matches the intervention assignment, which yields probability one for the intervention event and probability zero for conflicting assignment events. | Use this algebra to sharpen `Marginalize` next, or further concretize `TruncatePMF` beyond support-level contracts. |

## Phase Transitions

| Date | Transition | Status | Notes |
| --- | --- | --- | --- |
| 2026-05-19 | Phase 0 -> Phase 1 | Ready | Ledger created in `docs/plans/2026-05-19-dafny-axiom-ledger.md`; next recommended proof batch is graph surgery compiled equivalence. |
| 2026-05-19 | Phase 2 -> Phase 3 | Ready | The three planned P2 items are complete through `6a89013`; next recommended proof batch is `DA-P3-001` (`RemoveNodesSM` well-formedness preservation). |
| 2026-05-19 | Phase 3 -> Phase 4 | Ready | The planned P3 items are complete through `c2dca87`; next recommended proof batch is `DA-P4-001` (`DSep_Decomposition`). |
| 2026-05-20 | Phase 4 -> Phase 5 | Ready | `DSep_Intersection` is now proved in the working tree, so the remaining graph-level Phase 4 agenda is closed; next recommended batch is `DA-P5-001` (the `ProbEvent` finite-sum proof-path decision). |

## Phase 1 Batch Notes

| Date | Batch | Status | Notes | Next recommended step |
| --- | --- | --- | --- | --- |
| 2026-05-19 | DA-P1-002 graph surgery compiled equivalence | Complete | Removed two declaration axioms from `dag.dfy`, kept the full Dafny stack green, and committed as `0e6fa23`. | Start `DA-P1-001` (`KahnsAlgorithm_Correct`) or `DA-P1-003` (`BidirectedBFS_ContainsSelf`). |
| 2026-05-19 | DA-P1-001 trivial Kahn correctness wrapper | Complete | Removed one declaration axiom from `dag.dfy` with a direct existential-witness proof, kept the full Dafny stack green, and committed as `1e46df7`. | Continue with `DA-P1-003` (`BidirectedBFS_ContainsSelf`). |
| 2026-05-19 | DA-P1-003 bidirected BFS contains start node | Complete | Removed one declaration axiom from `semi_markovian.dfy` by proving the more general `BidirectedBFS_FrontierSubset` lemma and deriving `BidirectedBFS_ContainsSelf` from it; full stack stayed green and the batch committed as `b8b2c03`. | Continue with `DA-P1-004` (`CComponent_Connected`). |
| 2026-05-19 | DA-P1-004 component membership implies connectivity | Complete | Removed one local `assume {:axiom}` from `semi_markovian.dfy` by unpacking the connectivity conjunct already present in the `CComponents` ghost definition; full stack stayed green and the batch committed as `d04e297`. | Start `DA-P2-001` (small C-component helper lemmas). |

## Phase 2 Batch Notes

| Date | Batch | Status | Notes | Next recommended step |
| --- | --- | --- | --- | --- |
| 2026-05-19 | DA-P2-001 / DA-P2-002 small C-component helpers and bow-arc cleanup | Complete | Added `SingletonNode_SingleCComponent` plus the `TwoNodeBidirected_*` helper lemmas in `semi_markovian.dfy`, then used them in `BowArc_NotIdentifiable` to replace the two local `|CComponents(...)| == 1` assumptions; full stack stayed green and the batch committed as `f3d459d`. | Start `DA-P2-003` (`Figure1b_NotIdentifiable`). |
| 2026-05-19 | DA-P2-003 Figure 1(b) explicit hedge witness | Complete | Replaced `Figure1b_NotIdentifiable` with an explicit hedge proof that reuses the bow-arc witness pattern on the `{2,3}` subgraph of `Figure1bGraph`; full stack stayed green and the batch committed as `6a89013`. | Start `DA-P3-001` (`RemoveNodesSM` preserves well-formedness). |

## Phase 3 Batch Notes

| Date | Batch | Status | Notes | Next recommended step |
| --- | --- | --- | --- | --- |
| 2026-05-19 | DA-P3-001 / DA-P3-002 node removal well-formedness and partition cleanup | Complete | Added a DAG-level `RemoveNodes` helper plus `RemoveNodes_PreservesDAG`, then used them in `RemoveNodesSM_PreservesWellFormedness` to replace the local `WellFormedSM(smX)` assumption inside `CComponentsWithout_Partition`; full stack stayed green and the batch committed as `c2dca87`. | Start `DA-P4-001` (`DSep_Decomposition`). |

## Phase 4 Batch Notes

| Date | Batch | Status | Notes | Next recommended step |
| --- | --- | --- | --- | --- |
| 2026-05-19 | DA-P4-001 d-separation decomposition | Complete | Replaced `DSep_Decomposition` with a direct proof from the `DSep` definition: a trail ending in `z in Z` is also a trail ending in `z in Z + Z'`; full stack stayed green and the batch committed as `5a4c783`. | Start `DA-P4-002` (trail reversal helpers for `DSep_Symmetry`). |
| 2026-05-19 | DA-P4-002 trail reversal helpers | Complete | Added the `ReverseTrail` helper surface plus validity and endpoint/connectivity lemmas in `dag.dfy`; full stack stayed green and the batch committed as `2157e42`. | Start `DA-P4-003` (`DSep_Symmetry`). |
| 2026-05-19 | DA-P4-003 d-separation symmetry | Complete | Proved `DSep_Symmetry` by reversing valid connected trails, mirroring collider status and blocking witnesses, and factoring `TrailBlockedAtPos`; a small downstream rewrite in `PureDSepErasesObservation` made the `Rule1_InsertDeleteObservation` precondition explicit. The full stack stayed green and the batch committed as `29c5895`. | Start `DA-P4-004` (`DSep_WeakUnion`). |
| 2026-05-19 | DA-P4-004 d-separation weak union | Complete | Revised `TrailBlocked` so only internal nodes block a trail, kept the new constructive ancestry/trail substrate, and proved `DSep_WeakUnion` by taking the first blocker under `W`, showing it must be a collider opened by some `z' in Z'`, and building a contradiction trail back to the premise. The full stack stayed green and the batch committed as `4c1fbd6`. | Start `DA-P4-005` (`DSep_Contraction`). |
| 2026-05-19 | DA-P4-005 d-separation contraction | Complete | Proved `DSep_Contraction` by taking the first blocker under `W + Z'` on a trail to `z in Z`, showing that any blocker newly created by extra conditioning must be a non-collider in `Z'`, and contradicting `DSep(G, Y, Z', W)` with the resulting unblocked prefix trail. The full stack stayed green and the batch committed as `f255893`. | Start `DA-P4-006` (review or restate `DSep_Intersection`). |
| 2026-05-19 | DA-P4-006 d-separation intersection review | Complete | Reviewed the remaining `DSep_Intersection` axiom and removed the misplaced positivity/faithfulness caveat from the graph-level `DSep` layer. The theorem remains axiomatic, but it is now documented as a pure graph statement pending a direct proof; the review landed in `1774828`. | Start `DA-P4-007` (reassess `LocalMarkov` proof path). |
| 2026-05-19 | DA-P4-007 local Markov path assessment | Complete | Reassessed `LocalMarkov` after the semi-graphoid work landed. The forward-prefix descendant helpers added for `DSep_WeakUnion` already cover the “all-forward implies descendant” side; the missing proof work is now sharply scoped to a first-step case split and a first forward-to-backward pivot lemma showing descendants of `v` cannot open through `Parents(v)` in a DAG. The assessment landed in `80dcddd`. | If staying in P4, start `DA-P4-008` (split `LocalMarkov` into helper lemmas). |
| 2026-05-19 | DA-P4-008 local Markov proof | Complete | Finished the `LocalMarkov` burn-down in two slices: `5b8559c` restated the theorem target and added the first entry helpers, and `17ac9a5` added the forward-to-backward pivot blocker, descendant-parent disjointness, topological-order bridge lemmas, and the final proof body. The full stack stayed green and the declaration-axiom count dropped to 70. | Decide whether to attack `DSep_Intersection` directly or move to the Phase 5 probability-boundary decision. |
| 2026-05-20 | DA-P4-009 d-separation intersection proof | Complete | Replaced `DSep_Intersection` with a direct graph proof: from any `W`-unblocked trail to `Z` or `Z'`, the corresponding premise forces the first newly blocked non-collider from the opposite set under added conditioning, yielding a strictly shorter `W`-unblocked prefix into that opposite endpoint set. Alternating this descent contradicts finite trail length. The full stack stayed green and the batch later committed as `31d624b`, dropping the declaration-axiom count to 69. | Start `DA-P5-001` (decide the probability finite-sum proof path). |

## Phase 5 Batch Notes

| Date | Batch | Status | Notes | Next recommended step |
| --- | --- | --- | --- | --- |
| 2026-05-21 | DA-P5-001 probability finite-sum decision | Complete | Recorded the decision in `docs/plans/2026-05-21-dafny-probability-finite-sum-decision.md`: make `ProbEvent` concrete as a finite sum over `A * p.Keys`, keep the Kolmogorov axioms explicit in the first slice, and defer the PMF outcome vs node-assignment alignment before constructor-level probability work. | Start the finite-support summation helper and concrete `ProbEvent` batch. |
| 2026-05-21 | DA-P5-002 finite-support event sums and derived laws | Complete | Added the axiomatic `SetToSequence` bridge, `SumOutcomeMasses`, and `FiniteSupportSum`; made `ProbEvent` concrete over `A * p.Keys`; and proved `ComplementRule`, `InclusionExclusion`, and `TotalProbability` from `Axiom_Additivity`, `Axiom_Normalization`, and `ChainRule`. The full stack stayed green, the declaration-axiom count dropped to 67, and the batch was committed as `c914de5`. | Start `DA-P5-003` on PMF outcome versus assignment alignment. |
| 2026-05-21 | DA-P5-003 PMF outcome versus assignment alignment note | Complete | Recorded `docs/plans/2026-05-21-dafny-probability-assignment-alignment.md`, which sharpens the semantic split between node values and joint outcomes, rejects an immediate in-place reinterpretation of `Prob.PMF`, and scopes the translation-layer prototype path; the note was committed as `87143dd`. | Start the narrow bridge prototype in `DA-P5-004`. |
| 2026-05-21 | DA-P5-004 assignment-event grounding | Complete | Added `Value`, axiomatic `OutcomeToAssignment`, and support-bounded `AssignmentEvent` in `interventional.dfy`; used them to define `IntProbConcrete` as a real conditional probability on the truncated PMF; and replaced the narrative `IntProb_Grounded` axiom with a checked equality contract. The full stack stayed green, the declaration-axiom count dropped to 66, and the batch was committed as `c148a9c`. | Extend the same bridge to `TruncatePMF`, `Marginalize`, or `ProductPMF`. |
| 2026-05-21 | DA-P5-005 assignment-event algebra and truncated-support facts | Complete | Added `CompatibleAssignments`, `ConflictingAssignments`, `ExtendsAssignment`, `MergeAssignments`, and the reusable event-algebra lemmas for compatible intersection, conflicting intersection, and strengthening. Then sharpened `TruncatePMF` with support-matching contracts and proved `TruncatePMF_InterventionProbabilityOne` plus `TruncatePMF_ConflictingAssignmentZero`. The full stack stayed green and the axiom count stayed flat at 66 declaration axioms. | Package this batch or carry the same algebra forward into `Marginalize`. |

## Next Update Checklist

When starting a new implementation batch, update:

1. Active sprint board status.
2. Commit queue row for the intended commit.
3. Axiom count snapshot after verification.
4. Verification log with exact command and result.
5. Commit ledger after the commit lands.