# Dafny Axiom Ledger

Date opened: 2026-05-19

Related documents:

- `docs/plans/2026-05-19-dafny-de-axiomitization-plan.md`
- `docs/plans/2026-05-19-dafny-de-axiomitization-progress.md`

This ledger is the Phase 0 inventory for declaration-level `{:axiom}` sitYes es in
`src/dafny/*.dfy`. It intentionally excludes local `assume {:axiom}` sites,
which are tracked separately as local proof gaps in the progress tracker.

Baseline: 80 declaration axioms at commit `ece2091`.

## Category Key

| Category | Meaning | Default action |
| --- | --- | --- |
| Foundational | Establishes the current abstract mathematical universe. | Keep explicit unless replacing the abstraction. |
| Abstract interface | Names an object whose representation is intentionally absent. | Keep until a concrete definition is chosen. |
| Derivable wrapper | Should follow from existing or planned lower-level definitions. | Prove when prerequisites are available. |
| Infrastructure equivalence | Connects executable/compiled helpers to ghost specs. | Prove with helper lemmas. |
| Local proof target | Looks tractable without changing major abstractions. | Prioritize early. |
| Deep theorem | Requires substantial semantic or inductive proof infrastructure. | Defer until dependencies land. |
| Contract gap | A lemma currently has no formal `ensures`; its axiom annotation proves no useful postcondition. | Add a real contract or demote to documentation/comment. |

## Phase 0 Findings

1. The ledger confirms 80 declaration axioms across six Dafny files.
2. Twelve axiom lemmas currently have no formal `ensures` clause. These are
   contract gaps, not proof obligations yet.
3. The best Phase 1 starting points remain the low-risk local proof targets in
   `dag.dfy` and `semi_markovian.dfy`; no probability or do-calculus semantic
   redesign is needed before starting those.

## Contract Gaps Found In Phase 0

These lemmas should not be treated as successful formal statements until they
receive real postconditions or are demoted to comments.

| Symbol | Location | Decision |
| --- | --- | --- |
| `Lemma2_CComponentFactorization` | `src/dafny/identification.dfy:176` | Demote to documentation or restate after `CComponentFactorization` has a formal product contract. |
| `ID_Line2` | `src/dafny/identification.dfy:539` | Add a formal recursive-result equality/bridge contract in P7; until then it is documentation. |
| `ID_Line7` | `src/dafny/identification.dfy:723` | Add a formal recursive-result equality/bridge contract in P7; until then it is documentation. |
| `Line1_Uses_Rule3` | `src/dafny/identification.dfy:920` | Demote to comment unless a do-calculus derivation object is introduced. |
| `Line2_Uses_Rule1` | `src/dafny/identification.dfy:935` | Demote to comment unless a do-calculus derivation object is introduced. |
| `Line3_Uses_Rule3` | `src/dafny/identification.dfy:952` | Demote to comment unless a do-calculus derivation object is introduced. |
| `Line4_Uses_Rules2and3` | `src/dafny/identification.dfy:967` | Demote to comment unless a do-calculus derivation object is introduced. |
| `IntProb_Grounded` | `src/dafny/interventional.dfy:181` | Add a pointwise grounding postcondition once events/assignments are represented formally enough. |
| `GlobalMarkov_From_Factorization` | `src/dafny/interventional.dfy:208` | Add a conditional-independence postcondition or keep as an explicitly documented deep theorem boundary. |
| `CComponentFactorization` | `src/dafny/semi_markovian.dfy:466` | Add a product-over-components contract after product/set-sum infrastructure exists; otherwise demote narrative. |
| `InterventionalFactorization` | `src/dafny/semi_markovian.dfy:497` | Add a product-over-components contract after product/set-sum infrastructure exists; otherwise demote narrative. |
| `QValue_Nested` | `src/dafny/semi_markovian.dfy:531` | Add a computability/equality contract for nested Q-values after Q semantics is concrete. |

## File Ledger

### `src/dafny/probability.dfy`

| Symbol | Line | Category | Phase | Reason / next action |
| --- | ---: | --- | --- | --- |
| `SetToSequence` | 60 | Abstract interface | P5 | Finite-support enumeration bridge for abstract unordered `Outcome`; kept axiomatic because functions cannot use nondeterministic set choice and the current model has no canonical outcome ordering. |
| `Axiom_NonNegativity` | 103 | Foundational | Keep | Kept explicit after concretizing `ProbEvent`; foundational probability boundary in the first P5 slice. |
| `Axiom_Normalization` | 109 | Foundational | Keep | Kept explicit after concretizing `ProbEvent`; foundational probability boundary in the first P5 slice. |
| `Axiom_Additivity` | 115 | Foundational | Keep | Kept explicit after concretizing `ProbEvent`; foundational probability boundary in the first P5 slice. |
| `ComplementRule` | 126 | Derivable wrapper | Done | Proved in the 2026-05-21 working tree from `Axiom_Additivity` and `Axiom_Normalization` after defining concrete `ProbEvent` over `A * p.Keys`. |
| `InclusionExclusion` | 198 | Derivable wrapper | Done | Proved in the 2026-05-21 working tree by isolating the `(A - B) * p.Keys` slice, applying finite additivity to `A` and `A + B`, and cancelling the shared contribution. |
| `TotalProbability` | 317 | Derivable wrapper | Done | Proved in the 2026-05-21 working tree by partitioning `A * p.Keys` into `A * B1` and `A * B2`, applying finite additivity, and rewriting the two joint terms with `ChainRule`. |
| `ProductPMF` | 408 | Abstract interface | P5 | Needs a concrete product-of-PMFs representation before proof work. |
| `ProductPMF_IsDistribution` | 410 | Derivable wrapper | P5 | Prove once `ProductPMF` is concrete. |

### `src/dafny/dag.dfy`

| Symbol | Line | Category | Phase | Reason / next action |
| --- | ---: | --- | --- | --- |
| `KahnsAlgorithm_Correct` | 158 | Local proof target | Done | Proved in `DA-P1-001` and committed as `1e46df7`; current contract remains the thin wrapper around `IsDAG`. |
| `AncestorsCompiled_Correct` | 261 | Infrastructure equivalence | P3 | Needs BFS reachability equivalence to bounded ancestry. |
| `DescendantsCompiled_Correct` | 264 | Infrastructure equivalence | P3 | Needs BFS reachability equivalence to bounded ancestry. |
| `RemoveIncomingCompiled_Correct` | 285 | Local proof target | Done | Proved in `DA-P1-002` and committed as `0e6fa23`. |
| `RemoveOutgoingCompiled_Correct` | 288 | Local proof target | Done | Proved in `DA-P1-002` and committed as `0e6fa23`. |
| `DSep_Symmetry` | 433 | Infrastructure equivalence | Done | Proved in `DA-P4-003` and committed as `29c5895`; the proof reverses connected trails and mirrors blocking witnesses with `TrailBlockedAtPos`. |
| `DSep_Decomposition` | 438 | Local proof target | Done | Proved in `DA-P4-001` and committed as `5a4c783`; the current proof is the direct subset step from `z in Z` to `z in Z + Z'`. |
| `DSep_WeakUnion` | 445 | Deep theorem | Done | Proved in `DA-P4-004` and committed as `4c1fbd6` after revising `TrailBlocked` so only internal nodes can block a trail. |
| `DSep_Contraction` | 452 | Deep theorem | Done | Proved in `DA-P4-005` and committed as `f255893`; the proof reduces new blocking under `W + Z'` to an unblocked prefix trail ending in a node of `Z'`. |
| `DSep_Intersection` | 463 | Deep theorem | Done | Proved in the 2026-05-20 working tree by alternating the two premises along the first blocker introduced by extra conditioning: each new non-collider blocker from the opposite set yields a strictly shorter `W`-unblocked prefix into that set, contradicting finite trail length. |
| `LocalMarkov` | 483 | Deep theorem | Done | Proved in `DA-P4-008` and committed as `17ac9a5`. The final proof uses the corrected target `NonDescendants(G, v) - Parents(G, v)`, an all-forward-to-descendant bridge, immediate parent-entry blocking, a first forward-to-backward pivot blocker, and a descendant-parent disjointness fact for DAGs. |

### `src/dafny/interventional.dfy`

| Symbol | Line | Category | Phase | Reason / next action |
| --- | ---: | --- | --- | --- |
| `ConditionalFactor` | 56 | Abstract interface | P5 | Needs concrete conditional factor representation. |
| `ConditionalFactor_NonNeg` | 64 | Derivable wrapper | P5 | Prove once `ConditionalFactor` is concrete. |
| `MarkovFactorization` | 87 | Abstract interface | P5 | Current graph-to-PMF semantic relation is intentionally abstract. |
| `MarkovFactorization_IsDistribution` | 90 | Derivable wrapper | P5 | Prove once `MarkovFactorization` has concrete semantics or keep as interface invariant. |
| `TruncatePMF` | 111 | Abstract interface | P5 | Needs concrete do-intervention PMF construction. |
| `TruncatePMF_IsDistribution` | 121 | Derivable wrapper | P5 | Prove once `TruncatePMF` is concrete. |
| `TruncatePMF_Empty` | 130 | Derivable wrapper | P5 | Prove once `TruncatePMF` is concrete. |
| `TruncatePMF_Markov` | 136 | Deep theorem | P5 | Needs Markov-factorization and graph-surgery semantics. |
| `IntProbConcrete` | 163 | Abstract interface | P6 | Needs event/assignment interpretation to become concrete. |
| `IntProb_Grounded` | 181 | Contract gap | P6 | Add formal pointwise equality; currently narrative only. |
| `GlobalMarkov_From_Factorization` | 208 | Contract gap | P6 | Intended deep theorem, but currently lacks a formal postcondition. |
| `Marginalize` | 230 | Abstract interface | P5 | Needs concrete summation over variables. |
| `Marginalize_IsDistribution` | 232 | Derivable wrapper | P5 | Prove once `Marginalize` is concrete. |

### `src/dafny/do_calculus.dfy`

| Symbol | Line | Category | Phase | Reason / next action |
| --- | ---: | --- | --- | --- |
| `IntProb` | 58 | Abstract interface | P6 | Main do-calculus distribution interface; decide whether it wraps concrete semantics. |
| `GlobalMarkov` | 80 | Derivable wrapper | P6 | Should delegate to formal `GlobalMarkov_From_Factorization` after witness/contract issues are solved. |
| `InterventionSemantics` | 103 | Derivable wrapper | P6 | Should delegate to `TruncatePMF` semantics after `IntProb` grounding is solved. |
| `Rule1_InsertDeleteObservation` | 137 | Deep theorem | P6 | Requires Global Markov, intervention semantics, and checked conditioning-set shape. |
| `Rule2_ActionObservationExchange` | 151 | Deep theorem | P6 | Requires graph-surgery and do-calculus semantics. |
| `Rule3_InsertDeleteAction` | 168 | Deep theorem | P6 | Requires graph-surgery and do-calculus semantics. |
| `BackdoorAdjustment` | 208 | Deep theorem | P6 | Prove after Rule 2/3 and adjustment-specific graph lemmas. |
| `FrontdoorCriterion` | 240 | Deep theorem | P6 | Prove after Rule 2/3 and frontdoor graph lemmas. |

### `src/dafny/semi_markovian.dfy`

| Symbol | Line | Category | Phase | Reason / next action |
| --- | ---: | --- | --- | --- |
| `CComponents_Partition` | 160 | Infrastructure equivalence | P3 | Needs proof from ghost set-comprehension or compiled BFS correctness. |
| `BidirectedBFS_ContainsSelf` | 228 | Local proof target | Done | Proved in `DA-P1-003` via the general `BidirectedBFS_FrontierSubset` helper and committed as `b8b2c03`. |
| `CComponentCompiled_Correct` | 296 | Infrastructure equivalence | P3 | Needs bidirected BFS soundness/completeness. |
| `TopoPredecessors` | 385 | Abstract interface | P5/P7 | Needs concrete topological-prefix definition before Q-value proofs. |
| `QValue` | 416 | Abstract interface | P5/P7 | Core c-factor object; keep abstract until product/conditional semantics are concrete. |
| `QValue_AllNodes` | 427 | Derivable wrapper | P5/P7 | Prove once `QValue` semantics are concrete. |
| `QValue_IsDistribution` | 438 | Derivable wrapper | P5/P7 | Prove once `QValue` semantics are concrete. |
| `CComponentFactorization` | 466 | Contract gap | P5/P7 | Add formal product-over-components postcondition or demote narrative. |
| `InterventionalFactorization` | 497 | Contract gap | P5/P7 | Add formal product-over-components postcondition or demote narrative. |
| `QValue_Nested` | 531 | Contract gap | P5/P7 | Add formal nested-Q contract or demote narrative. |

### `src/dafny/identification.dfy`

| Symbol | Line | Category | Phase | Reason / next action |
| --- | ---: | --- | --- | --- |
| `Lemma1_NonIdentifiabilityWitness` | 123 | Deep theorem | P7 | Requires two-model semantic construction. |
| `Lemma2_CComponentFactorization` | 176 | Contract gap | P0/P7 | Demote or restate after semi-Markovian factorization has formal content. |
| `Lemma3_QValueDerivation` | 217 | Derivable wrapper | P1/P7 | Likely follows from `QValue_IsDistribution` once preconditions are exposed. |
| `SetOfSetsToSeq` | 257 | Abstract interface | P3/P7 | Sequence choice over sets; keep until deterministic ordering bridge exists. |
| `IDLine4ComponentsReady` | 281 | Local proof target | P1 | Should follow from `CComponentsWithout_Partition` and set-to-sequence membership. |
| `ID_Line1` | 509 | Derivable wrapper | P7 | Prove from concrete ID control flow after fuel/empty-graph edge cases are checked. |
| `ID_Line2` | 539 | Contract gap | P7 | Needs formal recursive equality/bridge postcondition. |
| `ID_Line3` | 574 | Deep theorem | P7 | Has equality contract; needs do-calculus/ancestry support. |
| `ID_Line4` | 620 | Deep theorem | P7 | Requires Line 4 recursive products and C-component factorization. |
| `ID_Line5` | 653 | Deep theorem | P7 | Requires hedge construction from single C-component conditions. |
| `ID_Line6` | 688 | Deep theorem | P7 | Requires Q-value semantics. |
| `ID_Line7` | 723 | Contract gap | P7 | Needs formal recursive equality/bridge postcondition. |
| `Theorem2_Soundness` | 769 | Deep theorem | P7 | Requires line-by-line induction over ID recursion. |
| `Theorem3_Completeness` | 817 | Deep theorem | P7 | Requires hedge completeness and ID recursion proof. |
| `Theorem3_HedgeIFF` | 832 | Derivable wrapper | P7 | Prove from `Theorem3_Completeness` after it exists. |
| `Theorem4_DoCalculusCompleteness` | 888 | Deep theorem | P7 | Requires do-calculus derivation semantics and ID soundness/completeness. |
| `Line1_Uses_Rule3` | 920 | Contract gap | P0/P6 | Demote or introduce a formal derivation object. |
| `Line2_Uses_Rule1` | 935 | Contract gap | P0/P6 | Demote or introduce a formal derivation object. |
| `Line3_Uses_Rule3` | 952 | Contract gap | P0/P6 | Demote or introduce a formal derivation object. |
| `Line4_Uses_Rules2and3` | 967 | Contract gap | P0/P6 | Demote or introduce a formal derivation object. |
| `Theorem5_AllIdentifiable` | 1030 | Deep theorem | P7 | Requires no-bidirected-to-child characterization proof. |
| `TianID` | 1049 | Abstract interface | P7 | Keep abstract unless Tian algorithm is modeled concretely. |
| `Corollary3_TianComplete` | 1065 | Deep theorem | P7 | Requires equivalence between Tian ID and ID. |
| `Frontdoor_Identifiable` | 1288 | Deep theorem | P7 | Universal no-hedge proof or ID completeness path; not a P2 target. |
| `Markovian_AllIdentifiable` | 1336 | Derivable wrapper | P7 | Prove from `Theorem5_AllIdentifiable` or direct no-hedge proof. |
| `Figure1a_Identifiable` | 1366 | Deep theorem | P7 | Requires no-hedge proof or ID completeness path. |
| `Figure1b_NotIdentifiable` | 1399 | Local proof target | Done | Proved in `DA-P2-003` by an explicit hedge witness on the `{2,3}` subgraph of `Figure1bGraph`; committed as `6a89013`. |
| `MarkovianCompleteness` | 1418 | Derivable wrapper | P7 | Prove from `Theorem5_AllIdentifiable` or direct no-hedge proof. |
| `IdentifiabilityMonotoneBidirected` | 1431 | Derivable wrapper | P2/P7 | Likely contradiction proof from hedge preservation under bidirected-edge removal. |
| `IdentifiedIsDistribution` | 1449 | Derivable wrapper | P7 | Prove from line soundness or full ID soundness. |

## Ready For Phase 1

Phase 0 is complete when this ledger and the progress tracker are checked in.
The recommended Phase 1 order is:

1. `RemoveIncomingCompiled_Correct` and `RemoveOutgoingCompiled_Correct`.
2. `KahnsAlgorithm_Correct` with the current weak contract.
3. `BidirectedBFS_ContainsSelf`.
4. `CComponent_Connected` local assume removal.
5. `IDLine4ComponentsReady`, if `CComponentsWithout_Partition` is already
   strong enough; otherwise defer it to P3.

The first Phase 1 commit message should likely be:

```text
proof(dafny): prove graph surgery compiled equivalence
```