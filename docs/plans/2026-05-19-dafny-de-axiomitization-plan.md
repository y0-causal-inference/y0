# Dafny De-Axiomitization Plan

Date: 2026-05-19

## Goal

Reduce `{:axiom}` and `assume {:axiom}` usage where doing so increases trust in
the Dafny development, while keeping honest foundational assumptions at explicit
trust boundaries. The aim is not to minimize the raw axiom count at all costs;
the aim is to replace tractable assumptions with proofs, demote non-formal
documentation lemmas, and leave genuinely foundational mathematical interfaces
clearly labeled.

## Current Inventory

Read-only count from `src/dafny/*.dfy` on 2026-05-19:

| File | Total `{:axiom}` sites | Local `assume {:axiom}` | Declaration axioms |
| --- | ---: | ---: | ---: |
| `src/dafny/dag.dfy` | 13 | 2 | 11 |
| `src/dafny/do_calculus.dfy` | 8 | 0 | 8 |
| `src/dafny/identification.dfy` | 48 | 18 | 30 |
| `src/dafny/interventional.dfy` | 13 | 0 | 13 |
| `src/dafny/probability.dfy` | 8 | 0 | 8 |
| `src/dafny/semi_markovian.dfy` | 13 | 3 | 10 |
| **Total** | **103** | **23** | **80** |

The axiom-free executable extraction files are not the main target of this plan;
the theorem/specification stack is.

## Triage Categories

Use these categories before touching any axiom:

| Category | Meaning | Default action |
| --- | --- | --- |
| Foundational | Defines the intended mathematical universe, e.g., Kolmogorov probability axioms under the current abstract `ProbEvent` interface. | Keep, label, and test against Python where possible. |
| Abstract interface | Introduces an object whose concrete representation is intentionally absent, e.g., `IntProb`, `MarkovFactorization`, `QValue`, `TianID`. | Keep until a concrete definition is introduced. |
| Derivable wrapper | Follows from lower-level axioms/theorems already accepted, e.g., result well-formedness or theorem corollaries. | Prove with a small body. |
| Local proof gap | A body uses `assume {:axiom}` for a finite graph, set partition, subgraph, or sequence fact. | Prioritize; these are usually tractable. |
| Infrastructure equivalence | Connects executable/compiled helpers to ghost specs, e.g., BFS reachability or C-component computation. | Prove after smaller helper lemmas exist. |
| Deep theorem | Requires induction over ID recursion, Bayes-ball/global Markov, or C-factor semantics. | Defer until dependencies are proved. |
| Documentation-only | A lemma has no formal `ensures`, so `{:axiom}` adds no mathematical content. | Add a real postcondition or convert to a non-axiom lemma/comment. |

## Dependency Shape

The proof stack is layered:

```text
Probability
  -> DAG
  -> Interventional
  -> DoCalculus
  -> SemiMarkovian
  -> Identification
```

But the de-axiomitization order should not be purely bottom-up. The best early
work is the tractable local proof debt that unlocks many callers without taking
on global semantics.

Critical dependencies:

- C-component proofs in `semi_markovian.dfy` unlock hedge examples, ID Line 4
  readiness, and later Line 5 proofs.
- DAG surgery and subgraph/topological-sort preservation unlock `ValidQuery`
  assumptions in recursive ID calls.
- PMF summation/product/marginalization definitions unlock probability-derived
  lemmas, `TruncatePMF`, and ID result distribution proofs.
- `GlobalMarkov_From_Factorization` and the d-separation semi-graphoid lemmas
  are prerequisites for de-axiomitizing do-calculus rules in earnest.
- ID soundness/completeness theorems should wait until the ID line lemmas and
  recursive precondition bridge lemmas are in place.

## Priority Backlog

### P0: Make The Axiom Ledger Honest

Effort: S. Dependencies: none.

1. Tag remaining declaration axioms by category in comments or a generated
   ledger.
2. Find lemmas with no `ensures` clauses and either give them formal content or
   remove the axiom annotation by adding an empty body.
3. Do not count foundational axioms as burn-down debt unless a concrete
   replacement representation is planned.

Likely documentation-only or near-documentation items include narrative lemmas
around C-component factorization, Q-value nesting, and the `Line*_Uses_*`
do-calculus correspondence lemmas in `identification.dfy`.

Exit criterion: every remaining axiom has an explicit category and reason.

### P1: Low-Risk Local Proofs

Effort: S to M. Dependencies: none or existing definitions only.

| Target | File | Tractability | Why first |
| --- | --- | --- | --- |
| `KahnsAlgorithm_Correct` as currently stated | `src/dafny/dag.dfy` | S | The current postcondition is essentially the definition of `IsDAG`; it does not prove full Kahn correctness. |
| `RemoveIncomingCompiled_Correct` | `src/dafny/dag.dfy` | S | Compiled and ghost definitions are identical. |
| `RemoveOutgoingCompiled_Correct` | `src/dafny/dag.dfy` | S | Same as above. |
| `BidirectedBFS_ContainsSelf` | `src/dafny/semi_markovian.dfy` | S | Should follow from one unfold plus `BidirectedBFS_VisitedSubset`. |
| `CComponent_Connected` body-local assume | `src/dafny/semi_markovian.dfy` | S/M | Should follow directly from the `CComponents` set-comprehension definition. |
| `IDLine4ComponentsReady` | `src/dafny/identification.dfy` | M | Should follow from `CComponentsWithout_Partition` plus `SetOfSetsToSeq` membership. |

Exit criterion: remove at least 5 low-risk axiom sites with no public contract
changes and full Dafny verification passing.

### P2: Concrete Examples And Hedge Witnesses

Effort: M. Dependencies: C-component singleton helper lemmas.

1. Add generic helper lemmas for one-node and two-node single C-components, so
   concrete hedge proofs do not need local `assume {:axiom}` for
   `|CComponents(F)| == 1`.
2. Remove the two local C-component assumes inside `BowArc_NotIdentifiable`.
3. Prove `Figure1b_NotIdentifiable` by explicit hedge construction.
4. Add `Figure1a_WellFormed` and `Figure1b_WellFormed` if they are still useful
   as standalone graph certificates.

Do not prioritize `Frontdoor_Identifiable` or `Figure1a_Identifiable` in this
phase. They require universal no-hedge reasoning or a proved ID completeness
path, which is much harder than constructing a finite hedge witness.

Exit criterion: example non-identifiability proofs are proof bodies, not axiom
declarations or local assumes.

### P3: SMGraph And C-Component Infrastructure

Effort: M to L. Dependencies: P1 helpers.

| Target | Tractability | Notes |
| --- | --- | --- |
| `CComponentsWithout_Partition` local `WellFormedSM(smX)` assume | M | Needs `RemoveNodesSM` preserves well-formedness and a subgraph/topological-sort preservation lemma. |
| `CComponents_Partition` | L | Either prove from the ghost set-comprehension or first prove `ComputeCComponents` BFS correctness. |
| `CComponentCompiled_Correct` | L | Needs bidirected BFS soundness and completeness. |
| `AncestorsCompiled_Correct` / `DescendantsCompiled_Correct` | L | Needs BFS reachability equivalence to the bounded recursive ancestor predicate. |

Suggested order:

1. Prove subgraph/topological-sort preservation for node deletion.
2. Prove `RemoveNodesSM` preserves `WellFormedSM` under a well-formed input.
3. Prove `CComponentsWithout_Partition` without the local assume.
4. Split bidirected BFS correctness into visited-subset, neighbor-step
   soundness, reachability completeness, then tackle compiled C-components.

Exit criterion: Line 4 and hedge proofs can rely on C-component facts without
ad hoc local assumes.

### P4: DAG D-Separation Algebra

Effort: M to XL. Dependencies: trail helper library.

Order by tractability:

1. `DSep_Decomposition`: likely easiest because it narrows `Z + Z'` to `Z`.
2. `DSep_Symmetry`: needs trail reversal definitions and proofs that validity,
   connection, collider status, and blocking are preserved under reversal.
3. `DSep_WeakUnion`: needs monotonic/conditioning-set reasoning for blocked
   trails; more subtle because colliders and non-colliders react differently to
   conditioning.
4. `DSep_Contraction`: depends on stronger trail decomposition reasoning.
5. `DSep_Intersection`: keep last; the comment already notes positivity-style
   assumptions, and the current pure graph statement may need careful review.
6. `LocalMarkov`: depends on substantial d-separation theory for DAGs.

Exit criterion: at least decomposition and symmetry are proved before attempting
do-calculus rule proofs.

### P5: Probability And Interventional Semantics

Effort: L to XL. Dependencies: concrete algebraic representations.

Keep the three Kolmogorov axioms as foundational under the current abstraction.
To prove more, first choose one of these designs:

1. Define `ProbEvent` as a finite sum over `PMF` support and prove derived laws.
2. Keep `ProbEvent` abstract and add exactly the missing algebraic axioms needed
   for derived laws, accepting a wider foundation.

Recommended tractable order if choosing concrete finite sums:

1. Define a reusable finite-map/set summation helper.
2. Prove `ComplementRule`, then `InclusionExclusion`, then `TotalProbability`.
3. Define `ProductPMF(ps)` over sequences and prove `ProductPMF_IsDistribution`.
4. Define `Marginalize` and prove `Marginalize_IsDistribution`.
5. Define `TruncatePMF`, then prove `TruncatePMF_Empty`,
   `TruncatePMF_IsDistribution`, and `TruncatePMF_Markov`.

Keep `GlobalMarkov_From_Factorization` as a long-horizon boundary unless the
team decides to formalize Bayes-ball/global Markov completeness in Dafny.

Exit criterion: PMF constructors used by ID have concrete definitions and basic
distribution preservation proofs.

### P6: Do-Calculus Layer

Effort: L to XL. Dependencies: P4 and P5.

Current `DoCalculus.IntProb` is abstract. Therefore the do-calculus rules cannot
be meaningfully proved until `IntProb` is either tied to `IntProbConcrete` or
replaced by a concrete interventional distribution interface.

Suggested order:

1. Decide whether `IntProb` remains abstract or becomes a wrapper around the
   concrete interventional semantics.
2. Prove `GlobalMarkov` and `InterventionSemantics` as wrappers only after the
   implicit PMF witness problem is solved.
3. Recheck the exact preconditions of Rule 1 against the available
   `GlobalMarkov` statement before attempting a proof; the current conditioning
   set shape deserves scrutiny.
4. Prove Rule 1, then Rule 3, then Rule 2.
5. Prove `BackdoorAdjustment` and `FrontdoorCriterion` from the rules.

Exit criterion: the three do-calculus rules are no longer axioms, or the layer
has a documented reason why they remain trusted mathematical boundaries.

### P7: Identification Theorems

Effort: XL. Dependencies: P3, P5, P6, and Q-value semantics.

Do not start with Theorem 2 or Theorem 3. Start by removing recursive bridge
assumptions and proving line-specific obligations.

Suggested order:

1. Prove subquery validity obligations currently assumed inside `IDImpl`:
   ancestral subgraphs, enlarged treatment sets, Line 4 subcalls, and Line 7
   subgraphs.
2. Prove distribution and Markov-factorization preservation obligations for
   `Marginalize`, `SubgraphSM`, and `QValue` inputs.
3. Prove line lemmas in this order: Line 1, Line 3, Line 2, Line 6, Line 7,
   Line 4, Line 5.
4. Prove `IdentifiedIsDistribution` as a wrapper once Theorem 2 or the line
   soundness lemmas are available.
5. Prove `Theorem3_HedgeIFF` as a wrapper from `Theorem3_Completeness`.
6. Prove `MarkovianCompleteness` and `Markovian_AllIdentifiable` as wrappers
   from `Theorem5_AllIdentifiable` only after the no-bidirected-to-child proof
   is automated.
7. Leave `Theorem2_Soundness`, `Theorem3_Completeness`,
   `Theorem4_DoCalculusCompleteness`, `Theorem5_AllIdentifiable`, and
   `Corollary3_TianComplete` until their supporting line and semantic lemmas
   exist.

Exit criterion: recursive `IDImpl` has no local `assume {:axiom}` precondition
bridges, and at least Lines 1, 3, and 6 have formal proof bodies.

## Recommended Near-Term Sprint

For the next 1-2 weeks, target this sequence:

1. P0 ledger cleanup and documentation-only axiom demotion.
2. P1 low-risk local proofs in `dag.dfy` and `semi_markovian.dfy`.
3. Generic finite C-component helper lemmas.
4. Remove local assumes from `BowArc_NotIdentifiable`.
5. Prove `Figure1b_NotIdentifiable` with an explicit hedge.
6. Prove `IDLine4ComponentsReady` once `CComponentsWithout_Partition` is strong
   enough.

This sequence gives visible burn-down without committing the project to the
much larger probability/global-Markov/do-calculus proof program.

## Verification Gates

After each batch, run from the repository root:

```bash
cd src/dafny && /opt/homebrew/bin/dafny verify \
  probability.dfy dag.dfy interventional.dfy do_calculus.dfy \
  semi_markovian.dfy identification.dfy
```

Track axiom count with:

```bash
rg -o "\{:axiom\}" src/dafny/*.dfy | wc -l
rg -o "assume\s+\{:axiom\}" src/dafny/*.dfy | wc -l
```

If executable extraction files or Python routing are touched, also run the
focused generated-ID parity and routing tests.

## Success Metrics

1. Every remaining axiom has a category and a reason.
2. Local `assume {:axiom}` count decreases monotonically.
3. No axiom is removed by weakening a theorem statement unless the change is
   explicitly documented as contract correction.
4. Concrete examples rely on explicit graph/hedge witnesses, not theorem-level
   axioms.
5. Deep semantic boundaries remain explicit until their prerequisite proof
   libraries exist.