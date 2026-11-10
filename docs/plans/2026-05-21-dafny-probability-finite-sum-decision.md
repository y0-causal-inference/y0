# Dafny Probability Finite-Sum Decision

Date: 2026-05-21

Related documents:

- `docs/plans/2026-05-19-dafny-de-axiomitization-plan.md`
- `docs/plans/2026-05-19-dafny-de-axiomitization-progress.md`
- `docs/plans/2026-05-19-dafny-axiom-ledger.md`

This note resolves `DA-P5-001`: whether to keep `ProbEvent` abstract or to
choose a concrete finite-sum design for the probability layer.

## Decision

Adopt a concrete finite-support design for `ProbEvent`.

Specifically:

1. Keep `Outcome` abstract and keep `PMF = map<Outcome, real>`. The finite
   support remains `p.Keys`.
2. Define `ProbEvent(p, A)` as a finite sum over `A * p.Keys`, not as an
   additional trusted algebraic primitive.
3. Introduce a reusable ghost finite-support summation helper as the enabling
   substrate for `ProbEvent`, derived probability laws, and later PMF
   constructors.
4. Keep `SumsToOne` and the three Kolmogorov axioms explicit in the first P5
   slice. The immediate goal is to de-axiomatize derived laws and build usable
   PMF constructors, not to force a full refactor of foundational assumptions in
   the same batch.

## Why This Direction

1. The probability module already presents PMFs as concrete finite maps. The
   missing piece is finite summation, not a missing representation for events.
2. Keeping `ProbEvent` abstract and adding more axioms would widen the trust
   boundary exactly where the current P5 work can instead make it smaller.
3. A concrete `ProbEvent` directly unlocks the planned derived-law burn-down:
   `ComplementRule`, `InclusionExclusion`, and `TotalProbability`.
4. No reusable finite sum helper exists in the current Dafny stack, so this is
   the right first implementation slice.

## Boundary Of This Decision

This decision does **not** mean the whole probability/interventional layer is
ready to become concrete in one jump.

Two representation issues remain outside `DA-P5-001`:

1. `Interventional.Assignment = map<Node, Prob.Outcome>` treats `Outcome` as a
   per-node value, while `Prob.PMF = map<Outcome, real>` currently treats it as
   the sample-space point. `ProbEvent` can be made concrete without resolving
   that mismatch, but `IntProbConcrete`, `Marginalize`, and `TruncatePMF` need a
   later representation decision.
2. `ProductPMF(ps): PMF` is still under-specified until the project decides
   whether factors live on a shared assignment space or on tuple/product sample
   spaces. This is a constructor-level design issue, not a reason to leave
   `ProbEvent` abstract.

## Chosen Technical Shape

The first concrete implementation slice should be:

1. Add a ghost finite-support fold/summation helper over finite sets or over a
   duplicate-free sequence extracted from a finite set.
2. Define `ProbEvent(p, A)` concretely as the sum of `p[omega]` over
   `omega in A * p.Keys`.
3. Leave `IsDistribution(p)` as `AllNonNeg(p) && SumsToOne(p)` for this first
   slice, so the batch does not simultaneously rewrite the foundational
   distribution invariant.
4. Use the new `ProbEvent` definition to prove `ComplementRule`, then
   `InclusionExclusion`, then `TotalProbability`.

## Immediate Next Step

The next implementation batch after this decision should be:

1. Add the reusable finite-support summation helper.
2. Replace the abstract `ProbEvent` declaration with a concrete definition over
   `A * p.Keys`.
3. Keep the foundational Kolmogorov axioms explicit for now.
4. Prove the three derived laws listed above before attempting constructor-level
   probability work such as `ProductPMF`, `Marginalize`, or `TruncatePMF`.

That sequencing keeps the trust boundary shrinking where the current model is
already concrete, while deferring the separate assignment/sample-space design
questions to the later interventional and do-calculus layers.
