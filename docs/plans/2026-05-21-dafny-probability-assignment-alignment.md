# Dafny Probability Assignment Alignment

Date: 2026-05-21

Related documents:

- `docs/plans/2026-05-19-dafny-de-axiomitization-progress.md`
- `docs/plans/2026-05-19-dafny-axiom-ledger.md`
- `docs/plans/2026-05-21-dafny-probability-finite-sum-decision.md`

This note starts `DA-P5-003`: resolving the representation mismatch between the
probability layer's PMF sample points and the interventional layer's node
assignments.

## Problem Statement

Today the same Dafny type, `Prob.Outcome`, is playing two incompatible roles:

1. In `probability.dfy`, `PMF = map<Outcome, real>` treats `Outcome` as a full
   sample-space point.
2. In `interventional.dfy`, `Assignment = map<Node, Prob.Outcome>` treats
   `Prob.Outcome` as the value taken by a single node.

Those two readings cannot both be the intended semantics for the same type. The
mismatch stayed harmless while `ProbEvent` was abstract, but it blocks any
concrete account of `ProductPMF`, `Marginalize`, `TruncatePMF`, and
`IntProbConcrete`.

## Local Evidence

The controlling code path is narrow:

1. `probability.dfy` defines `Outcome` and `PMF` concretely enough that event
   probabilities now sum over `p.Keys`.
2. `interventional.dfy` defines `Assignment = map<Node, Prob.Outcome>` and then
   uses `Prob.PMF` in `ConditionalFactor`, `TruncatePMF`, `IntProbConcrete`, and
   `Marginalize`.

The cheap disconfirming check was whether a direct `PMF` refactor would stay
local. It does not: `Prob.PMF` is threaded through `do_calculus.dfy`,
`semi_markovian.dfy`, and especially `identification.dfy`. Reinterpreting the
existing `Prob.PMF` type in place would therefore be a broad migration, not a
small constructor-level fix.

## Rejected Immediate Move

Reject the immediate rewrite "just make `Prob.PMF` assignment-keyed now".

That path is too wide for the next P5 slice because:

1. It would change the meaning of a type already used across the full oracle
   stack.
2. It would force a simultaneous redesign of event semantics in the probability
   layer.
3. It would mix a representation migration with the still-open constructor
   proofs for `ProductPMF`, `Marginalize`, `TruncatePMF`, and `IntProbConcrete`.

## Staged Hypothesis

The least disruptive next step is to introduce an explicit boundary between:

1. atomic node values, and
2. full sample-space points / world assignments.

In practice, that means the next design slice should choose one of these two
directions explicitly before more constructor work:

1. a shared joint sample-space representation, where PMFs are ultimately keyed
   by full assignments and node values are a separate type, or
2. a verified translation layer between abstract probability sample points and
   assignment-level objects used by the interventional layer.

Either direction requires a distinct notion of "joint outcome". The current
overloading of `Prob.Outcome` for both roles should not be extended further.

## Immediate Next Slice

The next implementation batch for `DA-P5-003` should stay design-scoped:

1. inventory the exact helper surface needed by `ProductPMF`, `Marginalize`,
   `TruncatePMF`, and `IntProbConcrete`;
2. decide whether the repo should standardize on assignment-keyed joint PMFs or
   on an explicit translation interface;
3. pick one narrow probe that tests the chosen boundary without rewriting the
   whole `Prob.PMF` stack.

The important outcome of this note is negative as well as positive: the repo now
has enough evidence to avoid an in-place reinterpretation of `Prob.PMF` as the
next move.
