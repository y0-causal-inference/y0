# Dafny TruncatePMF Factor-Level Design

Date: 2026-05-21

Related documents:

- `docs/plans/2026-05-19-dafny-de-axiomitization-progress.md`
- `docs/plans/2026-05-19-dafny-axiom-ledger.md`
- `docs/plans/2026-05-21-dafny-probability-assignment-alignment.md`
- `docs/plans/2026-05-22-dafny-proof-dependency-dags.md`

This note records why the current support-filtering probe should stop short of
becoming the public semantics of `TruncatePMF`, and sketches the factor-level
construction that better matches Pearl's truncated factorization.

## Why Stop The Conditioning Probe

The recent probe established a useful negative result.

The helper that:

1. keeps only outcomes matching `xVals`, and
2. renormalizes by `AssignmentProb(p, G, xVals)`

is locally definable, and some support-level properties verify cleanly.

However, it is still the wrong public target for `TruncatePMF`.

The reasons are semantic, not just technical:

1. it hardens an observational-conditioning reading of `do(X = xVals)`;
2. it requires a positivity premise `AssignmentProb(p, G, xVals) > 0.0` that
   the current `TruncatePMF` interface does not require; and
3. it does not explain the mutilated-graph semantics directly.

That probe was therefore useful as a discriminator, but it should remain an
auxiliary experiment rather than the next public constructor definition.

## Target Construction

The better-matched construction is factor-level rather than support-level.

For a full assignment `a : Assignment` over `Nodes(G)`, define a truncated
local factor at node `v` by:

1. if `v in X` and `a[v] == xVals[v]`, return `1.0`;
2. if `v in X` and `a[v] != xVals[v]`, return `0.0`;
3. if `v !in X`, return `ConditionalFactor(p, v, Parents(G, v), a)`.

Then define the interventional assignment mass as the product of those local
factors over a topological order of `G`.

In pseudocode:

```text
TruncatedLocalFactor(G, p, X, xVals, a, v) =
  if v in X then
    if a[v] == xVals[v] then 1.0 else 0.0
  else
    ConditionalFactor(p, v, Parents(G, v), a)

TruncatedAssignmentMass(G, p, X, xVals, a, topo) =
  product_{v in topo} TruncatedLocalFactor(G, p, X, xVals, a, v)
```

This matches the intended causal semantics more closely:

1. intervened nodes stop reading parent conditionals;
2. non-intervened nodes keep their original conditional factors; and
3. the resulting distribution is aligned with `RemoveIncoming(G, X)` rather
   than with observational conditioning on the event `X = xVals`.

## Why This Better Matches The Current Surface

This design fits the abstractions already present in `interventional.dfy`:

1. `ConditionalFactor` already names the local probability object the
   truncated formula should manipulate;
2. `MarkovFactorization` already states that the observational joint is built
   from such local factors; and
3. `TruncatePMF_Markov` is already phrased in terms of the mutilated graph
   `RemoveIncoming(G, X)`.

The missing piece is representation. The current bridge goes from outcome to
assignment, but not from assignment back to a canonical outcome. That means a
true factor-level constructor naturally wants to live over full assignments,
not over the observational support `p.Keys` alone.

## Expected Proof Payoff

If this route works, several current goals line up with the semantics more
cleanly:

1. `TruncatePMF_InterventionProbabilityOne` becomes structural, because the
   intervened-node factors force agreement with `xVals`;
2. `TruncatePMF_Empty` becomes the theorem that an empty intervention recovers
   the original factorization;
3. `TruncatePMF_Markov` matches the constructor by design, instead of being a
   separate semantic jump; and
4. `TruncatePMF_IsDistribution` becomes a factor-normalization theorem rather
   than a renormalization theorem.

## Prerequisites For The First Code Slice

The next code batch should stay narrow and add only the scaffolding required by
the factor-level route:

1. a finite product helper over a topological order;
2. a ghost helper such as `TruncatedLocalFactor`;
3. a ghost helper such as `TruncatedAssignmentMass`; and
4. no replacement of the public `TruncatePMF` boundary yet.

The bigger unresolved obligations should stay explicit:

1. a normalization law for `ConditionalFactor` over child values;
2. a locality law showing `ConditionalFactor` depends only on the node and its
   parent coordinates;
3. a representation choice between assignment-keyed PMFs and a trusted
   assignment-to-outcome bridge.

## Immediate Recommendation

Record the conditioning probe as a useful rejected move, and make the next
substantive proof batch a factor-level scaffolding batch rather than another
support-filtering batch.

That updates the operational recommendation in
`2026-05-22-dafny-proof-dependency-dags.md`: support filtering was a useful
local test, but it should not be treated as the long-term public semantics of
`TruncatePMF`.