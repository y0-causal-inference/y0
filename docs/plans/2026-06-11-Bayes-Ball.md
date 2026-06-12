# Bayes-Ball Proof Plan: Closing `GlobalMarkovConcrete`

Date: 2026-06-11

## Objective

Prove `GlobalMarkovConcrete` in `src/dafny/interventional.dfy` — the last
major axiom blocking the Global Markov property — by using the Bayes-Ball
algorithm as the graph-theoretic bridge.

Once `GlobalMarkovConcrete` is proved, `GlobalMarkov_From_Factorization`
(already a two-line proof via `PMFToAssignmentPMF_MarkovTransfer`) becomes
fully non-axiomatic, completing the causal-Markov chain.

## Background

`GlobalMarkovConcrete` states:

```
DSep(G, Y, Z, W)
  ∧ MarkovFactorizationConcrete(G, PMFToAssignmentPMF(G, p))
  ∧ CompatibleAssignments(zAssign, wAssign)
  ∧ AssignmentProb(p, G, wAssign) > 0
  ∧ AssignmentProb(p, G, MergeAssignments(zAssign, wAssign)) > 0
  ⟹
  AssignmentCondProb(p, G, yAssign, MergeAssignments(zAssign, wAssign))
    == AssignmentCondProb(p, G, yAssign, wAssign)
```

Its proof comment identifies two blocked sub-steps:

1. **Graph lemma (Bayes-Ball soundness):**
   `DSep(G, Y, Z, W) ⟹ ∀ v ∈ Y, Z ⊆ NonDescendants(G,v) \ Parents(G,v)`
   (relative to W — i.e., the d-separation premise forces every Z-node to be
   a non-descendant non-parent of every Y-node).

2. **Probabilistic telescope over Y:**
   Induct on `|Y|`, peel one node `v`, apply `LocalMarkovConcrete` at `v`,
   then recurse on `Y \ {v}`.

## Already-Proved Prerequisites

| Lemma | File | Status |
|---|---|---|
| `LocalMarkov` | `dag.dfy` | ✅ proved |
| `DSep_NonDesc_Subset` | `interventional.dfy` | ✅ proved |
| `DSep_Decomposition` | `dag.dfy` | ✅ proved |
| `DSep_Symmetry` | `dag.dfy` | ✅ proved |
| `DSep_ExtractSingle` | `dag.dfy` | ✅ proved |
| `LocalMarkovConcrete` | `interventional.dfy` | ✅ proved |
| `MarginalMass_FactorOut` | `interventional.dfy` | ✅ proved (`376f3ce`) |
| `PMFToAssignmentPMF_MarginalCoherence` | `interventional.dfy` | ✅ proved (`d745f40`) |
| `AssignmentCondProb_As_MarginalMass_Ratio` | `interventional.dfy` | ✅ proved |
| `GlobalMarkov_From_Factorization` | `interventional.dfy` | ✅ proved (2-line wrapper) |

## Phase A — Bayes-Ball ↔ d-separation equivalence (`dag.dfy`)

The `dag.dfy` §7b scaffolding already defines:
- `BallDir = IntoNode | OutOfNode`
- `BallState = BallState(node, dir)`
- `BallStep(G, W, s)` — local propagation table
- `BallSideBFS(G, W, frontier, visited, fuel)` — BFS over `(node, dir)` states
- `BallSide(G, v, W)` — nodes reachable from `v` given `W`

### A1 — `BallSide_Sound`

**Statement:**
```
∀ z ∈ BallSide(G, v, W) \ {v},
  ∃ trail :: ValidTrail(G, trail) ∧ TrailConnects(trail, v, z) ∧ ¬TrailBlocked(G, trail, W)
```

**Proof sketch:**
- Induct on the BFS depth at which `z` was discovered.
- At each BFS step, the `BallStep` propagation rule corresponds exactly to
  one trail extension that is locally unblocked:
  - `IntoNode` at unobserved `u`: the ball came down (Forward step), and
    `u ∉ W` means the non-collider is unblocked.
  - `OutOfNode` at activated `u` (u ∈ W or Desc(u)∩W ≠ {}): the ball came
    up (Backward step), and the collider is activated.
- Concatenate the per-step trail extensions to build the full active trail.

**Key helper:** `BallSideBFS_TrailWitness` — for every state `(z, d)` in the
BFS visited set, there exists an active trail from `v` to `z` whose last step
has direction consistent with `d`.

### A2 — `BallSide_Complete`

**Statement:**
```
ValidTrail(G, trail) ∧ TrailConnects(trail, v, z) ∧ ¬TrailBlocked(G, trail, W)
  ⟹ z ∈ BallSide(G, v, W)
```

**Proof sketch:**
- Induct on `|trail|`.
- At each step, the trail's local structure (direction of arrival, collider
  status, observation status) matches exactly one `BallStep` case that
  propagates the ball forward.
- Show the corresponding `BallState` is added to the BFS frontier, and by
  `BallSideBFS_Mono` it eventually enters the visited set.

### A3 — `DSep_iff_NotBallSide`

**Statement:**
```
DSep(G, {v}, Z, W) ⟺ Z ∩ BallSide(G, v, W) = {}
```

Follows immediately from A1 + A2 by contrapositive.

## Phase B — The Bayes-Ball graph lemma (`dag.dfy`)

### B1 — `DSep_Implies_NonDesc`

**Statement:**
```
DSep(G, Y, Z, W) ⟹ ∀ v ∈ Y, ∀ z ∈ Z, z ∉ Descendants(G, {v}) ∨ z ∈ W
```

**Proof sketch:**
- Use `DSep_ExtractSingle` to reduce to `DSep(G, {v}, Z, W)`.
- If `z ∈ Descendants(G, {v})` and `z ∉ W`, build a forward trail from `v`
  to `z` via `IsAncestorBounded_ImpliesForwardTrail`. A forward trail through
  unobserved nodes is never blocked (all non-colliders are unobserved since
  they are descendants of `v` and `z ∉ W`). Contradiction with `DSep`.

### B2 — `DSep_Implies_NonParent` (already implicit in `DSep_NoDirectEdge`)

If `z ∈ Parents(G, v)` and `z ∉ W`, the 1-step backward trail `v ← z` is
unblocked. `DSep_NoDirectEdge` already proves this.

### B3 — `DSep_Forward_NonDesc` (the full graph lemma)

**Statement:**
```
DSep(G, Y, Z, W) ⟹ ∀ v ∈ Y, Z ⊆ NonDescendants(G, v) - Parents(G, v)
```

Combine B1 and B2 (with the observation that `W`-members in `Z` are
irrelevant to the telescope because conditioning on them is vacuous).

*Note:* The exact statement needed inside `GlobalMarkovConcrete` is the
single-node version: `DSep(G, {v}, Z, W) ⟹ Z ⊆ NonDesc(v) \ Pa(v)`.
This is the converse of `DSep_NonDesc_Subset` (which goes the other way).

## Phase C — Probabilistic telescope (`interventional.dfy`)

### C1 — `GlobalMarkovConcrete_SingleNode`

**Statement:**
```
DSep(G, {v}, Z, W)
  ∧ MarkovFactorizationConcrete(G, q)
  ∧ CompatibleAssignments(zAssign, wAssign)
  ∧ AssignmentProb(p, G, wAssign) > 0
  ∧ AssignmentProb(p, G, MergeAssignments(zAssign, wAssign)) > 0
  ⟹
  AssignmentCondProb(p, G, vAssign, MergeAssignments(zAssign, wAssign))
    == AssignmentCondProb(p, G, vAssign, wAssign)
```

**Proof sketch:**
1. Apply Phase B (`DSep_Forward_NonDesc`) to get `Z ⊆ NonDesc(v) \ Pa(v)`.
2. Apply `LocalMarkovConcrete` (which calls `MarginalMass_FactorOut`) to get
   the cross-ratio identity for `v`.
3. Convert via `AssignmentCondProb_As_MarginalMass_Ratio` to get the
   conditional-probability equality.

### C2 — `GlobalMarkovConcrete` (induction over Y)

**Proof sketch:**
- Base `Y = {}`: vacuously true (no `yAssign` to constrain).
- Step: pick `v ∈ Y`. Use `DSep_ExtractSingle` to get `DSep(G, {v}, Z, W)`.
  Apply C1 to factor out `v`. Then apply the IH on `Y \ {v}` with the
  updated conditioning set.

The induction is over `|Y|` (finite set cardinality), which Dafny supports
via `decreases |Y|`.

## Implementation Order

1. **Write plan file** (this document). ✅
2. **Phase A1:** `BallSideBFS_TrailWitness` + `BallSide_Sound` in `dag.dfy`.
3. **Phase A2:** `BallSide_Complete` in `dag.dfy`.
4. **Phase A3:** `DSep_iff_NotBallSide` corollary in `dag.dfy`.
5. **Phase B3:** `DSep_Forward_NonDesc` in `dag.dfy`.
6. **Phase C1:** `GlobalMarkovConcrete_SingleNode` in `interventional.dfy`.
7. **Phase C2:** Replace `GlobalMarkovConcrete` axiom with proof body.
8. **Verify:** focused `dag.dfy`, then focused `interventional.dfy`, then
   full 6-file stack.
9. **Update** axiom ledger and progress tracker.

## Expected Axiom-Count Impact

`GlobalMarkovConcrete` is 1 declaration axiom. Removing it drops the count
from ~62 to ~61 total / ~57 declaration axioms.

`GlobalMarkov_From_Factorization` is already non-axiomatic (proved wrapper),
so no additional change there.

## Files Modified

- `src/dafny/dag.dfy` — Phase A and B lemmas (§7b extension).
- `src/dafny/interventional.dfy` — Phase C lemmas + axiom removal.
- `docs/plans/2026-05-19-dafny-de-axiomitization-progress.md` — tracker update.
- `docs/plans/2026-05-19-dafny-axiom-ledger.md` — ledger update.
