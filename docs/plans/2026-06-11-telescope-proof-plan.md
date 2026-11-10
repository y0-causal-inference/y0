# Telescope Proof Plan: `GlobalMarkovConcrete` Phase C2

Date: 2026-06-11

## Objective

Discharge the final proof obligation for `GlobalMarkovConcrete` in
`src/dafny/interventional.dfy`: the induction from a single-node result
(`GlobalMarkovConcrete_SingleNode`) to the full multi-node statement over
all `v ∈ Y` via telescoping.

This document focuses on **Phase C2** of the Bayes-Ball plan.  Phase C1
(the single-node case) is treated as a prerequisite and its proof is
described in the companion plan `2026-06-11-Bayes-Ball.md`.

---

## 1. The Statement to Prove

```dafny
lemma GlobalMarkovConcrete(
  G: Graph, p: Prob.PMF,
  Y Z W: set<Node>,
  yAssign zAssign wAssign: Assignment
)
  requires IsDAG(G)
  requires Prob.IsDistribution(p)
  requires MarkovFactorizationConcrete(G, PMFToAssignmentPMF(G, p))
  requires DSep(G, Y, Z, W)
  requires Y <= Nodes(G); Z <= Nodes(G); W <= Nodes(G)
  requires yAssign.Keys <= Y; zAssign.Keys <= Z; wAssign.Keys <= W
  requires CompatibleAssignments(zAssign, wAssign)
  requires AssignmentProb(p, G, wAssign) > 0.0
  requires AssignmentProb(p, G, MergeAssignments(zAssign, wAssign)) > 0.0
  ensures
    AssignmentCondProb(p, G, yAssign, MergeAssignments(zAssign, wAssign))
      == AssignmentCondProb(p, G, yAssign, wAssign)
```

In words: **P(Y=y | Z=z, W=w) = P(Y=y | W=w)** whenever Y ⊥ Z | W in G
and the distribution satisfies the Markov factorization.

---

## 2. The Circularity Problem

The existing `MarginalMass_FactorOut_Core` (lines 690–1034) is supposed to
prove the cross-ratio identity

```
MM(pa+Z+{v}) · MM(pa) == MM(pa+{v}) · MM(pa+Z)
```

but its current proof body calls `GlobalMarkov_From_Factorization` (line 883),
which calls `GlobalMarkovConcrete` (line 3822) — the very axiom we want to
prove.  The dependency chain is:

```
MarginalMass_FactorOut_Core
  └─ GlobalMarkov_From_Factorization
       └─ GlobalMarkovConcrete   ← axiom we are discharging
```

This means we **cannot** use `LocalMarkovConcrete` / `MarginalMass_FactorOut`
as a stepping stone to prove `GlobalMarkovConcrete` without first breaking
this cycle.

---

## 3. Breaking the Cycle: `MarginalMass_FactorOut_Direct`

### 3.1 What the current `_Core` proof does (incorrectly)

The non-zero branch of `_Core` (lines 780–1033) proceeds as:

1. Establishes `DSep(G, {v}, Z, pa)` from `Z ⊆ NonDesc(v)\Pa(v)`.
2. Calls `GlobalMarkov_From_Factorization` to get `acp_zp == acp_pa`
   (i.e., `P(v|Z,pa) = P(v|pa)`).
3. Uses P2 (`AssignmentCondProb_As_MarginalMass_Ratio`) twice to convert
   `acp_zp` and `acp_pa` back to marginal-mass ratios.
4. Closes the cross-ratio by algebra.

Steps 3–4 are fine.  Step 2 is the circular call.

### 3.2 The direct proof

The cross-ratio identity is a consequence of `MarkovFactorizationConcrete`
alone, without any conditional-probability reasoning.  The key algebraic
fact is:

> Under the Markov factorization, `MarginalMass(q, S, ref)` is a sum of
> products of local conditional factors.  When `Z ⊆ NonDesc(v)\Pa(v)`,
> the Z-nodes appear in the factorization **before** `v` in topological
> order and their factors are independent of `v`'s factor.  Summing over
> Z-values therefore factors out of the `v`-sum, giving the cross-ratio.

Concretely, introduce a new lemma:

```dafny
lemma MarginalMass_FactorOut_Direct(
  G: Graph, p: Prob.PMF, q: AssignmentPMF,
  v: Node, Z: set<Node>, ref: Assignment, pa: set<Node>
)
  requires Prob.IsDistribution(p)
  requires q == PMFToAssignmentPMF(G, p)
  requires MarkovFactorizationConcrete(G, q)   // ← uses concrete layer directly
  requires IsDAG(G)
  requires v in Nodes(G)
  requires ref.Keys == Nodes(G)
  requires Z <= NonDescendants(G, v) - Parents(G, v)
  requires pa == Parents(G, v) * Nodes(G)
  ensures
    MarginalMass(q, pa + Z + {v}, ref) * MarginalMass(q, pa, ref)
 == MarginalMass(q, pa + {v}, ref) * MarginalMass(q, pa + Z, ref)
```

**Proof sketch** (no call to `GlobalMarkovConcrete` or
`GlobalMarkov_From_Factorization`):

1. **Factorization expansion.**  By `MarkovFactorizationConcrete`, every
   full assignment `a` with `a.Keys == Nodes(G)` satisfies
   `AssignmentMass(q, a) == ∏_{u ∈ ord} ConditionalFactorConcrete(G, q, u, a)`.

2. **MarginalMass as a sum.**  `MarginalMass(q, S, ref)` sums
   `AssignmentMass(q, a)` over all `a` that agree with `ref` on `S`.
   Substituting the factorization, this becomes a sum of products of
   local factors.

3. **Factor independence.**  Because `Z ⊆ NonDesc(v)\Pa(v)`, the
   topological order places all Z-nodes before `v`, and no Z-node is a
   parent of `v`.  Therefore `ConditionalFactorConcrete(G, q, v, a)`
   depends only on `a` restricted to `Pa(v)` — it is independent of
   `a[z]` for `z ∈ Z`.

4. **Sum factoring.**  The double sum over Z-values and v-values splits:
   ```
   Σ_{a: a|pa+Z+{v}=ref|pa+Z+{v}} ∏ factors
     = (Σ_{z-vals} Σ_{v-val} factor_v · factor_z · rest)
     = (Σ_{v-val} factor_v) · (Σ_{z-vals} factor_z · rest)
   ```
   This gives `MM(pa+Z+{v}) = MM(pa+{v}) · (MM(pa+Z)/MM(pa))` when
   `MM(pa) > 0`, which rearranges to the cross-ratio.

5. **Zero cases.**  When `MM(pa) = 0` or `MM(pa+Z) = 0`, the
   `MarginalMass_SubsetMonotone` + `MarginalMass_Nonneg` argument from
   the existing `_Core` proof applies unchanged.

**Implementation note.**  Steps 2–4 require a helper lemma
`MarginalMass_Factorization_Expand` that rewrites `MarginalMass(q, S, ref)`
as a sum over the factorized form.  This is a pure combinatorial identity
about `SumAssignmentMasses` and `ConditionalFactorProductConcrete`; it does
not involve `GlobalMarkovConcrete`.

---

## 4. Phase C1: `GlobalMarkovConcrete_SingleNode`

Once `MarginalMass_FactorOut_Direct` is proved, the single-node case
follows in three steps:

```
DSep(G, {v}, Z, W)
  ──[DSep_Implies_NonDesc / DSep_Implies_NonDesc_IntersectedParents]──▶
  Z ⊆ NonDesc(v) \ Pa(v)
  ──[MarginalMass_FactorOut_Direct]──▶
  MM(pa+Z+{v}) · MM(pa) == MM(pa+{v}) · MM(pa+Z)
  ──[AssignmentCondProb_As_MarginalMass_Ratio (P2)]──▶
  AssignmentCondProb(p, G, vAssign, merge(zAssign, wAssign))
    == AssignmentCondProb(p, G, vAssign, wAssign)
```

The P2 step converts the cross-ratio into the conditional-probability
equality by the already-proved `AssignmentCondProb_As_MarginalMass_Ratio`
lemma (line 525).

---

## 5. Phase C2: The Telescope (`GlobalMarkovConcrete`)

### 5.1 What the telescope proves

`GlobalMarkovConcrete` states `P(Y=y | Z=z, W=w) = P(Y=y | W=w)` for
**arbitrary** Y.  The single-node result (Phase C1) gives this for `|Y|=1`.
The telescope extends it to all `|Y|`.

### 5.2 The key mathematical identity

`AssignmentCondProb(p, G, yAssign, given)` is defined as
`ProbCond(p, AssignmentEvent(yAssign), AssignmentEvent(given))`.  This is
**not** a product over Y-nodes; it is a single ratio of events.

The telescope works at the **MarginalMass** level.  Via P2, the goal
`P(Y=y | Z=z, W=w) = P(Y=y | W=w)` is equivalent to:

```
MM(y ∪ z ∪ w) · MM(w) == MM(y ∪ w) · MM(z ∪ w)
```

(where `y = yAssign.Keys`, etc., and `MM(S)` abbreviates
`MarginalMass(q, S, merge(yAssign, zAssign, wAssign))`).

We prove this cross-ratio identity by induction on `|Y|`.

### 5.3 Induction structure

**Decreases:** `|Y|` (finite set cardinality; Dafny supports `decreases |Y|`).

**Base case `Y = {}`:**
- `yAssign = map[]` (empty assignment, since `yAssign.Keys <= Y = {}`).
- `AssignmentCondProb(p, G, map[], given)` = `ProbCond(p, Ω, given)` = 1.0
  on both sides (the event `AssignmentEvent(map[])` is the full sample space).
- Need helper: `AssignmentEvent_Empty` — `AssignmentEvent(p, G, map[]) == p.Keys`.
- Then `ProbCond(p, p.Keys, B) = 1.0` for any `B` with positive probability
  (from `Prob.ProbCond_Full`).

**Inductive step `Y ≠ {}`:**

Pick any `v ∈ Y`.  Let `Y' = Y \ {v}`, `y'Assign = yAssign - {v}`,
`vAssign = map[v := yAssign[v]]` (if `v ∈ yAssign.Keys`; otherwise `vAssign = map[]`).

The proof proceeds in two sub-steps:

**Sub-step A (peel v from the conditioning):**
Apply `DSep_ExtractSingle` (dag.dfy:2749) to get `DSep(G, {v}, Z, W)`.
Apply Phase C1 (`GlobalMarkovConcrete_SingleNode`) to get:
```
P(v=vAssign | Z=z, W=w) = P(v=vAssign | W=w)
```

**Sub-step B (chain rule split):**
Use the chain rule to split the Y-event:
```
P(Y=y | given) = P(v=vAssign | given) · P(Y'=y' | v=vAssign, given)
```
This requires a helper lemma `AssignmentCondProb_ChainRule`:
```dafny
lemma AssignmentCondProb_ChainRule(
  p: Prob.PMF, G: Graph,
  vAssign: Assignment, restAssign: Assignment, given: Assignment
)
  requires CompatibleAssignments(vAssign, restAssign)
  requires CompatibleAssignments(vAssign, given)
  requires CompatibleAssignments(restAssign, given)
  requires AssignmentProb(p, G, given) > 0.0
  requires AssignmentProb(p, G, MergeAssignments(vAssign, given)) > 0.0
  ensures
    AssignmentCondProb(p, G, MergeAssignments(vAssign, restAssign), given)
      == AssignmentCondProb(p, G, vAssign, given)
       * AssignmentCondProb(p, G, restAssign, MergeAssignments(vAssign, given))
```
This follows from `Prob.ChainRule` applied to the three events
`AssignmentEvent(merge(v,rest))`, `AssignmentEvent(v)`, `AssignmentEvent(given)`.

**Combining A and B:**

```
P(Y=y | Z=z, W=w)
  = P(v | Z=z, W=w) · P(Y'=y' | v, Z=z, W=w)   [chain rule, sub-step B]
  = P(v | W=w)      · P(Y'=y' | v, Z=z, W=w)   [sub-step A]
```

Now apply the IH to `P(Y'=y' | v, Z=z, W=w) = P(Y'=y' | v, W=w)`:
- IH precondition: `DSep(G, Y', Z, W ∪ {v})`.
  This follows from `DSep_ExtractSingle` + `DSep_Decomposition` (or a
  `DSep_Subset_Conditioning` lemma that says adding a Y-node to W preserves
  d-separation of the remaining Y-nodes from Z).
- IH conclusion: `P(Y'=y' | v, Z=z, W=w) = P(Y'=y' | v, W=w)`.

Then:
```
P(v | W=w) · P(Y'=y' | v, Z=z, W=w)
  = P(v | W=w) · P(Y'=y' | v, W=w)   [IH]
  = P(Y=y | W=w)                       [chain rule, sub-step B reversed]
```

### 5.4 Required helper lemmas

| Lemma | Purpose | Status |
|---|---|---|
| `AssignmentEvent_Empty` | `AssignmentEvent(p, G, map[]) == p.Keys` | Need to add |
| `Prob.ProbCond_Full` | `ProbCond(p, p.Keys, B) == 1.0` | Check if in `Prob` module |
| `AssignmentCondProb_ChainRule` | Chain rule for `AssignmentCondProb` | Need to add |
| `DSep_Subset_Conditioning` | `DSep(G, Y, Z, W) ∧ v ∈ Y ⟹ DSep(G, Y\{v}, Z, W∪{v})` | Need to add |
| `MarginalMass_FactorOut_Direct` | Cross-ratio without circularity | Need to add |

### 5.5 `DSep_Subset_Conditioning`

This is the d-separation property that moving a Y-node into the conditioning
set preserves d-separation of the remaining Y-nodes from Z.

**Statement:**
```dafny
lemma DSep_Subset_Conditioning(G: Graph, Y Z W: set<Node>, v: Node)
  requires DSep(G, Y, Z, W)
  requires v in Y
  ensures DSep(G, Y - {v}, Z, W + {v})
```

**Proof sketch:**
- Take any trail from `u ∈ Y\{v}` to `z ∈ Z`.
- By `DSep(G, Y, Z, W)`, this trail is blocked by W.
- Adding `v` to W can only block more trails (more nodes observed).
- Formally: `TrailBlocked(G, trail, W) ⟹ TrailBlocked(G, trail, W + {v})`
  because `W ⊆ W + {v}` and blocking is monotone in the conditioning set.
- Need helper: `TrailBlocked_Monotone` — `W ⊆ W' ⟹ TrailBlocked(G, trail, W) ⟹ TrailBlocked(G, trail, W')`.

This lemma should live in `dag.dfy` alongside `DSep_ExtractSingle`.

---

## 6. Implementation Order

1. **`dag.dfy`**: Add `TrailBlocked_Monotone` and `DSep_Subset_Conditioning`.
2. **`interventional.dfy`**: Add `MarginalMass_FactorOut_Direct` (the
   circularity-free cross-ratio proof from `MarkovFactorizationConcrete`).
3. **`interventional.dfy`**: Add `AssignmentEvent_Empty` and
   `AssignmentCondProb_ChainRule`.
4. **`interventional.dfy`**: Add `GlobalMarkovConcrete_SingleNode` (Phase C1).
5. **`interventional.dfy`**: Replace `GlobalMarkovConcrete {:axiom}` with
   the `decreases |Y|` induction body (Phase C2).
6. **Verify**: focused `dag.dfy`, then focused `interventional.dfy`, then
   full stack.
7. **Update** axiom ledger and progress tracker.

---

## 7. Dependency Graph

```
TrailBlocked_Monotone (dag.dfy)
  └─▶ DSep_Subset_Conditioning (dag.dfy)
        └─▶ GlobalMarkovConcrete [IH step] (interventional.dfy)

DSep_Implies_NonDesc (dag.dfy) [already proved]
  └─▶ MarginalMass_FactorOut_Direct (interventional.dfy)
        └─▶ GlobalMarkovConcrete_SingleNode (interventional.dfy)
              └─▶ GlobalMarkovConcrete [base + step] (interventional.dfy)

AssignmentCondProb_ChainRule (interventional.dfy)
  └─▶ GlobalMarkovConcrete [step] (interventional.dfy)

AssignmentEvent_Empty (interventional.dfy)
  └─▶ GlobalMarkovConcrete [base case] (interventional.dfy)
```

---

## 8. Dafny Proof Sketch for Phase C2

```dafny
lemma GlobalMarkovConcrete(
  G: Graph, p: Prob.PMF,
  Y: set<Node>, Z: set<Node>, W: set<Node>,
  yAssign: Assignment, zAssign: Assignment, wAssign: Assignment
)
  // ... preconditions as in axiom ...
  ensures AssignmentCondProb(p, G, yAssign, MergeAssignments(zAssign, wAssign))
       == AssignmentCondProb(p, G, yAssign, wAssign)
  decreases |Y|
{
  var q := PMFToAssignmentPMF(G, p);
  var zw := MergeAssignments(zAssign, wAssign);

  if Y == {} {
    // yAssign.Keys <= {} so yAssign == map[].
    // AssignmentEvent(p, G, map[]) == p.Keys (full sample space).
    AssignmentEvent_Empty(p, G);
    // ProbCond(p, p.Keys, B) == 1.0 for any positive-probability B.
    Prob.ProbCond_Full(p, AssignmentEvent(p, G, zw));
    Prob.ProbCond_Full(p, AssignmentEvent(p, G, wAssign));
  } else {
    // Pick v ∈ Y.
    var v :| v in Y;
    var vAssign := if v in yAssign.Keys then map[v := yAssign[v]] else map[];
    var yRest := yAssign - {v};
    var Yrest := Y - {v};

    // Sub-step A: single-node d-separation.
    DSep_ExtractSingle(G, Y, Z, W, v);
    // DSep(G, {v}, Z, W) now holds.
    GlobalMarkovConcrete_SingleNode(G, p, v, Z, W, vAssign, zAssign, wAssign);
    // P(v | Z,W) == P(v | W).

    // Sub-step B: chain rule split.
    // Need: AssignmentProb(p, G, MergeAssignments(vAssign, zw)) > 0.
    // (follows from AssignmentProb(p, G, zw) > 0 and the single-node result)
    AssignmentCondProb_ChainRule(p, G, vAssign, yRest, zw);
    // P(Y=y | Z,W) = P(v | Z,W) * P(Y'=y' | v, Z,W)
    AssignmentCondProb_ChainRule(p, G, vAssign, yRest, wAssign);
    // P(Y=y | W) = P(v | W) * P(Y'=y' | v, W)

    // IH: DSep(G, Y', Z, W ∪ {v}) ⟹ P(Y'=y' | v, Z,W) = P(Y'=y' | v, W).
    DSep_Subset_Conditioning(G, Y, Z, W, v);
    // DSep(G, Y - {v}, Z, W + {v}) holds.
    var wv := MergeAssignments(vAssign, wAssign);
    GlobalMarkovConcrete(G, p, Yrest, Z, W + {v}, yRest, zAssign, wv);
    // P(Y'=y' | v, Z,W) == P(Y'=y' | v, W).

    // Algebra: combine A, B, IH.
    // P(Y | Z,W) = P(v|Z,W) * P(Y'|v,Z,W)
    //            = P(v|W)   * P(Y'|v,W)    [A + IH]
    //            = P(Y | W)                [B reversed]
  }
}
```

---

## 9. Expected Axiom-Count Impact

Removing `{:axiom}` from `GlobalMarkovConcrete` drops the declaration-axiom
count by 1 (from ~62 to ~61 total / ~57 declaration axioms).

`GlobalMarkov_From_Factorization` is already a proved wrapper; no change there.

---

## 10. Files Modified

- `src/dafny/dag.dfy` — `TrailBlocked_Monotone`, `DSep_Subset_Conditioning`
- `src/dafny/interventional.dfy` — `MarginalMass_FactorOut_Direct`,
  `AssignmentEvent_Empty`, `AssignmentCondProb_ChainRule`,
  `GlobalMarkovConcrete_SingleNode`, `GlobalMarkovConcrete` (axiom → proof)
- `docs/plans/2026-05-19-dafny-de-axiomitization-progress.md` — tracker update
- `docs/plans/2026-05-19-dafny-axiom-ledger.md` — ledger update
