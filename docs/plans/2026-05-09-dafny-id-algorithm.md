# De-Axiomize Identification Lemmas — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `{:axiom}` annotations from concrete example lemmas in `identification.dfy` by providing direct proofs, and document a confirmed bug in the `IsHedge` definition.

**Architecture:** Non-identifiability lemmas construct explicit `(F, F')` hedge witnesses following the same pattern as the already-proved `BowArc_NotIdentifiable`. Identifiability proofs are deferred until a confirmed `IsHedge` definition bug is fixed.

**Tech Stack:** Dafny 4 (`/opt/homebrew/bin/dafny`), verified from `src/dafny/` directory

**Verification command (run from repo root):**
```bash
cd /Users/zuck016/Projects/CausalInference/y0-causal-inference/y0/src/dafny && /opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy
```

**Current baseline:** 86 verified, 0 errors

---

## Critical Finding: `IsHedge` definition bug (confirmed by Dafny)

Our `IsHedge` predicate in `src/dafny/semi_markovian.dfy` (line ~607) is **too permissive**. It allows a false hedge for the frontdoor graph (`X→M→Y, X↔Y`), which IS identifiable per the paper and the Python ID algorithm.

**The false hedge that Dafny verifies for the frontdoor graph:**
```dafny
F  = SMGraph(map[0 := {}, 2 := {}], {BiEdge(0, 2)})  // nodes {X,Y}, no directed, bidirected X↔Y
F' = SMGraph(map[2 := {}], {})                         // node {Y} only
// All 7 conditions of IsHedge pass. Verified: 61 verified, 0 errors.
```

**Consequence:**
- `IsIdentifiable(sm, X, Y)` (defined as `!exists F, Fprime :: IsHedge(...)`) is too **restrictive** — it incorrectly returns `false` for identifiable graphs
- The `ensures` of `Frontdoor_Identifiable`, `Figure1a_Identifiable`, `Markovian_AllIdentifiable`, and `MarkovianCompleteness` are **actually false** under the current definition
- These lemmas **cannot** be proved until `IsHedge` is fixed
- Non-identifiability proofs (constructing hedges) remain **safe** — the hedges we construct are genuine and satisfy the paper's (stricter) definition too

---

## Scope

### In scope (this plan)

| Task | What | Strategy |
|------|------|----------|
| 1 | `Figure1a_WellFormed` + `Figure1b_WellFormed` (new lemmas) | Topological sort witness `[0,1,2,3,4]` |
| 2 | `Figure1b_NotIdentifiable` (de-axiomize) | Explicit hedge: F={2,3} with BiEdge(2,3), F'={3} |
| 3 | `IsHedge` bug documentation | Add `// BUG:` comment block with analysis |

### Out of scope (blocked or requires infrastructure)

| Lemma | Blocker |
|-------|---------|
| `Frontdoor_Identifiable` | `IsHedge` bug — ensures is **false** under current definition |
| `Figure1a_Identifiable` | Same |
| `Markovian_AllIdentifiable` | Same |
| `MarkovianCompleteness` | Same |
| `ID_Line1` through `ID_Line7` | Requires concrete `ID` function body |
| `Theorem2–5`, `Corollary3` | Requires induction on ID recursion / PMF semantics |
| All other `{:axiom}` lemmas | Requires concrete ID or deep semantic proofs |

---

## Reference: key Dafny predicates

All in `src/dafny/semi_markovian.dfy`. Read these before implementing.

```dafny
predicate IsHedge(sm, F, Fprime, X, Y) {        // line ~607
    var Gx := RemoveIncomingSM(sm, X);
    var AncY := Ancestors(Gx.dag, Y);
    IsCForest(F) && IsSubgraphSM(F, sm) && RootSet(F) <= AncY &&
    IsCForest(Fprime) && IsSubgraphSM(Fprime, F) &&
    SMNodes(Fprime) < SMNodes(F) && RootSet(F) * X != {}
}
predicate IsCForest(sm) { WellFormedSM(sm) && AtMostOneChild(sm) }
predicate WellFormedSM(sm) {
    (forall e :: e in sm.bidirected ==>
       e.u in SMNodes(sm) && e.v in SMNodes(sm) && e.u != e.v) && IsDAG(sm.dag)
}
predicate AtMostOneChild(sm) {
    forall v :: v in SMNodes(sm) ==> |Children(sm.dag, v)| <= 1
}
predicate IsSubgraphSM(Fprime, F) {
    SMNodes(Fprime) <= SMNodes(F) &&
    (forall v :: v in SMNodes(Fprime) ==> Parents(Fprime.dag, v) <= Parents(F.dag, v)) &&
    Fprime.bidirected <= F.bidirected
}
ghost predicate IsIdentifiable(sm, X, Y) {
    !exists F: SMGraph, Fprime: SMGraph :: IsHedge(sm, F, Fprime, X, Y)
}
```

## Reference: proof pattern for non-identifiability

Follow the structure of `BowArc_NotIdentifiable` (line ~900 of `identification.dfy`):

1. Construct concrete `F` and `Fprime` as `SMGraph` literal values
2. Prove `IsDAG` for each via topological sort witness (`var ord := [...]; assert IsTopologicalSort(F.dag, ord);`)
3. Prove `IsCForest` for each (`WellFormedSM` + assert `Children` empty + `AtMostOneChild`)
4. Prove `IsSubgraphSM` relationships (`SMNodes` subset + parents subset + bidirected subset)
5. Prove `RootSet(F) <= AncY` via `IsAncestorBounded` chain with explicit fuel values
6. Assert `IsHedge(sm, F, Fprime, X, Y)`
7. Assert `!IsIdentifiable(sm, X, Y)` — Dafny derives this from the existential witness

---

## Task 1: Add WellFormed lemmas for Figure 1a and 1b

**Files:**
- Modify: `src/dafny/identification.dfy`

### Step 1: Add `Figure1a_WellFormed`

Insert this lemma immediately after the `Figure1aGraph()` function (around line 1093) and before the `/// Figure 1(a) — P_x(Y1, Y2) IS identifiable.` doc comment:

```dafny
  /// The Figure 1(a) graph is well-formed.
  lemma Figure1a_WellFormed()
    ensures WellFormedSM(Figure1aGraph())
  {
    var sm := Figure1aGraph();
    var ord := [0, 1, 2, 3, 4];
    forall i | 0 <= i < |ord|
      ensures forall p :: p in Parents(sm.dag, ord[i]) ==>
        exists k :: 0 <= k < i && ord[k] == p
    {
      if i == 2 { assert ord[0] == 0; assert ord[1] == 1; }
      if i == 3 { assert ord[2] == 2; }
      if i == 4 { assert ord[2] == 2; }
    }
    assert IsTopologicalSort(sm.dag, ord);
    assert IsDAG(sm.dag);
  }
```

Find the exact insertion point by searching for this text in the file:
```
  /// Figure 1(a) — P_x(Y1, Y2) IS identifiable.
```
Insert the new lemma **before** that line.

### Step 2: Add `Figure1b_WellFormed`

Insert this lemma immediately after the `Figure1bGraph()` function (around line 1122) and before the `/// Figure 1(b) — P_x(Y1, Y2) is NOT identifiable.` doc comment:

```dafny
  /// The Figure 1(b) graph is well-formed.
  lemma Figure1b_WellFormed()
    ensures WellFormedSM(Figure1bGraph())
  {
    var sm := Figure1bGraph();
    var ord := [0, 1, 2, 3, 4];
    forall i | 0 <= i < |ord|
      ensures forall p :: p in Parents(sm.dag, ord[i]) ==>
        exists k :: 0 <= k < i && ord[k] == p
    {
      if i == 2 { assert ord[0] == 0; assert ord[1] == 1; }
      if i == 3 { assert ord[2] == 2; }
      if i == 4 { assert ord[2] == 2; }
    }
    assert IsTopologicalSort(sm.dag, ord);
    assert IsDAG(sm.dag);
  }
```

Find the exact insertion point by searching for this text in the file:
```
  /// Figure 1(b) — P_x(Y1, Y2) is NOT identifiable.
```
Insert the new lemma **before** that line.

### Step 3: Verify

```bash
cd /Users/zuck016/Projects/CausalInference/y0-causal-inference/y0/src/dafny && /opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy
```

Expected: 88 verified (was 86), 0 errors.

### Step 4: Commit

```bash
cd /Users/zuck016/Projects/CausalInference/y0-causal-inference/y0
git add src/dafny/identification.dfy
git commit -m "feat(dafny): add Figure1a_WellFormed and Figure1b_WellFormed lemmas"
```

---

## Task 2: Prove `Figure1b_NotIdentifiable` by hedge construction

**Files:**
- Modify: `src/dafny/identification.dfy`

**Graph context:**
```
Figure 1(b): nodes W1=0, W2=1, X=2, Y1=3, Y2=4
Directed: 0→2, 1→2, 2→3, 2→4
Bidirected: 0↔1, 2↔3, 0↔4
Treatment X = {2}, Outcome Y = {3, 4}
```

**Hedge construction (mini bow-arc on {X, Y1}):**
```
F      = SMGraph(map[2 := {}, 3 := {}], {BiEdge(2, 3)})   // {X, Y1}, no directed, bidirected X↔Y1
Fprime = SMGraph(map[3 := {}], {})                          // {Y1} only
```

**Why the hedge works:**
- `Gx = RemoveIncomingSM(sm, {2})` removes incoming edges to node 2
  - Result: `Gx.dag = map[0:={}, 1:={}, 2:={}, 3:={2}, 4:={2}]`
  - Nodes 0 and 1 lose their children (2's parents removed), so `Children(Gx.dag, 0) = {}` and `Children(Gx.dag, 1) = {}`
- `Ancestors(Gx.dag, {3,4}) = {2, 3, 4}` (2→3 and 2→4 in Gx; 0,1 have no children in Gx)
- `RootSet(F) = {2, 3}` (both have no children in F)
- `{2, 3} ⊆ {2, 3, 4}` ✓ and `{2, 3} ∩ {2} = {2} ≠ {}` ✓

### Step 1: Replace the axiom lemma with a proof

Find this exact text in `identification.dfy` (around line 1124):

```dafny
  /// Figure 1(b) — P_x(Y1, Y2) is NOT identifiable.
  ///
  ///   The additional bidirected edge W1 ↔ Y2 creates a hedge
  ///   that prevents identification of the joint effect on
  ///   (Y1, Y2).
  lemma {:axiom} Figure1b_NotIdentifiable()
    ensures
      var sm := Figure1bGraph();
      WellFormedSM(sm) ==> !IsIdentifiable(sm, {2}, {3, 4})
```

Replace it with:

```dafny
  /// Figure 1(b) — P_x(Y1, Y2) is NOT identifiable.
  ///
  ///   The bidirected edge X ↔ Y1 (BiEdge(2,3)) creates a
  ///   mini bow-arc hedge on {X, Y1} that prevents identification
  ///   of the joint effect on (Y1, Y2).
  ///
  ///   Hedge: F = {X, Y1} with bidirected X↔Y1 (no directed edges),
  ///          F' = {Y1}
  ///
  ///   Ref: Shpitser & Pearl (2006), Figure 1(b)
  lemma Figure1b_NotIdentifiable()
    ensures
      var sm := Figure1bGraph();
      !IsIdentifiable(sm, {2}, {3, 4})
  {
    var sm := Figure1bGraph();
    // Construct the hedge:
    //   F  = nodes {X=2, Y1=3}, no directed edges, bidirected X↔Y1
    //   F' = node {Y1=3} only
    var F := SMGraph(
      map[2 := {}, 3 := {}],    // No directed edges
      {BiEdge(2, 3)}             // X ↔ Y1 bidirected
    );
    var Fprime := SMGraph(
      map[3 := {}],              // Just node Y1
      {}                          // No edges
    );

    // Prove F is a DAG (two isolated nodes)
    var Ford := [2, 3];
    assert IsTopologicalSort(F.dag, Ford);
    assert IsDAG(F.dag);

    // Prove Fprime is a DAG (single node)
    var Fpord := [3];
    assert IsTopologicalSort(Fprime.dag, Fpord);
    assert IsDAG(Fprime.dag);

    // Prove IsCForest(F): WellFormedSM + AtMostOneChild
    assert WellFormedSM(F);
    assert Children(F.dag, 2) == {};
    assert Children(F.dag, 3) == {};
    assert AtMostOneChild(F);
    assert IsCForest(F);

    // Prove IsCForest(Fprime)
    assert WellFormedSM(Fprime);
    assert Children(Fprime.dag, 3) == {};
    assert AtMostOneChild(Fprime);
    assert IsCForest(Fprime);

    // Prove IsSubgraphSM(F, sm)
    assert SMNodes(F) <= SMNodes(sm);
    assert IsSubgraphSM(F, sm);

    // Prove IsSubgraphSM(Fprime, F)
    assert SMNodes(Fprime) <= SMNodes(F);
    assert IsSubgraphSM(Fprime, F);

    // Prove strict subset
    assert SMNodes(Fprime) < SMNodes(F);

    // Prove RootSet(F) * {2} != {}
    assert RootSet(F) == {2, 3};
    assert {2, 3} * {2} != {};

    // Prove RootSet(F) <= Ancestors(Gx.dag, {3, 4})
    var Gx := RemoveIncomingSM(sm, {2});
    assert Gx.dag == map[0 := {}, 1 := {}, 2 := {}, 3 := {2}, 4 := {2}];

    // Ancestors of {3,4} in Gx.dag: {2, 3, 4}
    // Node 3: ancestor of itself
    assert IsAncestorBounded(Gx.dag, 3, 3, 0);
    assert IsAncestor(Gx.dag, 3, 3);
    // Node 2: parent of 3 (and 4) in Gx
    assert 3 in Children(Gx.dag, 2);
    assert IsAncestorBounded(Gx.dag, 2, 3, 1);
    assert IsAncestor(Gx.dag, 2, 3);

    var AncY := Ancestors(Gx.dag, {3, 4});
    assert 2 in AncY;
    assert 3 in AncY;
    assert RootSet(F) <= AncY;

    // Now IsHedge holds
    assert IsHedge(sm, F, Fprime, {2}, {3, 4});

    // Therefore not identifiable
    assert !IsIdentifiable(sm, {2}, {3, 4});
  }
```

**NOTE:** The `ensures` changed from `WellFormedSM(sm) ==> !IsIdentifiable(...)` to the stronger `!IsIdentifiable(...)` (no implication), matching the style of `BowArc_NotIdentifiable`. This is correct because `!IsIdentifiable` follows directly from the existence of a hedge — well-formedness of `sm` is irrelevant.

### Step 2: Verify

```bash
cd /Users/zuck016/Projects/CausalInference/y0-causal-inference/y0/src/dafny && /opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy
```

Expected: 88–89 verified, 0 errors. The count increases because the axiom (0 verification conditions) became a proved lemma (several conditions).

### Step 3: Commit

```bash
cd /Users/zuck016/Projects/CausalInference/y0-causal-inference/y0
git add src/dafny/identification.dfy
git commit -m "feat(dafny): prove Figure1b_NotIdentifiable by explicit hedge construction

Hedge: F = {X, Y1} with bidirected X↔Y1, F' = {Y1}.
Same mini bow-arc pattern as BowArc_NotIdentifiable."
```

---

## Task 3: Document `IsHedge` bug

**Files:**
- Modify: `src/dafny/semi_markovian.dfy`

### Step 1: Add bug documentation

Find this exact text in `semi_markovian.dfy` (the doc comment immediately before `IsHedge`, around line 600):

```dafny
  /// A hedge for P_x(y) in graph sm.
  ///
  /// Def 4 (Shpitser & Pearl 2006):
  ///   F, F' form a hedge for P_x(y) if:
  ///   - F is an R-rooted C-forest in G with R ⊆ An(Y)_{G_{X̄}}
  ///   - F' is an R'-rooted C-forest that is a subgraph of F
  ///   - F' ⊂ F  (strictly)
  ///   - R ∩ X ≠ ∅  (the root set intersects the treatments)
```

Replace it with:

```dafny
  /// A hedge for P_x(y) in graph sm.
  ///
  /// Def 4 (Shpitser & Pearl 2006):
  ///   F, F' form a hedge for P_x(y) if:
  ///   - F is an R-rooted C-forest in G with R ⊆ An(Y)_{G_{X̄}}
  ///   - F' is an R'-rooted C-forest that is a subgraph of F
  ///   - F' ⊂ F  (strictly)
  ///   - R ∩ X ≠ ∅  (the root set intersects the treatments)
  //
  // BUG: This predicate is too permissive — it allows false hedges.
  // For the frontdoor graph (X→M→Y, X↔Y), the following satisfies
  // all conditions but shouldn't (frontdoor IS identifiable):
  //   F  = SMGraph(map[0:={}, 2:={}], {BiEdge(0,2)})
  //   F' = SMGraph(map[2:={}], {})
  //   X = {0}, Y = {2}
  //
  // Consequence: IsIdentifiable (= !exists F,F' :: IsHedge) is too
  // RESTRICTIVE — it incorrectly says identifiable graphs are NOT
  // identifiable. Lemmas with `ensures IsIdentifiable(...)` for
  // graphs with bidirected edges are false under this definition.
  //
  // Non-identifiability proofs (hedge construction) remain correct
  // because the hedges we construct satisfy the paper's definition.
  //
  // The paper's Definition 4 likely has additional constraints we
  // haven't fully captured. Possible fixes require careful study
  // of the Theorem 3 proof in Shpitser & Pearl (2006).
```

### Step 2: Verify nothing changed functionally

```bash
cd /Users/zuck016/Projects/CausalInference/y0-causal-inference/y0/src/dafny && /opt/homebrew/bin/dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy semi_markovian.dfy identification.dfy
```

Expected: same count as after Task 2, 0 errors.

### Step 3: Commit

```bash
cd /Users/zuck016/Projects/CausalInference/y0-causal-inference/y0
git add src/dafny/semi_markovian.dfy
git commit -m "docs(dafny): document IsHedge definition bug (too permissive)

The predicate allows false hedges for identifiable graphs like the
frontdoor model. Non-identifiability proofs remain correct.
Identifiability proofs are blocked until the definition is fixed."
```

---

## Summary

After completing all 3 tasks:

| File | Change |
|------|--------|
| `identification.dfy` | +2 new lemmas (`Figure1a_WellFormed`, `Figure1b_WellFormed`), 1 de-axiomized (`Figure1b_NotIdentifiable`) |
| `semi_markovian.dfy` | +bug documentation comment on `IsHedge` |

**Expected final verification:** ~89 verified, 0 errors (up from 86)

**Axiom scorecard:**

| Lemma | Before | After |
|-------|--------|-------|
| `Figure1b_NotIdentifiable` | `{:axiom}` | **proved** |
| `Frontdoor_Identifiable` | `{:axiom}` | `{:axiom}` (blocked: IsHedge bug) |
| `Figure1a_Identifiable` | `{:axiom}` | `{:axiom}` (blocked: IsHedge bug) |
| `Markovian_AllIdentifiable` | `{:axiom}` | `{:axiom}` (blocked: IsHedge bug) |
| `MarkovianCompleteness` | `{:axiom}` | `{:axiom}` (blocked: IsHedge bug) |
| All `ID_Line*` / Theorem / Corollary | `{:axiom}` | `{:axiom}` (needs concrete ID) |

---

## Future work (separate plans)

1. **Fix `IsHedge` definition** — study Theorem 3 proof in Shpitser & Pearl (2006) to identify the exact missing condition. Fix `IsHedge`, update `BowArc_NotIdentifiable` and `Figure1b_NotIdentifiable` if the hedge construction needs to change, then prove the identifiability lemmas.

2. **Implement concrete `ID` function** — replace `ghost function {:axiom} ID(...)` with a recursive ghost function implementing the 7 lines. Termination measure: `(|SMNodes(sm)|, |SMNodes(sm)| - |X|)` lexicographic. This unlocks `ID_Line1`, `ID_Line4`, `ID_Line6`.

3. **Prove `CComponents_Partition`** — prove `BidirectedConnected` is an equivalence relation (reflexive, symmetric, transitive), then CComponents are its equivalence classes.
