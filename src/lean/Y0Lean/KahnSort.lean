/-
  KahnSort.lean — L2-001: computable topological sort via Kahn's algorithm.
  Port of: dag.dfy — KahnsAlgorithm, IsTopologicalSort
  Phase L2: computable algorithms.

  Dafny's `method KahnsAlgorithm` is a while-loop with mutable state.
  Here we use well-founded recursion on `remaining.card` — strictly
  decreasing because each step removes exactly one node (the minimum
  zero-in-degree node).

  Key design choices vs. Dafny:
  · Functional, not imperative — no `var` or mutation.
  · `zeros.min'` gives a deterministic (computable) choice of v, so
    the output is a unique canonical topological ordering (not just "some"
    ordering), which is convenient for `#guard` conformance tests.
  · `inDegreeIn G v remaining` uses `Finset.inter` rather than iterating
    over the parent set, so the definition is a single card call.
-/
import Y0Lean.Graph
import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Max
import Mathlib.Data.Finset.Filter

namespace Y0Lean

-- ── Predicate ────────────────────────────────────────────────────────────────

/-- An ordering is a topological sort of G when it
    (a) covers every node in G.keys,
    (b) is contained in G.keys (no spurious nodes),
    (c) has no duplicates, and
    (d) every parent precedes its child (by `List.indexOf`).
    Corresponds to `IsTopologicalSort` in dag.dfy. -/
def isTopologicalSort (G : Graph) (ord : List Node) : Prop :=
  (∀ v ∈ G.keys, v ∈ ord) ∧
  (∀ v ∈ ord, v ∈ G.keys) ∧
  ord.Nodup ∧
  ∀ v ∈ ord, ∀ p ∈ (G.lookup v).getD ∅,
    ord.idxOf p < ord.idxOf v

-- ── In-degree helper ─────────────────────────────────────────────────────────

/-- Number of parents of v that are still unprocessed (present in `remaining`).
    Corresponds to `InDegree` / the degree invariant in dag.dfy. -/
def inDegreeIn (G : Graph) (v : Node) (remaining : Finset Node) : ℕ :=
  ((G.lookup v).getD ∅ ∩ remaining).card

-- ── Kahn's algorithm ─────────────────────────────────────────────────────────

/-- Recursive Kahn step: repeatedly extract the minimum zero-in-degree node.
    · Returns `some` of the constructed prefix when `remaining` is emptied.
    · Returns `none` if no zero-in-degree node exists while `remaining ≠ ∅`
      (cycle detected).
    Terminates because `remaining.card` strictly decreases by 1 per step. -/
def kahnAux (G : Graph) (remaining : Finset Node) : Option (List Node) :=
  if h_empty : remaining = ∅ then some []
  else
    let zeros := remaining.filter (fun v => inDegreeIn G v remaining = 0)
    if hz : zeros = ∅ then none   -- cycle detected
    else
      have hne : zeros.Nonempty := Finset.nonempty_iff_ne_empty.mpr hz
      have hv : zeros.min' hne ∈ remaining :=
        (Finset.mem_filter.mp (Finset.min'_mem zeros hne)).1
      (kahnAux G (remaining.erase (zeros.min' hne))).map (zeros.min' hne :: ·)
termination_by remaining.card
decreasing_by
  -- `zeros.min' hne` is syntactically the same term in hv and the recursive call.
  exact Finset.card_erase_lt_of_mem hv

/-- Kahn's topological sort of a graph.
    Returns `some ord` (a topological ordering of G.keys) if G is a DAG,
    `none` if G contains a cycle.
    Corresponds to `KahnsAlgorithm` in dag.dfy. -/
def kahnSort (G : Graph) : Option (List Node) :=
  kahnAux G G.keys

-- ── Correctness spec ─────────────────────────────────────────────────────────

/-- Kahn's algorithm produces a valid topological sort when it returns `some`.
    Corresponds to the postcondition `IsTopologicalSort(G, result.value)` of
    `KahnsAlgorithm` in dag.dfy.

    Proof obligation: derive the four clauses of `isTopologicalSort` from the
    `kahnAux` loop invariants:
      I1. remaining.keys ⊆ G.keys
      I2. ordered ++ remaining.keys = G.keys (partition)
      I3. ordered ∩ remaining.keys = ∅       (disjoint)
      I4. ordered is duplicate-free
      I5. ordered satisfies the parent-before-child condition (partial topo)
    Deferred: requires structural induction on `remaining.card` following the
    same invariant chain used in the Dafny proof. -/
theorem kahnSort_spec (G : Graph) (ord : List Node)
    (h : kahnSort G = some ord) : isTopologicalSort G ord := by
  sorry

end Y0Lean
