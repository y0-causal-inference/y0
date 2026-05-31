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

-- ── idxOf helpers ────────────────────────────────────────────────────────────

/-- When `u ≠ v₀`, prepending `v₀` shifts the index of `u` by one. -/
private lemma idxOf_ne_head {v₀ : Node} (rest : List Node) {u : Node}
    (h : u ≠ v₀) : (v₀ :: rest).idxOf u = rest.idxOf u + 1 := by
  have hbeq : (v₀ == u) = false := beq_eq_false_iff_ne.mpr (Ne.symm h)
  simp [List.idxOf_cons, hbeq]

-- ── kahnAux correctness (generalized) ────────────────────────────────────────

/-- Core correctness of kahnAux, generalized over arbitrary `remaining`.
    Three invariants proved together by induction on `remaining.card`:
      (a) membership: `v ∈ result ↔ v ∈ remaining`
      (b) nodup:      `result.Nodup`
      (c) parent-before-child for parents *within* remaining:
          `v ∈ result → p ∈ parents(v) → p ∈ remaining → result.idxOf p < result.idxOf v`
    Clause (c) restricts to parents in `remaining`; the full `isTopologicalSort`
    clause follows at the top level by the well-formedness assumption `h_wf`. -/
private theorem kahnAux_spec (G : Graph) (remaining : Finset Node)
    {result : List Node}
    (h : kahnAux G remaining = some result) :
    (∀ v, v ∈ result ↔ v ∈ remaining) ∧
    result.Nodup ∧
    (∀ v ∈ result, ∀ p ∈ (G.lookup v).getD ∅,
      p ∈ remaining → result.idxOf p < result.idxOf v) := by
  induction h_card : remaining.card generalizing remaining result with
  | zero =>
    -- Base: remaining = ∅ → result = []
    have hr : remaining = ∅ := Finset.card_eq_zero.mp h_card
    rw [kahnAux.eq_1, dif_pos hr] at h
    simp only [Option.some.injEq] at h   -- h : [] = result
    subst h                               -- substitute result := []
    simp [hr]
  | succ n ih =>
    have h_ne : remaining ≠ ∅ := by intro heq; simp [heq] at h_card
    -- Unfold kahnAux one step, then zeta-reduce the let
    rw [kahnAux.eq_1, dif_neg h_ne] at h
    simp (config := { zeta := true }) only [] at h
    -- Name the zero-in-degree set
    set zeros := ({v ∈ remaining | inDegreeIn G v remaining = 0} : Finset Node)
      with hzeros_def
    -- Case: cycle detected (zeros empty) → contradicts h : some result
    by_cases hz : zeros = ∅
    · rw [dif_pos hz] at h; simp at h
    -- Case: zeros nonempty
    · rw [dif_neg hz] at h
      simp only [Option.map_eq_some_iff] at h
      obtain ⟨rest, h_rest, h_eq⟩ := h
      -- h_rest : kahnAux G (remaining.erase v₀) = some rest
      -- h_eq   : v₀ :: rest = result
      have hne : zeros.Nonempty := Finset.nonempty_iff_ne_empty.mpr hz
      set v₀ := zeros.min' hne with hv₀_def
      -- Key properties of v₀
      have hv₀_mem : v₀ ∈ remaining :=
        (Finset.mem_filter.mp (Finset.min'_mem zeros hne)).1
      have hv₀_zero : inDegreeIn G v₀ remaining = 0 :=
        (Finset.mem_filter.mp (Finset.min'_mem zeros hne)).2
      -- v₀ has no parents in remaining (its in-degree in remaining is 0)
      have hv₀_no_par : ∀ p ∈ (G.lookup v₀).getD ∅, p ∉ remaining := by
        intro p hp hpr
        have : 0 < inDegreeIn G v₀ remaining := by
          simp only [inDegreeIn]
          exact Finset.card_pos.mpr ⟨p, Finset.mem_inter.mpr ⟨hp, hpr⟩⟩
        omega
      -- Card decreases by exactly 1
      have h_card' : (remaining.erase v₀).card = n := by
        have := Finset.card_erase_of_mem hv₀_mem; omega
      -- Apply IH to the tail
      obtain ⟨hmem, hnd, htopo⟩ := ih (remaining.erase v₀) h_rest h_card'
      -- Substitute result = v₀ :: rest
      subst h_eq
      refine ⟨?_, ?_, ?_⟩
      · -- (a) Membership: v ∈ (v₀ :: rest) ↔ v ∈ remaining
        intro u
        simp only [List.mem_cons]
        constructor
        · rintro (rfl | hu)
          · exact hv₀_mem
          · exact Finset.mem_of_mem_erase ((hmem u).mp hu)
        · intro hu
          rcases eq_or_ne u v₀ with rfl | hne_u
          · exact Or.inl rfl
          · exact Or.inr ((hmem u).mpr (Finset.mem_erase.mpr ⟨hne_u, hu⟩))
      · -- (b) Nodup: v₀ ∉ rest (since v₀ ∉ remaining.erase v₀)
        rw [List.nodup_cons]
        exact ⟨fun hv₀r =>
          Finset.notMem_erase v₀ remaining ((hmem v₀).mp hv₀r), hnd⟩
      · -- (c) Parent-before-child
        intro u hu_cons p hp_par hp_rem
        simp only [List.mem_cons] at hu_cons
        rcases hu_cons with rfl | hu_rest
        · -- u = v₀: impossible — v₀ has no parents in remaining
          exact absurd hp_rem (hv₀_no_par p hp_par)
        · -- u ∈ rest
          have hu_ne : u ≠ v₀ := fun heq =>
            Finset.notMem_erase v₀ remaining ((hmem v₀).mp (heq ▸ hu_rest))
          rcases eq_or_ne p v₀ with rfl | hp_ne
          · -- p = v₀: at index 0, u at index ≥ 1
            rw [List.idxOf_cons_self, idxOf_ne_head rest hu_ne]; omega
          · -- p ≠ v₀: p ∈ remaining.erase v₀; use IH
            have hp_erase : p ∈ remaining.erase v₀ :=
              Finset.mem_erase.mpr ⟨hp_ne, hp_rem⟩
            have := htopo u hu_rest p hp_par hp_erase
            rw [idxOf_ne_head rest hp_ne, idxOf_ne_head rest hu_ne]; omega

-- ── Correctness spec ─────────────────────────────────────────────────────────

/-- Kahn's algorithm produces a valid topological sort when it returns `some`.
    Corresponds to the postcondition `IsTopologicalSort(G, result.value)` of
    `KahnsAlgorithm` in dag.dfy.

    Requires the well-formedness assumption that every parent of every node in
    `G.keys` is itself in `G.keys`.  Without this, `isTopologicalSort` clause (d)
    would be vacuously satisfied for out-of-graph parents via the sentinel value
    `ord.idxOf p = ord.length` (≥ every in-graph index), but the < direction
    would fail for hypothetical parents not in `ord`.

    Proof: `kahnAux_spec` gives (a)–(c) above.  For (d), `h_wf` supplies
    `p ∈ G.keys` for every parent `p`, which matches the `p ∈ remaining`
    hypothesis of clause (c) when `remaining = G.keys`. -/
theorem kahnSort_spec (G : Graph) (ord : List Node)
    (h_wf : ∀ v ∈ G.keys, ∀ p ∈ (G.lookup v).getD ∅, p ∈ G.keys)
    (h : kahnSort G = some ord) : isTopologicalSort G ord := by
  obtain ⟨hmem, hnd, htopo⟩ := kahnAux_spec G G.keys h
  exact ⟨fun v hv => (hmem v).mpr hv,
         fun v hv => (hmem v).mp hv,
         hnd,
         fun v hv p hp => htopo v hv p hp (h_wf v ((hmem v).mp hv) p hp)⟩

end Y0Lean
