/-
  DSeparation.lean — D-separation predicate for causal graphs.
  Port of: dag.dfy (DSep, ValidTrail, TrailBlocked, IsCollider)
  Phase L2-004: Prop-valued definition (not computable) matching the
  Dafny `ghost predicate DSep`.

  Phase L4-003: Trail reversal machinery; proofs of
    · dSep_decomposition  (trivial set inclusion)
    · dSep_symmetry       (via trail reversal)
  Remaining (deferred — require descendants-aware blocking):
    · dSep_weakUnion, dSep_contraction
  See also: CausalQIF (OstensibleParadox/CausalQIF) which takes a
  Bayes-Ball / ReflTransGen approach as an alternative proof strategy.
-/
import Y0Lean.Graph

namespace Y0Lean

-- ======================================================================
-- Trail definitions (mirror dag.dfy §5)
-- ======================================================================

/-- A `Trail` is a sequence of `TrailStep`s through the graph. -/
abbrev Trail := List TrailStep

/-- The step is *valid* in `G`: the directed edge corresponding to the step
    exists in the graph.
    - Forward step  (u → v): v must have u in its parent set.
    - Backward step (u ← v, i.e. traversed v→u in reverse): u must have v in its parent set. -/
def stepValid (G : Graph) (s : TrailStep) : Prop :=
  match s.dir with
  | EdgeDir.Forward  => s.src ∈ (G.lookup s.dst).getD ∅
  | EdgeDir.Backward => s.dst ∈ (G.lookup s.src).getD ∅

/-- All steps in the trail are valid in `G`. -/
def trailValid (G : Graph) (t : Trail) : Prop :=
  ∀ s ∈ t, stepValid G s

/-- Consecutive steps are connected: the destination of step `i` equals
    the source of step `i+1`. -/
def trailConnected (t : Trail) : Prop :=
  ∀ i : Fin (t.length - 1),
    (t.get ⟨i.val,     by omega⟩).dst =
    (t.get ⟨i.val + 1, by omega⟩).src

/-- The trail connects node `y` to node `z`:
    the source of the first step is `y` and the destination of the last is `z`. -/
def trailConnects (t : Trail) (y z : Node) : Prop :=
  t ≠ [] ∧
  (t.head?.map TrailStep.src = some y) ∧
  (t.getLast?.map TrailStep.dst = some z)

/-- A step at position `pos` (1-indexed, 1 ≤ pos < |t|) is a **collider**:
    the node at `pos` is reached from both sides by inward-pointing edges: → node ←.
    · step pos-1 is Forward  (edge from left  → node)
    · step pos   is Backward (edge from right → node, traversed in reverse)
    Corresponds to Dafny `predicate IsCollider`. -/
def isCollider (t : Trail) (pos : ℕ) : Prop :=
  ∃ (h1 : pos - 1 < t.length) (h2 : pos < t.length),
    (t.get ⟨pos - 1, h1⟩).dir = EdgeDir.Forward ∧
    (t.get ⟨pos,     h2⟩).dir = EdgeDir.Backward

/-- The *node at position `pos`* in the trail (the source of step `pos`). -/
def trailNodeAt (t : Trail) (pos : ℕ) (h : pos < t.length) : Node :=
  (t.get ⟨pos, h⟩).src

/-- A trail is **blocked at position `pos`** by `W`:
    - If the node at `pos` is a collider (→ node ←): the node is **not** in `W`
      (simplified — full definition requires `node ∉ W ∧ Descendants(G,{node}) ∩ W = ∅`;
       dSep_weakUnion and dSep_contraction are deferred until that stronger condition
       is added, as weak union is false with this simplified version).
    - If the node is a non-collider: the node **is** in `W`. -/
def trailBlockedAtPos (G : Graph) (t : Trail) (pos : ℕ) (W : Finset Node) : Prop :=
  1 ≤ pos ∧ ∃ h : pos < t.length,
    let node := trailNodeAt t pos h
    (isCollider t pos → node ∉ W) ∧
    (¬isCollider t pos → node ∈ W)

/-- A trail is **blocked** by `W` if there exists a position where it is blocked. -/
def trailBlocked (G : Graph) (t : Trail) (W : Finset Node) : Prop :=
  ∃ pos, 1 ≤ pos ∧ pos < t.length ∧ trailBlockedAtPos G t pos W

-- ======================================================================
-- L2-004: D-Separation predicate
-- ======================================================================

/-- D-separation: Y and Z are **d-separated** given W in G iff every trail
    from any y ∈ Y to any z ∈ Z is blocked by W.

    Corresponds to Dafny's `ghost predicate DSep`.

    This is a `Prop` (not computable). -/
def dSep (G : Graph) (Y Z W : Finset Node) : Prop :=
  ∀ (t : Trail) (y z : Node),
    y ∈ Y → z ∈ Z →
    trailValid G t →
    trailConnected t →
    trailConnects t y z →
    trailBlocked G t W

-- ======================================================================
-- L4-003: Trail reversal (for dSep_symmetry)
-- ======================================================================

/-- Flip edge direction. -/
private def flipDir : EdgeDir → EdgeDir
  | EdgeDir.Forward  => EdgeDir.Backward
  | EdgeDir.Backward => EdgeDir.Forward

private lemma flipDir_invol (d : EdgeDir) : flipDir (flipDir d) = d := by
  cases d <;> rfl

/-- Flip a trail step: swap src/dst and flip direction.
    Corresponds to Dafny's `ReverseStep`. -/
private def flipStep (s : TrailStep) : TrailStep :=
  { src := s.dst, dst := s.src, dir := flipDir s.dir }

private lemma flipStep_invol (s : TrailStep) : flipStep (flipStep s) = s := by
  obtain ⟨_, _, d⟩ := s; cases d <;> simp [flipStep, flipDir]

/-- Reverse a trail: reverse step order and flip each step.
    Corresponds to Dafny's `ReverseTrail`. -/
private def reverseTrail (t : Trail) : Trail := t.reverse.map flipStep

private lemma reverseTrail_length (t : Trail) :
    (reverseTrail t).length = t.length := by simp [reverseTrail]

private lemma reverseTrail_invol (t : Trail) : reverseTrail (reverseTrail t) = t := by
  simp only [reverseTrail, ← List.map_reverse, List.map_map]
  conv_lhs => simp only [show flipStep ∘ flipStep = id from funext flipStep_invol]
  simp

/-- Helper: index equality for `getElem` — avoids dependent rewrite issues. -/
private lemma getElem_idx_eq {α : Type*} (l : List α) {i j : ℕ} (hij : i = j)
    (hi : i < l.length) : l[i]'hi = l[j]'(hij ▸ hi) := by subst hij; rfl

private lemma reverseTrail_getElem (t : Trail) (j : ℕ) (hj : j < t.length) :
    (reverseTrail t)[j]'(by simp [reverseTrail_length]; exact hj) =
    flipStep (t[t.length - 1 - j]'(by omega)) := by
  simp [reverseTrail, List.getElem_map, List.getElem_reverse]

/-- Variant of `reverseTrail_getElem` accepting a bound proof against `(reverseTrail t).length`
    directly, which is needed when the proof comes from an existential in `isCollider`. -/
private lemma reverseTrail_getElem' (t : Trail) (j : ℕ) (hj : j < (reverseTrail t).length) :
    (reverseTrail t)[j]'hj =
    flipStep (t[t.length - 1 - j]'(by rw [reverseTrail_length] at hj; omega)) := by
  simp [reverseTrail, List.getElem_map, List.getElem_reverse]

/-- Reversed trail is valid. Corresponds to Dafny `ReverseTrail_Valid`. -/
private lemma reverseTrail_valid (G : Graph) (t : Trail) (hv : trailValid G t) :
    trailValid G (reverseTrail t) := by
  intro s hs
  simp only [reverseTrail, List.mem_map, List.mem_reverse] at hs
  obtain ⟨s', hmem, rfl⟩ := hs
  have hv' := hv s' hmem
  simp only [stepValid] at hv' ⊢
  cases h : s'.dir <;> simp only [flipStep, flipDir, h] at * <;> exact hv'

/-- Reversed trail is connected. Corresponds to Dafny `ReverseTrail_Valid` (connectivity). -/
private lemma reverseTrail_connected (t : Trail) (hc : trailConnected t) :
    trailConnected (reverseTrail t) := by
  intro ⟨j, hj⟩
  simp only [reverseTrail_length] at hj
  simp only [List.get_eq_getElem]
  rw [reverseTrail_getElem t j (by omega), reverseTrail_getElem t (j + 1) (by omega)]
  simp only [flipStep]
  -- Goal: t[n-1-j].src = t[n-1-(j+1)].dst
  -- From hc at k = n-1-j-1: t[k].dst = t[k+1].src = t[n-1-j].src
  set n := t.length
  have hk : n - 1 - j - 1 < n - 1 := by omega
  have hck := hc ⟨n - 1 - j - 1, hk⟩
  simp only [List.get_eq_getElem] at hck
  -- hck: t[n-1-j-1].dst = t[n-1-j-1+1].src; rewrite +1 index to n-1-j
  rw [getElem_idx_eq t (show n - 1 - j - 1 + 1 = n - 1 - j by omega)] at hck
  rw [getElem_idx_eq t (show n - 1 - (j + 1) = n - 1 - j - 1 by omega)]
  exact hck.symm

/-- Reversed trail connects z→y when original connects y→z.
    Corresponds to Dafny `ReverseTrail_Connects`. -/
private lemma reverseTrail_connects (t : Trail) (y z : Node)
    (htc : trailConnects t y z) : trailConnects (reverseTrail t) z y := by
  obtain ⟨hne, hhead, hlast⟩ := htc
  refine ⟨by simp [reverseTrail, hne], ?_, ?_⟩
  · -- head of reverseTrail t has src = z
    simp only [reverseTrail, List.head?_map, List.head?_reverse, Option.map_map]
    simpa [Function.comp, flipStep] using hlast
  · -- getLast of reverseTrail t has dst = y
    simp only [reverseTrail, List.getLast?_map, List.getLast?_reverse, Option.map_map]
    simpa [Function.comp, flipStep] using hhead

/-- Blocking is preserved under reversal.
    Combines Dafny's `ReverseTrail_Blocked` + involution. -/
private lemma reverseTrail_blocked (G : Graph) (t : Trail) (W : Finset Node)
    (hc : trailConnected t) (hb : trailBlocked G t W) :
    trailBlocked G (reverseTrail t) W := by
  obtain ⟨pos, h1, h2, _, ⟨hlt, hcol, hncol⟩⟩ := hb
  have hn : (reverseTrail t).length = t.length := reverseTrail_length t
  have hrev_pos : t.length - pos < (reverseTrail t).length := by omega
  have hj1 : t.length - pos - 1 < (reverseTrail t).length := by omega
  -- Node equality: trailNodeAt (reverseTrail t) (t.length - pos) = trailNodeAt t pos
  -- (reverseTrail t)[t.length-pos] = flipStep(t[pos-1]), so .src = t[pos-1].dst = t[pos].src
  have hnode : trailNodeAt (reverseTrail t) (t.length - pos) hrev_pos =
               trailNodeAt t pos hlt := by
    simp only [trailNodeAt, List.get_eq_getElem,
               reverseTrail_getElem' t (t.length - pos) hrev_pos, flipStep]
    -- Goal: (t[t.length-1-(t.length-pos)]'_).dst = (t[pos]'hlt).src
    have hcc := hc ⟨pos - 1, by omega⟩
    simp only [List.get_eq_getElem] at hcc
    -- hcc: t[pos-1].dst = t[pos-1+1].src (getElem form)
    have lhs : t[t.length - 1 - (t.length - pos)]'(by rw [hn] at hrev_pos; omega) =
               t[pos - 1]'(by omega) :=
      (getElem_idx_eq t (by omega : t.length - 1 - (t.length - pos) = pos - 1) _).trans rfl
    have rhs : t[pos - 1 + 1]'(by omega) = t[pos]'hlt :=
      (getElem_idx_eq t (by omega : pos - 1 + 1 = pos) _).trans rfl
    simp only [lhs, hcc, rhs]
  -- isCollider equivalence
  -- (reverseTrail t)[t.length-pos-1] = flipStep(t[pos])   → .dir = flipDir(t[pos].dir)
  -- (reverseTrail t)[t.length-pos]   = flipStep(t[pos-1]) → .dir = flipDir(t[pos-1].dir)
  have hcol_iff : isCollider t pos ↔ isCollider (reverseTrail t) (t.length - pos) := by
    constructor
    · rintro ⟨_, h_orig2, hdF, hdB⟩
      -- Normalize from `t.get ⟨k, _⟩` to `t[k]'_` form
      simp only [List.get_eq_getElem] at hdF hdB
      refine ⟨hj1, hrev_pos, ?_, ?_⟩
      · -- (reverseTrail t)[t.length-pos-1].dir = Forward
        --   = flipDir(t[pos].dir) = flipDir Backward = Forward ✓
        simp only [List.get_eq_getElem, reverseTrail_getElem' t (t.length-pos-1) hj1, flipStep]
        -- goal: flipDir (t[t.length-1-(t.length-pos-1)]'_).dir = Forward
        have heq : t[t.length - 1 - (t.length - pos - 1)]'(by rw [hn] at hj1; omega) =
                   t[pos]'h_orig2 :=
          (getElem_idx_eq t (by omega : t.length - 1 - (t.length - pos - 1) = pos) _).trans rfl
        simp only [heq, hdB]  -- reduces to flipDir Backward = Forward
        rfl
      · -- (reverseTrail t)[t.length-pos].dir = Backward
        --   = flipDir(t[pos-1].dir) = flipDir Forward = Backward ✓
        simp only [List.get_eq_getElem, reverseTrail_getElem' t (t.length-pos) hrev_pos, flipStep]
        have heq : t[t.length - 1 - (t.length - pos)]'(by rw [hn] at hrev_pos; omega) =
                   t[pos - 1]'(by omega) :=
          (getElem_idx_eq t (by omega : t.length - 1 - (t.length - pos) = pos - 1) _).trans rfl
        simp only [heq, hdF]  -- reduces to flipDir Forward = Backward
        rfl
    · rintro ⟨hr1, hr2, hdF', hdB'⟩
      simp only [List.get_eq_getElem, reverseTrail_getElem' t (t.length-pos-1) hr1,
                 flipStep] at hdF'
      simp only [List.get_eq_getElem, reverseTrail_getElem' t (t.length-pos) hr2,
                 flipStep] at hdB'
      -- hdF' : flipDir (t[t.length-1-(t.length-pos-1)]'_).dir = Forward
      -- hdB' : flipDir (t[t.length-1-(t.length-pos)]'_).dir = Backward
      -- Normalize indices to pos and pos-1
      have heqF : t[t.length - 1 - (t.length - pos - 1)]'(by rw [hn] at hr1; omega) =
                  t[pos]'hlt :=
        (getElem_idx_eq t (by omega : t.length - 1 - (t.length - pos - 1) = pos) _).trans rfl
      have heqB : t[t.length - 1 - (t.length - pos)]'(by rw [hn] at hr2; omega) =
                  t[pos - 1]'(by omega) :=
        (getElem_idx_eq t (by omega : t.length - 1 - (t.length - pos) = pos - 1) _).trans rfl
      simp only [heqF] at hdF'; simp only [heqB] at hdB'
      -- hdF' : flipDir (t[pos]'hlt).dir = Forward  → t[pos].dir = Backward
      -- hdB' : flipDir (t[pos-1]'_).dir = Backward → t[pos-1].dir = Forward
      have hdirPos : (t.get ⟨pos, hlt⟩).dir = EdgeDir.Backward := by
        rw [List.get_eq_getElem]
        cases h : (t[pos]'hlt).dir <;> simp_all [flipDir]
      have hdirPred : (t.get ⟨pos - 1, by omega⟩).dir = EdgeDir.Forward := by
        rw [List.get_eq_getElem]
        cases h : (t[pos - 1]'(by omega)).dir <;> simp_all [flipDir]
      exact ⟨by omega, hlt, hdirPred, hdirPos⟩
  -- Construct the blocked position at t.length - pos in reverseTrail t
  refine ⟨t.length - pos, by omega, hrev_pos, by omega, hrev_pos, ?_, ?_⟩
  · intro hcol_rev; rw [hnode]; exact hcol (hcol_iff.mpr hcol_rev)
  · intro hncol_rev; rw [hnode]; exact hncol (fun hco => hncol_rev (hcol_iff.mp hco))

-- ======================================================================
-- L4-003: Semi-Graphoid Axioms
-- ======================================================================

/-- **Decomposition**: (Y ⊥ Z∪Z' | W) → (Y ⊥ Z | W).
    Trivial: z ∈ Z implies z ∈ Z∪Z', so the blocking trail exists.
    Corresponds to Dafny `DSep_Decomposition`. -/
theorem dSep_decomposition (G : Graph) (Y Z Z' W : Finset Node) :
    dSep G Y (Z ∪ Z') W → dSep G Y Z W := by
  intro h t y z hy hz hv hc htc
  exact h t y z hy (Finset.mem_union_left Z' hz) hv hc htc

/-- **Symmetry**: (Y ⊥ Z | W) → (Z ⊥ Y | W).
    Proved by reversing the trail and applying the hypothesis.
    Corresponds to Dafny `DSep_Symmetry` / `ReverseTrail_Blocked`. -/
theorem dSep_symmetry (G : Graph) (Y Z W : Finset Node) :
    dSep G Y Z W → dSep G Z Y W := by
  intro h t z y hz hy hv hc htc
  -- Apply h to the reversed trail (which connects y → z)
  have hrev_v  := reverseTrail_valid G t hv
  have hrev_c  := reverseTrail_connected t hc
  have hrev_tc := reverseTrail_connects t z y htc
  have hblk    := h (reverseTrail t) y z hy hz hrev_v hrev_c hrev_tc
  -- Blocked in reverseTrail(reverseTrail(t)) = t, via involution
  rw [← reverseTrail_invol t]
  exact reverseTrail_blocked G (reverseTrail t) W hrev_c hblk

/-- **Weak Union**: (Y ⊥ Z∪Z' | W) → (Y ⊥ Z | W∪Z').
    Deferred: requires the full descendants-aware blocking condition
    (node ∉ W ∧ Descendants(G,{node}) ∩ W = ∅ for colliders).
    With the current simplified definition, weak union is false in general.
    See also: CausalQIF `BayesBall` approach for an alternative proof strategy. -/
theorem dSep_weakUnion (G : Graph) (Y Z Z' W : Finset Node) :
    dSep G Y (Z ∪ Z') W → dSep G Y Z (W ∪ Z') := by sorry

/-- **Contraction**: (Y ⊥ Z | W) ∧ (Y ⊥ Z' | W∪Z) → (Y ⊥ Z∪Z' | W).
    Deferred: most complex semi-graphoid axiom; requires descendant
    trail construction (see Dafny `DSep_Contraction`). -/
theorem dSep_contraction (G : Graph) (Y Z Z' W : Finset Node) :
    dSep G Y Z W → dSep G Y Z' (W ∪ Z) → dSep G Y (Z ∪ Z') W := by sorry

end Y0Lean
