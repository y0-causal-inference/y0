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
import Y0Lean.Traversal

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

/-- A trail is **blocked at position `pos`** by `W` (Pearl's full d-separation condition):
    - If the node at `pos` is a **collider** (→ node ←): the node is not in `W`
      **and** none of its descendants are in `W`
      (i.e. `node ∉ W ∧ descendants G {node} ∩ W = ∅`).
    - If the node is a **non-collider**: the node **is** in `W`.
    Corresponds to Dafny's `TrailBlockedAtPos`. -/
def trailBlockedAtPos (G : Graph) (t : Trail) (pos : ℕ) (W : Finset Node) : Prop :=
  1 ≤ pos ∧ ∃ h : pos < t.length,
    let node := trailNodeAt t pos h
    (isCollider t pos → node ∉ W ∧ descendants G {node} ∩ W = ∅) ∧
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
-- L4-003: Phase 1 Infrastructure — trail prefix helpers
-- ======================================================================

/-- If a trail is not blocked by W, then no individual position is blocked.
    Contrapositive of the existential in `trailBlocked`. -/
private lemma trailNotBlockedAtPos (G : Graph) (t : Trail) (pos : ℕ) (W : Finset Node)
    (h : ¬trailBlocked G t W) : ¬trailBlockedAtPos G t pos W := by
  intro hbp
  exact h ⟨pos, hbp.1, hbp.2.choose, hbp⟩

/-- Taking a prefix of a valid trail gives a valid trail. -/
private lemma trailValid_prefix (G : Graph) (t : Trail) (pos : ℕ)
    (hv : trailValid G t) : trailValid G (t.take pos) :=
  fun s hs => hv s (List.mem_of_mem_take hs)

/-- Taking a prefix of a connected trail gives a connected trail.
    Key step: `List.getElem_take` (a `@[simp]` lemma) strips `take` from indexed
    access, reducing the prefix connectivity to the original. -/
private lemma trailConnected_prefix (t : Trail) (pos : ℕ)
    (hc : trailConnected t) : trailConnected (t.take pos) := by
  intro ⟨j, hj⟩
  simp only [List.length_take] at hj
  have hj1_t : j + 1 < t.length := by omega
  have key := hc ⟨j, by omega⟩
  simp only [List.get_eq_getElem] at key ⊢
  simp only [List.getElem_take]
  exact key

-- ======================================================================
-- L4-004: Trail concatenation, prefix-blocking, first-blocked helpers
-- (Dafny port for weak union / contraction)
-- ======================================================================

/-- Concatenation of two valid trails is valid. -/
private lemma trailValid_concat (G : Graph) (l r : Trail)
    (hl : trailValid G l) (hr : trailValid G r) :
    trailValid G (l ++ r) := by
  intro s hs
  rcases List.mem_append.mp hs with h | h
  exacts [hl s h, hr s h]

/-- Connecting endpoints: if `l` connects `y→mid` and `r` connects `mid→z`,
    then `l ++ r` connects `y→z`. -/
private lemma trailConnects_concat (l r : Trail) (y mid z : Node)
    (hl : trailConnects l y mid) (hr : trailConnects r mid z) :
    trailConnects (l ++ r) y z := by
  obtain ⟨hln, hlh, hll⟩ := hl
  obtain ⟨hrn, _, hrl⟩ := hr
  refine ⟨?_, ?_, ?_⟩
  · simp [List.append_eq_nil_iff, hln, hrn]
  · rw [List.head?_append_of_ne_nil _ hln]; exact hlh
  · rw [List.getLast?_append_of_ne_nil _ hrn]; exact hrl

/-- Last-step destination of a trail that connects `y → z`. -/
private lemma trailConnects_getLast_dst
    {t : Trail} {y z : Node} (htc : trailConnects t y z) :
    ∃ h : 0 < t.length, (t[t.length - 1]'(by omega)).dst = z := by
  obtain ⟨hne, _, hlast⟩ := htc
  have hlen : 0 < t.length := by
    cases t with | nil => exact absurd rfl hne | cons _ _ => simp
  refine ⟨hlen, ?_⟩
  rw [List.getLast?_eq_getElem?,
      List.getElem?_eq_getElem (show t.length - 1 < t.length by omega)] at hlast
  simpa using hlast

/-- First-step source of a trail that connects `y → z`. -/
private lemma trailConnects_get_src
    {t : Trail} {y z : Node} (htc : trailConnects t y z) :
    ∃ h : 0 < t.length, (t[0]'h).src = y := by
  obtain ⟨hne, hhead, _⟩ := htc
  have hlen : 0 < t.length := by
    cases t with | nil => exact absurd rfl hne | cons _ _ => simp
  refine ⟨hlen, ?_⟩
  rw [List.head?_eq_getElem?, List.getElem?_eq_getElem hlen] at hhead
  simpa using hhead

/-- Connectedness of a concatenation, given the middle node from `trailConnects`
    on each side (which forces the boundary step's `dst` to equal the next
    step's `src`). -/
private lemma trailConnected_concat (l r : Trail) (y mid z : Node)
    (hcl : trailConnected l) (hcr : trailConnected r)
    (hl : trailConnects l y mid) (hr : trailConnects r mid z) :
    trailConnected (l ++ r) := by
  have hl_last := trailConnects_getLast_dst hl
  have hr_first := trailConnects_get_src hr
  obtain ⟨hl_pos, hl_eq⟩ := hl_last
  obtain ⟨hr_pos, hr_eq⟩ := hr_first
  intro ⟨j, hj⟩
  simp only [List.length_append] at hj
  simp only [List.get_eq_getElem]
  by_cases h1 : j + 1 < l.length
  · -- Both indices fall inside `l`.
    have hjl : j < l.length := by omega
    have key := hcl ⟨j, by omega⟩
    simp only [List.get_eq_getElem] at key
    rw [List.getElem_append_left hjl, List.getElem_append_left h1]
    exact key
  · push_neg at h1
    by_cases hj_in_l : j < l.length
    · -- Boundary: `j` in `l`, `j+1` in `r`.
      have e1 : (l ++ r)[j]'(by simp [List.length_append]; omega) =
                l[l.length - 1]'(by omega) := by
        rw [List.getElem_append_left hj_in_l]
        exact getElem_idx_eq l (by omega : j = l.length - 1) hj_in_l
      have hj1_idx : j + 1 - l.length < r.length := by omega
      have e2 : (l ++ r)[j + 1]'(by simp [List.length_append]; omega) =
                r[0]'hr_pos := by
        rw [List.getElem_append_right h1]
        exact getElem_idx_eq r (by omega : j + 1 - l.length = 0) hj1_idx
      rw [e1, e2, hl_eq, ← hr_eq]
    · -- Both indices in `r`.
      push_neg at hj_in_l
      have hjr : j - l.length < r.length := by omega
      have hjr1 : j + 1 - l.length < r.length := by omega
      have key := hcr ⟨j - l.length, by omega⟩
      simp only [List.get_eq_getElem] at key
      have e1 : (l ++ r)[j]'(by simp [List.length_append]; omega) =
                r[j - l.length]'hjr := by
        rw [List.getElem_append_right hj_in_l]
      have e2 : (l ++ r)[j + 1]'(by simp [List.length_append]; omega) =
                r[(j - l.length) + 1]'(by omega) := by
        rw [List.getElem_append_right h1]
        exact getElem_idx_eq r (by omega : j + 1 - l.length = (j - l.length) + 1) hjr1
      rw [e1, e2]
      exact key

/-- `(t.take k)[i]` agrees with `t[i]` for `i < k ∧ i < t.length`. -/
private lemma getElem_take_eq (t : Trail) {k i : ℕ}
    (hp_take : i < (t.take k).length) (hp_t : i < t.length) :
    (t.take k)[i]'hp_take = t[i]'hp_t := by
  simp [List.getElem_take]

/-- The prefix `t.take pos` connects `y` to the destination of step `pos-1`. -/
private lemma trailConnects_prefix
    {t : Trail} {y z : Node} {pos : ℕ}
    (htc : trailConnects t y z) (hpos1 : 1 ≤ pos) (hpos_t : pos ≤ t.length) :
    trailConnects (t.take pos) y ((t[pos - 1]'(by omega)).dst) := by
  obtain ⟨hne, hhead, _⟩ := htc
  have hlen_t : 0 < t.length := by
    cases t with | nil => exact absurd rfl hne | cons _ _ => simp
  have hlen_take : (t.take pos).length = pos := by
    rw [List.length_take]; omega
  have hne_take : t.take pos ≠ [] := by
    intro habs
    have : (t.take pos).length = 0 := by rw [habs]; rfl
    omega
  refine ⟨hne_take, ?_, ?_⟩
  · have hp0_take : 0 < (t.take pos).length := by omega
    have h_head_take : (t.take pos).head? = some ((t.take pos)[0]'hp0_take) := by
      rw [List.head?_eq_getElem?]; exact List.getElem?_eq_getElem hp0_take
    have h_head_t : t.head? = some (t[0]'hlen_t) := by
      rw [List.head?_eq_getElem?]; exact List.getElem?_eq_getElem hlen_t
    rw [h_head_t] at hhead
    rw [h_head_take]
    simp only [Option.map_some, Option.some.injEq] at hhead ⊢
    rw [getElem_take_eq t hp0_take hlen_t]; exact hhead
  · have hp_last_take : (t.take pos).length - 1 < (t.take pos).length := by omega
    have h_last_take :
        (t.take pos).getLast? = some ((t.take pos)[(t.take pos).length - 1]'hp_last_take) := by
      rw [List.getLast?_eq_getElem?]; exact List.getElem?_eq_getElem hp_last_take
    rw [h_last_take]
    simp only [Option.map_some, Option.some.injEq]
    have hidx : (t.take pos).length - 1 = pos - 1 := by omega
    have hpm1_take : pos - 1 < (t.take pos).length := by omega
    have hpm1_t : pos - 1 < t.length := by omega
    rw [getElem_idx_eq _ hidx hp_last_take, getElem_take_eq t hpm1_take hpm1_t]

-- ----------------------------------------------------------------------
-- Prefix-blocking helpers
-- ----------------------------------------------------------------------



/-- Being a collider at `pos` is invariant under taking a long-enough prefix. -/
private lemma isCollider_take_iff (t : Trail) {k pos : ℕ}
    (h1 : 1 ≤ pos) (hp_t : pos < t.length) (hk : pos < k) :
    isCollider (t.take k) pos ↔ isCollider t pos := by
  have hp_take : pos < (t.take k).length := by
    rw [List.length_take]; omega
  have hpm1_take : pos - 1 < (t.take k).length := by
    rw [List.length_take]; omega
  have hpm1_t : pos - 1 < t.length := by omega
  constructor
  · rintro ⟨_, _, hF, hB⟩
    simp only [List.get_eq_getElem] at hF hB
    refine ⟨hpm1_t, hp_t, ?_, ?_⟩
    · simp only [List.get_eq_getElem]
      rw [← getElem_take_eq t hpm1_take hpm1_t]; exact hF
    · simp only [List.get_eq_getElem]
      rw [← getElem_take_eq t hp_take hp_t]; exact hB
  · rintro ⟨_, _, hF, hB⟩
    simp only [List.get_eq_getElem] at hF hB
    refine ⟨hpm1_take, hp_take, ?_, ?_⟩
    · simp only [List.get_eq_getElem]
      rw [getElem_take_eq t hpm1_take hpm1_t]; exact hF
    · simp only [List.get_eq_getElem]
      rw [getElem_take_eq t hp_take hp_t]; exact hB

/-- Being blocked-at-pos by `W` is invariant under long-enough prefixes. -/
private lemma trailBlockedAtPos_take_iff (G : Graph) (t : Trail) (W : Finset Node)
    {k pos : ℕ} (h1 : 1 ≤ pos) (hp_t : pos < t.length) (hk : pos < k) :
    trailBlockedAtPos G (t.take k) pos W ↔ trailBlockedAtPos G t pos W := by
  have hp_take : pos < (t.take k).length := by
    rw [List.length_take]; omega
  have hnode_eq : trailNodeAt (t.take k) pos hp_take = trailNodeAt t pos hp_t := by
    simp only [trailNodeAt, List.get_eq_getElem]
    rw [getElem_take_eq t hp_take hp_t]
  have hcol_iff := isCollider_take_iff t (k := k) (pos := pos) h1 hp_t hk
  constructor
  · rintro ⟨_, _, hcol, hncol⟩
    refine ⟨h1, hp_t, ?_, ?_⟩
    · intro hc_t
      have := hcol (hcol_iff.mpr hc_t)
      rwa [hnode_eq] at this
    · intro hnc_t
      have := hncol (fun h => hnc_t (hcol_iff.mp h))
      rwa [hnode_eq] at this
  · rintro ⟨_, _, hcol, hncol⟩
    refine ⟨h1, hp_take, ?_, ?_⟩
    · intro hc_take
      have := hcol (hcol_iff.mp hc_take)
      rwa [← hnode_eq] at this
    · intro hnc_take
      have := hncol (fun h => hnc_take (hcol_iff.mpr h))
      rwa [← hnode_eq] at this

/-- If no position `1 ≤ j < k` is blocked in `t`, then the prefix `t.take k`
    is unblocked. Corresponds to Dafny `PrefixWithoutBlockedPos_NotBlocked`. -/
private lemma prefixWithoutBlockedPos_notBlocked
    (G : Graph) (t : Trail) (k : ℕ) (W : Finset Node)
    (hno : ∀ j, 1 ≤ j → j < k → ¬ trailBlockedAtPos G t j W) :
    ¬ trailBlocked G (t.take k) W := by
  rintro ⟨pos, hpos1, hposlt, hbp⟩
  rw [List.length_take] at hposlt
  have hpos_k : pos < k := lt_of_lt_of_le hposlt (Nat.min_le_left _ _)
  have hpos_t : pos < t.length := lt_of_lt_of_le hposlt (Nat.min_le_right _ _)
  exact hno pos hpos1 hpos_k
    ((trailBlockedAtPos_take_iff G t W hpos1 hpos_t hpos_k).mp hbp)

/-- The least blocked position in a blocked trail (Dafny `FirstBlockedPos`). -/
private lemma firstBlockedPos (G : Graph) (t : Trail) (W : Finset Node)
    (h : trailBlocked G t W) :
    ∃ pos, 1 ≤ pos ∧ pos < t.length ∧ trailBlockedAtPos G t pos W ∧
      ∀ j, 1 ≤ j → j < pos → ¬ trailBlockedAtPos G t j W := by
  classical
  let P : ℕ → Prop := fun k => 1 ≤ k ∧ k < t.length ∧ trailBlockedAtPos G t k W
  have hex : ∃ k, P k := by
    obtain ⟨n, hn1, hnlt, hnbp⟩ := h
    exact ⟨n, hn1, hnlt, hnbp⟩
  refine ⟨Nat.find hex, (Nat.find_spec hex).1, (Nat.find_spec hex).2.1,
          (Nat.find_spec hex).2.2, ?_⟩
  intro j hj1 hjlt hbj
  have hj_t : j < t.length := lt_of_lt_of_le hjlt
    (le_of_lt (Nat.find_spec hex).2.1 |>.trans (le_refl _))
  -- Simpler: j < Nat.find hex ≤ t.length implicitly; just use Nat.find_min'
  have : ¬ P j := Nat.find_min hex hjlt
  exact this ⟨hj1, by
    -- Need j < t.length. From hjlt: j < Nat.find hex, and Nat.find_spec gives Nat.find hex < t.length
    have := (Nat.find_spec hex).2.1; omega, hbj⟩

-- ----------------------------------------------------------------------
-- Conditioning-change lemmas (Dafny `BlockingAddedByConditioningAtPos`
-- and `ColliderOpenedByNewConditioning`).
-- ----------------------------------------------------------------------

/-- If `pos` is not blocked under `W` but becomes blocked under `W ∪ Z'`,
    then `pos` is a non-collider whose node lies in `Z' \ W`. -/
private lemma blockingAddedByConditioningAtPos
    (G : Graph) (t : Trail) (pos : ℕ) (W Z' : Finset Node)
    (hnb : ¬ trailBlockedAtPos G t pos W)
    (hb : trailBlockedAtPos G t pos (W ∪ Z')) :
    ∃ h : pos < t.length,
      ¬ isCollider t pos ∧
      trailNodeAt t pos h ∈ Z' ∧
      trailNodeAt t pos h ∉ W := by
  obtain ⟨h1, hp, hcol_b, hncol_b⟩ := hb
  -- ¬ isCollider t pos
  have hncol : ¬ isCollider t pos := by
    intro hcol
    have hWZ' := hcol_b hcol
    apply hnb
    refine ⟨h1, hp, ?_, fun hnc => absurd hcol hnc⟩
    intro _
    refine ⟨fun hin => hWZ'.1 (Finset.mem_union_left Z' hin), ?_⟩
    apply Finset.eq_empty_iff_forall_notMem.mpr
    intro x hx
    obtain ⟨hxd, hxw⟩ := Finset.mem_inter.mp hx
    have hxd' : x ∈ descendants G {trailNodeAt t pos hp} ∩ (W ∪ Z') :=
      Finset.mem_inter.mpr ⟨hxd, Finset.mem_union_left Z' hxw⟩
    rw [hWZ'.2] at hxd'
    exact (Finset.notMem_empty _) hxd'
  -- node ∈ W∪Z'
  have hnode_WZ' := hncol_b hncol
  -- node ∉ W
  have hnode_nW : trailNodeAt t pos hp ∉ W := by
    intro hin
    apply hnb
    exact ⟨h1, hp, fun hcol => absurd hcol hncol, fun _ => hin⟩
  refine ⟨hp, hncol, ?_, hnode_nW⟩
  rcases Finset.mem_union.mp hnode_WZ' with h | h
  · exact absurd h hnode_nW
  · exact h

/-- If a collider `pos` is blocked under `W` but unblocked under `W ∪ Z'`,
    then some `zPrime ∈ Z' \ W` equals the collider node or is a descendant. -/
private lemma colliderOpenedByNewConditioning
    (G : Graph) (t : Trail) (pos : ℕ) (W Z' : Finset Node)
    (h1 : 1 ≤ pos) (hp : pos < t.length)
    (hcol : isCollider t pos)
    (hb : trailBlockedAtPos G t pos W)
    (hnb : ¬ trailBlockedAtPos G t pos (W ∪ Z')) :
    ∃ zPrime ∈ Z', zPrime ∉ W ∧
      (zPrime = trailNodeAt t pos hp ∨
       zPrime ∈ descendants G {trailNodeAt t pos hp}) := by
  classical
  obtain ⟨_, _, hcol_b, _⟩ := hb
  obtain ⟨hnode_nW, hdesc_W⟩ := hcol_b hcol
  by_contra hne
  apply hnb
  refine ⟨h1, hp, ?_, fun hnc => absurd hcol hnc⟩
  intro _
  refine ⟨?_, ?_⟩
  · -- node ∉ W∪Z'
    intro hin
    rcases Finset.mem_union.mp hin with hw | hz
    · exact hnode_nW hw
    · exact hne ⟨trailNodeAt t pos hp, hz, hnode_nW, Or.inl rfl⟩
  · -- desc ∩ (W∪Z') = ∅
    apply Finset.eq_empty_iff_forall_notMem.mpr
    intro x hx
    obtain ⟨hxd, hxwz⟩ := Finset.mem_inter.mp hx
    rcases Finset.mem_union.mp hxwz with hw | hz
    · have : x ∈ descendants G {trailNodeAt t pos hp} ∩ W :=
        Finset.mem_inter.mpr ⟨hxd, hw⟩
      rw [hdesc_W] at this
      exact (Finset.notMem_empty _) this
    · refine hne ⟨x, hz, ?_, Or.inr hxd⟩
      intro hxw
      have : x ∈ descendants G {trailNodeAt t pos hp} ∩ W :=
        Finset.mem_inter.mpr ⟨hxd, hxw⟩
      rw [hdesc_W] at this
      exact (Finset.notMem_empty _) this

-- ----------------------------------------------------------------------
-- Forward-trail axiom (option (i): minimum axiom needed to discharge
-- `dSep_weakUnion`; corresponds to Dafny `IsAncestorBounded_ImpliesForwardTrail`
-- + `ForwardTrail_NodeInDescendants` bundled into one statement).
-- ----------------------------------------------------------------------

/-- **Axiom**: For any `d ∈ descendants G {n}` with `d ≠ n`, there is an
    explicit *forward-only* trail from `n` to `d` whose every step lands
    inside `descendants G {n}`.

    Discharging this constructively requires recovering a witnessing path
    from the BFS expansion in `Traversal.descendants`; we keep it as an
    axiom here (the agreed "option (i)" of the de-axiomatization plan). -/
axiom forwardTrail_of_mem_descendants
    (G : Graph) (n d : Node) (h : d ∈ descendants G {n}) (hne : d ≠ n) :
    ∃ t : Trail, trailValid G t ∧ trailConnected t ∧ trailConnects t n d ∧
      (∀ s ∈ t, s.dir = EdgeDir.Forward) ∧
      (∀ s ∈ t, s.dst ∈ descendants G {n})

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
    Corresponds to Dafny `DSep_WeakUnion`. -/
theorem dSep_weakUnion (G : Graph) (Y Z Z' W : Finset Node) :
    dSep G Y (Z ∪ Z') W → dSep G Y Z (W ∪ Z') := by
  intro hUnion t y z hy hzZ hv hc htc
  by_contra hnb
  have hzZZ' : z ∈ Z ∪ Z' := Finset.mem_union_left _ hzZ
  -- Whole trail blocked under W (by hypothesis with z∈Z⊆Z∪Z').
  have hbW : trailBlocked G t W := hUnion t y z hy hzZZ' hv hc htc
  obtain ⟨pos, hpos1, hposlt, hbpos, hmin⟩ := firstBlockedPos G t W hbW
  have hnbpos_WZ' : ¬ trailBlockedAtPos G t pos (W ∪ Z') :=
    trailNotBlockedAtPos G t pos (W ∪ Z') hnb
  -- Destructure block at pos.
  obtain ⟨_, _, hbcol_W, hbncol_W⟩ := hbpos
  -- pos must be a collider; otherwise node ∈ W ⊆ W∪Z' opens the block.
  have hcol : isCollider t pos := by
    by_contra hncol
    have hnode_W := hbncol_W hncol
    apply hnbpos_WZ'
    refine ⟨hpos1, hposlt, fun hc' => absurd hc' hncol,
            fun _ => Finset.mem_union_left _ hnode_W⟩
  obtain ⟨hnode_nW, hdesc_W⟩ := hbcol_W hcol
  -- The collider node.
  set colNode : Node := trailNodeAt t pos hposlt with hcolNode_def
  -- Use Collider-opened-by-new-conditioning.
  obtain ⟨zPrime, hzP_Z', hzP_nW, hzP_or⟩ :=
    colliderOpenedByNewConditioning G t pos W Z' hpos1 hposlt hcol
      ⟨hpos1, hposlt, hbcol_W, hbncol_W⟩ hnbpos_WZ'
  -- Prefix machinery: prefix = t.take pos.
  have hv_pref := trailValid_prefix G t pos hv
  have hc_pref := trailConnected_prefix t pos hc
  -- Connectivity: prefix connects y → colNode.
  have hcc := hc ⟨pos - 1, by omega⟩
  simp only [List.get_eq_getElem] at hcc
  have hidx : t[pos - 1 + 1]'(by omega) = t[pos]'hposlt :=
    getElem_idx_eq t (by omega : pos - 1 + 1 = pos) (by omega)
  rw [hidx] at hcc
  have htc_pref_raw := trailConnects_prefix htc hpos1 (le_of_lt hposlt)
  have htc_pref : trailConnects (t.take pos) y colNode := by
    simp only [hcolNode_def, trailNodeAt, List.get_eq_getElem]
    rw [← hcc]; exact htc_pref_raw
  have hpref_nb_W : ¬ trailBlocked G (t.take pos) W :=
    prefixWithoutBlockedPos_notBlocked G t pos W (fun j hj1 hjlt => hmin j hj1 hjlt)
  have hzP_in : zPrime ∈ Z ∪ Z' := Finset.mem_union_right _ hzP_Z'
  -- Case split: zPrime = colNode or zPrime ∈ descendants properly.
  by_cases hzP_eq : zPrime = colNode
  · -- Case A: zPrime is the collider node itself; use the prefix.
    have htc_pref_zP : trailConnects (t.take pos) y zPrime := by
      rw [hzP_eq]; exact htc_pref
    exact hpref_nb_W (hUnion (t.take pos) y zPrime hy hzP_in hv_pref hc_pref htc_pref_zP)
  · -- Case B: zPrime ≠ colNode, zPrime ∈ descendants colNode.
    have hzP_desc : zPrime ∈ descendants G {colNode} := by
      rcases hzP_or with h | h
      · exact absurd h hzP_eq
      · exact h
    -- Forward descendant trail.
    obtain ⟨descTrail, hdv, hdc, hdtc, hdfwd, hddst⟩ :=
      forwardTrail_of_mem_descendants G colNode zPrime hzP_desc hzP_eq
    -- Joined trail.
    have hv_join := trailValid_concat G _ _ hv_pref hdv
    have htc_join : trailConnects (t.take pos ++ descTrail) y zPrime :=
      trailConnects_concat _ _ y colNode zPrime htc_pref hdtc
    have hc_join : trailConnected (t.take pos ++ descTrail) :=
      trailConnected_concat _ _ y colNode zPrime hc_pref hdc htc_pref hdtc
    -- Lengths.
    have hpref_len : (t.take pos).length = pos := by
      rw [List.length_take]; omega
    have hdT_pos : 0 < descTrail.length := by
      obtain ⟨hdne, _, _⟩ := hdtc
      cases descTrail with | nil => exact absurd rfl hdne | cons _ _ => simp
    -- descTrail[0].src = colNode (from hdtc).
    obtain ⟨hdT_src_pos, hdT_src_eq⟩ := trailConnects_get_src hdtc
    -- Apply hUnion to the joined trail to get blocked under W.
    have hbjoined_W := hUnion _ y zPrime hy hzP_in hv_join hc_join htc_join
    -- Now derive contradiction: joined is unblocked under W.
    obtain ⟨q, hq1, hqlt, hbq⟩ := hbjoined_W
    simp only [List.length_append, hpref_len] at hqlt
    obtain ⟨_, _, hcolq_b, hncolq_b⟩ := hbq
    -- Three cases on q vs pos.
    rcases lt_trichotomy q pos with hqlt_pos | hqeq | hqgt
    · -- q < pos: position is fully in the prefix, contradicts hmin.
      have hq_t : q < t.length := by omega
      have hq_left : q < (t.take pos).length := by rw [hpref_len]; exact hqlt_pos
      have hqm1_left : q - 1 < (t.take pos).length := by rw [hpref_len]; omega
      have hqm1_t : q - 1 < t.length := by omega
      -- Build trailBlockedAtPos t q W.
      apply hmin q hq1 hqlt_pos
      refine ⟨hq1, hq_t, ?_, ?_⟩
      · -- collider t q → ...
        intro hcq_t
        -- Translate collider in t to collider in joined at q.
        have hcq_join : isCollider (t.take pos ++ descTrail) q := by
          obtain ⟨_, _, hF, hB⟩ := hcq_t
          simp only [List.get_eq_getElem] at hF hB
          refine ⟨by simp [List.length_append, hpref_len]; omega,
                  by simp [List.length_append, hpref_len]; omega, ?_, ?_⟩
          · simp only [List.get_eq_getElem, List.getElem_append_left hqm1_left,
                       getElem_take_eq t hqm1_left hqm1_t]
            exact hF
          · simp only [List.get_eq_getElem, List.getElem_append_left hq_left,
                       getElem_take_eq t hq_left hq_t]
            exact hB
        have hblk := hcolq_b hcq_join
        -- trailNodeAt joined q = trailNodeAt t q.
        have hnode_eq :
            trailNodeAt (t.take pos ++ descTrail) q
              (by simp [List.length_append, hpref_len]; omega) =
            trailNodeAt t q hq_t := by
          simp only [trailNodeAt, List.get_eq_getElem,
                     List.getElem_append_left hq_left,
                     getElem_take_eq t hq_left hq_t]
        rw [hnode_eq] at hblk
        exact hblk
      · intro hncq_t
        have hncq_join : ¬ isCollider (t.take pos ++ descTrail) q := by
          intro hcq_join
          apply hncq_t
          obtain ⟨_, _, hF, hB⟩ := hcq_join
          simp only [List.get_eq_getElem] at hF hB
          refine ⟨hqm1_t, hq_t, ?_, ?_⟩
          · simp only [List.get_eq_getElem]
            have := hF
            rw [List.getElem_append_left hqm1_left,
                getElem_take_eq t hqm1_left hqm1_t] at this
            exact this
          · simp only [List.get_eq_getElem]
            have := hB
            rw [List.getElem_append_left hq_left,
                getElem_take_eq t hq_left hq_t] at this
            exact this
        have hblk := hncolq_b hncq_join
        have hnode_eq :
            trailNodeAt (t.take pos ++ descTrail) q
              (by simp [List.length_append, hpref_len]; omega) =
            trailNodeAt t q hq_t := by
          simp only [trailNodeAt, List.get_eq_getElem,
                     List.getElem_append_left hq_left,
                     getElem_take_eq t hq_left hq_t]
        rw [hnode_eq] at hblk
        exact hblk
    · -- q = pos: at the boundary; non-collider, node = colNode ∉ W.
      subst q
      have hqm1_left : pos - 1 < (t.take pos).length := by rw [hpref_len]; omega
      have hqm1_t : pos - 1 < t.length := by omega
      -- step pos-1 in t is Forward (from collider def).
      obtain ⟨_, _, hF_t, _⟩ := hcol
      simp only [List.get_eq_getElem] at hF_t
      have hdT_dir0 : (descTrail[0]'hdT_pos).dir = EdgeDir.Forward :=
        hdfwd _ (List.getElem_mem hdT_pos)
      -- Identify the boundary index.
      have hpos_le : (t.take pos).length ≤ pos := by omega
      have hidx_qpos : pos - (t.take pos).length = 0 := by omega
      have hbnd_qpos : pos - (t.take pos).length < descTrail.length := by omega
      have hjoined_pos_eq :
          (t.take pos ++ descTrail)[pos]'(by simp [List.length_append, hpref_len]; omega) =
          descTrail[0]'hdT_pos := by
        rw [List.getElem_append_right hpos_le]
        first
        | rfl
        | exact getElem_idx_eq descTrail hidx_qpos hbnd_qpos
      -- joined is NOT a collider at pos.
      have hncq_join : ¬ isCollider (t.take pos ++ descTrail) pos := by
        rintro ⟨_, _, _, hB_j⟩
        simp only [List.get_eq_getElem] at hB_j
        rw [hjoined_pos_eq, hdT_dir0] at hB_j
        cases hB_j
      have hblk := hncolq_b hncq_join
      -- Node at pos in joined is descTrail[0].src = colNode.
      have hnode_eq :
          trailNodeAt (t.take pos ++ descTrail) pos
            (by simp [List.length_append, hpref_len]; omega) = colNode := by
        simp only [trailNodeAt, List.get_eq_getElem]
        rw [hjoined_pos_eq]; exact hdT_src_eq
      rw [hnode_eq] at hblk
      exact hnode_nW hblk
    · -- q > pos: position is in descTrail; both adjacent steps Forward → non-collider.
      set dPos : ℕ := q - pos with hdPos_def
      have hdPos_lo : 1 ≤ dPos := by omega
      have hdPos_hi : dPos < descTrail.length := by omega
      have hdPosm1_hi : dPos - 1 < descTrail.length := by omega
      have hq_le : (t.take pos).length ≤ q := by omega
      have hidx_q : q - (t.take pos).length = dPos := by omega
      have hbnd_q : q - (t.take pos).length < descTrail.length := by omega
      have hdT_dir : (descTrail[dPos]'hdPos_hi).dir = EdgeDir.Forward :=
        hdfwd _ (List.getElem_mem hdPos_hi)
      have hjoined_q_eq :
          (t.take pos ++ descTrail)[q]'(by simp [List.length_append, hpref_len]; omega) =
          descTrail[dPos]'hdPos_hi := by
        rw [List.getElem_append_right hq_le]
        first
        | rfl
        | exact getElem_idx_eq descTrail hidx_q hbnd_q
      have hncq_join : ¬ isCollider (t.take pos ++ descTrail) q := by
        rintro ⟨_, _, _, hB_j⟩
        simp only [List.get_eq_getElem] at hB_j
        rw [hjoined_q_eq, hdT_dir] at hB_j
        cases hB_j
      have hblk := hncolq_b hncq_join
      have hnode_eq :
          trailNodeAt (t.take pos ++ descTrail) q
            (by simp [List.length_append, hpref_len]; omega) =
          (descTrail[dPos]'hdPos_hi).src := by
        simp only [trailNodeAt, List.get_eq_getElem]
        rw [hjoined_q_eq]
      rw [hnode_eq] at hblk
      -- (descTrail[dPos]).src = (descTrail[dPos-1]).dst (by hdc).
      have hdcc := hdc ⟨dPos - 1, by omega⟩
      simp only [List.get_eq_getElem] at hdcc
      have hdidx : descTrail[dPos - 1 + 1]'(by omega) = descTrail[dPos]'hdPos_hi :=
        getElem_idx_eq descTrail (by omega : dPos - 1 + 1 = dPos) (by omega)
      rw [hdidx] at hdcc
      rw [← hdcc] at hblk
      have hdst_in : (descTrail[dPos - 1]'hdPosm1_hi).dst ∈ descendants G {colNode} :=
        hddst _ (List.getElem_mem hdPosm1_hi)
      have hwm : (descTrail[dPos - 1]'hdPosm1_hi).dst ∈
                 descendants G {colNode} ∩ W :=
        Finset.mem_inter.mpr ⟨hdst_in, hblk⟩
      rw [hdesc_W] at hwm
      exact (Finset.notMem_empty _) hwm

/-- **Contraction**: (Y ⊥ Z | W) ∧ (Y ⊥ Z' | W∪Z) → (Y ⊥ Z∪Z' | W).
    Corresponds to Dafny `DSep_Contraction`. -/
theorem dSep_contraction (G : Graph) (Y Z Z' W : Finset Node) :
    dSep G Y Z W → dSep G Y Z' (W ∪ Z) → dSep G Y (Z ∪ Z') W := by
  intro hZ hZ' t y z hy hz hv hc htc
  by_contra hnb
  rcases Finset.mem_union.mp hz with hzZ | hzZ'
  · -- Easy case: z ∈ Z.
    exact hnb (hZ t y z hy hzZ hv hc htc)
  · -- z ∈ Z': trail blocked under W ∪ Z.
    have hbWZ := hZ' t y z hy hzZ' hv hc htc
    obtain ⟨pos, hpos1, hposlt, hbpos, _⟩ :=
      firstBlockedPos G t (W ∪ Z) hbWZ
    have hnbpos : ¬ trailBlockedAtPos G t pos W := trailNotBlockedAtPos G t pos W hnb
    obtain ⟨_, _, hnode_Z, _⟩ :=
      blockingAddedByConditioningAtPos G t pos W Z hnbpos hbpos
    -- Build the prefix `t.take pos`.
    have hv_pref := trailValid_prefix G t pos hv
    have hc_pref := trailConnected_prefix t pos hc
    -- prefix connects y → node = trail[pos].src, since prev connector ends at it.
    have hcc := hc ⟨pos - 1, by omega⟩
    simp only [List.get_eq_getElem] at hcc
    have hidx : t[pos - 1 + 1]'(by omega) = t[pos]'hposlt :=
      getElem_idx_eq t (by omega : pos - 1 + 1 = pos) (by omega)
    rw [hidx] at hcc
    have raw := trailConnects_prefix htc hpos1 (le_of_lt hposlt)
    have htc_pref : trailConnects (t.take pos) y (trailNodeAt t pos hposlt) := by
      simp only [trailNodeAt, List.get_eq_getElem]
      rw [← hcc]; exact raw
    -- Prefix is unblocked under W.
    have hpref_nb : ¬ trailBlocked G (t.take pos) W :=
      prefixWithoutBlockedPos_notBlocked G t pos W
        (fun j _ _ => trailNotBlockedAtPos G t j W hnb)
    -- But `hZ` says any trail to a node in Z is blocked under W.
    exact hpref_nb
      (hZ (t.take pos) y (trailNodeAt t pos hposlt) hy hnode_Z hv_pref hc_pref htc_pref)

end Y0Lean
