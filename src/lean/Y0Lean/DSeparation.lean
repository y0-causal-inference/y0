/-
  DSeparation.lean — D-separation predicate for causal graphs.
  Port of: dag.dfy (DSep, ValidTrail, TrailBlocked, IsCollider)
  Phase L2-004: Prop-valued definition (not computable) matching the
  Dafny `ghost predicate DSep`.  Computable decision procedure deferred
  to L4-003 (requires path enumeration with a bound).
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
    the node shared between step `pos-1` (its destination) and step `pos` (its source)
    is reached by a Forward step and departed by a Forward step: → node ←.
    Corresponds to Dafny `predicate IsCollider`. -/
def isCollider (t : Trail) (pos : ℕ) : Prop :=
  ∃ (h1 : pos - 1 < t.length) (h2 : pos < t.length),
    (t.get ⟨pos - 1, h1⟩).dir = EdgeDir.Forward ∧
    (t.get ⟨pos,     h2⟩).dir = EdgeDir.Forward

/-- The *node at position `pos`* in the trail (the source of step `pos`). -/
def trailNodeAt (t : Trail) (pos : ℕ) (h : pos < t.length) : Node :=
  (t.get ⟨pos, h⟩).src

/-- A trail is **blocked at position `pos`** by `W`:
    - If the node at `pos` is a collider: the node itself is not in `W`
      (simplified; full definition needs descendants — deferred to L4-003).
    - If the node is a non-collider: the node is in `W`. -/
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

    This is a `Prop` (not computable) — a computable decision procedure
    using BFS is deferred to L4-003. -/
def dSep (G : Graph) (Y Z W : Finset Node) : Prop :=
  ∀ (t : Trail) (y z : Node),
    y ∈ Y → z ∈ Z →
    trailValid G t →
    trailConnected t →
    trailConnects t y z →
    trailBlocked G t W

-- Semi-Graphoid axioms (deferred to L4-003)
-- These correspond to DSep_Symmetry, DSep_Decomposition, DSep_WeakUnion,
-- DSep_Contraction, DSep_Intersection from dag.dfy.

theorem dSep_symmetry (G : Graph) (Y Z W : Finset Node) :
    dSep G Y Z W → dSep G Z Y W := by sorry

theorem dSep_decomposition (G : Graph) (Y Z Z' W : Finset Node) :
    dSep G Y (Z ∪ Z') W → dSep G Y Z W := by sorry

theorem dSep_weakUnion (G : Graph) (Y Z Z' W : Finset Node) :
    dSep G Y (Z ∪ Z') W → dSep G Y Z (W ∪ Z') := by sorry

theorem dSep_contraction (G : Graph) (Y Z Z' W : Finset Node) :
    dSep G Y Z W → dSep G Y Z' (W ∪ Z) → dSep G Y (Z ∪ Z') W := by sorry

end Y0Lean
