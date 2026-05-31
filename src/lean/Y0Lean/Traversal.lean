/-
  Traversal.lean — Computable BFS traversals for causal graphs.
  Port of: dag.dfy (ReachableBFS, ReachableParentBFS) and
           semi_markovian.dfy (BidirectedBFSLoop, CComponentCompiled)
  Phase L2-002 (Ancestors/Descendants) + L2-003 (C-components).
-/
import Y0Lean.Graph
import Y0Lean.SemiMarkovian

namespace Y0Lean

-- ======================================================================
-- L2-002: Ancestors and Descendants (fuel-based BFS)
-- ======================================================================

/-- BFS over child edges: compute all nodes reachable from `frontier` by
    following directed edges *forward* (parent → child).
    Corresponds to Dafny's `function ReachableBFS`.  Terminates by fuel. -/
def reachableBFS (G : Graph) (frontier visited : Finset Node) (fuel : ℕ) : Finset Node :=
  match fuel with
  | 0 => visited ∪ frontier
  | fuel' + 1 =>
    -- children of a node v are NOT stored directly; G maps each node to its *parents*.
    -- So "children of v" = { w | v ∈ G.lookup w }.
    -- We implement this by iterating over G.keys and collecting those whose parent
    -- set intersects frontier.
    let newNodes : Finset Node :=
      G.keys.filter (fun w =>
        (G.lookup w).getD ∅ ∩ frontier ≠ ∅ ∧ w ∉ visited)
    if newNodes = ∅ then visited ∪ frontier
    else reachableBFS G newNodes (visited ∪ frontier) fuel'
termination_by fuel

/-- BFS over parent edges: compute all nodes reachable from `frontier` by
    following directed edges *backward* (child → parent).
    Corresponds to Dafny's `function ReachableParentBFS`. -/
def reachableParentBFS (G : Graph) (frontier visited : Finset Node) (fuel : ℕ) : Finset Node :=
  match fuel with
  | 0 => visited ∪ frontier
  | fuel' + 1 =>
    -- parents of frontier nodes: union of G.lookup w for w ∈ frontier
    let newNodes : Finset Node :=
      frontier.biUnion (fun w => (G.lookup w).getD ∅) \ visited
    if newNodes = ∅ then visited ∪ frontier
    else reachableParentBFS G newNodes (visited ∪ frontier) fuel'
termination_by fuel

/-- All descendants of node set `W` in `G`: nodes reachable by following
    directed edges forward.  Initial fuel = |G.keys| (enough for a DAG).
    Corresponds to Dafny's `function DescendantsCompiled`. -/
def descendants (G : Graph) (W : Finset Node) : Finset Node :=
  let start := W ∩ G.keys
  reachableBFS G start ∅ G.keys.card

/-- All ancestors of node set `W` in `G`: nodes reachable by following
    directed edges backward.
    Corresponds to Dafny's `function AncestorsCompiled`. -/
def ancestors (G : Graph) (W : Finset Node) : Finset Node :=
  let start := W ∩ G.keys
  reachableParentBFS G start ∅ G.keys.card

-- ── Monotonicity helpers ─────────────────────────────────────────────────────

/-- The BFS result always contains the initial `visited ∪ frontier`. -/
private lemma reachableParentBFS_mono (G : Graph) (frontier visited : Finset Node)
    (fuel : ℕ) : visited ∪ frontier ⊆ reachableParentBFS G frontier visited fuel := by
  induction fuel generalizing frontier visited with
  | zero => simp [reachableParentBFS]
  | succ n ih =>
    -- Unfold one step; simp with zeta also reduces the let-binding
    simp (config := { zeta := true }) only [reachableParentBFS]
    by_cases h : frontier.biUnion (fun w => (G.lookup w).getD ∅) \ visited = ∅
    · simp [h]
    · rw [if_neg h]
      exact Finset.subset_union_left.trans (ih _ _)

private lemma reachableBFS_mono (G : Graph) (frontier visited : Finset Node)
    (fuel : ℕ) : visited ∪ frontier ⊆ reachableBFS G frontier visited fuel := by
  induction fuel generalizing frontier visited with
  | zero => simp [reachableBFS]
  | succ n ih =>
    simp (config := { zeta := true }) only [reachableBFS]
    by_cases h : G.keys.filter (fun w =>
        (G.lookup w).getD ∅ ∩ frontier ≠ ∅ ∧ w ∉ visited) = ∅
    · simp [h]
    · rw [if_neg h]
      exact Finset.subset_union_left.trans (ih _ _)

-- Correctness spec: W ∩ G.keys (the starting nodes) are in the BFS result.
theorem ancestors_correct (G : Graph) (W : Finset Node) :
    W ∩ G.keys ⊆ ancestors G W := by
  simp only [ancestors]
  have h := reachableParentBFS_mono G (W ∩ G.keys) ∅ G.keys.card
  simpa using h

theorem descendants_correct (G : Graph) (W : Finset Node) :
    W ∩ G.keys ⊆ descendants G W := by
  simp only [descendants]
  have h := reachableBFS_mono G (W ∩ G.keys) ∅ G.keys.card
  simpa using h

-- ======================================================================
-- Relational ancestor/descendant predicates and BFS soundness
-- ======================================================================

/-- One-step **child** edge: `childOf G p c` iff there is a directed edge
    `p → c` in `G`, i.e. `p` is in the recorded parent set of `c`. -/
def childOf (G : Graph) (p c : Node) : Prop :=
  p ∈ (G.lookup c).getD ∅

/-- One-step **parent** edge (dual of `childOf`).
    `parentOf G c p` iff `p → c` is an edge in `G`. -/
def parentOf (G : Graph) (c p : Node) : Prop :=
  p ∈ (G.lookup c).getD ∅

/-- `isAncestor G a v` — `a = v` or `a` is reached from `v` by following
    parent-edges (reflexive-transitive closure).  Corresponds to Dafny's
    `predicate IsAncestor`. -/
def isAncestor (G : Graph) (a v : Node) : Prop :=
  Relation.ReflTransGen (parentOf G) v a

/-- `isDescendant G d v` — `d = v` or `d` is reached from `v` by following
    child-edges (reflexive-transitive closure). -/
def isDescendant (G : Graph) (d v : Node) : Prop :=
  Relation.ReflTransGen (childOf G) v d

/-- BFS soundness invariant for `reachableBFS` (child direction):
    every node produced by the BFS is a descendant of some original starting node. -/
private lemma reachableBFS_sound
    (G : Graph) (W0 : Finset Node) :
    ∀ (frontier visited : Finset Node) (fuel : ℕ),
      (∀ v ∈ visited ∪ frontier, ∃ w ∈ W0, isDescendant G v w) →
      ∀ v ∈ reachableBFS G frontier visited fuel, ∃ w ∈ W0, isDescendant G v w := by
  intro frontier visited fuel
  induction fuel generalizing frontier visited with
  | zero =>
    intro hinv v hv
    simp [reachableBFS] at hv
    exact hinv v (Finset.mem_union.mpr hv)
  | succ n ih =>
    intro hinv v hv
    simp (config := { zeta := true }) only [reachableBFS] at hv
    by_cases hempty :
        G.keys.filter (fun w => (G.lookup w).getD ∅ ∩ frontier ≠ ∅ ∧ w ∉ visited) = ∅
    · simp [hempty] at hv; exact hinv v (Finset.mem_union.mpr hv)
    · rw [if_neg hempty] at hv
      apply ih _ _ ?_ v hv
      intro u hu
      rcases Finset.mem_union.mp hu with hu | hu
      · exact hinv u hu
      · rcases Finset.mem_filter.mp hu with ⟨_, hpar, _⟩
        have hpne : ((G.lookup u).getD ∅ ∩ frontier).Nonempty :=
          Finset.nonempty_iff_ne_empty.mpr hpar
        obtain ⟨p, hp⟩ := hpne
        rcases Finset.mem_inter.mp hp with ⟨hppar, hpfront⟩
        have hpinv := hinv p (Finset.mem_union_right _ hpfront)
        rcases hpinv with ⟨w, hwW0, hwd⟩
        -- hwd : isDescendant G p w = ReflTransGen (childOf G) w p
        -- hppar : p ∈ (G.lookup u).getD ∅ = childOf G p u
        exact ⟨w, hwW0, hwd.tail hppar⟩

/-- BFS soundness invariant for `reachableParentBFS` (parent direction):
    every node produced is an ancestor of some original starting node. -/
private lemma reachableParentBFS_sound
    (G : Graph) (W0 : Finset Node) :
    ∀ (frontier visited : Finset Node) (fuel : ℕ),
      (∀ v ∈ visited ∪ frontier, ∃ w ∈ W0, isAncestor G v w) →
      ∀ v ∈ reachableParentBFS G frontier visited fuel, ∃ w ∈ W0, isAncestor G v w := by
  intro frontier visited fuel
  induction fuel generalizing frontier visited with
  | zero =>
    intro hinv v hv
    simp [reachableParentBFS] at hv
    exact hinv v (Finset.mem_union.mpr hv)
  | succ n ih =>
    intro hinv v hv
    simp (config := { zeta := true }) only [reachableParentBFS] at hv
    by_cases hempty :
        frontier.biUnion (fun w => (G.lookup w).getD ∅) \ visited = ∅
    · simp [hempty] at hv; exact hinv v (Finset.mem_union.mpr hv)
    · rw [if_neg hempty] at hv
      apply ih _ _ ?_ v hv
      intro u hu
      rcases Finset.mem_union.mp hu with hu | hu
      · exact hinv u hu
      · rcases Finset.mem_sdiff.mp hu with ⟨huU, _⟩
        rcases Finset.mem_biUnion.mp huU with ⟨c, hcfront, hupar⟩
        have hcinv := hinv c (Finset.mem_union_right _ hcfront)
        rcases hcinv with ⟨w, hwW0, hwa⟩
        -- hwa : isAncestor G c w = ReflTransGen (parentOf G) w c
        -- hupar : u ∈ (G.lookup c).getD ∅ = parentOf G c u
        exact ⟨w, hwW0, hwa.tail hupar⟩

/-- Public BFS soundness: every node in `descendants G W` is a graph-theoretic
    descendant (`isDescendant`) of some node in `W ∩ G.keys`. -/
theorem descendants_sound (G : Graph) (W : Finset Node) :
    ∀ v ∈ descendants G W, ∃ w ∈ W, isDescendant G v w := by
  intro v hv
  have hkey : ∀ u ∈ (∅ : Finset Node) ∪ (W ∩ G.keys),
      ∃ w ∈ W ∩ G.keys, isDescendant G u w := by
    intro u hu
    have hu' : u ∈ W ∩ G.keys := by simpa using hu
    exact ⟨u, hu', Relation.ReflTransGen.refl⟩
  have := reachableBFS_sound G (W ∩ G.keys) (W ∩ G.keys) ∅ G.keys.card hkey v hv
  rcases this with ⟨w, hw, hd⟩
  exact ⟨w, (Finset.mem_inter.mp hw).1, hd⟩

/-- Public BFS soundness for ancestors. -/
theorem ancestors_sound (G : Graph) (W : Finset Node) :
    ∀ v ∈ ancestors G W, ∃ w ∈ W, isAncestor G v w := by
  intro v hv
  have hkey : ∀ u ∈ (∅ : Finset Node) ∪ (W ∩ G.keys),
      ∃ w ∈ W ∩ G.keys, isAncestor G u w := by
    intro u hu
    have hu' : u ∈ W ∩ G.keys := by simpa using hu
    exact ⟨u, hu', Relation.ReflTransGen.refl⟩
  have := reachableParentBFS_sound G (W ∩ G.keys) (W ∩ G.keys) ∅ G.keys.card hkey v hv
  rcases this with ⟨w, hw, ha⟩
  exact ⟨w, (Finset.mem_inter.mp hw).1, ha⟩

-- ======================================================================
-- Graph surgery helpers (used by IDImpl)
-- ======================================================================

/-- Remove all incoming (parent) edges to nodes in `X` from graph `G`.
    Corresponds to Dafny's `function RemoveIncomingSM` / `G_{X̄}` notation. -/
def removeIncoming (G : Graph) (X : Finset Node) : Graph :=
  (Finset.sort G.keys).foldl
    (fun acc v =>
      if v ∈ X then
        acc.insert v ∅
      else
        match G.lookup v with
        | none   => acc
        | some s => acc.insert v s)
    ∅

/-- Restrict graph `G` to the node set `S` (remove all nodes not in `S` and
    any edges whose endpoints are outside `S`).
    Corresponds to Dafny's `function SubgraphSM`. -/
def subgraph (G : Graph) (S : Finset Node) : Graph :=
  (Finset.sort S).foldl
    (fun acc v =>
      match G.lookup v with
      | none   => acc
      | some ps => acc.insert v (ps ∩ S))
    ∅

-- ======================================================================
-- L2-003: Bidirected BFS and C-components
-- ======================================================================

/-- Neighbours of node `u` via bidirected edges in `sm`.
    Corresponds to Dafny's `function BidirectedNeighbors`. -/
def bidirectedNeighbors (sm : SMGraph) (u : Node) : Finset Node :=
  sm.bidirected.biUnion (fun e =>
    if e.u = u then {e.v}
    else if e.v = u then {e.u}
    else ∅)

/-- BFS over bidirected edges from `frontier`, avoiding `visited`.
    Corresponds to Dafny's `function BidirectedBFSLoop`. -/
def bidirectedBFSLoop (sm : SMGraph) (frontier visited : Finset Node) (fuel : ℕ) : Finset Node :=
  match fuel with
  | 0 => visited ∪ frontier
  | fuel' + 1 =>
    let newNodes : Finset Node :=
      frontier.biUnion (bidirectedNeighbors sm) \ (visited ∪ frontier)
    if newNodes = ∅ then visited ∪ frontier
    else bidirectedBFSLoop sm newNodes (visited ∪ frontier) fuel'
termination_by fuel

/-- The c-component containing node `v` in semi-Markovian graph `sm`.
    Corresponds to Dafny's `function CComponentCompiled`. -/
def cComponent (sm : SMGraph) (v : Node) : Finset Node :=
  bidirectedBFSLoop sm {v} ∅ sm.bidirected.card

/-- All c-components of `sm` restricted to node set `S`, returned in topological order.
    Each node in `S` appears in exactly one component. -/
def cComponentsOf (sm : SMGraph) (S : Finset Node) (ord : List Node) : List (Finset Node) :=
  let (comps, _) : List (Finset Node) × Finset Node :=
    ord.foldl
      (fun (acc : List (Finset Node) × Finset Node) v =>
        let (cs, visited) := acc
        if v ∈ S ∧ v ∉ visited then
          let c := cComponent sm v ∩ S
          (cs ++ [c], visited ∪ c)
        else acc)
      ([], ∅)
  comps

-- ── Monotonicity + correctness (L4-004) ─────────────────────────────────────

private lemma bidirectedBFSLoop_mono (sm : SMGraph) (frontier visited : Finset Node)
    (fuel : ℕ) : visited ∪ frontier ⊆ bidirectedBFSLoop sm frontier visited fuel := by
  induction fuel generalizing frontier visited with
  | zero => simp [bidirectedBFSLoop]
  | succ n ih =>
    simp (config := { zeta := true }) only [bidirectedBFSLoop]
    by_cases h : frontier.biUnion (bidirectedNeighbors sm) \ (visited ∪ frontier) = ∅
    · simp [h]
    · rw [if_neg h]
      exact Finset.subset_union_left.trans (ih _ _)

theorem cComponent_correct (sm : SMGraph) (v : Node) :
    v ∈ cComponent sm v := by
  simp only [cComponent]
  have h := bidirectedBFSLoop_mono sm {v} ∅ sm.bidirected.card
  simp only [Finset.empty_union] at h
  exact h (Finset.mem_singleton_self v)

end Y0Lean
