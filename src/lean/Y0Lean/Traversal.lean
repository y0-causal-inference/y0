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
  | 0 => visited
  | fuel' + 1 =>
    -- children of a node v are NOT stored directly; G maps each node to its *parents*.
    -- So "children of v" = { w | v ∈ G.lookup w }.
    -- We implement this by iterating over G.keys and collecting those whose parent
    -- set intersects frontier.
    let newNodes : Finset Node :=
      G.keys.filter (fun w =>
        (G.lookup w).getD ∅ ∩ frontier ≠ ∅ ∧ w ∉ visited)
    if newNodes = ∅ then visited
    else reachableBFS G newNodes (visited ∪ frontier) fuel'
termination_by fuel

/-- BFS over parent edges: compute all nodes reachable from `frontier` by
    following directed edges *backward* (child → parent).
    Corresponds to Dafny's `function ReachableParentBFS`. -/
def reachableParentBFS (G : Graph) (frontier visited : Finset Node) (fuel : ℕ) : Finset Node :=
  match fuel with
  | 0 => visited
  | fuel' + 1 =>
    -- parents of frontier nodes: union of G.lookup w for w ∈ frontier
    let newNodes : Finset Node :=
      frontier.biUnion (fun w => (G.lookup w).getD ∅) \ visited
    if newNodes = ∅ then visited
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

-- Correctness spec (sorry — deferred to L4-002)
theorem ancestors_correct (G : Graph) (W : Finset Node) :
    ∀ v ∈ ancestors G W, ∃ w ∈ W, True := by sorry

theorem descendants_correct (G : Graph) (W : Finset Node) :
    ∀ v ∈ descendants G W, ∃ w ∈ W, True := by sorry

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

-- Correctness spec (sorry — deferred to L4-004)
theorem cComponent_correct (sm : SMGraph) (v : Node) :
    v ∈ cComponent sm v := by sorry

end Y0Lean
