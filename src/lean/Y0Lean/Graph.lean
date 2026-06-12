/-
  Graph.lean — Base graph types.
  Port of: dag.dfy (Node, Graph, EdgeDir, TrailStep)
  Phase L1: type layer — no proof obligations.
-/
import Mathlib.Data.Finmap
import Mathlib.Data.Finset.Sort

namespace Y0Lean

/-- A node (variable) identifier.  Natural numbers give decidable equality.
    Corresponds to Dafny's `type Node = nat`. -/
abbrev Node := ℕ

/-- A directed graph: for each node in the domain, the finite set of its parents.
    `G.lookup v = some S` means `v` is present in the graph with parent set `S`.
    Corresponds to Dafny's `type Graph = map<Node, set<Node>>`. -/
abbrev Graph := Finmap (fun _ : Node => Finset Node)

/-- Display a `Graph` as a sorted list of `(node, sorted-parents)` pairs.
    Uses `Finset.sort` (computable merge-sort) so the instance is usable in `#eval`.
    Both keys and parent sets are sorted by node index. -/
instance : Repr Graph where
  reprPrec g _ :=
    -- Finset.sort (s : Finset α) (r := fun a b => a ≤ b) — relation is 2nd arg
    let ks : List Node := Finset.sort g.keys
    let pairs : List (Node × List Node) :=
      ks.filterMap (fun k =>
        (g.lookup k).map (fun v => (k, Finset.sort v)))
    reprPrec pairs 0

/-- Display a `Finset Node` as a sorted list (computable via `Finset.sort`). -/
instance : Repr (Finset Node) where
  reprPrec s _ := reprPrec (Finset.sort s) 0

/-- Direction of an edge step used in d-separation trails.
    Corresponds to Dafny's `datatype EdgeDir = Forward | Backward`. -/
inductive EdgeDir : Type
  | Forward  : EdgeDir   -- follows a directed edge u → v
  | Backward : EdgeDir   -- traverses a directed edge in reverse v ← u
  deriving DecidableEq, Repr

/-- A single step in a d-separation trail.
    Corresponds to Dafny's `datatype TrailStep = TrailStep(from, to, dir)`. -/
structure TrailStep : Type where
  src : Node    -- 'from' node (renamed to avoid keyword collision)
  dst : Node    -- 'to' node
  dir : EdgeDir
  deriving DecidableEq, Repr

end Y0Lean
