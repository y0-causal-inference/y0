/-
  Graph.lean — Base graph types.
  Port of: dag.dfy (Node, Graph, EdgeDir, TrailStep)
  Phase L1: type layer — no proof obligations.
-/
import Mathlib.Data.Finmap

namespace Y0Lean

/-- A node (variable) identifier.  Natural numbers give decidable equality.
    Corresponds to Dafny's `type Node = nat`. -/
abbrev Node := ℕ

/-- A directed graph: for each node in the domain, the finite set of its parents.
    `G.lookup v = some S` means `v` is present in the graph with parent set `S`.
    Corresponds to Dafny's `type Graph = map<Node, set<Node>>`. -/
abbrev Graph := Finmap (fun _ : Node => Finset Node)

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
