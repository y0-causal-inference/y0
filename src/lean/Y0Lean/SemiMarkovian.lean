/-
  SemiMarkovian.lean — Semi-Markovian graph types.
  Port of: semi_markovian.dfy (BiEdge, SMGraph)
  Phase L1: type layer — no proof obligations.
-/
import Y0Lean.Graph

namespace Y0Lean

/-- A bidirected edge between two nodes, representing a latent common cause.
    Corresponds to Dafny's `datatype BiEdge = BiEdge(u: Node, v: Node)`. -/
structure BiEdge : Type where
  u : Node
  v : Node
  deriving DecidableEq, Repr

/-- A Semi-Markovian graph: a DAG plus a set of bidirected (hidden-confounder) edges.
    Corresponds to Dafny's
      `datatype SMGraph = SMGraph(dag: Graph, bidirected: set<BiEdge>)`. -/
structure SMGraph : Type where
  dag        : Graph         -- directed acyclic component
  bidirected : Finset BiEdge -- bidirected (latent confounder) edges

/-- Display a `Finset BiEdge` as a list.
    Noncomputable because `Multiset.toList` uses classical choice for a canonical representative. -/
noncomputable instance : Repr (Finset BiEdge) where
  reprPrec s _ := reprPrec s.val.toList 0

/-- Display an `SMGraph` as `(dag, bidirected_list)`.
    Noncomputable because `Repr (Finset BiEdge)` is noncomputable. -/
noncomputable instance : Repr SMGraph where
  reprPrec g p := reprPrec (g.dag, g.bidirected.val.toList) p

end Y0Lean
