/-
  SemiMarkovian.lean — Semi-Markovian graph types.
  Port of: semi_markovian.dfy (BiEdge, SMGraph)
  Phase L1: type layer — no proof obligations.
-/
import Y0Lean.Graph
import Mathlib.Data.Prod.Lex

namespace Y0Lean

/-- A bidirected edge between two nodes, representing a latent common cause.
    Corresponds to Dafny's `datatype BiEdge = BiEdge(u: Node, v: Node)`. -/
@[ext]
structure BiEdge : Type where
  u : Node
  v : Node
  deriving DecidableEq, Repr

/-- Lexicographic linear order on `BiEdge`, lifted from `ℕ ×ₗ ℕ`.
    `toLex = Equiv.refl`, so the lift is computable and enables `Finset.sort`. -/
instance : LinearOrder BiEdge :=
  LinearOrder.lift' (fun e : BiEdge => toLex (e.u, e.v)) fun a b h => by
    have h' : (a.u, a.v) = (b.u, b.v) := toLex.injective h
    exact BiEdge.ext (Prod.mk.inj h').1 (Prod.mk.inj h').2

/-- A Semi-Markovian graph: a DAG plus a set of bidirected (hidden-confounder) edges.
    Corresponds to Dafny's
      `datatype SMGraph = SMGraph(dag: Graph, bidirected: set<BiEdge>)`. -/
structure SMGraph : Type where
  dag        : Graph         -- directed acyclic component
  bidirected : Finset BiEdge -- bidirected (latent confounder) edges

/-- Display a `Finset BiEdge` sorted lexicographically. Computable via `Finset.sort`. -/
instance : Repr (Finset BiEdge) where
  reprPrec s _ := reprPrec (Finset.sort s) 0

/-- Display an `SMGraph` as `(dag, sorted_bidirected_list)`. Computable. -/
instance : Repr SMGraph where
  reprPrec g p := reprPrec (g.dag, Finset.sort g.bidirected) p

end Y0Lean
