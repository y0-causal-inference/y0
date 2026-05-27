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

end Y0Lean
