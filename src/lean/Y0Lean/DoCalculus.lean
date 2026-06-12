/-
  DoCalculus.lean — Interventional kernel type.
  Port of: do_calculus.dfy (InterventionalKernel)
  Phase L1: type layer — no proof obligations.
-/
import Y0Lean.Graph
import Y0Lean.Interventional
import Y0Lean.Probability

namespace Y0Lean

/-- Packages all state needed to evaluate a conditional interventional probability.
    Corresponds to Dafny's `datatype InterventionalKernel`.

    Fields:
    - `assignmentGraph`: the graph under the current assignment context
    - `modelGraph`: the underlying causal model graph
    - `basePMF`: the observational distribution P(V)
    - `xAssign`: the intervention assignment do(X = x)
    - `wAssign`: the conditioning assignment W = w
-/
structure InterventionalKernel : Type where
  assignmentGraph : Graph
  modelGraph      : Graph
  basePMF         : PMF Outcome
  xAssign         : Assignment
  wAssign         : Assignment

end Y0Lean
