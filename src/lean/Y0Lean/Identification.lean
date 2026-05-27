/-
  Identification.lean — Causal query and ID result types.
  Port of: identification.dfy (CausalQuery, IDResult)
  Phase L1: type layer — no proof obligations.
-/
import Y0Lean.SemiMarkovian
import Y0Lean.Probability

namespace Y0Lean

/-- A causal effect query: (G, X, Y) asking for P_x(Y).
    Corresponds to Dafny's
      `datatype CausalQuery = CausalQuery(graph, treatments, outcomes)`. -/
structure CausalQuery : Type where
  graph      : SMGraph       -- the Semi-Markovian causal graph G
  treatments : Finset Node   -- X — variables intervened on (do(X = x))
  outcomes   : Finset Node   -- Y — variables whose effect we seek

/-- Display a `CausalQuery` as `(graph, treatments, outcomes)`.
    Noncomputable because `Repr SMGraph` is noncomputable. -/
noncomputable instance : Repr CausalQuery where
  reprPrec q p := reprPrec (q.graph, q.treatments, q.outcomes) p

/-- The result of the ID algorithm.
    Corresponds to Dafny's `datatype IDResult = Identified(...) | NotIdentified(...)`.

    - `Identified pmf`: the effect P_x(Y) was successfully computed.
    - `NotIdentified F Fprime`: a hedge witness (F, F') certifying
      non-identifiability (Shpitser & Pearl 2006, Theorem 2). -/
inductive IDResult : Type
  | Identified    (pmf : PMF Outcome)                     : IDResult
  | NotIdentified (F : SMGraph) (Fprime : SMGraph)        : IDResult

end Y0Lean
