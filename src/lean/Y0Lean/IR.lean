/-
  IR.lean — Intermediate representation for the ID algorithm extractor.
  Port of: identification_executable_core.dfy (IRNode, IRQuery, IRDoc)
  Phase L1: type layer — no proof obligations.
-/

namespace Y0Lean

/-- An IR expression node representing one sub-formula in the identified expression.
    Corresponds to Dafny's `datatype IRNode`.

    Constructors:
    - `IRSum over body`         — Σ_{over} body
    - `IRProduct factors`       — ∏ factors
    - `IRProb vars given interv` — P(vars | given; do(interv))
    - `IRFrac numer denom`      — numer / denom
    - `IRFailHedge F Fprime`    — FAIL with hedge witness (F, F') -/
inductive IRNode : Type
  | IRSum      (over : List String) (body : IRNode)               : IRNode
  | IRProduct  (factors : List IRNode)                            : IRNode
  | IRProb     (vars given intervened : List String)              : IRNode
  | IRFrac     (numer denom : IRNode)                             : IRNode
  | IRFailHedge (F_nodes Fprime_nodes : List String)              : IRNode
  deriving Repr

/-- Metadata for a causal query submitted to the ID extractor.
    Corresponds to Dafny's `datatype IRQuery`. -/
structure IRQuery : Type where
  graph_id   : String
  outcomes   : List String
  treatments : List String
  ordering   : List String
  deriving Repr

/-- A full ID result document (query metadata + computed expression).
    Corresponds to Dafny's `datatype IRDoc`. -/
structure IRDoc : Type where
  version : String
  engine  : String
  query   : IRQuery
  result  : IRNode
  deriving Repr

end Y0Lean
