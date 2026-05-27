/-
  Interventional.lean — Value and Assignment types.
  Port of: interventional.dfy (Value, Assignment)
  Phase L1: type layer — no proof obligations.
-/
import Y0Lean.Graph
import Y0Lean.Probability

namespace Y0Lean

/-- The value taken by a single node variable.
    Corresponds to Dafny's `type Value = Prob.Outcome`. -/
abbrev Value := Outcome

/-- A full or partial assignment mapping nodes to values.
    `a.lookup v = some val` means node `v` is assigned `val`.
    Corresponds to Dafny's `type Assignment = map<Node, Value>`. -/
abbrev Assignment := Finmap (fun _ : Node => Value)

end Y0Lean
