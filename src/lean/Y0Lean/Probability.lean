/-
  Probability.lean — Abstract outcome type.
  Port of: probability.dfy (Outcome, PMF note)
  Phase L1: type layer — no proof obligations.

  Dafny's `type Outcome(==, !new)` is an abstract sample-space point.
  We fix a concrete representation as ℕ so that the type is computable
  and carries decidable equality.

  Dafny's `type PMF = map<Outcome, real>` is replaced throughout by
  Mathlib's `PMF Outcome`, which is `{f : Outcome → ℝ≥0∞ // HasSum f 1}`.
  No alias is defined here — callers import Mathlib.Probability.PMF and use
  `PMF Outcome` directly.  This collapses L3-001 and L3-002.
-/
import Mathlib.Probability.ProbabilityMassFunction.Basic

namespace Y0Lean

/-- An abstract point in the sample space Ω.
    Corresponds to Dafny's abstract `type Outcome(==, !new)`. -/
abbrev Outcome := ℕ

end Y0Lean
