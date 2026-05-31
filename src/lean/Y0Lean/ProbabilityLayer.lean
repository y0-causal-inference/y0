/-
  ProbabilityLayer.lean — L3: Kolmogorov axioms + probability layer.
  Port of: probability.dfy, interventional.dfy (probability-related axioms)
  Phase L3: replace Dafny {:axiom} declarations with Mathlib theorems or
            opaque/deferred placeholders for items deferred to L6.

  Dafny's `type PMF = map<Outcome, real>` is replaced by Mathlib's
  `PMF Outcome` (= `{f : Outcome → ℝ≥0∞ // HasSum f 1}`).
  Non-negativity is free: ℝ≥0∞ values are non-negative by type.
-/
import Y0Lean.Interventional
import Mathlib.Probability.ProbabilityMassFunction.Constructions

namespace Y0Lean

-- ── L3-001  Non-negativity ────────────────────────────────────────────────
-- Dafny: `lemma {:axiom} Axiom_NonNegativity(p: PMF, A: Event)
--             ensures IsDistribution(p) ==> 0.0 <= ProbEvent(p, A)`
-- Lean:  trivial — PMF values live in ℝ≥0∞ which is non-negative by type.

/-- Every point probability is non-negative.
    Trivial: `p a : ℝ≥0∞` is non-negative by type. -/
theorem pmf_apply_nonneg (p : PMF Outcome) (a : Outcome) : 0 ≤ p a :=
  zero_le

/-- The probability of any finite event is non-negative. -/
theorem pmf_event_nonneg (p : PMF Outcome) (s : Finset Outcome) :
    0 ≤ ∑ a ∈ s, p a :=
  Finset.sum_nonneg fun _ _ => zero_le

-- ── L3-002  Normalization + Finite Additivity ─────────────────────────────
-- Dafny: `Axiom_Normalization` + `Axiom_Additivity`

/-- Every PMF sums to 1.
    Corresponds to Dafny's `Axiom_Normalization`. -/
theorem pmf_tsum_one (p : PMF Outcome) : ∑' a, p a = 1 :=
  p.tsum_coe

/-- PMF is finitely additive over disjoint events.
    Corresponds to Dafny's `Axiom_Additivity`. -/
theorem pmf_additivity (p : PMF Outcome) (A B : Finset Outcome) (h : Disjoint A B) :
    ∑ a ∈ A ∪ B, p a = ∑ a ∈ A, p a + ∑ a ∈ B, p a :=
  Finset.sum_union h

-- ── L3-003  Product PMF ───────────────────────────────────────────────────
-- Dafny: `ghost function {:axiom} ProductPMF(ps: seq<PMF>): PMF`
--        `lemma {:axiom} ProductPMF_IsDistribution`
-- Lean:  independent product via the PMF monad (bind + map).

/-- Independent product distribution: P(X, Y) = P(X) · P(Y).
    Corresponds to Dafny's `ProductPMF` (two-argument case).
    The n-ary version is an iterated application of this. -/
noncomputable def pmfProd (p q : PMF Outcome) : PMF (Outcome × Outcome) :=
  p.bind fun a => q.map (Prod.mk a)

/-- The product of two PMFs sums to 1.
    Corresponds to Dafny's `ProductPMF_IsDistribution`. -/
theorem pmfProd_tsum_one (p q : PMF Outcome) : ∑' a, (pmfProd p q) a = 1 :=
  (pmfProd p q).tsum_coe

-- ── L3-005  TruncatePMF — discrete do-operator ───────────────────────────
-- Dafny: `ghost function {:axiom} TruncatePMF(G, p, X, xVals): PMF`
--        (conditions p on the event {ω : MatchesAssignment(G, ω, xVals)})
-- Lean:  `PMF.filter` restricts a PMF to a predicate and renormalizes.

/-- Condition (truncate) a PMF on an event `s`.
    Requires that `s` intersects `p.support`; otherwise P(s) = 0 and
    conditioning is undefined.
    Corresponds to Dafny's `TruncatePMF` (the discrete do-operator). -/
noncomputable def truncatePMF (p : PMF Outcome) (s : Set Outcome)
    (hs : ∃ a ∈ s, a ∈ p.support) : PMF Outcome :=
  p.filter s hs

-- ── L3-006  TruncatePMF is a distribution ────────────────────────────────
-- Trivial: `PMF.filter` always returns a valid PMF (sums to 1).

/-- A truncated PMF sums to 1.
    Corresponds to Dafny's `TruncatePMF_IsDistribution`. -/
theorem truncatePMF_tsum_one (p : PMF Outcome) (s : Set Outcome)
    (hs : ∃ a ∈ s, a ∈ p.support) : ∑' a, (truncatePMF p s hs) a = 1 :=
  (truncatePMF p s hs).tsum_coe

-- ── L3-007  SetToSequence ─────────────────────────────────────────────────
-- Dafny: `ghost function {:axiom} SetToSequence(s: set<Outcome>): seq<Outcome>`
--        (distinct, covering sequence)
-- Lean:  `Finset.sort s` is the computable canonical replacement.

/-- `Finset.sort s` is duplicate-free.
    Matches Dafny's `SetToSequence` distinctness guarantee. -/
theorem setToSequence_nodup (s : Finset Outcome) : (Finset.sort s).Nodup :=
  Finset.sort_nodup s (· ≤ ·)

/-- `Finset.sort s` contains exactly the elements of `s`.
    Matches Dafny's `SetToSequence` coverage guarantee. -/
theorem setToSequence_mem (a : Outcome) (s : Finset Outcome) :
    a ∈ Finset.sort s ↔ a ∈ s :=
  Finset.mem_sort (· ≤ ·)

-- ── L3-004  MarkovFactorization predicate ────────────────────────────────
-- Dafny: `ghost predicate {:axiom} MarkovFactorization(G: Graph, p: Prob.PMF)`
--        P(V) = ∏ᵢ P(Vᵢ | Pa_G(Vᵢ))
-- Status: opaque placeholder.
--   Proper definition deferred to L6: requires OutcomeToAssignment
--   machinery (axiomatized in Dafny's interventional.dfy) to connect the
--   abstract Outcome ℕ to the concrete node-value assignment space.
-- TODO L3-004: replace body with
--   ∀ ω, p ω = ∏ v ∈ G.keys, condFactor G p v (outcomeToAssignment G ω)

/-- A joint distribution `p` satisfies the Markov factorization condition
    w.r.t. DAG `G` (P decomposes as a product of local conditionals).
    Opaque placeholder — proper definition requires `OutcomeToAssignment`. -/
opaque MarkovFactorization (G : Graph) (p : PMF Outcome) : Prop := True

-- ── L3-009  MarkovFactorization → distribution ───────────────────────────
-- Trivial: every PMF sums to 1 regardless of any additional structure.

/-- Every Markov-factorized distribution is a valid PMF (sums to 1).
    Corresponds to Dafny's `MarkovFactorization_IsDistribution`. -/
theorem markovFactorization_isDistribution (G : Graph) (p : PMF Outcome)
    (_ : MarkovFactorization G p) : ∑' a, p a = 1 :=
  p.tsum_coe

-- ── L3-008  ConditionalLocalPMF / ConditionalFactor ──────────────────────
-- Dafny: `ghost function {:axiom} ConditionalLocalPMF(G, p, v, full): PMF`
--        P(Xᵥ | Pa_G(Xᵥ) = full(Pa)) — local conditional distribution
-- Status: TODO L3-008 — requires OutcomeToAssignment machinery.
--   Will be implemented as `truncatePMF` applied to the fibre over a
--   parent assignment, once the assignment-to-outcome bijection is in place.

-- ── L3-010  TruncatePMF preserves Markov condition ───────────────────────
-- Dafny: truncating by an intervention still satisfies Markov factorization.
-- Status: DEFERRED — requires L6-level reasoning about graph surgery
--         (Global Markov Property for mutilated graphs).

/-- Truncating (do-operator) a Markov-factorized distribution yields a
    distribution that still satisfies the Markov condition for the
    mutilated graph.
    Corresponds to Dafny's `TruncatePMF_Markov`.
    Proof deferred to L6 (requires Global Markov Property). -/
theorem truncatePMF_markov (G : Graph) (p : PMF Outcome)
    (_ : MarkovFactorization G p) (s : Set Outcome)
    (hs : ∃ a ∈ s, a ∈ p.support) :
    MarkovFactorization G (truncatePMF p s hs) := by
  sorry

end Y0Lean
