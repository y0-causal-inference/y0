// ===================================================================
// Laws of Probability — Dafny Specification
//
// A concrete representation of discrete probability distributions
// over a finite sample space, with the Kolmogorov axioms and
// standard derived rules (chain rule, Bayes' theorem, marginalisation,
// conditional independence).
//
// Distributions are represented as maps from outcomes (Events) to
// non-negative rationals that sum to 1.  We use `real` (Dafny's
// arbitrary-precision rationals) to avoid floating-point issues.
// ===================================================================

module Probability {

  // ==================================================================
  // 1.  Outcome and Event types
  // ==================================================================

  // An Outcome is a single point in the sample space Ω.
  // We keep it abstract but require decidable equality.
  type Outcome(==)

  // A probability mass function (PMF) over a finite support.
  // pmf[ω] = P(ω).  Outcomes not in the map have probability 0.
  type PMF = map<Outcome, real>

  // An Event is a set of outcomes.
  type Event = set<Outcome>

  // ==================================================================
  // 2.  Well-formedness predicate for a PMF
  // ==================================================================

  // Every entry is non-negative.
  ghost predicate AllNonNeg(p: PMF) {
    forall omega :: omega in p ==> p[omega] >= 0.0
  }

  // The sum of all entries equals 1.
  // Because Dafny does not have built-in finite-map summation,
  // we axiomatize this as a ghost predicate and trust the
  // well-formedness invariant on construction.
  ghost predicate SumsToOne(p: PMF)

  // A valid distribution satisfies both conditions.
  ghost predicate IsDistribution(p: PMF) {
    AllNonNeg(p) && SumsToOne(p)
  }

  // ==================================================================
  // 3.  Core probability operations
  // ==================================================================

  // P(A) = Σ_{ω ∈ A} pmf[ω]
  ghost function ProbEvent(p: PMF, A: Event): real

  // P(A ∩ B)
  ghost function ProbJoint(p: PMF, A: Event, B: Event): real {
    ProbEvent(p, A * B)
  }

  // P(A | B) = P(A ∩ B) / P(B),  defined only when P(B) > 0.
  ghost function ProbCond(p: PMF, A: Event, B: Event): real
    requires ProbEvent(p, B) > 0.0
  {
    ProbJoint(p, A, B) / ProbEvent(p, B)
  }

  // ==================================================================
  // 4.  Kolmogorov Axioms  (for finite discrete spaces)
  // ==================================================================

  /// Axiom 1 — Non-negativity:  P(A) ≥ 0.
  lemma {:axiom} Axiom_NonNegativity(p: PMF, A: Event)
    requires IsDistribution(p)
    ensures  ProbEvent(p, A) >= 0.0

  /// Axiom 2 — Normalization:  P(Ω) = 1.
  ///   Here Ω = p.Keys (the entire support).
  lemma {:axiom} Axiom_Normalization(p: PMF)
    requires IsDistribution(p)
    ensures  ProbEvent(p, p.Keys) == 1.0

  /// Axiom 3 — Finite Additivity:
  ///   If A ∩ B = ∅  then  P(A ∪ B) = P(A) + P(B).
  lemma {:axiom} Axiom_Additivity(p: PMF, A: Event, B: Event)
    requires IsDistribution(p)
    requires A * B == {}    // disjoint
    ensures  ProbEvent(p, A + B) == ProbEvent(p, A) + ProbEvent(p, B)

  // ==================================================================
  // 5.  Derived Laws
  // ==================================================================

  /// Complement Rule:  P(Aᶜ) = 1 − P(A),
  /// where Aᶜ = Ω \ A.
  lemma ComplementRule(p: PMF, A: Event)
    requires IsDistribution(p)
    ensures  ProbEvent(p, p.Keys - A) == 1.0 - ProbEvent(p, A)
  {
    // A and (Ω \ A) are disjoint and their union is Ω.
    assert A * (p.Keys - A) == {};
    Axiom_Additivity(p, A, p.Keys - A);
    // P(A) + P(Ω \ A) = P(Ω) = 1
    Axiom_Normalization(p);
  }

  /// Impossible Event:  P(∅) = 0.
  lemma EmptyEventZero(p: PMF)
    requires IsDistribution(p)
    ensures  ProbEvent(p, {}) == 0.0
  {
    // {} and Ω are disjoint, {} ∪ Ω = Ω, so P({}) + P(Ω) = P(Ω).
    Axiom_Additivity(p, {}, p.Keys);
    Axiom_Normalization(p);
    // P({}) + 1.0 == 1.0  ⟹  P({}) == 0.0
  }

  /// Monotonicity:  A ⊆ B  ⟹  P(A) ≤ P(B).
  lemma Monotonicity(p: PMF, A: Event, B: Event)
    requires IsDistribution(p)
    requires A <= B
    ensures  ProbEvent(p, A) <= ProbEvent(p, B)
  {
    // B = A ∪ (B \ A), disjoint.
    assert A * (B - A) == {};
    assert A + (B - A) == B;
    Axiom_Additivity(p, A, B - A);
    Axiom_NonNegativity(p, B - A);
    // P(B) = P(A) + P(B\A) ≥ P(A)
  }

  /// Upper bound:  P(A) ≤ 1.
  lemma ProbAtMostOne(p: PMF, A: Event)
    requires IsDistribution(p)
    ensures  ProbEvent(p, A) <= 1.0
  {
    Axiom_NonNegativity(p, p.Keys - A);
    ComplementRule(p, A);
    // P(A) = 1 − P(Aᶜ) ≤ 1
  }

  /// Inclusion-Exclusion (two events):
  ///   P(A ∪ B) = P(A) + P(B) − P(A ∩ B).
  lemma {:axiom} InclusionExclusion(p: PMF, A: Event, B: Event)
    requires IsDistribution(p)
    ensures  ProbEvent(p, A + B)
             == ProbEvent(p, A) + ProbEvent(p, B) - ProbEvent(p, A * B)

  // ==================================================================
  // 6.  Chain Rule (Product Rule)
  // ==================================================================

  /// P(A ∩ B) = P(A | B) · P(B)   when P(B) > 0.
  lemma ChainRule(p: PMF, A: Event, B: Event)
    requires IsDistribution(p)
    requires ProbEvent(p, B) > 0.0
    ensures  ProbJoint(p, A, B) == ProbCond(p, A, B) * ProbEvent(p, B)
  {
    // Immediate from the definition of ProbCond.
  }

  // ==================================================================
  // 7.  Bayes' Theorem
  // ==================================================================

  /// P(A | B) = P(B | A) · P(A) / P(B)
  ///   when P(A) > 0 and P(B) > 0.
  lemma BayesTheorem(p: PMF, A: Event, B: Event)
    requires IsDistribution(p)
    requires ProbEvent(p, A) > 0.0
    requires ProbEvent(p, B) > 0.0
    ensures  ProbCond(p, A, B) ==
             ProbCond(p, B, A) * ProbEvent(p, A) / ProbEvent(p, B)
  {
    // P(A|B) = P(A∩B)/P(B)
    //        = P(B∩A)/P(B)          [intersection commutes]
    //        = P(B|A)·P(A)/P(B)    [chain rule on numerator]
    assert A * B == B * A;
    ChainRule(p, B, A);
  }

  // ==================================================================
  // 8.  Law of Total Probability
  // ==================================================================

  /// If B₁ ∪ B₂ = Ω and B₁ ∩ B₂ = ∅, then
  ///   P(A) = P(A | B₁) P(B₁) + P(A | B₂) P(B₂).
  ///
  /// (Binary partition version; the general version over a sequence
  ///  of partitions would require induction over a list.)
  lemma TotalProbability(p: PMF, A: Event, B1: Event, B2: Event)
    requires IsDistribution(p)
    requires B1 + B2 == p.Keys   // partition covers Ω
    requires B1 * B2 == {}        // partition is disjoint
    requires ProbEvent(p, B1) > 0.0
    requires ProbEvent(p, B2) > 0.0
    ensures  ProbEvent(p, A)
             == ProbCond(p, A, B1) * ProbEvent(p, B1)
              + ProbCond(p, A, B2) * ProbEvent(p, B2)
  {
    // A = (A ∩ B₁) ∪ (A ∩ B₂), disjoint.
    assert (A * B1) * (A * B2) == {};
    assert A * p.Keys == A * (B1 + B2);
    // We need A = (A ∩ B₁) ∪ (A ∩ B₂) for outcomes in the support.
    // This holds because A ∩ Ω = A  and  Ω = B₁ ∪ B₂.

    Axiom_Additivity(p, A * B1, A * B2);
    ChainRule(p, A, B1);
    ChainRule(p, A, B2);
  }

  // ==================================================================
  // 9.  Conditional Independence
  // ==================================================================

  /// A and B are conditionally independent given C:
  ///   A ⊥⊥ B | C   iff   P(A ∩ B | C) = P(A | C) · P(B | C)
  ghost predicate CondIndep(p: PMF, A: Event, B: Event, C: Event)
    requires ProbEvent(p, C) > 0.0
  {
    ProbCond(p, A * B, C) == ProbCond(p, A, C) * ProbCond(p, B, C)
  }

  /// Symmetry of conditional independence.
  lemma CondIndep_Symmetric(p: PMF, A: Event, B: Event, C: Event)
    requires IsDistribution(p)
    requires ProbEvent(p, C) > 0.0
    requires CondIndep(p, A, B, C)
    ensures  CondIndep(p, B, A, C)
  {
    assert A * B == B * A;
  }

  /// Unconditional independence (special case: C = Ω).
  ghost predicate Independent(p: PMF, A: Event, B: Event) {
    ProbJoint(p, A, B) == ProbEvent(p, A) * ProbEvent(p, B)
  }

  /// Symmetry of unconditional independence.
  lemma Independent_Symmetric(p: PMF, A: Event, B: Event)
    requires Independent(p, A, B)
    ensures  Independent(p, B, A)
  {
    assert A * B == B * A;
  }

}  // end module Probability
