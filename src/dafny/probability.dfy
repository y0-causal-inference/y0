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
  type Outcome(==, !new)

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
  // Even with a concrete finite-support event sum below, we keep this
  // foundational normalization boundary explicit in the first P5 slice.
  ghost predicate SumsToOne(p: PMF)

  // A valid distribution satisfies both conditions.
  ghost predicate IsDistribution(p: PMF) {
    AllNonNeg(p) && SumsToOne(p)
  }

  // ==================================================================
  // 3.  Core probability operations
  // ==================================================================

  ghost function OutcomeMass(p: PMF, omega: Outcome): real {
    if omega in p then p[omega] else 0.0
  }

  // A finite set of abstract outcomes has no canonical enumeration.
  // This bridge supplies a duplicate-free sequence view of a finite support.
  ghost function {:axiom} SetToSequence(s: set<Outcome>): seq<Outcome>
    ensures |SetToSequence(s)| == |s|
    ensures forall omega :: omega in s <==> omega in SetToSequence(s)
    ensures forall i, j :: 0 <= i < j < |SetToSequence(s)| ==>
      SetToSequence(s)[i] != SetToSequence(s)[j]

  ghost function SumOutcomeMasses(p: PMF, ws: seq<Outcome>): real {
    if |ws| == 0 then 0.0
    else OutcomeMass(p, ws[0]) + SumOutcomeMasses(p, ws[1..])
  }

  ghost function FiniteSupportSum(p: PMF, support: set<Outcome>): real {
    SumOutcomeMasses(p, SetToSequence(support))
  }

  // P(A) = Σ_{ω ∈ A ∩ p.Keys} pmf[ω]
  ghost function ProbEvent(p: PMF, A: Event): real {
    FiniteSupportSum(p, A * p.Keys)
  }

  lemma ProbEvent_RestrictToSupport(p: PMF, A: Event)
    ensures ProbEvent(p, A) == ProbEvent(p, A * p.Keys)
  {
    assert (A * p.Keys) * p.Keys == A * p.Keys;
  }

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
    var inSupport := A * p.Keys;
    assert (p.Keys - A) * inSupport == {};
    assert (p.Keys - A) + inSupport == p.Keys by {
      assert forall omega :: omega in ((p.Keys - A) + inSupport) <==> omega in p.Keys by {
        forall omega
          ensures omega in ((p.Keys - A) + inSupport) <==> omega in p.Keys
        {
          if omega in ((p.Keys - A) + inSupport) {
            if omega in p.Keys - A {
              assert omega in p.Keys;
            } else {
              assert omega in inSupport;
              assert omega in p.Keys;
            }
          } else if omega in p.Keys {
            if omega in A {
              assert omega in inSupport;
            } else {
              assert omega in p.Keys - A;
            }
          }
        }
      }
    }
    Axiom_Additivity(p, p.Keys - A, inSupport);
    Axiom_Normalization(p);
    ProbEvent_RestrictToSupport(p, A);
  }

  /// Impossible Event:  P(∅) = 0.
  lemma EmptyEventZero(p: PMF)
    requires IsDistribution(p)
    ensures  ProbEvent(p, {}) == 0.0
  {
    // {} and Ω are disjoint, {} ∪ Ω = Ω, so P({}) + P(Ω) = P(Ω).
    assert {} * p.Keys == {};
    assert {} + p.Keys == p.Keys;
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
  lemma InclusionExclusion(p: PMF, A: Event, B: Event)
    requires IsDistribution(p)
    ensures  ProbEvent(p, A + B)
             == ProbEvent(p, A) + ProbEvent(p, B) - ProbEvent(p, A * B)
  {
    var onlyA := (A - B) * p.Keys;
    var both := (A * B) * p.Keys;
    var bSupport := B * p.Keys;

    assert onlyA * both == {};
    assert onlyA + both == A * p.Keys by {
      assert forall omega :: omega in (onlyA + both) <==> omega in A * p.Keys by {
        forall omega
          ensures omega in (onlyA + both) <==> omega in A * p.Keys
        {
          if omega in onlyA + both {
            if omega in onlyA {
              assert omega in A - B;
              assert omega in A;
              assert omega in p.Keys;
            } else {
              assert omega in both;
              assert omega in A * B;
              assert omega in A;
              assert omega in p.Keys;
            }
          } else if omega in A * p.Keys {
            if omega in B {
              assert omega in both;
            } else {
              assert omega in A - B;
              assert omega in onlyA;
            }
          }
        }
      }
    }

    assert onlyA * bSupport == {};
    assert onlyA + bSupport == (A + B) * p.Keys by {
      assert forall omega :: omega in (onlyA + bSupport) <==> omega in (A + B) * p.Keys by {
        forall omega
          ensures omega in (onlyA + bSupport) <==> omega in (A + B) * p.Keys
        {
          if omega in onlyA + bSupport {
            if omega in onlyA {
              assert omega in A - B;
              assert omega in A;
              assert omega in A + B;
              assert omega in p.Keys;
            } else {
              assert omega in bSupport;
              assert omega in B;
              assert omega in A + B;
              assert omega in p.Keys;
            }
          } else if omega in (A + B) * p.Keys {
            if omega in B {
              assert omega in bSupport;
            } else {
              assert omega in A;
              assert omega in A - B;
              assert omega in onlyA;
            }
          }
        }
      }
    }

    Axiom_Additivity(p, onlyA, both);
    Axiom_Additivity(p, onlyA, bSupport);
    ProbEvent_RestrictToSupport(p, A);
    ProbEvent_RestrictToSupport(p, B);
    ProbEvent_RestrictToSupport(p, A + B);
    ProbEvent_RestrictToSupport(p, A * B);
  }

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
    var inB1 := A * B1;
    var inB2 := A * B2;

    assert inB1 * inB2 == {};
    assert inB1 + inB2 == A * p.Keys by {
      assert forall omega :: omega in (inB1 + inB2) <==> omega in A * p.Keys by {
        forall omega
          ensures omega in (inB1 + inB2) <==> omega in A * p.Keys
        {
          if omega in inB1 + inB2 {
            if omega in inB1 {
              assert omega in A;
              assert omega in B1;
              assert omega in B1 + B2;
              assert omega in p.Keys;
            } else {
              assert omega in inB2;
              assert omega in A;
              assert omega in B2;
              assert omega in B1 + B2;
              assert omega in p.Keys;
            }
          } else if omega in A * p.Keys {
            assert omega in B1 + B2;
            if omega in B1 {
              assert omega in inB1;
            } else {
              assert omega in B2;
              assert omega in inB2;
            }
          }
        }
      }
    }

    Axiom_Additivity(p, inB1, inB2);
    ProbEvent_RestrictToSupport(p, A);
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

  // ==================================================================
  // 8.  Products of PMFs
  // ==================================================================

  // Product of a sequence of PMFs.
  ghost function {:axiom} ProductPMF(ps: seq<PMF>): PMF

  lemma {:axiom} ProductPMF_IsDistribution(ps: seq<PMF>)
    requires forall i :: 0 <= i < |ps| ==> IsDistribution(ps[i])
    ensures IsDistribution(ProductPMF(ps))

}  // end module Probability
