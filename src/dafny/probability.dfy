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
  ghost predicate SumsToOne(p: PMF) {
    ProbEvent(p, p.Keys) == 1.0
  }

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

  lemma SumOutcomeMasses_Concat(p: PMF, ws1: seq<Outcome>, ws2: seq<Outcome>)
    ensures SumOutcomeMasses(p, ws1 + ws2)
      == SumOutcomeMasses(p, ws1) + SumOutcomeMasses(p, ws2)
  {
    if |ws1| == 0 {
      assert ws1 == [];
      assert ws1 + ws2 == ws2;
      assert SumOutcomeMasses(p, ws1) == 0.0;
    } else {
      SumOutcomeMasses_Concat(p, ws1[1..], ws2);
      assert |ws1 + ws2| != 0;
      assert (ws1 + ws2)[0] == ws1[0];
      assert (ws1 + ws2)[1..] == ws1[1..] + ws2;
      assert SumOutcomeMasses(p, ws1 + ws2)
        == OutcomeMass(p, (ws1 + ws2)[0]) + SumOutcomeMasses(p, (ws1 + ws2)[1..]);
      assert SumOutcomeMasses(p, ws1 + ws2)
        == OutcomeMass(p, ws1[0]) + SumOutcomeMasses(p, ws1[1..] + ws2);
      assert SumOutcomeMasses(p, ws1)
        == OutcomeMass(p, ws1[0]) + SumOutcomeMasses(p, ws1[1..]);
      calc {
        SumOutcomeMasses(p, ws1 + ws2);
        ==
        OutcomeMass(p, ws1[0]) + SumOutcomeMasses(p, ws1[1..] + ws2);
        == { SumOutcomeMasses_Concat(p, ws1[1..], ws2); }
        OutcomeMass(p, ws1[0]) + (SumOutcomeMasses(p, ws1[1..]) + SumOutcomeMasses(p, ws2));
        ==
        (OutcomeMass(p, ws1[0]) + SumOutcomeMasses(p, ws1[1..])) + SumOutcomeMasses(p, ws2);
        ==
        SumOutcomeMasses(p, ws1) + SumOutcomeMasses(p, ws2);
      }
    }
  }

  ghost function RemoveAt(ws: seq<Outcome>, i: nat): seq<Outcome>
    requires i < |ws|
  {
    ws[..i] + ws[i + 1..]
  }

  lemma RemoveAt_Distinct(ws: seq<Outcome>, i: nat)
    requires i < |ws|
    requires forall a, b :: 0 <= a < b < |ws| ==> ws[a] != ws[b]
    ensures forall a, b :: 0 <= a < b < |RemoveAt(ws, i)| ==> RemoveAt(ws, i)[a] != RemoveAt(ws, i)[b]
  {
    var prefix := ws[..i];
    var suffix := ws[i + 1..];

    assert forall a, b :: 0 <= a < b < |prefix| ==> prefix[a] != prefix[b] by {
      forall a, b | 0 <= a < b < |prefix|
        ensures prefix[a] != prefix[b]
      {
        assert prefix[a] == ws[a];
        assert prefix[b] == ws[b];
      }
    }

    assert forall a, b :: 0 <= a < b < |suffix| ==> suffix[a] != suffix[b] by {
      forall a, b | 0 <= a < b < |suffix|
        ensures suffix[a] != suffix[b]
      {
        assert suffix[a] == ws[i + 1 + a];
        assert suffix[b] == ws[i + 1 + b];
      }
    }

    assert forall a, b :: 0 <= a < |prefix| && 0 <= b < |suffix| ==> prefix[a] != suffix[b] by {
      forall a, b | 0 <= a < |prefix| && 0 <= b < |suffix|
        ensures prefix[a] != suffix[b]
      {
        assert prefix[a] == ws[a];
        assert suffix[b] == ws[i + 1 + b];
      }
    }

    assert forall a, b :: 0 <= a < b < |RemoveAt(ws, i)| ==> RemoveAt(ws, i)[a] != RemoveAt(ws, i)[b] by {
      forall a, b | 0 <= a < b < |RemoveAt(ws, i)|
        ensures RemoveAt(ws, i)[a] != RemoveAt(ws, i)[b]
      {
        if b < |prefix| {
          assert RemoveAt(ws, i)[a] == prefix[a];
          assert RemoveAt(ws, i)[b] == prefix[b];
        } else if a >= |prefix| {
          assert RemoveAt(ws, i)[a] == suffix[a - |prefix|];
          assert RemoveAt(ws, i)[b] == suffix[b - |prefix|];
        } else {
          assert RemoveAt(ws, i)[a] == prefix[a];
          assert RemoveAt(ws, i)[b] == suffix[b - |prefix|];
        }
      }
    }
  }

  lemma RemoveAt_Membership(ws: seq<Outcome>, i: nat, omega: Outcome)
    requires i < |ws|
    requires forall a, b :: 0 <= a < b < |ws| ==> ws[a] != ws[b]
    ensures omega in RemoveAt(ws, i) <==> omega in ws && omega != ws[i]
  {
    var prefix := ws[..i];
    var suffix := ws[i + 1..];

    if omega in RemoveAt(ws, i) {
      var k :| 0 <= k < |RemoveAt(ws, i)| && RemoveAt(ws, i)[k] == omega;
      if k < |prefix| {
        assert RemoveAt(ws, i)[k] == prefix[k];
        assert prefix[k] == ws[k];
        assert omega in ws;
        assert omega != ws[i];
      } else {
        assert RemoveAt(ws, i)[k] == suffix[k - |prefix|];
        assert suffix[k - |prefix|] == ws[i + 1 + (k - |prefix|)];
        assert omega in ws;
        assert omega != ws[i];
      }
    } else if omega in ws && omega != ws[i] {
      var k :| 0 <= k < |ws| && ws[k] == omega;
      assert k != i;
      if k < i {
        assert prefix[k] == omega;
        assert omega in prefix;
      } else {
        assert k > i;
        assert suffix[k - (i + 1)] == omega;
        assert omega in suffix;
      }
    }
  }

  lemma SumOutcomeMasses_DistinctSequenceInvariant(
    p: PMF,
    ws1: seq<Outcome>,
    ws2: seq<Outcome>
  )
    requires forall i, j :: 0 <= i < j < |ws1| ==> ws1[i] != ws1[j]
    requires forall i, j :: 0 <= i < j < |ws2| ==> ws2[i] != ws2[j]
    requires forall omega :: omega in ws1 <==> omega in ws2
    ensures SumOutcomeMasses(p, ws1) == SumOutcomeMasses(p, ws2)
  {
    if |ws1| == 0 {
      assert |ws2| == 0 by {
        if |ws2| != 0 {
          assert ws2[0] in ws2;
          assert ws2[0] in ws1;
        }
      }
    } else {
      var omega0 := ws1[0];
      assert omega0 in ws1;
      assert omega0 in ws2;
      var j :| 0 <= j < |ws2| && ws2[j] == omega0;
      var ws1Rest := RemoveAt(ws1, 0);
      var ws2Rest := RemoveAt(ws2, j);

      assert ws1Rest == ws1[1..];
      assert ws2 == ws2[..j] + [omega0] + ws2[j + 1..];

      RemoveAt_Distinct(ws1, 0);
      RemoveAt_Distinct(ws2, j);

      assert forall omega :: omega in ws1Rest <==> omega in ws2Rest by {
        forall omega
          ensures omega in ws1Rest <==> omega in ws2Rest
        {
          RemoveAt_Membership(ws1, 0, omega);
          RemoveAt_Membership(ws2, j, omega);
        }
      }

      SumOutcomeMasses_DistinctSequenceInvariant(p, ws1Rest, ws2Rest);
      SumOutcomeMasses_Concat(p, ws2[..j], ws2[j + 1..]);
      SumOutcomeMasses_Concat(p, ws2[..j], [omega0]);
      SumOutcomeMasses_Concat(p, ws2[..j] + [omega0], ws2[j + 1..]);

      calc {
        SumOutcomeMasses(p, ws1);
        ==
        OutcomeMass(p, omega0) + SumOutcomeMasses(p, ws1Rest);
        == { SumOutcomeMasses_DistinctSequenceInvariant(p, ws1Rest, ws2Rest); }
        OutcomeMass(p, omega0) + SumOutcomeMasses(p, ws2Rest);
        == { SumOutcomeMasses_Concat(p, ws2[..j], ws2[j + 1..]); }
        OutcomeMass(p, omega0) + (SumOutcomeMasses(p, ws2[..j]) + SumOutcomeMasses(p, ws2[j + 1..]));
        ==
        SumOutcomeMasses(p, ws2[..j]) + OutcomeMass(p, omega0) + SumOutcomeMasses(p, ws2[j + 1..]);
        ==
        SumOutcomeMasses(p, ws2[..j]) + SumOutcomeMasses(p, [omega0]) + SumOutcomeMasses(p, ws2[j + 1..]);
        == { SumOutcomeMasses_Concat(p, ws2[..j], [omega0]); }
        SumOutcomeMasses(p, ws2[..j] + [omega0]) + SumOutcomeMasses(p, ws2[j + 1..]);
        == { SumOutcomeMasses_Concat(p, ws2[..j] + [omega0], ws2[j + 1..]); }
        SumOutcomeMasses(p, ws2);
      }
    }
  }

  ghost function FiniteSupportSum(p: PMF, support: set<Outcome>): real {
    SumOutcomeMasses(p, SetToSequence(support))
  }

  lemma FiniteSupportSum_AnyDistinctEnumeration(
    p: PMF,
    support: set<Outcome>,
    ws: seq<Outcome>
  )
    requires forall omega :: omega in ws <==> omega in support
    requires forall i, j :: 0 <= i < j < |ws| ==> ws[i] != ws[j]
    ensures FiniteSupportSum(p, support) == SumOutcomeMasses(p, ws)
  {
    SumOutcomeMasses_DistinctSequenceInvariant(p, SetToSequence(support), ws);
  }

  lemma SumOutcomeMasses_AllZero(p: PMF, ws: seq<Outcome>)
    requires forall i :: 0 <= i < |ws| ==> OutcomeMass(p, ws[i]) == 0.0
    ensures SumOutcomeMasses(p, ws) == 0.0
  {
    if |ws| != 0 {
      assert forall i :: 0 <= i < |ws[1..]| ==> OutcomeMass(p, ws[1..][i]) == 0.0 by {
        forall i | 0 <= i < |ws[1..]|
          ensures OutcomeMass(p, ws[1..][i]) == 0.0
        {
          assert ws[1..][i] == ws[i + 1];
        }
      }
      SumOutcomeMasses_AllZero(p, ws[1..]);
    }
  }

  lemma FiniteSupportSum_AllZero(p: PMF, support: set<Outcome>)
    requires forall omega :: omega in support ==> OutcomeMass(p, omega) == 0.0
    ensures FiniteSupportSum(p, support) == 0.0
  {
    assert forall i :: 0 <= i < |SetToSequence(support)| ==>
      OutcomeMass(p, SetToSequence(support)[i]) == 0.0 by {
      forall i | 0 <= i < |SetToSequence(support)|
        ensures OutcomeMass(p, SetToSequence(support)[i]) == 0.0
      {
        assert SetToSequence(support)[i] in support;
      }
    }
    SumOutcomeMasses_AllZero(p, SetToSequence(support));
  }

  lemma SumOutcomeMasses_PositiveCompleteNormalized(p: PMF, ws: seq<Outcome>)
    requires IsDistribution(p)
    requires forall i, j :: 0 <= i < j < |ws| ==> ws[i] != ws[j]
    requires forall i :: 0 <= i < |ws| ==> ws[i] in p.Keys
    requires forall omega :: omega in p.Keys && OutcomeMass(p, omega) > 0.0 ==> omega in ws
    ensures SumOutcomeMasses(p, ws) == 1.0
  {
    var support := set omega: Outcome | omega in ws :: omega;
    assert support <= p.Keys by {
      assert forall omega :: omega in support ==> omega in p.Keys by {
        forall omega | omega in support
          ensures omega in p.Keys
        {
          var i :| 0 <= i < |ws| && ws[i] == omega;
        }
      }
    }
    FiniteSupportSum_AnyDistinctEnumeration(p, support, ws);

    assert forall omega :: omega in p.Keys - support ==> OutcomeMass(p, omega) == 0.0 by {
      forall omega | omega in p.Keys - support
        ensures OutcomeMass(p, omega) == 0.0
      {
        assert OutcomeMass(p, omega) >= 0.0;
        if OutcomeMass(p, omega) > 0.0 {
          assert omega in ws;
          assert omega in support;
          assert false;
        }
      }
    }
    FiniteSupportSum_AllZero(p, p.Keys - support);
    assert (p.Keys - support) * p.Keys == p.Keys - support;
    assert ProbEvent(p, p.Keys - support) == 0.0;

    assert support * (p.Keys - support) == {};
    assert support + (p.Keys - support) == p.Keys;
    Axiom_Additivity(p, support, p.Keys - support);
    Axiom_Normalization(p);
    assert ProbEvent(p, support) == 1.0;
    assert support * p.Keys == support;
  }

  lemma SumOutcomeMasses_RemoveZeroAt(p: PMF, ws: seq<Outcome>, i: nat)
    requires i < |ws|
    requires OutcomeMass(p, ws[i]) == 0.0
    ensures SumOutcomeMasses(p, RemoveAt(ws, i)) == SumOutcomeMasses(p, ws)
  {
    var prefix := ws[..i];
    var suffix := ws[i + 1..];
    SumOutcomeMasses_Concat(p, prefix, suffix);
    SumOutcomeMasses_Concat(p, prefix, [ws[i]]);
    SumOutcomeMasses_Concat(p, prefix + [ws[i]], suffix);
    assert ws == prefix + [ws[i]] + suffix;
    assert SumOutcomeMasses(p, [ws[i]]) == OutcomeMass(p, ws[i]);
    calc {
      SumOutcomeMasses(p, RemoveAt(ws, i));
      ==
      SumOutcomeMasses(p, prefix + suffix);
      == { SumOutcomeMasses_Concat(p, prefix, suffix); }
      SumOutcomeMasses(p, prefix) + SumOutcomeMasses(p, suffix);
      ==
      SumOutcomeMasses(p, prefix) + 0.0 + SumOutcomeMasses(p, suffix);
      ==
      SumOutcomeMasses(p, prefix) + SumOutcomeMasses(p, [ws[i]]) + SumOutcomeMasses(p, suffix);
      == { SumOutcomeMasses_Concat(p, prefix, [ws[i]]); }
      SumOutcomeMasses(p, prefix + [ws[i]]) + SumOutcomeMasses(p, suffix);
      == { SumOutcomeMasses_Concat(p, prefix + [ws[i]], suffix); }
      SumOutcomeMasses(p, ws);
    }
  }

  lemma SumOutcomeMasses_PositiveCompleteNormalized_WithZeroExtras(
    p: PMF,
    ws: seq<Outcome>
  )
    requires IsDistribution(p)
    requires forall i, j :: 0 <= i < j < |ws| ==> ws[i] != ws[j]
    requires forall omega :: omega in p.Keys && OutcomeMass(p, omega) > 0.0 ==> omega in ws
    ensures SumOutcomeMasses(p, ws) == 1.0
  {
    if forall i :: 0 <= i < |ws| ==> ws[i] in p.Keys && OutcomeMass(p, ws[i]) > 0.0 {
      SumOutcomeMasses_PositiveCompleteNormalized(p, ws);
    } else {
      var i :| 0 <= i < |ws| && !(ws[i] in p.Keys && OutcomeMass(p, ws[i]) > 0.0);
      assert OutcomeMass(p, ws[i]) == 0.0 by {
        if ws[i] in p.Keys {
          assert OutcomeMass(p, ws[i]) >= 0.0;
          if OutcomeMass(p, ws[i]) != 0.0 {
            assert OutcomeMass(p, ws[i]) > 0.0;
            assert false;
          }
        }
      }
      RemoveAt_Distinct(ws, i);
      assert forall omega :: omega in p.Keys && OutcomeMass(p, omega) > 0.0 ==> omega in RemoveAt(ws, i) by {
        forall omega | omega in p.Keys && OutcomeMass(p, omega) > 0.0
          ensures omega in RemoveAt(ws, i)
        {
          assert omega in ws;
          assert omega != ws[i] by {
            if omega == ws[i] {
              assert OutcomeMass(p, ws[i]) > 0.0;
              assert false;
            }
          }
          RemoveAt_Membership(ws, i, omega);
        }
      }
      SumOutcomeMasses_PositiveCompleteNormalized_WithZeroExtras(p, RemoveAt(ws, i));
      SumOutcomeMasses_RemoveZeroAt(p, ws, i);
    }
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
  lemma Axiom_Normalization(p: PMF)
    requires IsDistribution(p)
    ensures  ProbEvent(p, p.Keys) == 1.0
  {
  }

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

  lemma DistributionHasSomeKey(p: PMF)
    requires IsDistribution(p)
    ensures p.Keys != {}
  {
    EmptyEventZero(p);
    Axiom_Normalization(p);
    if p.Keys == {} {
      assert ProbEvent(p, p.Keys) == ProbEvent(p, {});
      assert false;
    }
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
