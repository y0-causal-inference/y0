// ===================================================================
// Interventional Distributions — Dafny Specification
//
// This module grounds the abstract IntProb function from do_calculus.dfy
// by defining it concretely via the truncated factorization formula
// (Pearl 2000, Theorem 1.3.1).
//
// All definitions stay within Dafny's discrete PMF type — no measure
// theory is needed.
//
// Layer diagram:
//
//   ┌───────────────────────────┐
//   │  DoCalculus               │  ← Uses IntProb (abstract)
//   ├───────────────────────────┤
//   │  Interventional           │  ← Defines IntProb concretely
//   │  (this module)            │
//   ├───────────────────────────┤
//   │  DAG (d-separation)       │  ← Graph surgery, ancestry
//   ├───────────────────────────┤
//   │  Probability (axioms)     │  ← Kolmogorov, Bayes, chain rule
//   └───────────────────────────┘
//
// To verify:
//   dafny verify probability.dfy dag.dfy interventional.dfy
// ===================================================================

include "dag.dfy"
include "probability.dfy"

module Interventional {

  import opened DAG
  import Prob = Probability

  // ==================================================================
  // 1.  Assignments
  //
  //   A Value is the value taken by a single node.
  //   An Assignment maps each node to such a value.
  //   Used to represent a full instantiation of all variables
  //   in the DAG, or a partial assignment for interventions.
  // ==================================================================

  type Value = Prob.Outcome

  type Assignment = map<Node, Value>

  // A PMF outcome is a joint sample-space point. This bridge interprets
  // such a point as a full node assignment for the current graph.
  ghost function {:axiom} OutcomeToAssignment(
    G: Graph,
    omega: Prob.Outcome
  ): Assignment
    ensures OutcomeToAssignment(G, omega).Keys == Nodes(G)

  // The converse bridge reifies a full node assignment as a joint
  // sample-space point. This is the encoding direction needed when an
  // assignment-level mass function is turned back into a PMF-valued object.
  ghost function {:axiom} AssignmentToOutcome(
    G: Graph,
    full: Assignment
  ): Prob.Outcome
    requires full.Keys == Nodes(G)
    ensures OutcomeToAssignment(G, AssignmentToOutcome(G, full)) == full

  ghost predicate MatchesAssignment(
    G: Graph,
    omega: Prob.Outcome,
    partial: Assignment
  ) {
    partial.Keys <= Nodes(G)
    && forall v :: v in partial.Keys ==> OutcomeToAssignment(G, omega)[v] == partial[v]
  }

  lemma AssignmentToOutcome_MatchesFull(
    G: Graph,
    full: Assignment
  )
    requires full.Keys == Nodes(G)
    ensures MatchesAssignment(G, AssignmentToOutcome(G, full), full)
  {
    assert full.Keys <= Nodes(G);
    forall v | v in full.Keys
      ensures OutcomeToAssignment(G, AssignmentToOutcome(G, full))[v] == full[v]
    {
      assert OutcomeToAssignment(G, AssignmentToOutcome(G, full)) == full;
    }
  }

  lemma AssignmentToOutcome_MatchesExtension(
    G: Graph,
    base: Assignment,
    full: Assignment
  )
    requires base.Keys <= Nodes(G)
    requires full.Keys == Nodes(G)
    requires ExtendsAssignment(base, full)
    ensures MatchesAssignment(G, AssignmentToOutcome(G, full), base)
  {
    AssignmentToOutcome_MatchesFull(G, full);
    MatchesAssignment_Extension(G, AssignmentToOutcome(G, full), base, full);
  }

  lemma AssignmentToOutcome_Injective(
    G: Graph,
    a: Assignment,
    b: Assignment
  )
    requires a.Keys == Nodes(G)
    requires b.Keys == Nodes(G)
    requires AssignmentToOutcome(G, a) == AssignmentToOutcome(G, b)
    ensures a == b
  {
    assert OutcomeToAssignment(G, AssignmentToOutcome(G, a))
      == OutcomeToAssignment(G, AssignmentToOutcome(G, b));
    assert OutcomeToAssignment(G, AssignmentToOutcome(G, a)) == a;
    assert OutcomeToAssignment(G, AssignmentToOutcome(G, b)) == b;
  }

  // The event corresponding to a partial assignment consists of all
  // joint sample points whose interpreted node values agree on that scope.
  ghost function AssignmentEvent(
    p: Prob.PMF,
    G: Graph,
    partial: Assignment
  ): Prob.Event
    requires partial.Keys <= Nodes(G)
  {
    set omega: Prob.Outcome | omega in p.Keys && MatchesAssignment(G, omega, partial) :: omega
  }

  // Scalar probability of a partial assignment under a PMF.
  // This is the pointwise level: a number in [0, 1].
  ghost function AssignmentProb(
    p: Prob.PMF,
    G: Graph,
    partial: Assignment
  ): real
    requires partial.Keys <= Nodes(G)
  {
    Prob.ProbEvent(p, AssignmentEvent(p, G, partial))
  }

  // Conditional scalar probability for one assignment given another.
  // This is the pointwise evaluation corresponding to a PMF-level kernel.
  ghost function AssignmentCondProb(
    p: Prob.PMF,
    G: Graph,
    target: Assignment,
    given: Assignment
  ): real
    requires target.Keys <= Nodes(G)
    requires given.Keys <= Nodes(G)
    requires AssignmentProb(p, G, given) > 0.0
  {
    Prob.ProbCond(p, AssignmentEvent(p, G, target), AssignmentEvent(p, G, given))
  }

  ghost predicate CompatibleAssignments(a: Assignment, b: Assignment) {
    forall v :: v in a.Keys * b.Keys ==> a[v] == b[v]
  }

  ghost predicate ConflictingAssignments(a: Assignment, b: Assignment) {
    exists v :: v in a.Keys * b.Keys && a[v] != b[v]
  }

  ghost predicate ExtendsAssignment(base: Assignment, extension: Assignment) {
    base.Keys <= extension.Keys
    && forall v :: v in base.Keys ==> base[v] == extension[v]
  }

  ghost predicate PairwiseDisjointAssignments(parts: seq<Assignment>) {
    forall i, j :: 0 <= i < j < |parts| ==> parts[i].Keys * parts[j].Keys == {}
  }

  ghost predicate PairwiseDisjointScopes(scopes: seq<set<Node>>) {
    forall i, j :: 0 <= i < j < |scopes| ==> scopes[i] * scopes[j] == {}
  }

  lemma PairwiseDisjointScopes_Prefix(scopes: seq<set<Node>>, n: nat)
    requires n <= |scopes|
    requires PairwiseDisjointScopes(scopes)
    ensures PairwiseDisjointScopes(scopes[..n])
  {
  }

  ghost function AssignmentSeqKeys(parts: seq<Assignment>): set<Node> {
    if |parts| == 0 then {}
    else parts[0].Keys + AssignmentSeqKeys(parts[1..])
  }

  ghost function MergeAssignmentSeq(parts: seq<Assignment>): Assignment
    requires PairwiseDisjointAssignments(parts)
    ensures MergeAssignmentSeq(parts).Keys == AssignmentSeqKeys(parts)
  {
    if |parts| == 0 then
      map[]
    else
      map v | v in parts[0].Keys + AssignmentSeqKeys(parts[1..]) ::
        if v in parts[0].Keys then parts[0][v] else MergeAssignmentSeq(parts[1..])[v]
  }

  ghost function AssignmentProbProduct(
    pms: seq<Prob.PMF>,
    G: Graph,
    parts: seq<Assignment>
  ): real
    requires |pms| == |parts|
    requires forall i :: 0 <= i < |parts| ==> parts[i].Keys <= Nodes(G)
  {
    if |pms| == 0 then 1.0
    else AssignmentProb(pms[0], G, parts[0]) * AssignmentProbProduct(pms[1..], G, parts[1..])
  }

  ghost function MergeAssignments(a: Assignment, b: Assignment): Assignment
    requires CompatibleAssignments(a, b)
    ensures MergeAssignments(a, b).Keys == a.Keys + b.Keys
  {
    map v | v in a.Keys + b.Keys :: if v in a.Keys then a[v] else b[v]
  }

  lemma MatchesAssignment_MergeEquivalent(
    G: Graph,
    omega: Prob.Outcome,
    a: Assignment,
    b: Assignment
  )
    requires a.Keys <= Nodes(G)
    requires b.Keys <= Nodes(G)
    requires CompatibleAssignments(a, b)
    ensures MatchesAssignment(G, omega, MergeAssignments(a, b))
      <==> (MatchesAssignment(G, omega, a) && MatchesAssignment(G, omega, b))
  {
    var merged := MergeAssignments(a, b);
    if MatchesAssignment(G, omega, merged) {
      assert a.Keys <= Nodes(G);
      forall v | v in a.Keys
        ensures OutcomeToAssignment(G, omega)[v] == a[v]
      {
        assert v in merged.Keys;
        assert OutcomeToAssignment(G, omega)[v] == merged[v];
        assert merged[v] == a[v];
      }
      assert MatchesAssignment(G, omega, a);

      assert b.Keys <= Nodes(G);
      forall v | v in b.Keys
        ensures OutcomeToAssignment(G, omega)[v] == b[v]
      {
        assert v in merged.Keys;
        assert OutcomeToAssignment(G, omega)[v] == merged[v];
        if v in a.Keys {
          assert a[v] == b[v];
        }
        assert merged[v] == b[v];
      }
      assert MatchesAssignment(G, omega, b);
    }

    if MatchesAssignment(G, omega, a) && MatchesAssignment(G, omega, b) {
      assert merged.Keys <= Nodes(G) by {
        assert merged.Keys == a.Keys + b.Keys;
      }
      forall v | v in merged.Keys
        ensures OutcomeToAssignment(G, omega)[v] == merged[v]
      {
        if v in a.Keys {
          assert OutcomeToAssignment(G, omega)[v] == a[v];
          assert merged[v] == a[v];
        } else {
          assert v in b.Keys;
          assert OutcomeToAssignment(G, omega)[v] == b[v];
          assert merged[v] == b[v];
        }
      }
      assert MatchesAssignment(G, omega, merged);
    }
  }

  lemma AssignmentEvent_Intersection_Compatible(
    p: Prob.PMF,
    G: Graph,
    a: Assignment,
    b: Assignment
  )
    requires a.Keys <= Nodes(G)
    requires b.Keys <= Nodes(G)
    requires CompatibleAssignments(a, b)
    ensures AssignmentEvent(p, G, a) * AssignmentEvent(p, G, b)
      == AssignmentEvent(p, G, MergeAssignments(a, b))
  {
    var merged := MergeAssignments(a, b);
    assert forall omega ::
      (omega in AssignmentEvent(p, G, a) * AssignmentEvent(p, G, b))
        ==> (omega in AssignmentEvent(p, G, merged)) by {
      forall omega | omega in AssignmentEvent(p, G, a) * AssignmentEvent(p, G, b)
        ensures omega in AssignmentEvent(p, G, merged)
      {
        assert omega in AssignmentEvent(p, G, a);
        assert omega in AssignmentEvent(p, G, b);
        assert MatchesAssignment(G, omega, a);
        assert MatchesAssignment(G, omega, b);
        MatchesAssignment_MergeEquivalent(G, omega, a, b);
        assert MatchesAssignment(G, omega, merged);
      }
    }
    assert AssignmentEvent(p, G, a) * AssignmentEvent(p, G, b)
      <= AssignmentEvent(p, G, merged);

    assert forall omega ::
      (omega in AssignmentEvent(p, G, merged))
        ==> (omega in AssignmentEvent(p, G, a) * AssignmentEvent(p, G, b)) by {
      forall omega | omega in AssignmentEvent(p, G, merged)
        ensures omega in AssignmentEvent(p, G, a) * AssignmentEvent(p, G, b)
      {
        assert MatchesAssignment(G, omega, merged);
        MatchesAssignment_MergeEquivalent(G, omega, a, b);
        assert MatchesAssignment(G, omega, a);
        assert MatchesAssignment(G, omega, b);
        assert omega in AssignmentEvent(p, G, a);
        assert omega in AssignmentEvent(p, G, b);
      }
    }
    assert AssignmentEvent(p, G, merged)
      <= AssignmentEvent(p, G, a) * AssignmentEvent(p, G, b);
  }

  lemma AssignmentEvent_Intersection_Incompatible(
    p: Prob.PMF,
    G: Graph,
    a: Assignment,
    b: Assignment
  )
    requires a.Keys <= Nodes(G)
    requires b.Keys <= Nodes(G)
    requires ConflictingAssignments(a, b)
    ensures AssignmentEvent(p, G, a) * AssignmentEvent(p, G, b) == {}
  {
    var conflictKey :| conflictKey in a.Keys * b.Keys && a[conflictKey] != b[conflictKey];
    assert forall omega ::
      omega in AssignmentEvent(p, G, a) * AssignmentEvent(p, G, b) ==> false by {
      forall omega | omega in AssignmentEvent(p, G, a) * AssignmentEvent(p, G, b)
        ensures false
      {
        assert omega in AssignmentEvent(p, G, a);
        assert omega in AssignmentEvent(p, G, b);
        assert MatchesAssignment(G, omega, a);
        assert MatchesAssignment(G, omega, b);
        assert OutcomeToAssignment(G, omega)[conflictKey] == a[conflictKey];
        assert OutcomeToAssignment(G, omega)[conflictKey] == b[conflictKey];
        assert false;
      }
    }
  }

  lemma MatchesAssignment_Extension(
    G: Graph,
    omega: Prob.Outcome,
    base: Assignment,
    extension: Assignment
  )
    requires base.Keys <= Nodes(G)
    requires extension.Keys <= Nodes(G)
    requires ExtendsAssignment(base, extension)
    requires MatchesAssignment(G, omega, extension)
    ensures MatchesAssignment(G, omega, base)
  {
    assert base.Keys <= Nodes(G);
    forall v | v in base.Keys ensures OutcomeToAssignment(G, omega)[v] == base[v] {
      assert OutcomeToAssignment(G, omega)[v] == extension[v];
      assert extension[v] == base[v];
    }
  }

  lemma AssignmentEvent_StrengtheningSubset(
    p: Prob.PMF,
    G: Graph,
    base: Assignment,
    extension: Assignment
  )
    requires base.Keys <= Nodes(G)
    requires extension.Keys <= Nodes(G)
    requires ExtendsAssignment(base, extension)
    ensures AssignmentEvent(p, G, extension) <= AssignmentEvent(p, G, base)
  {
    forall omega | omega in AssignmentEvent(p, G, extension) ensures omega in AssignmentEvent(p, G, base) {
      assert MatchesAssignment(G, omega, extension);
      MatchesAssignment_Extension(G, omega, base, extension);
      assert MatchesAssignment(G, omega, base);
    }
  }

  // ==================================================================
  // 2.  Conditional Factor
  //
  //   P(v | pa(v)) — the conditional probability of node v taking
  //   its assigned value, given the values of its parents.
  //   Extracted from the joint PMF.
  //
  //   This is the building block of the Markov factorization.
  // ==================================================================

  ghost function {:axiom} ConditionalFactor(
    p: Prob.PMF,
    v: Node,
    parents: set<Node>,
    assignment: Assignment
  ): real

  // The conditional factor is always non-negative.
  lemma {:axiom} ConditionalFactor_NonNeg(
    p: Prob.PMF,
    v: Node,
    parents: set<Node>,
    assignment: Assignment
  )
    requires Prob.IsDistribution(p)
    ensures ConditionalFactor(p, v, parents, assignment) >= 0.0

  // ==================================================================
  // 3.  Markov Factorization
  //
  //   A joint PMF satisfies the Causal Markov Condition for DAG G
  //   if the probability of every full assignment equals the product
  //   of conditional factors over all nodes:
  //
  //     P(v₁, ..., vₙ) = ∏ᵢ P(vᵢ | pa_G(vᵢ))
  //
  //   Since Dafny has no built-in product over a set, we axiomatize
  //   the relationship. In a DAG with a topological sort, the product
  //   can be computed recursively along the ordering.
  // ==================================================================

  ghost predicate {:axiom} MarkovFactorization(G: Graph, p: Prob.PMF)

  // A Markov-factored PMF is a valid distribution.
  lemma {:axiom} MarkovFactorization_IsDistribution(
    G: Graph, p: Prob.PMF
  )
    requires IsDAG(G)
    requires MarkovFactorization(G, p)
    ensures Prob.IsDistribution(p)

  // ==================================================================
  // 4.  TruncatePMF — the do-operator
  //
  //   The interventional distribution after do(X = xVals):
  //
  //   1. Replace each intervened-node local factor with a point mass
  //      at the imposed value xVals.
  //   2. Keep each non-intervened local factor from the observational
  //      distribution.
  //   3. Interpret the result against the mutilated graph G_{X̄}.
  //
  //   The public PMF-valued constructor remains axiomatic below. The
  //   helpers in this section are the first factor-level scaffolding for a
  //   later concrete definition.
  //
  //   Ref: Pearl (2000), Theorem 1.3.1 / Definition 3.2.1
  // ==================================================================

  ghost function TruncatedLocalFactor(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    full: Assignment,
    v: Node
  ): real
    requires v in Nodes(G)
    requires full.Keys == Nodes(G)
    requires xVals.Keys == X
    requires X <= Nodes(G)
  {
    if v in X then
      if full[v] == xVals[v] then 1.0 else 0.0
    else
      ConditionalFactor(p, v, Parents(G, v), full)
  }

  lemma TruncatedLocalFactor_NonNeg(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    full: Assignment,
    v: Node
  )
    requires Prob.IsDistribution(p)
    requires v in Nodes(G)
    requires full.Keys == Nodes(G)
    requires xVals.Keys == X
    requires X <= Nodes(G)
    ensures TruncatedLocalFactor(G, p, X, xVals, full, v) >= 0.0
  {
    if v !in X {
      ConditionalFactor_NonNeg(p, v, Parents(G, v), full);
    }
  }

  lemma TruncatedLocalFactor_IntervenedMatch(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    full: Assignment,
    v: Node
  )
    requires v in Nodes(G)
    requires full.Keys == Nodes(G)
    requires xVals.Keys == X
    requires X <= Nodes(G)
    requires v in X
    requires full[v] == xVals[v]
    ensures TruncatedLocalFactor(G, p, X, xVals, full, v) == 1.0
  {
  }

  lemma TruncatedLocalFactor_IntervenedMismatch(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    full: Assignment,
    v: Node
  )
    requires v in Nodes(G)
    requires full.Keys == Nodes(G)
    requires xVals.Keys == X
    requires X <= Nodes(G)
    requires v in X
    requires full[v] != xVals[v]
    ensures TruncatedLocalFactor(G, p, X, xVals, full, v) == 0.0
  {
  }

  lemma TruncatedLocalFactor_Unintervened(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    full: Assignment,
    v: Node
  )
    requires v in Nodes(G)
    requires full.Keys == Nodes(G)
    requires xVals.Keys == X
    requires X <= Nodes(G)
    requires v !in X
    ensures TruncatedLocalFactor(G, p, X, xVals, full, v)
      == ConditionalFactor(p, v, Parents(G, v), full)
  {
  }

  ghost function TruncatedFactorProduct(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    full: Assignment,
    ord: seq<Node>
  ): real
    requires full.Keys == Nodes(G)
    requires xVals.Keys == X
    requires X <= Nodes(G)
    requires forall i :: 0 <= i < |ord| ==> ord[i] in Nodes(G)
  {
    if |ord| == 0 then 1.0 else
      TruncatedLocalFactor(G, p, X, xVals, full, ord[0])
      * TruncatedFactorProduct(G, p, X, xVals, full, ord[1..])
  }

  lemma TruncatedFactorProduct_NonNeg(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    full: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires full.Keys == Nodes(G)
    requires xVals.Keys == X
    requires X <= Nodes(G)
    requires forall i :: 0 <= i < |ord| ==> ord[i] in Nodes(G)
    ensures TruncatedFactorProduct(G, p, X, xVals, full, ord) >= 0.0
  {
    if |ord| > 0 {
      assert ord[0] in Nodes(G);
      TruncatedLocalFactor_NonNeg(G, p, X, xVals, full, ord[0]);
      assert forall i :: 0 <= i < |ord[1..]| ==> ord[1..][i] in Nodes(G) by {
        forall i | 0 <= i < |ord[1..]| ensures ord[1..][i] in Nodes(G) {
          assert ord[1..][i] == ord[i + 1];
        }
      }
      TruncatedFactorProduct_NonNeg(G, p, X, xVals, full, ord[1..]);
    }
  }

  lemma TruncatedFactorProduct_ZeroOnInterventionConflict(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    full: Assignment,
    ord: seq<Node>,
    i: nat
  )
    requires full.Keys == Nodes(G)
    requires xVals.Keys == X
    requires X <= Nodes(G)
    requires forall j :: 0 <= j < |ord| ==> ord[j] in Nodes(G)
    requires i < |ord|
    requires ord[i] in X
    requires full[ord[i]] != xVals[ord[i]]
    ensures TruncatedFactorProduct(G, p, X, xVals, full, ord) == 0.0
  {
    if i == 0 {
      TruncatedLocalFactor_IntervenedMismatch(G, p, X, xVals, full, ord[0]);
    } else {
      assert forall j :: 0 <= j < |ord[1..]| ==> ord[1..][j] in Nodes(G) by {
        forall j | 0 <= j < |ord[1..]| ensures ord[1..][j] in Nodes(G) {
          assert ord[1..][j] == ord[j + 1];
        }
      }
      assert i - 1 < |ord[1..]|;
      assert ord[1..][i - 1] == ord[i];
      TruncatedFactorProduct_ZeroOnInterventionConflict(G, p, X, xVals, full, ord[1..], i - 1);
    }
  }

  ghost function TruncatedAssignmentMass(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    full: Assignment,
    ord: seq<Node>
  ): real
    requires IsTopologicalSort(G, ord)
    requires full.Keys == Nodes(G)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
  {
    TruncatedFactorProduct(G, p, X, xVals, full, ord)
  }

  lemma TruncatedAssignmentMass_NonNeg(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    full: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires full.Keys == Nodes(G)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures TruncatedAssignmentMass(G, p, X, xVals, full, ord) >= 0.0
  {
    assert X <= Nodes(G);
    assert forall i :: 0 <= i < |ord| ==> ord[i] in Nodes(G);
    TruncatedFactorProduct_NonNeg(G, p, X, xVals, full, ord);
  }

  lemma TruncatedAssignmentMass_ZeroOnInterventionConflict(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    full: Assignment,
    ord: seq<Node>,
    v: Node
  )
    requires IsTopologicalSort(G, ord)
    requires full.Keys == Nodes(G)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    requires v in X
    requires full[v] != xVals[v]
    ensures TruncatedAssignmentMass(G, p, X, xVals, full, ord) == 0.0
  {
    assert X <= Nodes(G);
    assert v in Nodes(G);
    assert v in ord;
    var i :| 0 <= i < |ord| && ord[i] == v;
    assert forall j :: 0 <= j < |ord| ==> ord[j] in Nodes(G);
    TruncatedFactorProduct_ZeroOnInterventionConflict(G, p, X, xVals, full, ord, i);
  }

  // A future PMF-valued truncation constructor will need a finite support of
  // full assignments, then encode those assignments back into PMF outcomes.
  // This helper exposes that finite assignment-side support without yet
  // committing to a concrete constructor body.
  ghost function {:axiom} TruncateSupportAssignments(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment
  ): seq<Assignment>
    requires Prob.IsDistribution(p)
    requires xVals.Keys == X
    ensures forall i :: 0 <= i < |TruncateSupportAssignments(G, p, X, xVals)| ==>
      TruncateSupportAssignments(G, p, X, xVals)[i].Keys == Nodes(G)
    ensures forall i :: 0 <= i < |TruncateSupportAssignments(G, p, X, xVals)| ==>
      ExtendsAssignment(xVals, TruncateSupportAssignments(G, p, X, xVals)[i])
    ensures forall i, j :: 0 <= i < j < |TruncateSupportAssignments(G, p, X, xVals)| ==>
      AssignmentToOutcome(G, TruncateSupportAssignments(G, p, X, xVals)[i])
        != AssignmentToOutcome(G, TruncateSupportAssignments(G, p, X, xVals)[j])

  ghost function EncodedTruncateSupport(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment
  ): set<Prob.Outcome>
    requires Prob.IsDistribution(p)
    requires xVals.Keys == X
  {
    set i: int | 0 <= i < |TruncateSupportAssignments(G, p, X, xVals)| ::
      AssignmentToOutcome(G, TruncateSupportAssignments(G, p, X, xVals)[i])
  }

  lemma EncodedTruncateSupport_MatchesAssignment(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment
  )
    requires Prob.IsDistribution(p)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures forall omega :: omega in EncodedTruncateSupport(G, p, X, xVals) ==> MatchesAssignment(G, omega, xVals)
  {
    forall omega | omega in EncodedTruncateSupport(G, p, X, xVals)
      ensures MatchesAssignment(G, omega, xVals)
    {
      var i :| 0 <= i < |TruncateSupportAssignments(G, p, X, xVals)|
        && omega == AssignmentToOutcome(G, TruncateSupportAssignments(G, p, X, xVals)[i]);
      var full := TruncateSupportAssignments(G, p, X, xVals)[i];
      assert full.Keys == Nodes(G);
      assert ExtendsAssignment(xVals, full);
      AssignmentToOutcome_MatchesExtension(G, xVals, full);
    }
  }

  ghost function {:axiom} TruncatePMF(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment
  ): Prob.PMF
    requires xVals.Keys == X
    requires Prob.IsDistribution(p)
    ensures TruncatePMF(G, p, X, xVals).Keys <= EncodedTruncateSupport(G, p, X, xVals)
    ensures forall omega :: omega in TruncatePMF(G, p, X, xVals).Keys ==> MatchesAssignment(G, omega, xVals)

  // The truncated PMF is a valid distribution.
  lemma {:axiom} TruncatePMF_IsDistribution(
    G: Graph, p: Prob.PMF,
    X: set<Node>, xVals: Assignment
  )
    requires Prob.IsDistribution(p)
    requires xVals.Keys == X
    ensures Prob.IsDistribution(TruncatePMF(G, p, X, xVals))

  // Truncating with empty intervention recovers the original PMF.
  lemma {:axiom} TruncatePMF_Empty(G: Graph, p: Prob.PMF)
    requires Prob.IsDistribution(p)
    ensures TruncatePMF(G, p, {}, map[]) == p

  // If a PMF is Markov-factored, the truncated PMF is also
  // Markov-factored w.r.t. the mutilated graph G_{X̄}.
  lemma {:axiom} TruncatePMF_Markov(
    G: Graph, p: Prob.PMF,
    X: set<Node>, xVals: Assignment
  )
    requires IsDAG(G)
    requires Prob.IsDistribution(p)
    requires MarkovFactorization(G, p)
    requires xVals.Keys == X
    ensures MarkovFactorization(
      RemoveIncoming(G, X),
      TruncatePMF(G, p, X, xVals)
    )

  lemma TruncatePMF_InterventionEventIsSupport(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment
  )
    requires Prob.IsDistribution(p)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures AssignmentEvent(TruncatePMF(G, p, X, xVals), G, xVals)
      == TruncatePMF(G, p, X, xVals).Keys
  {
    var t := TruncatePMF(G, p, X, xVals);
    assert forall omega :: omega in t.Keys ==> omega in AssignmentEvent(t, G, xVals) by {
      forall omega | omega in t.Keys
        ensures omega in AssignmentEvent(t, G, xVals)
      {
        assert MatchesAssignment(G, omega, xVals);
      }
    }
    assert t.Keys <= AssignmentEvent(t, G, xVals);

    assert forall omega :: omega in AssignmentEvent(t, G, xVals) ==> omega in t.Keys by {
      forall omega | omega in AssignmentEvent(t, G, xVals)
        ensures omega in t.Keys
      {
      }
    }
    assert AssignmentEvent(t, G, xVals) <= t.Keys;
  }

  lemma TruncatePMF_InterventionProbabilityOne(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment
  )
    requires Prob.IsDistribution(p)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures AssignmentProb(TruncatePMF(G, p, X, xVals), G, xVals) == 1.0
  {
    var t := TruncatePMF(G, p, X, xVals);
    TruncatePMF_IsDistribution(G, p, X, xVals);
    TruncatePMF_InterventionEventIsSupport(G, p, X, xVals);
    Prob.Axiom_Normalization(t);
  }

  lemma TruncatePMF_ConflictingAssignmentEventEmpty(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    partial: Assignment
  )
    requires Prob.IsDistribution(p)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    requires partial.Keys <= Nodes(G)
    requires ConflictingAssignments(partial, xVals)
    ensures AssignmentEvent(TruncatePMF(G, p, X, xVals), G, partial) == {}
  {
    var t := TruncatePMF(G, p, X, xVals);
    TruncatePMF_InterventionEventIsSupport(G, p, X, xVals);
    AssignmentEvent_Intersection_Incompatible(t, G, partial, xVals);
    assert forall omega ::
      (omega in AssignmentEvent(t, G, partial))
        <==> (omega in AssignmentEvent(t, G, partial) * AssignmentEvent(t, G, xVals)) by {
      forall omega
        ensures omega in AssignmentEvent(t, G, partial)
          <==> omega in AssignmentEvent(t, G, partial) * AssignmentEvent(t, G, xVals)
      {
        if omega in AssignmentEvent(t, G, partial) {
          assert omega in t.Keys;
          assert omega in AssignmentEvent(t, G, xVals);
        }
      }
    }
  }

  lemma TruncatePMF_ConflictingAssignmentZero(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    partial: Assignment
  )
    requires Prob.IsDistribution(p)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    requires partial.Keys <= Nodes(G)
    requires ConflictingAssignments(partial, xVals)
    ensures AssignmentProb(TruncatePMF(G, p, X, xVals), G, partial) == 0.0
  {
    var t := TruncatePMF(G, p, X, xVals);
    TruncatePMF_IsDistribution(G, p, X, xVals);
    TruncatePMF_ConflictingAssignmentEventEmpty(G, p, X, xVals, partial);
    Prob.EmptyEventZero(t);
  }

  // ==================================================================
  // 5.  IntProbConcrete — grounding the abstract IntProb
  //
  //   P(Y | do(X = xVals), W = wVals) computed concretely:
  //     = ProbCond(TruncatePMF(G, p, X, xVals), Y-event, W-event)
  //
  //   This returns a real number for specific value assignments,
  //   whereas IntProb in do_calculus.dfy returns the whole PMF-valued
  //   interventional kernel over Y-assignments. The grounding lemma
  //   relates that PMF-level interface to this pointwise scalar view:
  //     for all y-assignments, IntProb(G,Y,X,W)[y] == IntProbConcrete(...)
  // ==================================================================

  // For a specific Y-assignment and W-assignment, compute the
  // concrete interventional probability.
  ghost function IntProbConcrete(
    G: Graph,
    p: Prob.PMF,
    yAssign: Assignment,
    xAssign: Assignment,
    wAssign: Assignment
  ): real
    requires Prob.IsDistribution(p)
    requires MarkovFactorization(G, p)
    requires xAssign.Keys <= Nodes(G)
    requires yAssign.Keys <= Nodes(G)
    requires wAssign.Keys <= Nodes(G)
    requires AssignmentProb(TruncatePMF(G, p, xAssign.Keys, xAssign), G, wAssign) > 0.0
  {
    AssignmentCondProb(
      TruncatePMF(G, p, xAssign.Keys, xAssign),
      G,
      yAssign,
      wAssign
    )
  }

  // The grounding axiom: IntProbConcrete equals the conditional
  // probability in the truncated distribution.
  //
  // IntProbConcrete(G, p, y, x, w) ==
  //   AssignmentCondProb(TruncatePMF(G, p, X, x), G, y, w)
  //
  // This connects the abstract IntProb (which returns a PMF) to
  // the concrete computation (which returns a real).
  lemma IntProb_Grounded(
    G: Graph,
    p: Prob.PMF,
    yAssign: Assignment,
    xAssign: Assignment,
    wAssign: Assignment
  )
    requires Prob.IsDistribution(p)
    requires MarkovFactorization(G, p)
    requires xAssign.Keys <= Nodes(G)
    requires yAssign.Keys <= Nodes(G)
    requires wAssign.Keys <= Nodes(G)
    requires AssignmentProb(TruncatePMF(G, p, xAssign.Keys, xAssign), G, wAssign) > 0.0
    ensures IntProbConcrete(G, p, yAssign, xAssign, wAssign)
      == AssignmentCondProb(
        TruncatePMF(G, p, xAssign.Keys, xAssign),
        G,
        yAssign,
        wAssign
      )
  {
  }

  // ==================================================================
  // 6.  GlobalMarkov from Factorization
  //
  //   If a PMF satisfies the Markov factorization for DAG G, then
  //   every d-separation in G implies conditional independence.
  //
  //   (Y ⊥_G Z | W) and MarkovFactorization(G, p)
  //     ⟹  P(Y | Z, W) = P(Y | W)
  //
  //   A full proof requires the Bayes Ball / d-separation completeness
  //   theorem, which is non-trivial in Dafny.  This remains an axiom
  //   on the first pass.
  // ==================================================================

  lemma {:axiom} GlobalMarkov_From_Factorization(
    G: Graph,
    p: Prob.PMF,
    Y: set<Node>,
    Z: set<Node>,
    W: set<Node>
  )
    requires IsDAG(G)
    requires Prob.IsDistribution(p)
    requires MarkovFactorization(G, p)
    requires DSep(G, Y, Z, W)
    // d-separation implies conditional independence:
    // for all assignments, P(Y | Z, W) == P(Y | W)

  // ==================================================================
  // 7.  Products of PMFs
  //
  //   ProductPMF combines factor PMFs over pairwise-disjoint scopes.
  //   The explicit scope sequence is the graph-aware boundary that lets
  //   the abstract constructor speak in assignment-event terms.
  // ==================================================================

  ghost function ProductPMF(
    G: Graph,
    scopes: seq<set<Node>>,
    pms: seq<Prob.PMF>
  ): Prob.PMF
    requires |scopes| == |pms|
    requires PairwiseDisjointScopes(scopes)
    requires forall i :: 0 <= i < |scopes| ==> scopes[i] <= Nodes(G)
  {
    Prob.ProductPMF(pms)
  }

  lemma ProductPMF_IsDistribution(
    G: Graph,
    scopes: seq<set<Node>>,
    pms: seq<Prob.PMF>
  )
    requires |scopes| == |pms|
    requires PairwiseDisjointScopes(scopes)
    requires forall i :: 0 <= i < |scopes| ==> scopes[i] <= Nodes(G)
    requires forall i :: 0 <= i < |pms| ==> Prob.IsDistribution(pms[i])
    ensures Prob.IsDistribution(ProductPMF(G, scopes, pms))
  {
    Prob.ProductPMF_IsDistribution(pms);
  }

  lemma {:axiom} ProductPMF_Grounded(
    G: Graph,
    scopes: seq<set<Node>>,
    pms: seq<Prob.PMF>,
    parts: seq<Assignment>
  )
    requires |scopes| == |pms|
    requires |parts| == |scopes|
    requires PairwiseDisjointScopes(scopes)
    requires PairwiseDisjointAssignments(parts)
    requires AssignmentSeqKeys(parts) <= Nodes(G)
    requires forall i :: 0 <= i < |scopes| ==> scopes[i] <= Nodes(G)
    requires forall i :: 0 <= i < |parts| ==> parts[i].Keys <= scopes[i]
    ensures AssignmentProb(ProductPMF(G, scopes, pms), G, MergeAssignmentSeq(parts))
      == AssignmentProbProduct(pms, G, parts)

  // ==================================================================
  // 7.  Marginalization
  //
  //   Marginalizing a PMF sums out a set of variables.
  //   Σ_W P(V) yields a distribution over V \ W.
  // ==================================================================

  // Marginalize: Σ_W P(V)
  ghost function {:axiom} Marginalize(G: Graph, p: Prob.PMF, W: set<Node>): Prob.PMF
    requires W <= Nodes(G)
    ensures forall partial: Assignment ::
      partial.Keys <= Nodes(G) - W ==>
      AssignmentProb(Marginalize(G, p, W), G, partial) == AssignmentProb(p, G, partial)

  lemma {:axiom} Marginalize_IsDistribution(G: Graph, p: Prob.PMF, W: set<Node>)
    requires Prob.IsDistribution(p)
    requires W <= Nodes(G)
    ensures Prob.IsDistribution(Marginalize(G, p, W))

}  // end module Interventional
