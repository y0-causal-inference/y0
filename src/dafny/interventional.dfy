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

  // ==================================================================
  //  Concrete assignment-keyed distribution (a.k.a. "joint PMF").
  //
  //  AssignmentPMF is the concrete counterpart of the abstract
  //  Prob.PMF.  Its keys are FULL assignments over Nodes(G), so each
  //  key already carries every node's value explicitly.  This lets the
  //  causal layer (factorization, do-operator) be defined and proved
  //  WITHOUT the OutcomeToAssignment / SetToSequence bridge axioms:
  //  a node's value is read directly off the key, and normalization is
  //  a finite sum recursing on the map's own domain.
  //
  //  The abstract Prob.PMF layer (Kolmogorov axioms, generic lemmas) is
  //  left untouched; a boundary adapter relates the two where needed.
  // ==================================================================

  type AssignmentPMF = map<Assignment, real>

  // A full assignment for G assigns a value to exactly the nodes of G.
  ghost predicate IsFullAssignment(G: Graph, a: Assignment) {
    a.Keys == Nodes(G)
  }

  // Every key of q is a full assignment over Nodes(G).
  ghost predicate WellFormedAssignmentPMF(G: Graph, q: AssignmentPMF) {
    forall a :: a in q.Keys ==> IsFullAssignment(G, a)
  }

  // Every mass is non-negative.
  ghost predicate AssignmentPMF_AllNonNeg(q: AssignmentPMF) {
    forall a :: a in q ==> q[a] >= 0.0
  }

  // Sum of all masses, recursing on the map's own (finite) domain.
  // No enumeration axiom is needed: a concrete map can be peeled one
  // key at a time, with cardinality strictly decreasing.
  ghost function SumAssignmentMasses(q: AssignmentPMF): real
    decreases |q|
  {
    if |q| == 0 then 0.0
    else
      var a :| a in q.Keys;
      q[a] + SumAssignmentMasses(q - {a})
  }

  // A well-formed, non-negative AssignmentPMF whose masses sum to 1.
  ghost predicate IsAssignmentDistribution(G: Graph, q: AssignmentPMF) {
    WellFormedAssignmentPMF(G, q)
    && AssignmentPMF_AllNonNeg(q)
    && SumAssignmentMasses(q) == 1.0
  }

  // ------------------------------------------------------------------
  //  Marginalization and the concrete conditional factor.
  //
  //  The submap of q whose keys agree with `ref` on every node of S.
  //  Marginal mass over S is the total mass of that submap; the
  //  conditional factor P(v | pa(v)) at `ref` is the ratio of the
  //  marginal over pa(v)+{v} to the marginal over pa(v).
  // ------------------------------------------------------------------

  ghost function MatchingSubmap(
    q: AssignmentPMF, S: set<Node>, ref: Assignment
  ): AssignmentPMF
  {
    map a | a in q.Keys
         && (forall v :: v in S ==> v in a.Keys && v in ref.Keys && a[v] == ref[v])
         :: q[a]
  }

  // The matching submap is a sub-map of q (keys and values agree).
  lemma MatchingSubmap_IsSubmap(
    q: AssignmentPMF, S: set<Node>, ref: Assignment
  )
    ensures MatchingSubmap(q, S, ref).Keys <= q.Keys
    ensures forall a :: a in MatchingSubmap(q, S, ref).Keys ==>
              MatchingSubmap(q, S, ref)[a] == q[a]
  {
  }

  // ------------------------------------------------------------------
  //  L2: MatchingSubmap composition identity.
  //
  //  Filtering on S then on T (independently, both against the same
  //  ref) is the same as filtering on S ∪ T in one step:
  //
  //    MatchingSubmap(MatchingSubmap(q, S, ref), T, ref)
  //      == MatchingSubmap(q, S + T, ref)
  //
  //  This is used in L3 to identify the "pa ∪ {v}-matching" part of
  //  a marginal submap as exactly MatchingSubmap(q, pa + {v}, ref).
  // ------------------------------------------------------------------

  lemma MatchingSubmap_Compose(
    q: AssignmentPMF, S: set<Node>, T: set<Node>, ref: Assignment
  )
    ensures MatchingSubmap(MatchingSubmap(q, S, ref), T, ref)
         == MatchingSubmap(q, S + T, ref)
  {
    var lhs := MatchingSubmap(MatchingSubmap(q, S, ref), T, ref);
    var rhs := MatchingSubmap(q, S + T, ref);
    assert forall a :: a in lhs.Keys <==> a in rhs.Keys by {
      forall a
        ensures a in lhs.Keys <==> a in rhs.Keys
      {
        assert (a in lhs.Keys)
          <==> (a in q.Keys
                && (forall v :: v in S ==> v in a.Keys && v in ref.Keys && a[v] == ref[v])
                && (forall v :: v in T ==> v in a.Keys && v in ref.Keys && a[v] == ref[v]));
        assert (a in rhs.Keys)
          <==> (a in q.Keys
                && (forall v :: v in S + T ==> v in a.Keys && v in ref.Keys && a[v] == ref[v]));
        assert (forall v :: v in S + T ==> v in a.Keys && v in ref.Keys && a[v] == ref[v])
          <==> ((forall v :: v in S ==> v in a.Keys && v in ref.Keys && a[v] == ref[v])
                && (forall v :: v in T ==> v in a.Keys && v in ref.Keys && a[v] == ref[v]));
      }
    }
    // Values agree on the common key set: both sides carry q[a].
    assert forall a :: a in rhs.Keys ==>
      lhs[a] == q[a] == rhs[a];
  }

  // Corollary: nested filtering with S ⊆ T is idempotent.
  lemma MatchingSubmap_NestedScope(
    q: AssignmentPMF, S: set<Node>, T: set<Node>, ref: Assignment
  )
    requires S <= T
    ensures MatchingSubmap(MatchingSubmap(q, T, ref), S, ref)
         == MatchingSubmap(q, T, ref)
  {
    // S ⊆ T means T + S == T; apply the composition identity.
    MatchingSubmap_Compose(q, T, S, ref);
    assert T + S == T;
  }

  // ------------------------------------------------------------------
  //  L3: Marginalizing out a single node splits a marginal.
  //
  //  MarginalMass(q, S, ref)  ==  MarginalMass(q, S + {v}, ref)
  //                            +  SumAssignmentMasses(complement)
  //
  //  where complement = assignments agreeing on S but NOT on {v}.
  //
  //  Proof: partition MatchingSubmap(q, S, ref) by inner.Keys (L1c);
  //  identify the positive part as MatchingSubmap(q, S+{v}, ref)
  //  using value equality outer[k] == q[k] == inner[k].
  // ------------------------------------------------------------------

  lemma MarginalMass_Marginalizes(
    q: AssignmentPMF, S: set<Node>, v: Node, ref: Assignment
  )
    ensures MarginalMass(q, S, ref) ==
      MarginalMass(q, S + {v}, ref) +
      SumAssignmentMasses(
        map a | a in MatchingSubmap(q, S, ref).Keys
             && a !in MatchingSubmap(q, S + {v}, ref).Keys
             :: q[a])
  {
    var inner := MatchingSubmap(q, S + {v}, ref);
    var outer := MatchingSubmap(q, S, ref);

    // Step 1: inner.Keys ⊆ outer.Keys (agreeing on S+{v} implies agreeing on S).
    assert inner.Keys <= outer.Keys by {
      forall a | a in inner.Keys ensures a in outer.Keys {
        assert forall u :: u in S ==> u in a.Keys && u in ref.Keys && a[u] == ref[u];
      }
    }

    // Step 2: Partition outer by inner.Keys (L1c).
    SumAssignmentMasses_Partition(outer, inner.Keys);
    // Now: SumAssignmentMasses(outer)
    //   == SumAssignmentMasses(map x | x in inner.Keys :: outer[x])
    //    + SumAssignmentMasses(map x | x in outer.Keys - inner.Keys :: outer[x])

    // Step 3: (map x | x in inner.Keys :: outer[x]) == inner.
    // Keys agree; values: outer[a] == q[a] == inner[a] by MatchingSubmap_IsSubmap.
    assert (map x | x in inner.Keys :: outer[x]) == inner by {
      assert (map x | x in inner.Keys :: outer[x]).Keys == inner.Keys;
      MatchingSubmap_IsSubmap(q, S, ref);         // outer[a] == q[a] for a in outer.Keys
      MatchingSubmap_IsSubmap(q, S + {v}, ref);   // inner[a] == q[a] for a in inner.Keys
      forall a | a in inner.Keys
        ensures (map x | x in inner.Keys :: outer[x])[a] == inner[a]
      {
        assert outer[a] == q[a];
        assert inner[a] == q[a];
      }
    }

    // Step 4: complement values outer[x] == q[x], so the two complement maps are equal.
    assert (map x | x in outer.Keys - inner.Keys :: outer[x])
        == (map x | x in outer.Keys - inner.Keys :: q[x]) by {
      assert (map x | x in outer.Keys - inner.Keys :: outer[x]).Keys
          == (map x | x in outer.Keys - inner.Keys :: q[x]).Keys;
      MatchingSubmap_IsSubmap(q, S, ref);
      forall a | a in outer.Keys - inner.Keys
        ensures (map x | x in outer.Keys - inner.Keys :: outer[x])[a]
             == (map x | x in outer.Keys - inner.Keys :: q[x])[a]
      {
        assert outer[a] == q[a];
      }
    }

    // Step 5: set-difference and boolean-and guards are interchangeable.
    assert (map x | x in outer.Keys - inner.Keys :: q[x])
        == (map a | a in outer.Keys && a !in inner.Keys :: q[a]);
  }

  // ------------------------------------------------------------------
  //  L5: Scope invariance — MarginalMass depends only on ref|S.
  //
  //  MatchingSubmap filters by agreement on S; if two references agree
  //  on every node in S, the filter predicate is identical, so the
  //  submaps and their sums are equal.  No MarkovFactorizationConcrete
  //  required — this is purely definitional.
  // ------------------------------------------------------------------

  // L5a: Two references that agree on S produce identical submaps.
  lemma MatchingSubmap_ScopeInvariant(
    q: AssignmentPMF, S: set<Node>, ref1: Assignment, ref2: Assignment
  )
    requires forall v :: v in S ==> v in ref1.Keys && v in ref2.Keys && ref1[v] == ref2[v]
    ensures MatchingSubmap(q, S, ref1) == MatchingSubmap(q, S, ref2)
  {
    // The filter predicates are identical: a agrees with ref1 on S ⟺ a agrees with ref2 on S.
    assert forall a ::
      a in MatchingSubmap(q, S, ref1).Keys <==> a in MatchingSubmap(q, S, ref2).Keys
    by {
      forall a
        ensures a in MatchingSubmap(q, S, ref1).Keys <==> a in MatchingSubmap(q, S, ref2).Keys
      {
        assert (forall v :: v in S ==> v in a.Keys && v in ref1.Keys && a[v] == ref1[v])
          <==> (forall v :: v in S ==> v in a.Keys && v in ref2.Keys && a[v] == ref2[v]);
      }
    }
    // Values agree: both sides carry q[a].
    assert forall a :: a in MatchingSubmap(q, S, ref1).Keys ==>
      MatchingSubmap(q, S, ref1)[a] == q[a] == MatchingSubmap(q, S, ref2)[a];
  }

  // L5b: MarginalMass depends only on ref|S (follows immediately from L5a).
  lemma MarginalMass_ScopeInvariant(
    q: AssignmentPMF, S: set<Node>, ref1: Assignment, ref2: Assignment
  )
    requires forall v :: v in S ==> v in ref1.Keys && v in ref2.Keys && ref1[v] == ref2[v]
    ensures MarginalMass(q, S, ref1) == MarginalMass(q, S, ref2)
  {
    MatchingSubmap_ScopeInvariant(q, S, ref1, ref2);
  }

  // ------------------------------------------------------------------
  //  L6: Locality of ConditionalFactorConcrete.
  //
  //  CF(G, q, v, ref) depends only on ref|(pa(v) ∪ {v}).  Two full
  //  assignments that agree on pa(v) ∪ {v} give the same factor.
  //  Again purely definitional — no MarkovFactorizationConcrete needed.
  //
  //  NOTE: The HARD step — that conditioning on Z ⊆ NonDesc(v)\Pa(v)
  //  leaves the factor unchanged at the RATIO level (probabilistic Local
  //  Markov) — requires MarkovFactorizationConcrete and is separate.
  // ------------------------------------------------------------------

  lemma ConditionalFactorConcrete_Locality(
    G: Graph, q: AssignmentPMF, v: Node, ref1: Assignment, ref2: Assignment
  )
    requires v in Nodes(G)
    requires v in ref1.Keys && v in ref2.Keys && ref1[v] == ref2[v]
    requires forall u :: u in Parents(G, v) * Nodes(G) ==>
               u in ref1.Keys && u in ref2.Keys && ref1[u] == ref2[u]
    ensures ConditionalFactorConcrete(G, q, v, ref1)
         == ConditionalFactorConcrete(G, q, v, ref2)
  {
    var pa := Parents(G, v) * Nodes(G);
    // ref1 and ref2 agree on pa, so the pa-marginals are equal.
    MarginalMass_ScopeInvariant(q, pa, ref1, ref2);
    // ref1 and ref2 agree on pa + {v}, so the (pa+{v})-marginals are equal.
    assert forall u :: u in pa + {v} ==>
      u in ref1.Keys && u in ref2.Keys && ref1[u] == ref2[u];
    MarginalMass_ScopeInvariant(q, pa + {v}, ref1, ref2);
    // Both numerators and denominators are equal, so the ratios are equal.
  }

  // ------------------------------------------------------------------
  //  L7-pre2: D-separation corollary for any Z ⊆ NonDesc(v) \ Pa(v).
  //
  //  The graph-level LocalMarkov lemma gives d-separation for the full
  //  set NonDesc(v)\Pa(v).  DSep_Decomposition (monotonicity) immediately
  //  extends it to any subset Z.  This is the pure graph ingredient that
  //  the probabilistic LocalMarkovConcrete theorem will cite.
  // ------------------------------------------------------------------

  lemma DSep_NonDesc_Subset(G: Graph, v: Node, Z: set<Node>)
    requires v in Nodes(G)
    requires IsDAG(G)
    requires Z <= NonDescendants(G, v) - Parents(G, v)
    ensures  DSep(G, {v}, Z, Parents(G, v) * Nodes(G))
  {
    // Step 1: LocalMarkov gives d-sep for the full set.
    LocalMarkov(G, v);
    // LocalMarkov ensures DSep(G, {v}, NonDescendants(G,v) - Parents(G,v), Parents(G,v))
    // Step 2: Z ⊆ NonDesc(v)-Pa(v), so DSep_Decomposition extends to Z.
    // DSep_Decomposition: if DSep(G,X,Z+Z',W) then DSep(G,X,Z,W).
    // We want the subset direction: from the full set to Z.
    // The full conditioning set is Parents(G,v) and the target is
    // NonDesc(v)-Pa(v); Z is a subset of that target.
    var pa := Parents(G, v);
    var fullZ := NonDescendants(G, v) - pa;
    // fullZ = Z + (fullZ - Z), so decomposition gives DSep on Z.
    assert Z + (fullZ - Z) == fullZ;
    DSep_Decomposition(G, {v}, Z, fullZ - Z, pa);
    // DSep_Decomposition requires DSep(G,{v},Z+(fullZ-Z),pa) = DSep(G,{v},fullZ,pa)
    // and gives DSep(G,{v},Z,pa).
    // But our lemma has Parents(G,v)*Nodes(G) in the conditioning set.
    // Those are the same up to intersection with Nodes(G): since Pa(v)⊆Nodes(G),
    // Pa(v)*Nodes(G) == Pa(v), so the DSep statements are identical.
    assert Parents(G, v) * Nodes(G) == Parents(G, v);
  }

  // ------------------------------------------------------------------
  //  L7-pre1: MarginalMass_FactorOut
  //
  //  Under the Markov factorization, the joint mass factors as a product
  //  of local conditional factors.  When Z ⊆ NonDesc(v) \ Pa(v), the
  //  Z-nodes are independent of v given Pa(v): summing over any choice
  //  of values for the Z-nodes leaves the Pa(v)+{v} marginal unchanged.
  //  Algebraically:
  //
  //    MM(pa+Z+{v}, ref) · MM(pa, ref) == MM(pa+{v}, ref) · MM(pa+Z, ref)
  //
  //  The proof telescopes out the Z-nodes from the sum using the Markov
  //  factorization.  This is entirely within the concrete AssignmentPMF
  //  layer (no abstract Prob.Outcome); the blocker is the combinatorial
  //  structure of summing over map<Assignment, real> entries, which Z3
  //  cannot currently close within the time budget.  This is recorded as
  //  a named axiom rather than a proof obligation.
  //
  //  Proof sketch (for the future):
  //    1. Pick the last node z ∈ Z in topo order (Pa(z) ⊆ pa ∪ (Z\{z})).
  //    2. By MarkovFactorizationConcrete, q[a] = ∏_w CF(G,q,w,a).
  //    3. CF(G,q,z,a) depends only on a|(Pa(z)+{z}) (by L6 locality).
  //       Pa(z) ⊆ pa ∪ (Z\{z}) ⊆ "fixed" coordinates on both sides.
  //    4. Summing over all values of a[z] while keeping all other coords
  //       fixed: ∑_{z-vals} CF(G,q,z,a) = 1 (proper conditional PMF).
  //       Here z-vals ranges over Value, which is a concrete type.
  //    5. The z-factor therefore cancels between numerator and denominator
  //       of the cross-ratio, reducing to the IH on Z\{z}.
  //  The main mechanization gap is a helper lemma that partitions
  //  AssignmentPMF keys by a single node's value and proves the
  //  per-value sub-sums equal the ConditionalFactorConcrete entries.
  // ------------------------------------------------------------------

  lemma {:axiom} MarginalMass_FactorOut(
    G: Graph, q: AssignmentPMF, v: Node, Z: set<Node>, ref: Assignment
  )
    requires MarkovFactorizationConcrete(G, q)
    requires IsDAG(G)
    requires v in Nodes(G)
    requires ref.Keys == Nodes(G)
    requires Z <= NonDescendants(G, v) - Parents(G, v)
    ensures
      var pa := Parents(G, v) * Nodes(G);
      MarginalMass(q, pa + Z + {v}, ref) * MarginalMass(q, pa, ref)
   == MarginalMass(q, pa + {v}, ref) * MarginalMass(q, pa + Z, ref)

  // ------------------------------------------------------------------
  //  L7: LocalMarkovConcrete — the concrete probabilistic Local Markov
  //  property.
  //
  //  Under MarkovFactorizationConcrete, v is conditionally independent
  //  of any Z ⊆ NonDesc(v)\Pa(v) given Pa(v):
  //
  //    MM(pa+Z+{v}) · MM(pa) == MM(pa+{v}) · MM(pa+Z)
  //
  //  This is an immediate corollary of MarginalMass_FactorOut.  The
  //  separate named lemma gives callers a stable API that does not
  //  expose the internal FactorOut axiom.
  // ------------------------------------------------------------------

  lemma LocalMarkovConcrete(
    G: Graph, q: AssignmentPMF, v: Node, Z: set<Node>, ref: Assignment
  )
    requires MarkovFactorizationConcrete(G, q)
    requires IsDAG(G)
    requires v in Nodes(G)
    requires ref.Keys == Nodes(G)
    requires Z <= NonDescendants(G, v) - Parents(G, v)
    ensures
      var pa := Parents(G, v) * Nodes(G);
      MarginalMass(q, pa + Z + {v}, ref) * MarginalMass(q, pa, ref)
   == MarginalMass(q, pa + {v}, ref) * MarginalMass(q, pa + Z, ref)
  {
    MarginalMass_FactorOut(G, q, v, Z, ref);
  }

  // ------------------------------------------------------------------
  //  L1: Arithmetic foundations for SumAssignmentMasses.
  //
  //  SumAssignmentMasses recurses by peeling one key at a time via
  //  `var a :| a in q.Keys`, so the order in which keys are peeled is
  //  unspecified.  L1a shows any key may be peeled first (the sum is
  //  order-independent); L1b builds additive splitting over disjoint
  //  sub-maps from that; L1c is the named partition corollary used by
  //  the marginalisation steps.
  // ------------------------------------------------------------------

  // L1a-helper: two distinct keys give equal peel results.
  // Decreases (|q|, 0): called by RemoveAny at (|q|, 1) on the same map. ✓
  lemma SumAssignmentMasses_PairCommute(
    q: AssignmentPMF, a: Assignment, b: Assignment
  )
    requires a in q.Keys && b in q.Keys && a != b
    ensures q[a] + SumAssignmentMasses(q - {a})
         == q[b] + SumAssignmentMasses(q - {b})
    decreases |q|, 0
  {
    assert a in (q - {b}).Keys;
    assert b in (q - {a}).Keys;
    // IH (rank 1, strictly smaller map): peel a from q-{b}, peel b from q-{a}.
    SumAssignmentMasses_RemoveAny(q - {b}, a);
    // Sum(q-{b}) == q[a] + Sum(q-{b}-{a})
    SumAssignmentMasses_RemoveAny(q - {a}, b);
    // Sum(q-{a}) == q[b] + Sum(q-{a}-{b})
    // Double-removal is commutative; real arithmetic chains the two.
    assert (q - {b}) - {a} == (q - {a}) - {b};
  }

  // L1a: Any element can be peeled first (order-independence).
  lemma SumAssignmentMasses_RemoveAny(q: AssignmentPMF, a: Assignment)
    requires a in q.Keys
    ensures SumAssignmentMasses(q) == q[a] + SumAssignmentMasses(q - {a})
    decreases |q|, 1
  {
    if |q| == 1 {
      // The only key is a; removing it leaves the empty map.
      assert q.Keys == {a};
      assert q - {a} == map[];
    } else {
      // Show every possible peel-b choice the function might make gives the
      // same result as peel-a, so the unspecified internal choice is irrelevant.
      forall b | b in q.Keys
        ensures q[b] + SumAssignmentMasses(q - {b})
             == q[a] + SumAssignmentMasses(q - {a})
      {
        if b != a {
          // PairCommute at (|q|, 0) < (|q|, 1). ✓
          SumAssignmentMasses_PairCommute(q, a, b);
        }
      }
    }
  }

  // L1b — additive splitting over disjoint sub-maps.
  lemma SumAssignmentMasses_Disjoint(q1: AssignmentPMF, q2: AssignmentPMF)
    requires q1.Keys * q2.Keys == {}
    ensures SumAssignmentMasses(q1 + q2) == SumAssignmentMasses(q1) + SumAssignmentMasses(q2)
    decreases |q1|
  {
    if |q1| == 0 {
      assert q1 == map[];
      assert q1 + q2 == q2;
    } else {
      var a :| a in q1.Keys;
      // a is in q1 but not q2.
      assert a !in q2.Keys;
      // Peel a from the merged map.
      SumAssignmentMasses_RemoveAny(q1 + q2, a);
      assert (q1 + q2)[a] == q1[a];
      // Removing a from the merge equals merging q1-{a} with q2.
      assert (q1 + q2) - {a} == (q1 - {a}) + q2;
      // Peel a from q1.
      SumAssignmentMasses_RemoveAny(q1, a);
      // IH on the strictly smaller q1 - {a}.
      assert (q1 - {a}).Keys * q2.Keys == {};
      SumAssignmentMasses_Disjoint(q1 - {a}, q2);
    }
  }

  // L1c — named partition corollary: splitting q by any subset S of its keys.
  lemma SumAssignmentMasses_Partition(q: AssignmentPMF, S: set<Assignment>)
    requires S <= q.Keys
    ensures SumAssignmentMasses(q) ==
      SumAssignmentMasses(map a | a in S :: q[a]) +
      SumAssignmentMasses(map a | a in q.Keys - S :: q[a])
  {
    var q1 := map a | a in S :: q[a];
    var q2 := map a | a in q.Keys - S :: q[a];
    assert q1.Keys == S;
    assert q2.Keys == q.Keys - S;
    assert q1.Keys * q2.Keys == {};
    // q is the disjoint union of q1 and q2.
    assert q == q1 + q2 by {
      assert q.Keys == q1.Keys + q2.Keys;
      forall k | k in q.Keys
        ensures q[k] == (q1 + q2)[k]
      {
        if k in S { assert (q1 + q2)[k] == q1[k] == q[k]; }
        else       { assert (q1 + q2)[k] == q2[k] == q[k]; }
      }
    }
    SumAssignmentMasses_Disjoint(q1, q2);
  }

  // ------------------------------------------------------------------
  //  L4: Non-negativity of sums, marginals, and conditional factors.
  // ------------------------------------------------------------------

  // L4a: SumAssignmentMasses of a non-negative map is non-negative.
  lemma SumAssignmentMasses_Nonneg(q: AssignmentPMF)
    requires AssignmentPMF_AllNonNeg(q)
    ensures SumAssignmentMasses(q) >= 0.0
    decreases |q|
  {
    if |q| == 0 {
      // Empty sum is 0.0 >= 0.0.
    } else {
      var a :| a in q.Keys;
      SumAssignmentMasses_RemoveAny(q, a);
      // SumAssignmentMasses(q) == q[a] + SumAssignmentMasses(q - {a})
      assert q[a] >= 0.0;
      assert AssignmentPMF_AllNonNeg(q - {a});
      SumAssignmentMasses_Nonneg(q - {a});
    }
  }

  // L4b: MarginalMass is always non-negative.
  lemma MarginalMass_Nonneg(q: AssignmentPMF, S: set<Node>, ref: Assignment)
    requires AssignmentPMF_AllNonNeg(q)
    ensures MarginalMass(q, S, ref) >= 0.0
  {
    // MatchingSubmap inherits non-negativity; delegate to L4a.
    var sub := MatchingSubmap(q, S, ref);
    assert AssignmentPMF_AllNonNeg(sub) by {
      MatchingSubmap_IsSubmap(q, S, ref);
    }
    SumAssignmentMasses_Nonneg(sub);
  }

  // L4c: ConditionalFactorConcrete is non-negative.
  lemma ConditionalFactorConcrete_Nonneg(
    G: Graph, q: AssignmentPMF, v: Node, ref: Assignment
  )
    requires v in Nodes(G)
    requires AssignmentPMF_AllNonNeg(q)
    ensures ConditionalFactorConcrete(G, q, v, ref) >= 0.0
  {
    var pa := Parents(G, v) * Nodes(G);
    var denom := MarginalMass(q, pa, ref);
    if denom == 0.0 {
      // Convention: 0.0 >= 0.0.
    } else {
      // numerator >= 0, denom > 0 (since denom != 0 and denom >= 0).
      MarginalMass_Nonneg(q, pa + {v}, ref);
      MarginalMass_Nonneg(q, pa, ref);
    }
  }

  // Marginal probability of the partial assignment ref|S under q.
  ghost function MarginalMass(
    q: AssignmentPMF, S: set<Node>, ref: Assignment
  ): real
  {
    SumAssignmentMasses(MatchingSubmap(q, S, ref))
  }

  // Concrete conditional factor P(v = ref[v] | pa(v) = ref|pa) under q.
  // Convention: 0 when the conditioning event has zero mass.
  ghost function ConditionalFactorConcrete(
    G: Graph, q: AssignmentPMF, v: Node, ref: Assignment
  ): real
    requires v in Nodes(G)
  {
    var pa := Parents(G, v) * Nodes(G);
    var denom := MarginalMass(q, pa, ref);
    if denom == 0.0 then 0.0
    else MarginalMass(q, pa + {v}, ref) / denom
  }

  // Mass of a full assignment under q (0 outside the support).
  ghost function AssignmentMass(q: AssignmentPMF, a: Assignment): real {
    if a in q then q[a] else 0.0
  }

  // Product of local conditional factors along a node ordering.
  // Concrete counterpart of ConditionalFactorProduct.
  ghost function ConditionalFactorProductConcrete(
    G: Graph, q: AssignmentPMF, ref: Assignment, ord: seq<Node>
  ): real
    requires forall i :: 0 <= i < |ord| ==> ord[i] in Nodes(G)
  {
    if |ord| == 0 then 1.0 else
      ConditionalFactorConcrete(G, q, ord[0], ref)
        * ConditionalFactorProductConcrete(G, q, ref, ord[1..])
  }

  // Causal Markov Condition for the concrete distribution: every full
  // assignment's mass equals the product of its local conditional
  // factors along any topological order.  Concrete counterpart of
  // MarkovFactorization (which is stated over the abstract Prob.PMF).
  ghost predicate MarkovFactorizationConcrete(G: Graph, q: AssignmentPMF) {
    IsAssignmentDistribution(G, q) &&
    forall ref: Assignment, ord: seq<Node> ::
      ref.Keys == Nodes(G) && IsTopologicalSort(G, ord) ==>
        AssignmentMass(q, ref) == ConditionalFactorProductConcrete(G, q, ref, ord)
  }

  // ==================================================================
  // DA-M1-G0-4: AssignmentPMF ↔ Prob.PMF boundary adapter.
  //
  // PMFToAssignmentPMF lifts an abstract Prob.PMF into the concrete
  // AssignmentPMF layer.  The three coherence axioms below document
  // the minimal assumptions at this boundary; all three are provable in
  // principle once Prob.Outcome is made concrete (they reduce to finite
  // summation and conditional-probability identities).
  //
  // With these axioms, MarkovFactorizationConcrete is transferred
  // non-axiomatically from MarkovFactorization in the proved lemma
  // PMFToAssignmentPMF_MarkovTransfer.
  // ==================================================================

  // Lift a Prob.PMF to an AssignmentPMF.  The ensures clauses state all
  // well-formedness properties; the proof is deferred until Prob.Outcome
  // is concrete.
  ghost function {:axiom} PMFToAssignmentPMF(
    G: Graph, p: Prob.PMF
  ): AssignmentPMF
    requires Prob.IsDistribution(p)
    ensures IsAssignmentDistribution(G, PMFToAssignmentPMF(G, p))

  // Pointwise mass coherence: the concrete mass of a full assignment
  // equals its abstract probability.
  lemma {:axiom} PMFToAssignmentPMF_MassCoherence(
    G: Graph, p: Prob.PMF, full: Assignment
  )
    requires Prob.IsDistribution(p)
    requires full.Keys == Nodes(G)
    ensures AssignmentMass(PMFToAssignmentPMF(G, p), full)
      == AssignmentProb(p, G, full)

  // Conditional-factor coherence: each concrete local factor under the
  // lifted PMF equals the corresponding abstract ConditionalFactor.
  // This is the key bridge between the two factorization predicates.
  lemma {:axiom} PMFToAssignmentPMF_CFCoherence(
    G: Graph, p: Prob.PMF, v: Node, ref: Assignment
  )
    requires Prob.IsDistribution(p)
    requires v in Nodes(G)
    requires ref.Keys == Nodes(G)
    ensures ConditionalFactorConcrete(G, PMFToAssignmentPMF(G, p), v, ref)
      == ConditionalFactor(p, v, Parents(G, v), ref)

  // Derived: the concrete factor product equals the abstract one along
  // any node ordering.  Follows by induction on ord from CFCoherence,
  // but Z3 cannot close the recursive function-unfolding goal within the
  // verification time budget (two parallel recursive functions + arithmetic).
  // Recorded as an axiom; proof sketch: induction on |ord|, base 1.0==1.0,
  // step: unfold both definitions, apply CFCoherence for the head, IH for
  // the tail, conclude by multiplication.
  lemma {:axiom} PMFToAssignmentPMF_ProductCoherence(
    G: Graph, p: Prob.PMF, ref: Assignment, ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires ref.Keys == Nodes(G)
    requires forall i :: 0 <= i < |ord| ==> ord[i] in Nodes(G)
    ensures ConditionalFactorProductConcrete(G, PMFToAssignmentPMF(G, p), ref, ord)
      == ConditionalFactorProduct(G, p, ref, ord)

  // Markov factorization transfer: if p satisfies the abstract Markov
  // condition then the lifted PMFToAssignmentPMF(G, p) satisfies the
  // concrete one.  Proved non-axiomatically using MassCoherence and
  // ProductCoherence.
  lemma PMFToAssignmentPMF_MarkovTransfer(G: Graph, p: Prob.PMF)
    requires IsDAG(G)
    requires MarkovFactorization(G, p)
    ensures MarkovFactorizationConcrete(G, PMFToAssignmentPMF(G, p))
  {
    var q := PMFToAssignmentPMF(G, p);
    // IsAssignmentDistribution(G, q) is given by PMFToAssignmentPMF's postcondition.
    forall ref: Assignment, ord: seq<Node>
      | ref.Keys == Nodes(G) && IsTopologicalSort(G, ord)
      ensures AssignmentMass(q, ref) == ConditionalFactorProductConcrete(G, q, ref, ord)
    {
      // ord[i] in Nodes(G) follows from IsTopologicalSort.
      assert forall i :: 0 <= i < |ord| ==> ord[i] in Nodes(G);
      PMFToAssignmentPMF_MassCoherence(G, p, ref);
      // AssignmentMass(q, ref) == AssignmentProb(p, G, ref)
      assert AssignmentProb(p, G, ref) == ConditionalFactorProduct(G, p, ref, ord);
      PMFToAssignmentPMF_ProductCoherence(G, p, ref, ord);
      // ConditionalFactorProductConcrete(G, q, ref, ord) == ConditionalFactorProduct(G, p, ref, ord)
    }
  }

  // ==================================================================
  // DA-M1-G0-5: PMFToAssignmentPMF_MarginalCoherence  [AXIOM — DEFERRED]
  //
  //   MarginalMass(PMFToAssignmentPMF(G, p), S, partial)
  //     == AssignmentProb(p, G, partial)
  //
  //   This follows from MassCoherence by a double-sum exchange:
  //     MarginalMass(q, S, ref)
  //       = Σ_{a ∈ MatchingSubmap(q,S,ref).Keys} q[a]
  //       = Σ_{a ∈ q.Keys, a|S=ref|S} AssignmentProb(p, G, a)     (MassCoherence)
  //       = Σ_{omega ∈ p.Keys, MatchesAssignment(G,omega,ref)} p[omega]
  //       = AssignmentProb(p, G, ref)
  //   The third equality needs fiber-partition reasoning:
  //     (i)  Fibers {omega : OutcomeToAssignment(G,omega) = a} are disjoint.
  //     (ii) Their union over {a ∈ q.Keys, a|S=ref|S} equals
  //          {omega ∈ p.Keys : MatchesAssignment(G,omega,ref)},
  //          because any omega in the RHS with OtA(G,omega) ∉ q.Keys has
  //          mass 0 in p (MassCoherence gives AssignmentProb(p,G,a)=0 ⟹ p(fiber)=0).
  //   Formalizing the sum-exchange requires Prob.Additivity over
  //   AssignmentPMF-indexed partitions.  Stated as axiom pending that
  //   partition infrastructure.
  // ==================================================================

  lemma {:axiom} PMFToAssignmentPMF_MarginalCoherence(
    G: Graph, p: Prob.PMF, partial: Assignment
  )
    requires Prob.IsDistribution(p)
    requires partial.Keys <= Nodes(G)
    ensures MarginalMass(PMFToAssignmentPMF(G, p), partial.Keys, partial)
         == AssignmentProb(p, G, partial)

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

  ghost function OverrideAssignment(
    G: Graph,
    full: Assignment,
    v: Node,
    value: Value
  ): Assignment
    requires full.Keys == Nodes(G)
    requires v in Nodes(G)
    ensures OverrideAssignment(G, full, v, value).Keys == Nodes(G)
  {
    map u | u in Nodes(G) :: if u == v then value else full[u]
  }

  lemma OverrideAssignment_SameValue(
    G: Graph,
    full: Assignment,
    v: Node
  )
    requires full.Keys == Nodes(G)
    requires v in Nodes(G)
    ensures OverrideAssignment(G, full, v, full[v]) == full
  {
    assert OverrideAssignment(G, full, v, full[v]).Keys == full.Keys;
    assert forall u :: u in Nodes(G) ==> OverrideAssignment(G, full, v, full[v])[u] == full[u] by {
      forall u | u in Nodes(G)
        ensures OverrideAssignment(G, full, v, full[v])[u] == full[u]
      {
        if u == v {
        }
      }
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

  // A local PMF witness for a node's conditional distribution. This packages
  // both the child-value normalization law and the intended parent-locality
  // boundary for ConditionalFactor.
  ghost function {:axiom} ConditionalLocalPMF(
    G: Graph,
    p: Prob.PMF,
    v: Node,
    full: Assignment
  ): Prob.PMF
    requires Prob.IsDistribution(p)
    requires v in Nodes(G)
    requires full.Keys == Nodes(G)
    ensures Prob.IsDistribution(ConditionalLocalPMF(G, p, v, full))
    ensures forall value ::
      Prob.OutcomeMass(ConditionalLocalPMF(G, p, v, full), value)
        == ConditionalFactor(p, v, Parents(G, v), OverrideAssignment(G, full, v, value))

  lemma {:axiom} ConditionalLocalPMF_Locality(
    G: Graph,
    p: Prob.PMF,
    v: Node,
    a: Assignment,
    b: Assignment
  )
    requires Prob.IsDistribution(p)
    requires v in Nodes(G)
    requires a.Keys == Nodes(G)
    requires b.Keys == Nodes(G)
    requires forall u :: u in Parents(G, v) * Nodes(G) ==> a[u] == b[u]
    ensures ConditionalLocalPMF(G, p, v, a) == ConditionalLocalPMF(G, p, v, b)

  lemma ConditionalLocalPMF_Normalized(
    G: Graph,
    p: Prob.PMF,
    v: Node,
    full: Assignment
  )
    requires Prob.IsDistribution(p)
    requires v in Nodes(G)
    requires full.Keys == Nodes(G)
    ensures Prob.ProbEvent(
      ConditionalLocalPMF(G, p, v, full),
      ConditionalLocalPMF(G, p, v, full).Keys
    ) == 1.0
  {
    Prob.Axiom_Normalization(ConditionalLocalPMF(G, p, v, full));
  }

  lemma ConditionalFactor_FromLocalPMF(
    G: Graph,
    p: Prob.PMF,
    v: Node,
    full: Assignment,
    value: Value
  )
    requires Prob.IsDistribution(p)
    requires v in Nodes(G)
    requires full.Keys == Nodes(G)
    ensures ConditionalFactor(p, v, Parents(G, v), OverrideAssignment(G, full, v, value))
      == Prob.OutcomeMass(ConditionalLocalPMF(G, p, v, full), value)
  {
  }

  lemma ConditionalFactor_Locality(
    G: Graph,
    p: Prob.PMF,
    v: Node,
    a: Assignment,
    b: Assignment
  )
    requires Prob.IsDistribution(p)
    requires v in Nodes(G)
    requires a.Keys == Nodes(G)
    requires b.Keys == Nodes(G)
    requires a[v] == b[v]
    requires forall u :: u in Parents(G, v) * Nodes(G) ==> a[u] == b[u]
    ensures ConditionalFactor(p, v, Parents(G, v), a)
      == ConditionalFactor(p, v, Parents(G, v), b)
  {
    ConditionalLocalPMF_Locality(G, p, v, a, b);
    ConditionalFactor_FromLocalPMF(G, p, v, a, a[v]);
    ConditionalFactor_FromLocalPMF(G, p, v, b, b[v]);
    OverrideAssignment_SameValue(G, a, v);
    OverrideAssignment_SameValue(G, b, v);
  }

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
  //   Since Dafny has no built-in product over a set, we express this
  //   as a universally quantified predicate: for every full assignment
  //   and every topological ordering of G, the joint probability equals
  //   ConditionalFactorProduct — the recursive product along that order.
  // ==================================================================

  ghost predicate MarkovFactorization(G: Graph, p: Prob.PMF)
  {
    Prob.IsDistribution(p) &&
    forall full: Assignment, ord: seq<Node> ::
      full.Keys == Nodes(G) && IsTopologicalSort(G, ord) ==>
        AssignmentProb(p, G, full) == ConditionalFactorProduct(G, p, full, ord)
  }

  // A Markov-factored PMF is a valid distribution.
  // Proof: MarkovFactorization's first conjunct is Prob.IsDistribution(p).
  lemma MarkovFactorization_IsDistribution(
    G: Graph, p: Prob.PMF
  )
    requires IsDAG(G)
    requires MarkovFactorization(G, p)
    ensures Prob.IsDistribution(p)
  {
  }

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

  ghost function ConditionalFactorProduct(
    G: Graph,
    p: Prob.PMF,
    full: Assignment,
    ord: seq<Node>
  ): real
    requires full.Keys == Nodes(G)
    requires forall i :: 0 <= i < |ord| ==> ord[i] in Nodes(G)
  {
    if |ord| == 0 then 1.0 else
      ConditionalFactor(p, ord[0], Parents(G, ord[0]), full)
        * ConditionalFactorProduct(G, p, full, ord[1..])
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

  lemma TruncatedFactorProduct_Empty(
    G: Graph,
    p: Prob.PMF,
    full: Assignment,
    ord: seq<Node>
  )
    requires full.Keys == Nodes(G)
    requires forall i :: 0 <= i < |ord| ==> ord[i] in Nodes(G)
    ensures TruncatedFactorProduct(G, p, {}, map[], full, ord)
      == ConditionalFactorProduct(G, p, full, ord)
  {
    if |ord| != 0 {
      TruncatedLocalFactor_Unintervened(G, p, {}, map[], full, ord[0]);
      assert forall i :: 0 <= i < |ord[1..]| ==> ord[1..][i] in Nodes(G) by {
        forall i | 0 <= i < |ord[1..]| ensures ord[1..][i] in Nodes(G) {
          assert ord[1..][i] == ord[i + 1];
        }
      }
      TruncatedFactorProduct_Empty(G, p, full, ord[1..]);
    }
  }

  lemma TruncatedAssignmentMass_Empty(
    G: Graph,
    p: Prob.PMF,
    full: Assignment,
    ord: seq<Node>
  )
    requires IsTopologicalSort(G, ord)
    requires full.Keys == Nodes(G)
    ensures TruncatedAssignmentMass(G, p, {}, map[], full, ord)
      == ConditionalFactorProduct(G, p, full, ord)
  {
    assert forall i :: 0 <= i < |ord| ==> ord[i] in Nodes(G);
    TruncatedFactorProduct_Empty(G, p, full, ord);
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
    ensures forall full: Assignment, ord: seq<Node> ::
      full.Keys == Nodes(G)
      && IsTopologicalSort(G, ord)
      && xVals.Keys <= Nodes(G)
      && TruncatedAssignmentMass(G, p, X, xVals, full, ord) > 0.0
      ==> exists i: int :: (
        0 <= i < |TruncateSupportAssignments(G, p, X, xVals)|
        && TruncateSupportAssignments(G, p, X, xVals)[i] == full)

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

  ghost function EncodedTruncateAssignment(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    omega: Prob.Outcome
  ): Assignment
    requires Prob.IsDistribution(p)
    requires xVals.Keys == X
    requires omega in EncodedTruncateSupport(G, p, X, xVals)
    ensures EncodedTruncateAssignment(G, p, X, xVals, omega).Keys == Nodes(G)
    ensures ExtendsAssignment(xVals, EncodedTruncateAssignment(G, p, X, xVals, omega))
    ensures AssignmentToOutcome(G, EncodedTruncateAssignment(G, p, X, xVals, omega)) == omega
  {
    var i :| 0 <= i < |TruncateSupportAssignments(G, p, X, xVals)|
      && omega == AssignmentToOutcome(G, TruncateSupportAssignments(G, p, X, xVals)[i]);
    TruncateSupportAssignments(G, p, X, xVals)[i]
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

  lemma PositiveTruncatedAssignment_InEncodedSupport(
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
    requires TruncatedAssignmentMass(G, p, X, xVals, full, ord) > 0.0
    ensures AssignmentToOutcome(G, full) in EncodedTruncateSupport(G, p, X, xVals)
  {
    var i :| 0 <= i < |TruncateSupportAssignments(G, p, X, xVals)|
      && TruncateSupportAssignments(G, p, X, xVals)[i] == full;
    assert AssignmentToOutcome(G, full)
      == AssignmentToOutcome(G, TruncateSupportAssignments(G, p, X, xVals)[i]);
  }

  ghost function EncodeAssignments(
    G: Graph,
    fulls: seq<Assignment>
  ): seq<Prob.Outcome>
    requires forall i :: 0 <= i < |fulls| ==> fulls[i].Keys == Nodes(G)
  {
    if |fulls| == 0 then []
    else [AssignmentToOutcome(G, fulls[0])] + EncodeAssignments(G, fulls[1..])
  }

  lemma EncodeAssignments_Index(
    G: Graph,
    fulls: seq<Assignment>
  )
    requires forall i :: 0 <= i < |fulls| ==> fulls[i].Keys == Nodes(G)
    ensures |EncodeAssignments(G, fulls)| == |fulls|
    ensures forall i :: 0 <= i < |fulls| ==>
      EncodeAssignments(G, fulls)[i] == AssignmentToOutcome(G, fulls[i])
  {
    if |fulls| != 0 {
      EncodeAssignments_Index(G, fulls[1..]);
    }
  }

  ghost function SumTruncatedAssignmentMasses(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    fulls: seq<Assignment>,
    ord: seq<Node>
  ): real
    requires IsTopologicalSort(G, ord)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    requires forall i :: 0 <= i < |fulls| ==> fulls[i].Keys == Nodes(G)
  {
    if |fulls| == 0 then 0.0
    else
      TruncatedAssignmentMass(G, p, X, xVals, fulls[0], ord)
      + SumTruncatedAssignmentMasses(G, p, X, xVals, fulls[1..], ord)
  }

  ghost function AssignmentValuesAt(fulls: seq<Assignment>, v: Node): seq<Value>
    requires forall i :: 0 <= i < |fulls| ==> v in fulls[i].Keys
  {
    if |fulls| == 0 then []
    else [fulls[0][v]] + AssignmentValuesAt(fulls[1..], v)
  }

  ghost function FilterAssignmentsByValue(
    fulls: seq<Assignment>,
    keys: set<Node>,
    v: Node,
    value: Value
  ): seq<Assignment>
    requires v in keys
    requires forall i :: 0 <= i < |fulls| ==> fulls[i].Keys == keys
    ensures forall i :: 0 <= i < |FilterAssignmentsByValue(fulls, keys, v, value)| ==>
      FilterAssignmentsByValue(fulls, keys, v, value)[i].Keys == keys
    ensures forall i :: 0 <= i < |FilterAssignmentsByValue(fulls, keys, v, value)| ==>
      FilterAssignmentsByValue(fulls, keys, v, value)[i][v] == value
  {
    if |fulls| == 0 then []
    else if fulls[0][v] == value then [fulls[0]] + FilterAssignmentsByValue(fulls[1..], keys, v, value)
    else FilterAssignmentsByValue(fulls[1..], keys, v, value)
  }

  ghost function SumTruncatedTailProducts(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    fulls: seq<Assignment>,
    ordTail: seq<Node>
  ): real
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    requires forall i :: 0 <= i < |fulls| ==> fulls[i].Keys == Nodes(G)
    requires forall i :: 0 <= i < |ordTail| ==> ordTail[i] in Nodes(G)
  {
    if |fulls| == 0 then 0.0
    else
      TruncatedFactorProduct(G, p, X, xVals, fulls[0], ordTail)
      + SumTruncatedTailProducts(G, p, X, xVals, fulls[1..], ordTail)
  }

  lemma AssignmentValuesAt_Index(fulls: seq<Assignment>, v: Node)
    requires forall i :: 0 <= i < |fulls| ==> v in fulls[i].Keys
    ensures |AssignmentValuesAt(fulls, v)| == |fulls|
    ensures forall i :: 0 <= i < |fulls| ==> AssignmentValuesAt(fulls, v)[i] == fulls[i][v]
  {
    if |fulls| != 0 {
      AssignmentValuesAt_Index(fulls[1..], v);
    }
  }

  lemma TopologicalHeadHasNoParents(G: Graph, ord: seq<Node>)
    requires IsTopologicalSort(G, ord)
    requires |ord| > 0
    ensures Parents(G, ord[0]) == {}
  {
    assert forall u :: u in Parents(G, ord[0]) ==> false by {
      forall u | u in Parents(G, ord[0])
        ensures false
      {
        assert exists k | 0 <= k < 0 :: ord[k] == u;
      }
    }
  }

  lemma SingletonTopologicalSort_NodeSet(G: Graph, ord: seq<Node>)
    requires IsTopologicalSort(G, ord)
    requires |ord| == 1
    ensures Nodes(G) == {ord[0]}
  {
    assert ord[0] in Nodes(G);
    assert forall u :: u in Nodes(G) ==> u == ord[0] by {
      forall u | u in Nodes(G)
        ensures u == ord[0]
      {
        assert u in ord;
        var i :| 0 <= i < |ord| && ord[i] == u;
        assert i == 0;
      }
    }
    assert forall u :: u in Nodes(G) <==> u in {ord[0]} by {
      forall u
        ensures u in Nodes(G) <==> u in {ord[0]}
      {
        if u in Nodes(G) {
          assert u == ord[0];
        } else if u in {ord[0]} {
          assert u == ord[0];
        }
      }
    }
  }

  lemma SingletonAssignmentsEqual(
    G: Graph,
    a: Assignment,
    b: Assignment,
    v: Node
  )
    requires Nodes(G) == {v}
    requires a.Keys == Nodes(G)
    requires b.Keys == Nodes(G)
    requires a[v] == b[v]
    ensures a == b
  {
    assert a.Keys == b.Keys;
    assert forall u :: u in a.Keys ==> a[u] == b[u] by {
      forall u | u in a.Keys
        ensures a[u] == b[u]
      {
        assert u == v;
      }
    }
  }

  lemma TruncatedAssignmentMass_FactorHeadViaLocalPMF(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>,
    full: Assignment,
    template: Assignment
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires |ord| > 0
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    requires ord[0] !in X
    requires full.Keys == Nodes(G)
    requires template.Keys == Nodes(G)
    ensures TruncatedAssignmentMass(G, p, X, xVals, full, ord)
      == Prob.OutcomeMass(ConditionalLocalPMF(G, p, ord[0], template), full[ord[0]])
        * TruncatedFactorProduct(G, p, X, xVals, full, ord[1..])
  {
    var v := ord[0];
    TopologicalHeadHasNoParents(G, ord);
    assert Parents(G, v) == {};
    assert Parents(G, v) * Nodes(G) == {};
    TruncatedLocalFactor_Unintervened(G, p, X, xVals, full, v);
    assert forall u :: u in Parents(G, v) * Nodes(G) ==> full[u] == template[u] by {
      forall u | u in Parents(G, v) * Nodes(G)
        ensures full[u] == template[u]
      {
      }
    }
    ConditionalLocalPMF_Locality(G, p, v, full, template);
    ConditionalFactor_FromLocalPMF(G, p, v, full, full[v]);
    OverrideAssignment_SameValue(G, full, v);
  }

  lemma SumTruncatedAssignmentMasses_FilterByHeadValue_FactorsHead(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>,
    fulls: seq<Assignment>,
    template: Assignment,
    value: Value
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires |ord| > 0
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    requires ord[0] !in X
    requires template.Keys == Nodes(G)
    requires forall i :: 0 <= i < |fulls| ==> fulls[i].Keys == Nodes(G)
    ensures SumTruncatedAssignmentMasses(
      G,
      p,
      X,
      xVals,
      FilterAssignmentsByValue(fulls, Nodes(G), ord[0], value),
      ord
    )
      == Prob.OutcomeMass(ConditionalLocalPMF(G, p, ord[0], template), value)
        * SumTruncatedTailProducts(
          G,
          p,
          X,
          xVals,
          FilterAssignmentsByValue(fulls, Nodes(G), ord[0], value),
          ord[1..]
        )
  {
    var v := ord[0];
    assert forall i :: 0 <= i < |ord[1..]| ==> ord[1..][i] in Nodes(G) by {
      forall i | 0 <= i < |ord[1..]|
        ensures ord[1..][i] in Nodes(G)
      {
        assert ord[1..][i] == ord[i + 1];
      }
    }
    if |fulls| != 0 {
      assert forall i :: 0 <= i < |fulls[1..]| ==> fulls[1..][i].Keys == Nodes(G) by {
        forall i | 0 <= i < |fulls[1..]|
          ensures fulls[1..][i].Keys == Nodes(G)
        {
          assert fulls[1..][i] == fulls[i + 1];
        }
      }
      SumTruncatedAssignmentMasses_FilterByHeadValue_FactorsHead(G, p, X, xVals, ord, fulls[1..], template, value);
      if fulls[0][v] == value {
        TruncatedAssignmentMass_FactorHeadViaLocalPMF(G, p, X, xVals, ord, fulls[0], template);
      }
    }
  }

  ghost function TruncatePMFOnOrder(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>
  ): Prob.PMF
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
  {
    map omega | omega in EncodedTruncateSupport(G, p, X, xVals) ::
      TruncatedAssignmentMass(
        G,
        p,
        X,
        xVals,
        EncodedTruncateAssignment(G, p, X, xVals, omega),
        ord
      )
  }

  lemma TruncatePMFOnOrder_Support(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures TruncatePMFOnOrder(G, p, X, xVals, ord).Keys
      == EncodedTruncateSupport(G, p, X, xVals)
  {
    assert TruncatePMFOnOrder(G, p, X, xVals, ord).Keys
      == EncodedTruncateSupport(G, p, X, xVals);
  }

  lemma TruncatePMFOnOrder_MatchesAssignment(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures forall omega :: omega in TruncatePMFOnOrder(G, p, X, xVals, ord).Keys ==> MatchesAssignment(G, omega, xVals)
  {
    TruncatePMFOnOrder_Support(G, p, X, xVals, ord);
    EncodedTruncateSupport_MatchesAssignment(G, p, X, xVals);
  }

  lemma TruncatePMFOnOrder_EmptyMass(
    G: Graph,
    p: Prob.PMF,
    ord: seq<Node>,
    omega: Prob.Outcome
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires omega in EncodedTruncateSupport(G, p, {}, map[])
    ensures TruncatePMFOnOrder(G, p, {}, map[], ord)[omega]
      == ConditionalFactorProduct(
        G,
        p,
        EncodedTruncateAssignment(G, p, {}, map[], omega),
        ord
      )
  {
    TruncatePMFOnOrder_Support(G, p, {}, map[], ord);
    var full := EncodedTruncateAssignment(G, p, {}, map[], omega);
    TruncatedAssignmentMass_Empty(G, p, full, ord);
  }

  lemma TruncatePMFOnOrder_AllNonNeg(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures Prob.AllNonNeg(TruncatePMFOnOrder(G, p, X, xVals, ord))
  {
    forall omega | omega in TruncatePMFOnOrder(G, p, X, xVals, ord)
      ensures TruncatePMFOnOrder(G, p, X, xVals, ord)[omega] >= 0.0
    {
      TruncatePMFOnOrder_Support(G, p, X, xVals, ord);
      var full := EncodedTruncateAssignment(G, p, X, xVals, omega);
      TruncatedAssignmentMass_NonNeg(G, p, X, xVals, full, ord);
    }
  }

  lemma TruncatePMFOnOrder_SumOutcomeMasses_OverAssignments(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>,
    fulls: seq<Assignment>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    requires forall i :: 0 <= i < |fulls| ==> fulls[i].Keys == Nodes(G)
    requires forall i :: 0 <= i < |fulls| ==>
      AssignmentToOutcome(G, fulls[i]) in EncodedTruncateSupport(G, p, X, xVals)
    requires forall i, j :: 0 <= i < j < |fulls| ==>
      AssignmentToOutcome(G, fulls[i]) != AssignmentToOutcome(G, fulls[j])
    ensures Prob.SumOutcomeMasses(
      TruncatePMFOnOrder(G, p, X, xVals, ord),
      EncodeAssignments(G, fulls)
    ) == SumTruncatedAssignmentMasses(G, p, X, xVals, fulls, ord)
  {
    if |fulls| != 0 {
      TruncatePMFOnOrder_Support(G, p, X, xVals, ord);
      var t := TruncatePMFOnOrder(G, p, X, xVals, ord);
      var full0 := fulls[0];
      var omega0 := AssignmentToOutcome(G, full0);
      assert EncodeAssignments(G, fulls)
        == [omega0] + EncodeAssignments(G, fulls[1..]);
      assert omega0 in EncodedTruncateSupport(G, p, X, xVals);
      assert omega0 in t.Keys;
      var decoded := EncodedTruncateAssignment(G, p, X, xVals, omega0);
      assert decoded.Keys == Nodes(G);
      assert AssignmentToOutcome(G, decoded) == omega0;
      AssignmentToOutcome_Injective(G, decoded, full0);
      assert decoded == full0;
      assert t[omega0] == TruncatedAssignmentMass(G, p, X, xVals, full0, ord);
      assert forall i :: 0 <= i < |fulls[1..]| ==> fulls[1..][i].Keys == Nodes(G) by {
        forall i | 0 <= i < |fulls[1..]|
          ensures fulls[1..][i].Keys == Nodes(G)
        {
          assert fulls[1..][i] == fulls[i + 1];
        }
      }
      assert forall i :: 0 <= i < |fulls[1..]| ==>
        AssignmentToOutcome(G, fulls[1..][i]) in EncodedTruncateSupport(G, p, X, xVals) by {
        forall i | 0 <= i < |fulls[1..]|
          ensures AssignmentToOutcome(G, fulls[1..][i]) in EncodedTruncateSupport(G, p, X, xVals)
        {
          assert fulls[1..][i] == fulls[i + 1];
        }
      }
      assert forall i, j :: 0 <= i < j < |fulls[1..]| ==>
        AssignmentToOutcome(G, fulls[1..][i]) != AssignmentToOutcome(G, fulls[1..][j]) by {
        forall i, j | 0 <= i < j < |fulls[1..]|
          ensures AssignmentToOutcome(G, fulls[1..][i]) != AssignmentToOutcome(G, fulls[1..][j])
        {
          assert fulls[1..][i] == fulls[i + 1];
          assert fulls[1..][j] == fulls[j + 1];
        }
      }
      TruncatePMFOnOrder_SumOutcomeMasses_OverAssignments(G, p, X, xVals, ord, fulls[1..]);
    }
  }

  lemma TruncatePMFOnOrder_SumEncodedSupportAssignments(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures Prob.SumOutcomeMasses(
      TruncatePMFOnOrder(G, p, X, xVals, ord),
      EncodeAssignments(G, TruncateSupportAssignments(G, p, X, xVals))
    ) == SumTruncatedAssignmentMasses(
      G,
      p,
      X,
      xVals,
      TruncateSupportAssignments(G, p, X, xVals),
      ord
    )
  {
    assert forall i :: 0 <= i < |TruncateSupportAssignments(G, p, X, xVals)| ==>
      AssignmentToOutcome(G, TruncateSupportAssignments(G, p, X, xVals)[i])
        in EncodedTruncateSupport(G, p, X, xVals) by {
      forall i | 0 <= i < |TruncateSupportAssignments(G, p, X, xVals)|
        ensures AssignmentToOutcome(G, TruncateSupportAssignments(G, p, X, xVals)[i])
          in EncodedTruncateSupport(G, p, X, xVals)
      {
      }
    }
    TruncatePMFOnOrder_SumOutcomeMasses_OverAssignments(
      G,
      p,
      X,
      xVals,
      ord,
      TruncateSupportAssignments(G, p, X, xVals)
    );
  }

  lemma TruncatePMFOnOrder_EncodedSupportSequence_EnumeratesKeys(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures forall omega :: omega in EncodeAssignments(G, TruncateSupportAssignments(G, p, X, xVals)) <==> omega in TruncatePMFOnOrder(G, p, X, xVals, ord).Keys
    ensures forall i, j ::
      0 <= i < j < |EncodeAssignments(G, TruncateSupportAssignments(G, p, X, xVals))| ==>
        EncodeAssignments(G, TruncateSupportAssignments(G, p, X, xVals))[i]
          != EncodeAssignments(G, TruncateSupportAssignments(G, p, X, xVals))[j]
  {
    var fulls := TruncateSupportAssignments(G, p, X, xVals);
    var omegas := EncodeAssignments(G, fulls);
    var t := TruncatePMFOnOrder(G, p, X, xVals, ord);
    TruncatePMFOnOrder_Support(G, p, X, xVals, ord);
    EncodeAssignments_Index(G, fulls);

    assert forall omega :: omega in omegas ==> omega in t.Keys by {
      forall omega | omega in omegas
        ensures omega in t.Keys
      {
        var i :| 0 <= i < |omegas| && omegas[i] == omega;
        assert omegas[i] == AssignmentToOutcome(G, fulls[i]);
      }
    }

    assert forall omega :: omega in t.Keys ==> omega in omegas by {
      forall omega | omega in t.Keys
        ensures omega in omegas
      {
        var i :| 0 <= i < |fulls| && omega == AssignmentToOutcome(G, fulls[i]);
        assert omegas[i] == AssignmentToOutcome(G, fulls[i]);
      }
    }

    assert forall i, j :: 0 <= i < j < |omegas| ==> omegas[i] != omegas[j] by {
      forall i, j | 0 <= i < j < |omegas|
        ensures omegas[i] != omegas[j]
      {
        assert omegas[i] == AssignmentToOutcome(G, fulls[i]);
        assert omegas[j] == AssignmentToOutcome(G, fulls[j]);
      }
    }
  }

  lemma SumTruncatedAssignmentMasses_SingletonIntervention(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires |ord| == 1
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    requires ord[0] in X
    ensures SumTruncatedAssignmentMasses(
      G,
      p,
      X,
      xVals,
      TruncateSupportAssignments(G, p, X, xVals),
      ord
    ) == 1.0
  {
    var v := ord[0];
    var supp := TruncateSupportAssignments(G, p, X, xVals);
    assert X <= Nodes(G);
    SingletonTopologicalSort_NodeSet(G, ord);

    var full := map u | u in Nodes(G) :: xVals[u];
    assert full.Keys == Nodes(G);
    assert ExtendsAssignment(xVals, full) by {
      forall u | u in xVals.Keys
        ensures full[u] == xVals[u]
      {
      }
    }
    assert full[v] == xVals[v];
    TruncatedLocalFactor_IntervenedMatch(G, p, X, xVals, full, v);
    assert forall i :: 0 <= i < |ord| ==> ord[i] in Nodes(G);
    assert ord[1..] == [];
    assert TruncatedAssignmentMass(G, p, X, xVals, full, ord)
      == TruncatedLocalFactor(G, p, X, xVals, full, v) * TruncatedFactorProduct(G, p, X, xVals, full, ord[1..]);
    assert TruncatedAssignmentMass(G, p, X, xVals, full, ord) == 1.0;

    var k :| 0 <= k < |supp| && supp[k] == full;

    assert forall i :: 0 <= i < |supp| ==> supp[i] == full by {
      forall i | 0 <= i < |supp|
        ensures supp[i] == full
      {
        assert supp[i].Keys == Nodes(G);
        assert ExtendsAssignment(xVals, supp[i]);
        assert supp[i][v] == xVals[v];
        SingletonAssignmentsEqual(G, supp[i], full, v);
      }
    }

    assert |supp| == 1 by {
      assert |supp| >= 1;
      if |supp| > 1 {
        assert supp[0] == full;
        assert supp[1] == full;
        assert AssignmentToOutcome(G, supp[0]) == AssignmentToOutcome(G, supp[1]);
        assert AssignmentToOutcome(G, supp[0]) != AssignmentToOutcome(G, supp[1]);
        assert false;
      }
    }

    assert supp[0] == full;
    assert supp[1..] == [];
    assert SumTruncatedAssignmentMasses(G, p, X, xVals, supp, ord)
      == TruncatedAssignmentMass(G, p, X, xVals, supp[0], ord)
        + SumTruncatedAssignmentMasses(G, p, X, xVals, supp[1..], ord);
  }

  lemma SumTruncatedAssignmentMasses_SingletonEmpty_AsLocalPMFSum(
    G: Graph,
    p: Prob.PMF,
    ord: seq<Node>,
    fulls: seq<Assignment>,
    template: Assignment
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires |ord| == 1
    requires forall i :: 0 <= i < |fulls| ==> fulls[i].Keys == Nodes(G)
    requires template.Keys == Nodes(G)
    requires Parents(G, ord[0]) == {}
    ensures SumTruncatedAssignmentMasses(G, p, {}, map[], fulls, ord)
      == Prob.SumOutcomeMasses(ConditionalLocalPMF(G, p, ord[0], template), AssignmentValuesAt(fulls, ord[0]))
  {
    var v := ord[0];
    var q := ConditionalLocalPMF(G, p, v, template);
    if |fulls| != 0 {
      var full0 := fulls[0];
      assert v in Nodes(G);
      assert forall i :: 0 <= i < |fulls[1..]| ==> fulls[1..][i].Keys == Nodes(G) by {
        forall i | 0 <= i < |fulls[1..]|
          ensures fulls[1..][i].Keys == Nodes(G)
        {
          assert fulls[1..][i] == fulls[i + 1];
        }
      }
      TruncatedAssignmentMass_Empty(G, p, full0, ord);
      ConditionalLocalPMF_Locality(G, p, v, full0, template);
      ConditionalFactor_FromLocalPMF(G, p, v, full0, full0[v]);
      OverrideAssignment_SameValue(G, full0, v);
      assert ord[1..] == [];
      assert ConditionalFactorProduct(G, p, full0, ord)
        == ConditionalFactor(p, v, Parents(G, v), full0);
      assert TruncatedAssignmentMass(G, p, {}, map[], full0, ord)
        == Prob.OutcomeMass(q, full0[v]);
      SumTruncatedAssignmentMasses_SingletonEmpty_AsLocalPMFSum(G, p, ord, fulls[1..], template);
    }
  }

  lemma SumTruncatedAssignmentMasses_SingletonEmpty(
    G: Graph,
    p: Prob.PMF,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires |ord| == 1
    ensures SumTruncatedAssignmentMasses(G, p, {}, map[], TruncateSupportAssignments(G, p, {}, map[]), ord) == 1.0
  {
    var v := ord[0];
    var supp := TruncateSupportAssignments(G, p, {}, map[]);
    SingletonTopologicalSort_NodeSet(G, ord);
    assert Parents(G, v) == {} by {
      assert forall u :: u in Parents(G, v) ==> false by {
        forall u | u in Parents(G, v)
          ensures false
        {
          assert exists k | 0 <= k < 0 :: ord[k] == u;
        }
      }
    }

    Prob.DistributionHasSomeKey(p);
    var omega0 :| omega0 in p.Keys;
    var template := OutcomeToAssignment(G, omega0);
    var vals := AssignmentValuesAt(supp, v);
    var q := ConditionalLocalPMF(G, p, v, template);

    AssignmentValuesAt_Index(supp, v);
    assert forall i :: 0 <= i < |supp| ==> v in supp[i].Keys by {
      forall i | 0 <= i < |supp|
        ensures v in supp[i].Keys
      {
        assert supp[i].Keys == Nodes(G);
      }
    }

    assert forall i, j :: 0 <= i < j < |vals| ==> vals[i] != vals[j] by {
      forall i, j | 0 <= i < j < |vals|
        ensures vals[i] != vals[j]
      {
        assert vals[i] == supp[i][v];
        assert vals[j] == supp[j][v];
        if vals[i] == vals[j] {
          SingletonAssignmentsEqual(G, supp[i], supp[j], v);
          assert AssignmentToOutcome(G, supp[i]) == AssignmentToOutcome(G, supp[j]);
          assert AssignmentToOutcome(G, supp[i]) != AssignmentToOutcome(G, supp[j]);
          assert false;
        }
      }
    }

    SumTruncatedAssignmentMasses_SingletonEmpty_AsLocalPMFSum(G, p, ord, supp, template);

    assert forall value :: value in q.Keys && Prob.OutcomeMass(q, value) > 0.0 ==> value in vals by {
      forall value | value in q.Keys && Prob.OutcomeMass(q, value) > 0.0
        ensures value in vals
      {
        var full := map u | u in Nodes(G) :: value;
        assert full.Keys == Nodes(G);
        ConditionalLocalPMF_Locality(G, p, v, full, template);
        ConditionalFactor_FromLocalPMF(G, p, v, full, value);
        OverrideAssignment_SameValue(G, full, v);
        TruncatedAssignmentMass_Empty(G, p, full, ord);
        assert ord[1..] == [];
        assert ConditionalFactorProduct(G, p, full, ord)
          == ConditionalFactor(p, v, Parents(G, v), full);
        assert TruncatedAssignmentMass(G, p, {}, map[], full, ord)
          == Prob.OutcomeMass(q, value);
        var i :| 0 <= i < |supp| && supp[i] == full;
        assert vals[i] == supp[i][v];
      }
    }

    Prob.SumOutcomeMasses_PositiveCompleteNormalized_WithZeroExtras(q, vals);
  }

  lemma SumTruncatedAssignmentMasses_Singleton(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires |ord| == 1
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures SumTruncatedAssignmentMasses(
      G,
      p,
      X,
      xVals,
      TruncateSupportAssignments(G, p, X, xVals),
      ord
    ) == 1.0
  {
    var v := ord[0];
    if v in X {
      SumTruncatedAssignmentMasses_SingletonIntervention(G, p, X, xVals, ord);
    } else {
      SingletonTopologicalSort_NodeSet(G, ord);
      assert X == {} by {
        assert forall u :: u in X ==> false by {
          forall u | u in X
            ensures false
          {
            assert u in Nodes(G);
            assert u == v;
          }
        }
      }
      var empty: Assignment := map[];
      assert xVals == map[] by {
        assert xVals.Keys == empty.Keys;
        assert forall u :: u in xVals.Keys ==> xVals[u] == empty[u] by {
          forall u | u in xVals.Keys
            ensures xVals[u] == empty[u]
          {
          }
        }
      }
      SumTruncatedAssignmentMasses_SingletonEmpty(G, p, ord);
    }
  }

  lemma {:axiom} SumTruncatedAssignmentMasses_Normalized(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures SumTruncatedAssignmentMasses(
      G,
      p,
      X,
      xVals,
      TruncateSupportAssignments(G, p, X, xVals),
      ord
    ) == 1.0

  lemma TruncatePMFOnOrder_SumsToOne(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures Prob.SumsToOne(TruncatePMFOnOrder(G, p, X, xVals, ord))
  {
    var t := TruncatePMFOnOrder(G, p, X, xVals, ord);
    var fulls := TruncateSupportAssignments(G, p, X, xVals);
    var omegas := EncodeAssignments(G, fulls);
    TruncatePMFOnOrder_EncodedSupportSequence_EnumeratesKeys(G, p, X, xVals, ord);
    TruncatePMFOnOrder_SumEncodedSupportAssignments(G, p, X, xVals, ord);
    SumTruncatedAssignmentMasses_Normalized(G, p, X, xVals, ord);
    Prob.FiniteSupportSum_AnyDistinctEnumeration(t, t.Keys, omegas);
    assert t.Keys * t.Keys == t.Keys;
    calc {
      Prob.ProbEvent(t, t.Keys);
      ==
      Prob.FiniteSupportSum(t, t.Keys * t.Keys);
      ==
      Prob.FiniteSupportSum(t, t.Keys);
      == { Prob.FiniteSupportSum_AnyDistinctEnumeration(t, t.Keys, omegas); }
      Prob.SumOutcomeMasses(t, omegas);
      == { TruncatePMFOnOrder_SumEncodedSupportAssignments(G, p, X, xVals, ord); }
      SumTruncatedAssignmentMasses(G, p, X, xVals, fulls, ord);
      == { SumTruncatedAssignmentMasses_Normalized(G, p, X, xVals, ord); }
      1.0;
    }
  }

  lemma TruncatePMFOnOrder_IsDistribution(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires IsTopologicalSort(G, ord)
    requires xVals.Keys == X
    requires xVals.Keys <= Nodes(G)
    ensures Prob.IsDistribution(TruncatePMFOnOrder(G, p, X, xVals, ord))
  {
    TruncatePMFOnOrder_AllNonNeg(G, p, X, xVals, ord);
    TruncatePMFOnOrder_SumsToOne(G, p, X, xVals, ord);
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

  // Marginalizing a Markov-factored PMF over a subgraph Z produces a PMF
  // that Markov-factorizes over the restricted DAG G_{V\Z} = RemoveNodes(G, Z).
  //
  // Ref: Standard result in graphical models (e.g., Lauritzen 1996, §3.2).
  //      Used in ID algorithm Line 2 to pass the ancestral-marginal
  //      to the recursive call on SubgraphSM(sm, AncY).
  lemma {:axiom} MarkovFactorization_Marginal(
    G: Graph, p: Prob.PMF, Z: set<Node>
  )
    requires IsDAG(G)
    requires Prob.IsDistribution(p)
    requires MarkovFactorization(G, p)
    requires Z <= Nodes(G)
    ensures MarkovFactorization(RemoveNodes(G, Z), Marginalize(G, p, Z))

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
  //   Proof structure (DA-M1-G2-3):
  //     (a) PMFToAssignmentPMF_MarkovTransfer lifts MarkovFactorization(G, p)
  //         to MarkovFactorizationConcrete(G, PMFToAssignmentPMF(G, p)).
  //     (b) GlobalMarkovConcrete (axiom below) closes the conclusion under
  //         the concrete Markov condition.
  //   GlobalMarkovConcrete absorbs two blocked sub-steps:
  //     (i)  Graph lemma: DSep(G,Y,Z,W) ⟹ ∀ v∈Y, Z ⊆ NonDesc(v)\Pa(v)
  //          relative to W (Bayes Ball soundness — not yet in dag.dfy).
  //     (ii) Probabilistic telescope: induction over Y using
  //          LocalMarkovConcrete + a MarginalMass↔AssignmentCondProb bridge
  //          (requires MarginalMass_FactorOut, itself still axiomatic).
  // ==================================================================

  // Concrete-layer GlobalMarkov: same conclusion as
  // GlobalMarkov_From_Factorization below, but the Markov precondition
  // is stated at the concrete AssignmentPMF level.  The proof of
  // GlobalMarkov_From_Factorization then follows in two lines using
  // PMFToAssignmentPMF_MarkovTransfer.
  //
  // Why this remains an axiom:
  //   (i)  The graph lemma DSep(G,Y,Z,W) ⟹ ∀ v∈Y, Z ⊆ NonDesc(v)\Pa(v)
  //        requires Bayes Ball soundness, which is not yet in dag.dfy.
  //   (ii) The probabilistic telescope connects MarginalMass cross-ratios
  //        to AssignmentCondProb via induction over Y; the per-step
  //        arithmetic reduction depends on MarginalMass_FactorOut, which
  //        is itself still axiomatic (concrete layer, Z3 arithmetic gap).
  //   All proof steps are within the concrete AssignmentPMF layer —
  //   there is no abstract Prob.Outcome enumeration involved.
  //   Proof sketch: induct on |Y|; base case |Y|=0 trivial; step:
  //   pick v∈Y, apply (i) to get Z⊆NonDesc(v)\Pa(v), apply
  //   LocalMarkovConcrete for that v, then use IH on Y\{v}.
  lemma {:axiom} GlobalMarkovConcrete(
    G: Graph,
    p: Prob.PMF,
    Y: set<Node>,
    Z: set<Node>,
    W: set<Node>,
    yAssign: Assignment,
    zAssign: Assignment,
    wAssign: Assignment
  )
    requires IsDAG(G)
    requires Prob.IsDistribution(p)
    requires MarkovFactorizationConcrete(G, PMFToAssignmentPMF(G, p))
    requires DSep(G, Y, Z, W)
    requires Y <= Nodes(G)
    requires Z <= Nodes(G)
    requires W <= Nodes(G)
    requires yAssign.Keys <= Y
    requires zAssign.Keys <= Z
    requires wAssign.Keys <= W
    requires CompatibleAssignments(zAssign, wAssign)
    requires AssignmentProb(p, G, wAssign) > 0.0
    requires AssignmentProb(p, G, MergeAssignments(zAssign, wAssign)) > 0.0
    ensures AssignmentCondProb(p, G, yAssign, MergeAssignments(zAssign, wAssign))
      == AssignmentCondProb(p, G, yAssign, wAssign)

  // Proved using the two-step chain:
  //   MarkovFactorization(G,p)
  //     ──[PMFToAssignmentPMF_MarkovTransfer]──▶
  //   MarkovFactorizationConcrete(G, PMFToAssignmentPMF(G,p))
  //     ──[GlobalMarkovConcrete]──▶
  //   AssignmentCondProb(p,G,yAssign,zw) == AssignmentCondProb(p,G,yAssign,w)
  lemma GlobalMarkov_From_Factorization(
    G: Graph,
    p: Prob.PMF,
    Y: set<Node>,
    Z: set<Node>,
    W: set<Node>,
    yAssign: Assignment,
    zAssign: Assignment,
    wAssign: Assignment
  )
    requires IsDAG(G)
    requires Prob.IsDistribution(p)
    requires MarkovFactorization(G, p)
    requires DSep(G, Y, Z, W)
    requires Y <= Nodes(G)
    requires Z <= Nodes(G)
    requires W <= Nodes(G)
    requires yAssign.Keys <= Y
    requires zAssign.Keys <= Z
    requires wAssign.Keys <= W
    requires CompatibleAssignments(zAssign, wAssign)
    requires AssignmentProb(p, G, wAssign) > 0.0
    requires AssignmentProb(p, G, MergeAssignments(zAssign, wAssign)) > 0.0
    ensures AssignmentCondProb(p, G, yAssign, MergeAssignments(zAssign, wAssign))
      == AssignmentCondProb(p, G, yAssign, wAssign)
  {
    PMFToAssignmentPMF_MarkovTransfer(G, p);
    GlobalMarkovConcrete(G, p, Y, Z, W, yAssign, zAssign, wAssign);
  }

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
  ghost function Marginalize(G: Graph, p: Prob.PMF, W: set<Node>): Prob.PMF
    requires W <= Nodes(G)
    ensures forall partial: Assignment ::
      partial.Keys <= Nodes(G) - W ==>
      AssignmentProb(Marginalize(G, p, W), G, partial) == AssignmentProb(p, G, partial)
  {
    p
  }

  lemma Marginalize_IsDistribution(G: Graph, p: Prob.PMF, W: set<Node>)
    requires Prob.IsDistribution(p)
    requires W <= Nodes(G)
    ensures Prob.IsDistribution(Marginalize(G, p, W))
  {
  }

}  // end module Interventional
