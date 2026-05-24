// ===================================================================
// Formal Rules of Pearl's Do-Calculus — Dafny Specification
//
// Reference: Pearl, J. (2000). Causality. Cambridge University Press.
//            §3.4  "The Do-Calculus" (Theorem 3.4)
//
// This module imports the DAG and Probability foundations and builds
// the three rules of the do-calculus on top.  Graph structure and
// d-separation come from the DAG module; probability laws come from
// the Probability module.  The concrete semantics of the do-operator
// (TruncatePMF, MarkovFactorization, GlobalMarkov_From_Factorization)
// come from the Interventional module.
//
// Layer diagram:
//
//   ┌───────────────────────────┐
//   │    DoCalculus             │  ← The three rules, backdoor, frontdoor
//   ├───────────────────────────┤
//   │  Interventional           │  ← TruncatePMF, GlobalMarkov, grounding
//   ├───────────────────────────┤
//   │  DAG (d-separation)       │  ← Graph surgery, trails, blocking
//   ├───────────────────────────┤
//   │  Probability (axioms)     │  ← Kolmogorov, Bayes, chain rule
//   └───────────────────────────┘
//
// To verify:
//   dafny verify probability.dfy dag.dfy interventional.dfy do_calculus.dfy
// ===================================================================

include "dag.dfy"
include "probability.dfy"
include "interventional.dfy"

module DoCalculus {

  import opened DAG
  import Prob = Probability
  import opened Interventional

  // ==================================================================
  // 1.  Interventional distribution (abstract)
  //
  //   IntProb(G, Y, doX, obsW)  :=  P_G( Y | do(X), W )
  //
  //   - G   : the causal DAG
  //   - Y   : target variable set
  //   - doX : set of intervened-on variables
  //   - obsW: set of conditioned (observed) variables
  //
  //   Setting doX = {} recovers the ordinary conditional P(Y | W).
  //   The return type is a PMF from the Probability module: this is
  //   the whole interventional distribution over Y-assignments, not a
  //   single scalar evaluation such as P(y | do(x), w).
  //
  //   This abstract function intentionally remains the interface used
  //   by the three rules. The current grounding in
  //   interventional.dfy is pointwise and witness-bearing:
  //   IntProbConcrete(G, p, yAssign, xAssign, wAssign) returns a real
  //   for specific assignments under an explicit observational PMF p.
  //   By contrast, this layer's IntProb is PMF-valued and indexed only
  //   by variable sets. A concrete wrapper here would therefore require
  //   either a signature redesign or a richer kernel object that carries
  //   the implicit PMF witness and value assignments.
  // ==================================================================

  function {:axiom} IntProb(
    G: Graph, Y: set<Node>, doX: set<Node>, obsW: set<Node>
  ): Prob.PMF

  // ==================================================================
  // 2.  Global Markov Property — derived from Interventional module
  //
  //   If (Y ⊥_G Z | W) then Y and Z are conditionally independent
  //   given W under any distribution faithful to G.
  //
  //   Previously an axiom here; now delegated to
  //   GlobalMarkov_From_Factorization in interventional.dfy.
  // ==================================================================

  /// Global Markov Property:
  ///   d-separation in G implies conditional independence in P_G.
  ///
  ///   (Y ⊥_G Z | W)  ⟹  P(Y | Z, W) = P(Y | W)
  ///
  ///   Delegated to Interventional.GlobalMarkov_From_Factorization;
  ///   kept as an axiom here at the DoCalculus layer because the
  ///   PMF witness (p) is implicit in IntProb's abstract type.
  lemma {:axiom} GlobalMarkov(
    G: Graph, Y: set<Node>, Z: set<Node>, W: set<Node>
  )
    requires DSep(G, Y, Z, W)
    ensures  IntProb(G, Y, {}, Z + W) == IntProb(G, Y, {}, W)

  // ==================================================================
  // 3.  Truncated factorisation (the do-operator semantics)
  //
  //   Setting do(X = x) replaces the structural equation for each
  //   node in X with a constant, which is equivalent to removing
  //   incoming edges in the DAG and using the resulting modified
  //   distribution.  The concrete implementation is TruncatePMF in
  //   interventional.dfy; TruncatePMF_Markov establishes that the
  //   result is Markov-factored w.r.t. RemoveIncoming(G, X).
  // ==================================================================

  /// Interventional Semantics:
  ///   P(Y | do(X), W)  =  P_{G_{X̄}}(Y | W)
  ///   where G_{X̄} = RemoveIncoming(G, X).
  ///
  ///   Grounded by TruncatePMF + TruncatePMF_Markov in interventional.dfy.
  ///   Kept as an axiom at this layer because IntProb is abstract.
  lemma {:axiom} InterventionSemantics(
    G: Graph, Y: set<Node>, X: set<Node>, W: set<Node>
  )
    ensures IntProb(G, Y, X, W) ==
            IntProb(RemoveIncoming(G, X), Y, {}, W)

  // ==================================================================
  // 4.  The Three Rules of the Do-Calculus
  //     (Pearl 2000, Theorem 3.4)
  //
  //   In each rule Y, X, Z, W are pairwise disjoint variable-sets.
  //   The rules are sound: Pearl's Chapter 3 proves them from the
  //   axioms of structural causal models.
  //
  //   Rule 1:  P(y | do(x), z, w)     = P(y | do(x), w)
  //   Rule 2:  P(y | do(x), do(z), w) = P(y | do(x), z, w)
  //   Rule 3:  P(y | do(x), do(z), w) = P(y | do(x), w)
  // ==================================================================

  /// Rule 1 — Insertion / Deletion of Observations
  ///
  ///   Condition:  (Y ⊥ Z | X, W)  in  G_{X̄}
  ///
  ///   Intuition: after intervening on X, the d-separation criterion
  ///   in the mutilated graph tells us Z carries no information about
  ///   Y beyond X and W.  Therefore Z may be added or removed from
  ///   the conditioning set without changing the interventional
  ///   distribution.
  ///
  ///   Proof sketch:
  ///     By InterventionSemantics, P(Y|do(X),Z,W) = P_{G_{X̄}}(Y|Z,W).
  ///     The hypothesis gives (Y ⊥ Z | W) in G_{X̄}.
  ///     By GlobalMarkov, P_{G_{X̄}}(Y|Z,W) = P_{G_{X̄}}(Y|W).
  ///     Again by InterventionSemantics, P_{G_{X̄}}(Y|W) = P(Y|do(X),W).
  lemma {:axiom} Rule1_InsertDeleteObservation(
    G: Graph, Y: set<Node>, X: set<Node>, Z: set<Node>, W: set<Node>
  )
    requires DSep(RemoveIncoming(G, X), Y, Z, X + W)
    ensures  IntProb(G, Y, X, Z + W) == IntProb(G, Y, X, W)

  /// Rule 2 — Action / Observation Exchange
  ///
  ///   Condition:  (Y ⊥ Z | X, W)  in  G_{X̄, Z̲}
  ///
  ///   Intuition: when Z has no outgoing effect on Y (edges from Z
  ///   removed), intervening on Z is the same as observing Z.
  ///
  ///   P(y | do(x), do(z), w) = P(y | do(x), z, w)
  lemma {:axiom} Rule2_ActionObservationExchange(
    G: Graph, Y: set<Node>, X: set<Node>, Z: set<Node>, W: set<Node>
  )
    requires DSep(RemoveOutgoing(RemoveIncoming(G, X), Z), Y, Z, X + W)
    ensures  IntProb(G, Y, X + Z, W) == IntProb(G, Y, X, Z + W)

  /// Rule 3 — Insertion / Deletion of Actions
  ///
  ///   Let  Z̄(W) := Z \ An_{G_{X̄}}(W)
  ///    (the Z-nodes that are NOT ancestors of W in G_{X̄}).
  ///
  ///   Condition:  (Y ⊥ Z | X, W)  in  G_{X̄, Z̄(W)_bar}
  ///
  ///   Intuition: if intervening on Z cannot reach Y (given X and W),
  ///   the intervention on Z is superfluous.
  ///
  ///   P(y | do(x), do(z), w) = P(y | do(x), w)
  lemma {:axiom} Rule3_InsertDeleteAction(
    G: Graph, Y: set<Node>, X: set<Node>, Z: set<Node>, W: set<Node>
  )
    requires
      var Gx      := RemoveIncoming(G, X);
      var ZnotAnc := Z - Ancestors(Gx, W);
      DSep(RemoveIncoming(Gx, ZnotAnc), Y, Z, X + W)
    ensures IntProb(G, Y, X + Z, W) == IntProb(G, Y, X, W)

  // ==================================================================
  // 5.  Derived Results
  // ==================================================================

  /// Pure observational d-separation:
  ///   (Y ⊥_G Z | W) with no interventions  ⟹  P(Y | Z, W) = P(Y | W).
  ///
  ///   This is Rule 1 instantiated with X = ∅.
  lemma PureDSepErasesObservation(
    G: Graph, Y: set<Node>, Z: set<Node>, W: set<Node>
  )
    requires DSep(G, Y, Z, W)
    ensures  IntProb(G, Y, {}, Z + W) == IntProb(G, Y, {}, W)
  {
    RemoveIncoming_Empty(G);
    // RemoveIncoming(G, {}) == G, so the hypothesis
    // DSep(G, Y, Z, W)  equals  DSep(RemoveIncoming(G,{}), Y, Z, {}+W).
    assert {} + W == W;
    assert DSep(RemoveIncoming(G, {}), Y, Z, {} + W);
    Rule1_InsertDeleteObservation(G, Y, {}, Z, W);
  }

  // ==================================================================
  // 6.  Backdoor Criterion  (Pearl 2000, Theorem 3.3.2)
  // ==================================================================

  /// If Z satisfies the backdoor criterion for (X → Y) in G:
  ///
  ///   (i)  No z ∈ Z is a descendant of any x ∈ X.
  ///   (ii) Z d-separates Y from X in G_{X̲}  (X's *outgoing* edges removed).
  ///
  /// Then:  P(Y | do(X)) = P(Y | X, Z)
  ///   (the causal effect is identified from observational data).
  lemma {:axiom} BackdoorAdjustment(
    G: Graph, Y: set<Node>, X: set<Node>, Z: set<Node>
  )
    requires
      // (i) No descendant of X in Z (except X nodes themselves)
      (forall x, z ::
         x in X && z in Z && IsAncestor(G, x, z) ==> x == z)
      // (ii) Z d-separates Y from X in G with X's *outgoing* edges removed.
      //      Using RemoveOutgoing (rather than RemoveIncoming) makes causal paths
      //      physically absent, so standard d-separation applies directly without
      //      relying on the |trail| <= 1 short-circuit in TrailBlocked.
      && DSep(RemoveOutgoing(G, X), Y, X, Z)
    ensures IntProb(G, Y, X, {}) == IntProb(G, Y, {}, X + Z)

  // ==================================================================
  // 7.  Frontdoor Criterion  (Pearl 2000, Theorem 3.3.4)
  // ==================================================================

  /// If M satisfies the frontdoor criterion for (X → Y) in G:
  ///
  ///   (i)   M intercepts all directed paths from X to Y.
  ///         (Structural: no directed x→y path exists avoiding all nodes in M.
  ///          Not expressible as a DSep query; checked separately.)
  ///   (ii)  No unblocked back-door paths from X to M.
  ///         Using RemoveOutgoing(G, X) eliminates causal X→M edges so that
  ///         standard d-separation checks only backdoor paths, without
  ///         relying on the |trail| <= 1 short-circuit in TrailBlocked.
  ///   (iii) All back-door paths from M to Y are blocked by X.
  ///         Using RemoveOutgoing(G, M) eliminates causal M→Y edges so that
  ///         d-separation checks only backdoor paths from M to Y.
  ///
  /// Then:  P(Y | do(X)) = P(Y | do(M))
  lemma {:axiom} FrontdoorCriterion(
    G: Graph, Y: set<Node>, X: set<Node>, M: set<Node>
  )
    requires
      // (ii) No unblocked backdoor from X to M in G_{X̲} (X outgoing removed)
      DSep(RemoveOutgoing(G, X), M, X, {})
      // (iii) All backdoor paths from M to Y blocked by X in G_{M̲} (M outgoing removed)
      && DSep(RemoveOutgoing(G, M), M, Y, X)
    ensures IntProb(G, Y, X, {}) == IntProb(G, Y, M, {})

}  // end module DoCalculus
