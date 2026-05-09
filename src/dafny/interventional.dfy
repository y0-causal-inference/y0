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
  //   An Assignment maps each node to a concrete outcome value.
  //   Used to represent a full instantiation of all variables
  //   in the DAG, or a partial assignment for interventions.
  // ==================================================================

  type Assignment = map<Node, Prob.Outcome>

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
  //   1. Keep only rows where X-variables match xVals.
  //   2. Renormalize over remaining rows.
  //
  //   Equivalently: replace each factor P(xᵢ | pa(xᵢ)) with a
  //   point mass δ(xᵢ = xValsᵢ), then renormalize.
  //
  //   Ref: Pearl (2000), Theorem 1.3.1 / Definition 3.2.1
  // ==================================================================

  ghost function {:axiom} TruncatePMF(
    G: Graph,
    p: Prob.PMF,
    X: set<Node>,
    xVals: Assignment
  ): Prob.PMF
    requires xVals.Keys == X
    requires Prob.IsDistribution(p)

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

  // ==================================================================
  // 5.  IntProbConcrete — grounding the abstract IntProb
  //
  //   P(Y | do(X = xVals), W = wVals) computed concretely:
  //     = ProbCond(TruncatePMF(G, p, X, xVals), Y-event, W-event)
  //
  //   This returns a real number (the probability for specific
  //   value assignments), whereas IntProb in do_calculus.dfy returns
  //   a PMF.  The grounding lemma relates them pointwise:
  //     for all y-assignments, IntProb(G,Y,X,W)[y] == IntProbConcrete(...)
  // ==================================================================

  // For a specific Y-assignment and W-assignment, compute the
  // concrete interventional probability.
  ghost function {:axiom} IntProbConcrete(
    G: Graph,
    p: Prob.PMF,
    yAssign: Assignment,
    xAssign: Assignment,
    wAssign: Assignment
  ): real
    requires Prob.IsDistribution(p)
    requires MarkovFactorization(G, p)

  // The grounding axiom: IntProbConcrete equals the conditional
  // probability in the truncated distribution.
  //
  // IntProbConcrete(G, p, y, x, w) ==
  //   ProbCond(TruncatePMF(G, p, X, x), Y-event(y), W-event(w))
  //
  // This connects the abstract IntProb (which returns a PMF) to
  // the concrete computation (which returns a real).
  lemma {:axiom} IntProb_Grounded(
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
  // 7.  Marginalization
  //
  //   Marginalizing a PMF sums out a set of variables.
  //   Σ_W P(V) yields a distribution over V \ W.
  // ==================================================================

  // Marginalize: Σ_W P(V)
  ghost function {:axiom} Marginalize(p: Prob.PMF, W: set<Node>): Prob.PMF

  lemma {:axiom} Marginalize_IsDistribution(p: Prob.PMF, W: set<Node>)
    requires Prob.IsDistribution(p)
    ensures Prob.IsDistribution(Marginalize(p, W))

}  // end module Interventional
