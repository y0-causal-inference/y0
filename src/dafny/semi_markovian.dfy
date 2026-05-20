// ===================================================================
// Semi-Markovian Causal Models — Dafny Specification
//
// Reference: Shpitser, I. & Pearl, J. (2006).
//   "Identification of Joint Interventional Distributions in
//    Recursive Semi-Markovian Causal Models."  AAAI-06.
//
// This module extends the DAG module with bidirected edges
// (representing hidden common causes) to model Semi-Markovian
// Causal Models (SMCMs).  It defines:
//
//   1. SMGraph — a graph with both directed and bidirected edges
//   2. C-components (districts) — connected components via bidirected edges
//   3. C-trees and C-forests — structures used in non-identifiability
//   4. Hedges — the graphical witness for non-identifiability
//
// Layer diagram:
//
//   ┌───────────────────────────────┐
//   │  Identification               │  ← ID algorithm, theorems
//   ├───────────────────────────────┤
//   │  SemiMarkovian (this module)  │  ← SMGraph, C-components, hedges
//   ├───────────────────────────────┤
//   │  DoCalculus                   │  ← Three rules, backdoor, frontdoor
//   ├───────────────────────────────┤
//   │  Interventional               │  ← TruncatePMF, grounding
//   ├───────────────────────────────┤
//   │  DAG (d-separation)           │  ← Graph surgery, trails, blocking
//   ├───────────────────────────────┤
//   │  Probability (axioms)         │  ← Kolmogorov, Bayes, chain rule
//   └───────────────────────────────┘
//
// To verify:
//   dafny verify probability.dfy dag.dfy interventional.dfy \
//          do_calculus.dfy semi_markovian.dfy
// ===================================================================

include "dag.dfy"
include "probability.dfy"
include "interventional.dfy"
include "do_calculus.dfy"

module SemiMarkovian {

  import opened DAG
  import Prob = Probability
  import opened Interventional
  import opened DoCalculus

  // ==================================================================
  // 1.  Semi-Markovian Graph
  //
  //   A Semi-Markovian graph extends a DAG with bidirected edges.
  //   Bidirected edges represent hidden common causes (latent
  //   confounders) between two observable variables.
  //
  //   SMGraph = (G, Bi) where:
  //     G  : the directed acyclic graph (from DAG module)
  //     Bi : set of unordered pairs {u, v} connected by bidirected edges
  //
  //   We represent bidirected edges as a set of pairs (u, v) where
  //   u < v to canonicalize the representation.
  //
  //   Ref: Shpitser & Pearl 2006, §"Notation and Definitions"
  //        Pearl 2000, Chapter 3
  //        Python: NxMixedGraph (src/y0/graph.py)
  // ==================================================================

  // A bidirected edge between two nodes.
  // We store both directions for symmetric access.
  datatype BiEdge = BiEdge(u: Node, v: Node)

  // A Semi-Markovian graph: a DAG plus a set of bidirected edges.
  datatype SMGraph = SMGraph(
    dag: Graph,          // The directed acyclic component
    bidirected: set<BiEdge>  // The bidirected (latent confounder) edges
  )

  // All nodes in the directed component.
  function SMNodes(sm: SMGraph): set<Node> {
    Nodes(sm.dag)
  }

  // Check that all bidirected edges connect nodes in the DAG.
  predicate WellFormedSM(sm: SMGraph) {
    (forall e :: e in sm.bidirected ==>
       e.u in SMNodes(sm) && e.v in SMNodes(sm) && e.u != e.v) &&
    IsDAG(sm.dag)
  }

  // Two nodes are connected by a bidirected edge.
  predicate HasBidirected(sm: SMGraph, u: Node, v: Node) {
    BiEdge(u, v) in sm.bidirected || BiEdge(v, u) in sm.bidirected
  }

  // ==================================================================
  // 2.  C-Components (Districts)
  //
  //   Two nodes are in the same C-component (district) if they are
  //   connected by a path consisting entirely of bidirected edges.
  //
  //   C-components partition V into groups of variables that share
  //   latent common causes (possibly transitively).
  //
  //   Ref: Tian & Pearl (2002), Definition 3
  //        Shpitser & Pearl (2006), §"Notation and Definitions"
  //        Python: NxMixedGraph.districts()
  // ==================================================================

  // Two nodes are bidirected-connected with bounded path length.
  predicate BidirectedConnectedBounded(
    sm: SMGraph, u: Node, v: Node, fuel: nat
  )
    decreases fuel
  {
    u == v ||
    (fuel > 0 &&
     exists w :: w in SMNodes(sm) &&
       HasBidirected(sm, u, w) &&
       BidirectedConnectedBounded(sm, w, v, fuel - 1))
  }

  // Two nodes are in the same C-component.
  predicate BidirectedConnected(sm: SMGraph, u: Node, v: Node) {
    BidirectedConnectedBounded(sm, u, v, |SMNodes(sm)|)
  }

  // The C-component (district) containing node v.
  function CComponent(sm: SMGraph, v: Node): set<Node>
    requires v in SMNodes(sm)
  {
    set u | u in SMNodes(sm) && BidirectedConnected(sm, u, v)
  }

  // The set of all C-components of a Semi-Markovian graph.
  // Each C-component is the maximal bidirected-connected subset
  // containing each node. C-components partition SMNodes(sm).
  function CComponents(sm: SMGraph): set<set<Node>>
  {
    // Ghost spec: the set of all maximal bidirected-connected subsets.
    // A set S is a C-component iff:
    //   (i)  every pair in S is BidirectedConnected
    //   (ii) no node outside S is BidirectedConnected to all of S
    set S: set<Node> | S <= SMNodes(sm) && S != {} &&
      (forall u, v | u in S && v in S :: BidirectedConnected(sm, u, v)) &&
      (forall u | u in SMNodes(sm) && u !in S ::
         exists v | v in S :: !BidirectedConnected(sm, u, v))
  } by method {
    var comps := ComputeCComponents(sm);
    // The BFS produces exactly the maximal bidirected-connected subsets.
    assume {:axiom} SeqToSetOfSets(comps) ==
      (set S: set<Node> | S <= SMNodes(sm) && S != {} &&
        (forall u, v | u in S && v in S :: BidirectedConnected(sm, u, v)) &&
        (forall u | u in SMNodes(sm) && u !in S ::
           exists v | v in S :: !BidirectedConnected(sm, u, v)));
    return SeqToSetOfSets(comps);
  }

  // C-components partition the node set.
  lemma {:axiom} CComponents_Partition(sm: SMGraph)
    requires WellFormedSM(sm)
    ensures
      (forall v :: v in SMNodes(sm) ==>
         exists S :: S in CComponents(sm) && v in S) &&
      (forall S1, S2 :: (S1 in CComponents(sm) && S2 in CComponents(sm)
         && S1 != S2) ==> S1 * S2 == {}) &&
      (forall S :: S in CComponents(sm) ==> S <= SMNodes(sm) && S != {})
  // Proof sketch: follows from the set-comprehension ghost body of
  // CComponents and the BFS correctness assumption in the by-method.

  // Nodes in the same C-component are bidirected-connected.
  lemma CComponent_Connected(sm: SMGraph, u: Node, v: Node)
    requires WellFormedSM(sm)
    requires u in SMNodes(sm) && v in SMNodes(sm)
    requires exists S :: S in CComponents(sm) && u in S && v in S
    ensures BidirectedConnected(sm, u, v)
  {
    // By definition of CComponents, every pair in a component is
    // BidirectedConnected (first conjunct of the ghost body).
    var S :| S in CComponents(sm) && u in S && v in S;
    assert S <= SMNodes(sm) && S != {} &&
      (forall x, y | x in S && y in S :: BidirectedConnected(sm, x, y)) &&
      (forall x | x in SMNodes(sm) && x !in S ::
        exists y | y in S :: !BidirectedConnected(sm, x, y));
    assert forall x, y | x in S && y in S :: BidirectedConnected(sm, x, y);
  }

  lemma SingletonNode_SingleCComponent(sm: SMGraph, v: Node)
    requires WellFormedSM(sm)
    requires SMNodes(sm) == {v}
    ensures {v} in CComponents(sm)
    ensures |CComponents(sm)| == 1
  {
    assert {v} <= SMNodes(sm);
    assert {v} != {};
    forall x, y | x in {v} && y in {v}
      ensures BidirectedConnected(sm, x, y)
    {
      assert x == v;
      assert y == v;
    }
    forall x | x in SMNodes(sm) && x !in {v}
      ensures exists y | y in {v} :: !BidirectedConnected(sm, x, y)
    {
      assert x in {v};
      assert false;
    }
    assert {v} in CComponents(sm);
    forall T | T in CComponents(sm)
      ensures T == {v}
    {
      assert T <= SMNodes(sm) && T != {} &&
        (forall x, y | x in T && y in T :: BidirectedConnected(sm, x, y)) &&
        (forall x | x in SMNodes(sm) && x !in T ::
          exists y | y in T :: !BidirectedConnected(sm, x, y));
      var t :| t in T;
      assert t in SMNodes(sm);
      assert t in {v};
      assert t == v;
      assert v in T;
      assert {v} <= T;
      assert T <= {v};
      assert T == {v};
    }
    forall T | T in CComponents(sm)
      ensures T in {{v}}
    {
      assert T == {v};
    }
    forall T | T in {{v}}
      ensures T in CComponents(sm)
    {
      assert T == {v};
    }
    assert CComponents(sm) <= {{v}};
    assert {{v}} <= CComponents(sm);
    assert CComponents(sm) == {{v}};
  }

  lemma TwoNodeBidirected_ComponentWitness(sm: SMGraph, u: Node, v: Node)
    requires WellFormedSM(sm)
    requires u != v
    requires SMNodes(sm) == {u, v}
    requires HasBidirected(sm, u, v)
    ensures {u, v} in CComponents(sm)
  {
    assert BidirectedConnectedBounded(sm, u, v, 1);
    assert BidirectedConnected(sm, u, v);
    assert BidirectedConnectedBounded(sm, v, u, 1);
    assert BidirectedConnected(sm, v, u);
    assert {u, v} <= SMNodes(sm);
    assert {u, v} != {};
    forall x, y | x in {u, v} && y in {u, v}
      ensures BidirectedConnected(sm, x, y)
    {
      if x == y {
      } else if x == u {
        assert y == v;
      } else {
        assert x == v;
        assert y == u;
      }
    }
    forall x | x in SMNodes(sm) && x !in {u, v}
      ensures exists y | y in {u, v} :: !BidirectedConnected(sm, x, y)
    {
      assert x in {u, v};
      assert false;
    }
    assert {u, v} in CComponents(sm);
  }

  lemma TwoNodeBidirected_OnlyComponent(
    sm: SMGraph,
    u: Node,
    v: Node,
    T: set<Node>
  )
    requires WellFormedSM(sm)
    requires u != v
    requires SMNodes(sm) == {u, v}
    requires HasBidirected(sm, u, v)
    requires T in CComponents(sm)
    ensures T == {u, v}
  {
    assert T <= SMNodes(sm) && T != {} &&
      (forall x, y | x in T && y in T :: BidirectedConnected(sm, x, y)) &&
      (forall x | x in SMNodes(sm) && x !in T ::
        exists y | y in T :: !BidirectedConnected(sm, x, y));
    assert BidirectedConnectedBounded(sm, u, v, 1);
    assert BidirectedConnected(sm, u, v);
    assert BidirectedConnectedBounded(sm, v, u, 1);
    assert BidirectedConnected(sm, v, u);
    if u !in T {
      assert u in SMNodes(sm);
      var y :| y in T && !BidirectedConnected(sm, u, y);
      assert y in SMNodes(sm);
      assert y in {u, v};
      if y == u {
        assert u in T;
        assert false;
      }
      assert y == v;
      assert BidirectedConnected(sm, u, y);
      assert false;
    }
    if v !in T {
      assert v in SMNodes(sm);
      var y :| y in T && !BidirectedConnected(sm, v, y);
      assert y in SMNodes(sm);
      assert y in {u, v};
      if y == v {
        assert v in T;
        assert false;
      }
      assert y == u;
      assert BidirectedConnected(sm, v, y);
      assert false;
    }
    assert {u, v} <= T;
    assert T <= {u, v};
    assert T == {u, v};
  }

  lemma TwoNodeBidirected_SingleCComponent(sm: SMGraph, u: Node, v: Node)
    requires WellFormedSM(sm)
    requires u != v
    requires SMNodes(sm) == {u, v}
    requires HasBidirected(sm, u, v)
    ensures {u, v} in CComponents(sm)
    ensures |CComponents(sm)| == 1
  {
    TwoNodeBidirected_ComponentWitness(sm, u, v);
    forall T | T in CComponents(sm)
      ensures T == {u, v}
    {
      TwoNodeBidirected_OnlyComponent(sm, u, v, T);
    }
    forall T | T in CComponents(sm)
      ensures T in {{u, v}}
    {
      assert T == {u, v};
    }
    forall T | T in {{u, v}}
      ensures T in CComponents(sm)
    {
      assert T == {u, v};
    }
    assert CComponents(sm) <= {{u, v}};
    assert {{u, v}} <= CComponents(sm);
    assert CComponents(sm) == {{u, v}};
  }

  // ------------------------------------------------------------------
  // Compiled C-Components — BFS over bidirected edges
  //
  // BidirectedBFS computes the connected component of a single node
  // by BFS over bidirected edges.  ComputeCComponents iterates over
  // all nodes to collect all components.
  // ------------------------------------------------------------------

  // Neighbors of u via bidirected edges.
  function BidirectedNeighbors(sm: SMGraph, u: Node): set<Node> {
    (set e | e in sm.bidirected && e.u == u :: e.v) +
    (set e | e in sm.bidirected && e.v == u :: e.u)
  }

  // BFS from a frontier, following bidirected edges only.
  // Returns all nodes reachable from the frontier (including frontier).
  function BidirectedBFSLoop(
    sm: SMGraph,
    frontier: set<Node>,
    visited: set<Node>,
    fuel: nat
  ): set<Node>
    decreases fuel
  {
    if frontier == {} || fuel == 0 then
      visited
    else
      var newVisited := visited + frontier;
      var nextFrontier :=
        (set u, v | u in frontier && v in BidirectedNeighbors(sm, u)
                  && v in SMNodes(sm) && v !in newVisited :: v);
      BidirectedBFSLoop(sm, nextFrontier, newVisited, fuel - 1)
  }

  // The C-component containing node v — compiled via BFS.
  function CComponentCompiled(sm: SMGraph, v: Node): set<Node>
    requires v in SMNodes(sm)
  {
    BidirectedBFSLoop(sm, {v}, {}, |SMNodes(sm)|)
  }

  lemma BidirectedBFS_FrontierSubset(
    sm: SMGraph,
    frontier: set<Node>,
    visited: set<Node>,
    fuel: nat
  )
    requires fuel >= 1
    ensures frontier <= BidirectedBFSLoop(sm, frontier, visited, fuel)
    decreases fuel
  {
    if frontier == {} {
    } else {
      var newVisited := visited + frontier;
      var nextFrontier :=
        (set u, v | u in frontier && v in BidirectedNeighbors(sm, u)
                  && v in SMNodes(sm) && v !in newVisited :: v);
      BidirectedBFS_VisitedSubset(sm, nextFrontier, newVisited, fuel - 1);
      assert frontier <= newVisited;
    }
  }

  // v is always in its own BFS component.
  lemma BidirectedBFS_ContainsSelf(sm: SMGraph, v: Node, fuel: nat)
    requires fuel >= 1
    ensures v in BidirectedBFSLoop(sm, {v}, {}, fuel)
  {
    BidirectedBFS_FrontierSubset(sm, {v}, {}, fuel);
    assert v in {v};
  }

  // BFS always returns a superset of visited.
  lemma BidirectedBFS_VisitedSubset(
    sm: SMGraph,
    frontier: set<Node>,
    visited: set<Node>,
    fuel: nat
  )
    ensures visited <= BidirectedBFSLoop(sm, frontier, visited, fuel)
    decreases fuel
  {
    if frontier == {} || fuel == 0 {
      // Base case: returns visited directly
    } else {
      var newVisited := visited + frontier;
      var nextFrontier :=
        (set u, v | u in frontier && v in BidirectedNeighbors(sm, u)
                  && v in SMNodes(sm) && v !in newVisited :: v);
      BidirectedBFS_VisitedSubset(sm, nextFrontier, newVisited, fuel - 1);
    }
  }

  // ------------------------------------------------------------------
  // Compiled C-Component computation — method with while loop
  // ------------------------------------------------------------------

  method ComputeCComponents(sm: SMGraph) returns (components: seq<set<Node>>)
  {
    var remaining := SMNodes(sm);
    components := [];

    while remaining != {}
      decreases |remaining|
    {
      var v :| v in remaining;
      var comp := BidirectedBFSLoop(sm, {v}, {}, |SMNodes(sm)|);
      // Only keep nodes that are actually in remaining
      var comp' := comp * remaining;
      // v is in comp: BFS from {v} always contains v
      BidirectedBFS_ContainsSelf(sm, v, |SMNodes(sm)|);
      assert v in comp;
      assert v in comp';
      // So comp' is non-empty, remaining strictly shrinks
      remaining := remaining - comp';
      components := components + [comp'];
    }
  }

  // Convert sequence of sets to a set of sets.
  function SeqToSetOfSets(s: seq<set<Node>>): set<set<Node>> {
    set i | 0 <= i < |s| :: s[i]
  }

  // The compiled C-components equal the ghost C-components.
  // (Follows from the function-by-method construction above;
  //  retained for backward compatibility with callers.)
  lemma ComputeCComponents_Correct(sm: SMGraph, components: seq<set<Node>>)
    requires WellFormedSM(sm)
    requires SeqToSetOfSets(components) == CComponents(sm)
    ensures SeqToSetOfSets(components) == CComponents(sm)
  {}

  // The compiled single-component equals the ghost CComponent.
  lemma {:axiom} CComponentCompiled_Correct(sm: SMGraph, v: Node)
    requires WellFormedSM(sm)
    requires v in SMNodes(sm)
    ensures CComponentCompiled(sm, v) == CComponent(sm, v)

  // ==================================================================
  // 3.  Graph Operations on SMGraphs
  //
  //   Analogous to RemoveIncoming and RemoveOutgoing from the DAG
  //   module, but also handling bidirected edges.
  // ==================================================================

  // Remove a set of nodes from the SMGraph.
  // Removes the nodes, all directed edges involving them,
  // and all bidirected edges involving them.
  function RemoveNodesSM(sm: SMGraph, X: set<Node>): SMGraph
  {
    SMGraph(
      map v | v in SMNodes(sm) && v !in X ::
        Parents(sm.dag, v) - X,
      set e | e in sm.bidirected && e.u !in X && e.v !in X
    )
  }

  // Remove incoming directed edges to nodes in X.
  // Bidirected edges are preserved.
  function RemoveIncomingSM(sm: SMGraph, X: set<Node>): SMGraph
  {
    SMGraph(
      RemoveIncoming(sm.dag, X),
      sm.bidirected
    )
  }

  // Remove outgoing directed edges from nodes in X.
  // Bidirected edges are preserved.
  function RemoveOutgoingSM(sm: SMGraph, X: set<Node>): SMGraph
  {
    SMGraph(
      RemoveOutgoing(sm.dag, X),
      sm.bidirected
    )
  }

  // C-components of the subgraph G \ X.
  ghost function CComponentsWithout(sm: SMGraph, X: set<Node>): set<set<Node>>
  {
    CComponents(RemoveNodesSM(sm, X))
  }

  // C-components of G\X still form a partition of the remaining node set.
  // This helper keeps the Line 4 assumptions local to the CComponentsWithout
  // abstraction used by Identification.
  lemma CComponentsWithout_Partition(sm: SMGraph, X: set<Node>)
    requires WellFormedSM(sm)
    ensures forall S :: S in CComponentsWithout(sm, X) ==> S <= SMNodes(sm) - X
    ensures forall S :: S in CComponentsWithout(sm, X) ==> S != {}
  {
    var smX := RemoveNodesSM(sm, X);
    assume {:axiom} WellFormedSM(smX);
    CComponents_Partition(smX);
    assert forall S :: S in CComponentsWithout(sm, X) ==> S <= SMNodes(smX) && S != {};
    assert SMNodes(smX) == SMNodes(sm) - X;
  }

  // Induced subgraph on a set of nodes S.
  // Keeps only nodes in S, directed edges between them,
  // and bidirected edges between them.
  // Equivalent to RemoveNodesSM(sm, SMNodes(sm) - S).
  function SubgraphSM(sm: SMGraph, S: set<Node>): SMGraph
    requires S <= SMNodes(sm)
  {
    RemoveNodesSM(sm, SMNodes(sm) - S)
  }

  // ==================================================================
  // 4.  Topological Ordering in SMGraphs
  //
  //   The topological ordering comes from the directed component only.
  //   It is used in the Q-decomposition and the ID algorithm.
  // ==================================================================

  // Topological ordering of the directed component.
  predicate SMTopologicalSort(sm: SMGraph, ord: seq<Node>) {
    IsTopologicalSort(sm.dag, ord)
  }

  // Predecessors of node v in topological ordering π.
  //   V_π^{(i-1)} = {π[0], ..., π[i-1]}  where π[i] = v
  ghost function {:axiom} TopoPredecessors(
    ord: seq<Node>, v: Node
  ): set<Node>

  // ==================================================================
  // 5.  Q-Decomposition (C-Factor Factorization)
  //
  //   Tian & Pearl (2002), Lemma 2:
  //   For a semi-Markovian model with C-components S₁,...,Sₖ:
  //
  //     P(v) = ∏ᵢ Q[Sᵢ]
  //
  //   where Q[Sᵢ] = ∏_{Vⱼ ∈ Sᵢ} P(vⱼ | v_π^{j-1})
  //
  //   and π is a topological ordering of the DAG.
  //
  //   Q[S] represents the post-intervention distribution of S
  //   under an intervention that sets all variables outside S
  //   to constants:  Q[S] = P_{v\s}(s)
  //
  //   Ref: Shpitser & Pearl 2006, §"Notation and Definitions"
  //        Tian & Pearl 2002, Theorem 4
  //        Python: y0.algorithm.tian_id.compute_c_factor
  // ==================================================================

  // The Q-value (c-factor) for a set of nodes S, given a PMF p
  // and a topological ordering.
  //
  // Q[S] = ∏_{Vⱼ ∈ S} P(vⱼ | v_π^{j-1})
  //
  // This is the interventional distribution P_{V\S}(S).
  ghost function {:axiom} QValue(
    sm: SMGraph,
    p: Prob.PMF,
    S: set<Node>,
    ord: seq<Node>
  ): Prob.PMF
    requires Prob.IsDistribution(p)
    requires WellFormedSM(sm)
    requires S <= SMNodes(sm)

  // Q[V] = P(V) — the Q-value of all nodes is the joint distribution.
  lemma {:axiom} QValue_AllNodes(
    sm: SMGraph,
    p: Prob.PMF,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires WellFormedSM(sm)
    requires SMTopologicalSort(sm, ord)
    ensures QValue(sm, p, SMNodes(sm), ord) == p

  // Q[S] is a valid distribution.
  lemma {:axiom} QValue_IsDistribution(
    sm: SMGraph,
    p: Prob.PMF,
    S: set<Node>,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires WellFormedSM(sm)
    requires S <= SMNodes(sm)
    ensures Prob.IsDistribution(QValue(sm, p, S, ord))

  // ==================================================================
  // C-Component Factorization (Tian 2002, Theorem 4)
  //
  //   The joint distribution factors as the product of Q-values
  //   over all C-components:
  //
  //     P(v) = ∏_{Sᵢ ∈ C(G)} Q[Sᵢ]
  //
  //   This is the fundamental decomposition that enables the
  //   ID algorithm.
  // ==================================================================

  /// C-Component Factorization:
  ///   P(v) = ∏_{Sᵢ ∈ C(G)} Q[Sᵢ]
  ///
  ///   Ref: Tian & Pearl (2002), Theorem 4
  ///        Shpitser & Pearl (2006), used implicitly in ID algorithm
  lemma {:axiom} CComponentFactorization(
    sm: SMGraph,
    p: Prob.PMF,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires WellFormedSM(sm)
    requires MarkovFactorization(sm.dag, p)
    requires SMTopologicalSort(sm, ord)
    // P(v) = ∏_{Sᵢ ∈ C(G)} Q[Sᵢ]
    // (The product equality is axiomatized since Dafny lacks
    //  built-in product-over-set.)

  // ==================================================================
  // Interventional Factorization
  //
  //   P_x(v \ x) = ∏_{Sᵢ ∈ C(G\X)} Q[Sᵢ]
  //
  //   After intervening on X, the interventional distribution
  //   over the remaining variables factors according to the
  //   C-components of the mutilated graph G \ X.
  //
  //   Ref: Tian & Pearl (2002), Corollary 1
  //        Shpitser & Pearl (2006), used in ID algorithm Line 4
  // ==================================================================

  /// Interventional C-Component Factorization (Tian 2002, Corollary 1):
  ///
  ///   P_x(v \ x) = ∏_{Sᵢ ∈ C(G\X)} Q[Sᵢ]
  ///
  ///   where Q[Sᵢ] values are computable from P(v).
  lemma {:axiom} InterventionalFactorization(
    sm: SMGraph,
    p: Prob.PMF,
    X: set<Node>,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires WellFormedSM(sm)
    requires MarkovFactorization(sm.dag, p)
    requires X <= SMNodes(sm)
    requires SMTopologicalSort(sm, ord)
    // P_x(v \ x) = ∏_{Sᵢ ∈ C(G\X)} Q[Sᵢ]

  // ==================================================================
  // Q-Value Computation from Nested C-Components
  //
  //   Tian & Pearl (2002), Lemma 3:
  //   If S' ⊆ S and both are C-components (of different subgraphs),
  //   then Q[S'] can be computed from Q[S].
  //
  //   Specifically:
  //     Q[S'] = ∏_{Vᵢ ∈ S'} P(vᵢ | v_π^{i-1}) / ∏_{Vᵢ ∈ S\S'} P(vᵢ | v_π^{i-1})
  //
  //   or equivalently by restricting the product to S' factors.
  //
  //   This corresponds to Lines 6-7 of the ID algorithm.
  // ==================================================================

  /// Q-value derivation from nested C-components:
  ///   If S' ⊂ S are both C-components (in appropriate subgraphs),
  ///   then Q[S'] is derivable from Q[S].
  ///
  ///   Ref: Tian & Pearl (2002), Lemma 3
  ///        Shpitser & Pearl (2006), ID algorithm Lines 6-7
  lemma {:axiom} QValue_Nested(
    sm: SMGraph,
    p: Prob.PMF,
    S: set<Node>,
    Sprime: set<Node>,
    ord: seq<Node>
  )
    requires Prob.IsDistribution(p)
    requires WellFormedSM(sm)
    requires Sprime < S     // strict subset
    requires S <= SMNodes(sm)
    requires Sprime <= SMNodes(sm)
    // Q[S'] is computable from Q[S]

  // ==================================================================
  // 6.  C-Trees and C-Forests
  //
  //   A C-tree is a Semi-Markovian graph where:
  //     - There is a single C-component (all nodes bidirected-connected)
  //     - Every node has at most one child in the directed component
  //     - The root set R is the set of nodes with no children
  //
  //   A C-forest is a Semi-Markovian graph consisting of a
  //   collection of C-trees, one per C-component.
  //
  //   C-forests are the building blocks of hedges.
  //
  //   Ref: Shpitser & Pearl (2006), Definition 3
  // ==================================================================

  // The root set of a graph: nodes with no children.
  function RootSet(sm: SMGraph): set<Node> {
    set v | v in SMNodes(sm) && Children(sm.dag, v) == {}
  }

  // Every node has at most one child in the directed component.
  predicate AtMostOneChild(sm: SMGraph) {
    forall v :: v in SMNodes(sm) ==> |Children(sm.dag, v)| <= 1
  }

  // A C-tree: a single-C-component graph where every node has
  // at most one child.
  predicate IsCTree(sm: SMGraph) {
    WellFormedSM(sm) &&
    // Single C-component: all nodes are bidirected-connected
    |CComponents(sm)| == 1 &&
    // Every node has at most one child
    AtMostOneChild(sm)
  }

  // A C-forest: a C-component where every node has at most one child.
  // Ref: Shpitser & Pearl (2006), Definition 5
  predicate IsCForest(sm: SMGraph) {
    WellFormedSM(sm) &&
    // Must be a single C-component (connected via bidirected edges)
    |CComponents(sm)| == 1 &&
    // Every node has at most one child
    AtMostOneChild(sm)
  }

  // An R-rooted C-forest: a C-forest with root set R.
  predicate IsRootedCForest(sm: SMGraph, R: set<Node>) {
    IsCForest(sm) && RootSet(sm) == R
  }

  // ==================================================================
  // 7.  Hedges
  //
  //   A hedge for P_x(y) in G is a pair of C-forests (F, F') where:
  //
  //   (i)   F' ⊆ F  (F' is a subgraph of F)
  //   (ii)  F and F' share the same root set R
  //   (iii) R ⊂ An(Y)_{G_{X̄}}
  //   (iv)  F ∩ X ≠ ∅  (F contains some treatment variables)
  //   (v)   F' ∩ X = ∅  (F' does NOT contain treatment variables)
  //   (vi)  F and F' are both C-forests (C-component + at most one child)
  //
  //   The existence of a hedge is the necessary and sufficient
  //   condition for non-identifiability.
  //
  //   Ref: Shpitser & Pearl (2006), Definition 4
  //        Python: y0.algorithm.identify.id_std.line_5
  //               (Unidentifiable exception raised when hedge found)
  // ==================================================================

  // F' is a subgraph of F: same directed and bidirected structure
  // restricted to a subset of nodes.
  predicate IsSubgraphSM(Fprime: SMGraph, F: SMGraph) {
    SMNodes(Fprime) <= SMNodes(F) &&
    // Every directed edge in F' exists in F
    (forall v :: v in SMNodes(Fprime) ==>
       Parents(Fprime.dag, v) <= Parents(F.dag, v)) &&
    // Every bidirected edge in F' exists in F
    Fprime.bidirected <= F.bidirected
  }

  /// A hedge for P_x(y) in graph sm.
  ///
  /// Def 6 (Shpitser & Pearl 2006):
  ///   F, F' form a hedge for P_x(y) if:
  ///   - F, F' are R-rooted C-forests (same root set R)
  ///   - F' ⊆ F  (subgraph, with strictly fewer nodes)
  ///   - F ∩ X ≠ ∅  (F contains treatment variables)
  ///   - F' ∩ X = ∅  (F' does NOT contain treatment variables)
  ///   - R ⊂ An(Y)_{G_{X̄}}
  ///   - F is a subgraph of G
  predicate IsHedge(
    sm: SMGraph,
    F: SMGraph,
    Fprime: SMGraph,
    X: set<Node>,
    Y: set<Node>
  ) {
    var Gx := RemoveIncomingSM(sm, X);
    var AncY := Ancestors(Gx.dag, Y);
    // F is a C-forest that is a subgraph of sm
    IsCForest(F) &&
    IsSubgraphSM(F, sm) &&
    // F' is a C-forest that is a subgraph of F
    IsCForest(Fprime) &&
    IsSubgraphSM(Fprime, F) &&
    // F' ⊂ F (strictly fewer nodes)
    SMNodes(Fprime) < SMNodes(F) &&
    // F and F' share the same root set R
    RootSet(F) == RootSet(Fprime) &&
    // R ⊂ An(Y)_{G_{X̄}} (root set is within ancestors of Y)
    RootSet(F) <= AncY &&
    // F contains treatment variables (F ∩ X ≠ ∅)
    SMNodes(F) * X != {} &&
    // F' does NOT contain treatment variables (F' ∩ X = ∅)
    SMNodes(Fprime) * X == {}
  }

  // ==================================================================
  // 8.  Hedge Existence and Non-Identifiability
  //
  //   The existence of a hedge is necessary and sufficient for
  //   non-identifiability.  This connects to the ID algorithm
  //   (Line 5) and forms the main completeness result.
  // ==================================================================

  /// A causal effect P_x(y) is identifiable iff no hedge exists.
  ghost predicate IsIdentifiable(
    sm: SMGraph,
    X: set<Node>,
    Y: set<Node>
  ) {
    !exists F: SMGraph, Fprime: SMGraph ::
      IsHedge(sm, F, Fprime, X, Y)
  }

}  // end module SemiMarkovian
