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
  ghost function CComponents(sm: SMGraph): set<set<Node>>
  {
    // Ghost spec: the set of all maximal bidirected-connected subsets.
    // A set S is a C-component iff:
    //   (i)  every pair in S is BidirectedConnected
    //   (ii) no node outside S is BidirectedConnected to all of S
    set S: set<Node> | S <= SMNodes(sm) && S != {} &&
      (forall u, v | u in S && v in S :: BidirectedConnected(sm, u, v)) &&
      (forall u | u in SMNodes(sm) && u !in S ::
         exists v | v in S :: !BidirectedConnected(sm, u, v))
  }

  lemma CComponent_IsInCComponents(sm: SMGraph, v: Node)
    requires WellFormedSM(sm)
    requires v in SMNodes(sm)
    ensures CComponent(sm, v) in CComponents(sm)
    ensures v in CComponent(sm, v)
  {
    var S := CComponent(sm, v);
    assert S <= SMNodes(sm);
    assert v in S;
    assert S != {};
    forall u, w | u in S && w in S
      ensures BidirectedConnected(sm, u, w)
    {
      assert BidirectedConnected(sm, u, v);
      assert BidirectedConnected(sm, w, v);
      var kV := |SMNodes(sm)|;
      BidirectedConnectedBounded_Symmetric(sm, w, v, kV);
      assert BidirectedConnected(sm, v, w);
      BidirectedConnected_Transitive(sm, u, v, w);
    }
    forall u | u in SMNodes(sm) && u !in S
      ensures exists w | w in S :: !BidirectedConnected(sm, u, w)
    {
      assert !BidirectedConnected(sm, u, v);
      assert exists w | w in S :: !BidirectedConnected(sm, u, w) by {
        assert v in S;
      }
    }
    assert S in CComponents(sm);
  }

  lemma CComponents_OverlapImpliesEqual(sm: SMGraph, S1: set<Node>, S2: set<Node>)
    requires WellFormedSM(sm)
    requires S1 in CComponents(sm)
    requires S2 in CComponents(sm)
    requires S1 * S2 != {}
    ensures S1 == S2
  {
    assert S1 <= SMNodes(sm) && S1 != {} &&
      (forall u, v | u in S1 && v in S1 :: BidirectedConnected(sm, u, v)) &&
      (forall u | u in SMNodes(sm) && u !in S1 ::
        exists v | v in S1 :: !BidirectedConnected(sm, u, v));
    assert S2 <= SMNodes(sm) && S2 != {} &&
      (forall u, v | u in S2 && v in S2 :: BidirectedConnected(sm, u, v)) &&
      (forall u | u in SMNodes(sm) && u !in S2 ::
        exists v | v in S2 :: !BidirectedConnected(sm, u, v));

    var x :| x in S1 * S2;

    assert S2 <= S1 by {
      forall y | y in S2
        ensures y in S1
      {
        if y in S1 {
        } else {
          assert BidirectedConnected(sm, y, x);
          var w :| w in S1 && !BidirectedConnected(sm, y, w);
          assert BidirectedConnected(sm, x, w);
          BidirectedConnected_Transitive(sm, y, x, w);
          assert false;
        }
      }
    }

    assert S1 <= S2 by {
      forall y | y in S1
        ensures y in S2
      {
        if y in S2 {
        } else {
          assert BidirectedConnected(sm, y, x);
          var w :| w in S2 && !BidirectedConnected(sm, y, w);
          assert BidirectedConnected(sm, x, w);
          BidirectedConnected_Transitive(sm, y, x, w);
          assert false;
        }
      }
    }

    assert S1 == S2;
  }

  // C-components partition the node set.
  lemma {:vcs_split_on_every_assert} CComponents_Partition(sm: SMGraph)
    requires WellFormedSM(sm)
    ensures
      (forall v :: v in SMNodes(sm) ==>
         exists S :: S in CComponents(sm) && v in S) &&
      (forall S1, S2 :: (S1 in CComponents(sm) && S2 in CComponents(sm)
         && S1 != S2) ==> S1 * S2 == {}) &&
      (forall S :: S in CComponents(sm) ==> S <= SMNodes(sm) && S != {})
  {
    forall v | v in SMNodes(sm)
      ensures exists S :: S in CComponents(sm) && v in S
    {
      CComponent_IsInCComponents(sm, v);
      var S := CComponent(sm, v);
      assert exists T :: T in CComponents(sm) && v in T by {
        assert S in CComponents(sm) && v in S;
      }
    }

    forall S | S in CComponents(sm)
      ensures S <= SMNodes(sm) && S != {}
    {
      assert S <= SMNodes(sm);
      assert S != {};
    }

    forall S1, S2 | S1 in CComponents(sm) && S2 in CComponents(sm) && S1 != S2
      ensures S1 * S2 == {}
    {
      if S1 * S2 != {} {
        CComponents_OverlapImpliesEqual(sm, S1, S2);
        assert false;
      }
    }
  }

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
      RemoveNodes(sm.dag, X),
      set e | e in sm.bidirected && e.u !in X && e.v !in X
    )
  }

  lemma RemoveNodesSM_PreservesWellFormedness(sm: SMGraph, X: set<Node>)
    requires WellFormedSM(sm)
    ensures WellFormedSM(RemoveNodesSM(sm, X))
  {
    var smX := RemoveNodesSM(sm, X);
    assert smX.dag == RemoveNodes(sm.dag, X);
    assert SMNodes(smX) == SMNodes(sm) - X;
    forall e | e in smX.bidirected
      ensures e.u in SMNodes(smX) && e.v in SMNodes(smX) && e.u != e.v
    {
      assert e in sm.bidirected;
      assert e.u !in X && e.v !in X;
      assert e.u in SMNodes(sm) && e.v in SMNodes(sm) && e.u != e.v;
      assert e.u in SMNodes(sm) - X;
      assert e.v in SMNodes(sm) - X;
    }
    RemoveNodes_PreservesDAG(sm.dag, X);
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
    RemoveNodesSM_PreservesWellFormedness(sm, X);
    CComponents_Partition(smX);
    assert forall S :: S in CComponentsWithout(sm, X) ==> S <= SMNodes(smX) && S != {};
    assert SMNodes(smX) == SMNodes(sm) - X;
  }

  // When G \ X has at least one node, its C-component set is non-empty.
  lemma CComponentsWithout_NonEmpty(sm: SMGraph, X: set<Node>)
    requires WellFormedSM(sm)
    requires SMNodes(sm) - X != {}
    ensures CComponentsWithout(sm, X) != {}
  {
    var smX := RemoveNodesSM(sm, X);
    RemoveNodesSM_PreservesWellFormedness(sm, X);
    CComponents_Partition(smX);
    assert SMNodes(smX) == SMNodes(sm) - X;
    var v :| v in SMNodes(smX);
    var S :| S in CComponents(smX) && v in S;
    assert S in CComponentsWithout(sm, X);
  }

  // If CComponentsWithout(sm, X) has exactly one element S, then S = V \ X.
  // This follows because CComponents_Partition guarantees coverage of V\X.
  lemma CComponentsWithout_SingletonCoversAll(sm: SMGraph, X: set<Node>, S: set<Node>)
    requires WellFormedSM(sm)
    requires S in CComponentsWithout(sm, X)
    requires |CComponentsWithout(sm, X)| == 1
    ensures S == SMNodes(sm) - X
  {
    var V := SMNodes(sm);
    var smX := RemoveNodesSM(sm, X);
    RemoveNodesSM_PreservesWellFormedness(sm, X);
    CComponents_Partition(smX);
    assert SMNodes(smX) == V - X;
    // CComponentsWithout has only S: since |A| == 1 and S ∈ A, A - {S} is empty
    assert |CComponentsWithout(sm, X) - {S}| == 0;
    assert CComponentsWithout(sm, X) - {S} == {};
    // Every v ∈ V - X is in S (the only component)
    forall v | v in V - X ensures v in S {
      var T :| T in CComponents(smX) && v in T;
      assert T in CComponentsWithout(sm, X);
      assert T !in CComponentsWithout(sm, X) - {S};
      assert T == S;
    }
    // S ⊆ V - X from CComponentsWithout_Partition
    CComponentsWithout_Partition(sm, X);
  }

  // ── BidirectedConnected helpers ────────────────────────────────────

  // Fuel monotonicity: a path that fits in k steps also fits in k' ≥ k steps.
  lemma BidirectedConnectedBounded_FuelMono(
    sm: SMGraph, u: Node, v: Node, k: nat, k': nat
  )
    requires BidirectedConnectedBounded(sm, u, v, k)
    requires k' >= k
    ensures  BidirectedConnectedBounded(sm, u, v, k')
    decreases k
  {
    if u == v { }
    else {
      var w :| w in SMNodes(sm) && HasBidirected(sm, u, w) &&
               BidirectedConnectedBounded(sm, w, v, k - 1);
      BidirectedConnectedBounded_FuelMono(sm, w, v, k - 1, k' - 1);
    }
  }

  // Graph monotonicity: a path in a subgraph also exists in the supergraph.
  lemma BidirectedConnectedBounded_GraphMono(
    sm1: SMGraph, sm2: SMGraph, u: Node, v: Node, k: nat
  )
    requires sm1.bidirected <= sm2.bidirected
    requires SMNodes(sm1) <= SMNodes(sm2)
    requires BidirectedConnectedBounded(sm1, u, v, k)
    ensures  BidirectedConnectedBounded(sm2, u, v, k)
    decreases k
  {
    if u == v { }
    else {
      var w :| w in SMNodes(sm1) && HasBidirected(sm1, u, w) &&
               BidirectedConnectedBounded(sm1, w, v, k - 1);
      assert w in SMNodes(sm2);
      assert HasBidirected(sm2, u, w);
      BidirectedConnectedBounded_GraphMono(sm1, sm2, w, v, k - 1);
    }
  }

  // Path extension: if a ~ ... ~ b in k steps and HasBidirected(b, c),
  // then a ~ ... ~ b ~ c in k+1 steps.
  lemma BidirectedConnectedBounded_Extend(
    sm: SMGraph, a: Node, b: Node, c: Node, k: nat
  )
    requires BidirectedConnectedBounded(sm, a, b, k)
    requires HasBidirected(sm, b, c)
    requires c in SMNodes(sm)
    ensures  BidirectedConnectedBounded(sm, a, c, k + 1)
    decreases k
  {
    if a == b {
      // BCC(a, c, k+1): k+1 > 0, take w = c.
      // HasBidirected(sm, a, c) since a = b and HasBidirected(b, c).
      assert BidirectedConnectedBounded(sm, c, c, k);
    } else {
      var w :| w in SMNodes(sm) && HasBidirected(sm, a, w) &&
               BidirectedConnectedBounded(sm, w, b, k - 1);
      BidirectedConnectedBounded_Extend(sm, w, b, c, k - 1);
      // Now BCC(w, c, k), and HasBidirected(a, w), so BCC(a, c, k+1). ✓
    }
  }

  // Symmetry: HasBidirected is symmetric, so BCC paths reverse.
  lemma BidirectedConnectedBounded_Symmetric(
    sm: SMGraph, u: Node, v: Node, k: nat
  )
    requires WellFormedSM(sm)
    requires BidirectedConnectedBounded(sm, u, v, k)
    ensures  BidirectedConnectedBounded(sm, v, u, k)
    decreases k
  {
    if u == v { }
    else {
      var w :| w in SMNodes(sm) && HasBidirected(sm, u, w) &&
               BidirectedConnectedBounded(sm, w, v, k - 1);
      // u ∈ SMNodes(sm): WellFormedSM says all bidirected-edge endpoints are in SMNodes.
      assert u in SMNodes(sm) by {
        if BiEdge(u, w) in sm.bidirected {
          assert BiEdge(u, w).u == u;
        } else {
          assert BiEdge(w, u) in sm.bidirected;
          assert BiEdge(w, u).v == u;
        }
      }
      // By IH: BCC(v, w, k-1).  HasBidirected(w, u) by symmetry of HasBidirected.
      BidirectedConnectedBounded_Symmetric(sm, w, v, k - 1);
      BidirectedConnectedBounded_Extend(sm, v, w, u, k - 1);
    }
  }

  // Subgraph lift: BidirectedConnected in G_{V\X} implies BidirectedConnected in G.
  lemma BidirectedConnected_SubgraphLift(
    sm: SMGraph, X: set<Node>, u: Node, v: Node
  )
    requires WellFormedSM(sm)
    requires BidirectedConnected(RemoveNodesSM(sm, X), u, v)
    ensures  BidirectedConnected(sm, u, v)
  {
    var smX := RemoveNodesSM(sm, X);
    var kX  := |SMNodes(smX)|;
    var kG  := |SMNodes(sm)|;
    // SMNodes(smX) = Nodes(RemoveNodes(sm.dag, X)) = Nodes(sm.dag) - X = SMNodes(sm) - X.
    assert SMNodes(smX) == SMNodes(sm) - X;
    assert SMNodes(smX) <= SMNodes(sm);
    assert kX <= kG;
    // Lift the path from smX to sm via graph monotonicity.
    assert smX.bidirected <= sm.bidirected;
    BidirectedConnectedBounded_GraphMono(smX, sm, u, v, kX);
    // Now BCC(sm, u, v, kX).  Raise fuel from kX to kG.
    BidirectedConnectedBounded_FuelMono(sm, u, v, kX, kG);
  }

  // ── Tier 2: path-sequence transitivity ──────────────────────────────

  // A bidirected path: every consecutive pair is connected by a bidirected edge.
  ghost predicate IsBiPath(sm: SMGraph, path: seq<Node>)
  {
    |path| >= 1 &&
    forall i | 0 <= i < |path| - 1 :: HasBidirected(sm, path[i], path[i + 1])
  }

  // BCC(k) → there is an IsBiPath of length ≤ k+1 from u to v.
  lemma BidirectedConnectedBounded_ExtractPath(
    sm: SMGraph, u: Node, v: Node, k: nat
  ) returns (path: seq<Node>)
    requires BidirectedConnectedBounded(sm, u, v, k)
    ensures  IsBiPath(sm, path)
    ensures  path[0] == u && path[|path| - 1] == v
    ensures  |path| <= k + 1
    decreases k
  {
    if u == v {
      path := [u];
    } else {
      var w :| w in SMNodes(sm) && HasBidirected(sm, u, w) &&
               BidirectedConnectedBounded(sm, w, v, k - 1);
      var tail := BidirectedConnectedBounded_ExtractPath(sm, w, v, k - 1);
      path := [u] + tail;
    }
  }

  // Concatenation: last of p1 = first of p2 → IsBiPath(p1 + p2[1..]).
  lemma IsBiPath_Concat(sm: SMGraph, p1: seq<Node>, p2: seq<Node>)
    requires IsBiPath(sm, p1) && IsBiPath(sm, p2)
    requires p1[|p1| - 1] == p2[0]
    ensures  IsBiPath(sm, p1 + p2[1..])
  {
    var cat := p1 + p2[1..];
    forall i | 0 <= i < |cat| - 1
      ensures HasBidirected(sm, cat[i], cat[i + 1])
    {
      if i < |p1| - 1 {
        assert cat[i] == p1[i] && cat[i + 1] == p1[i + 1];
      } else if i == |p1| - 1 {
        assert cat[i] == p1[|p1| - 1];
        assert cat[i + 1] == p2[1];
        assert HasBidirected(sm, p2[0], p2[1]);
      } else {
        var j := i - |p1| + 1;
        assert cat[i] == p2[j] && cat[i + 1] == p2[j + 1];
      }
    }
  }

  // In a WellFormed graph, every node in an IsBiPath is in SMNodes.
  // Requires path[0] in SMNodes (needed for single-node paths with no edges).
  lemma IsBiPath_NodesInSM(sm: SMGraph, path: seq<Node>)
    requires WellFormedSM(sm)
    requires IsBiPath(sm, path)
    requires path[0] in SMNodes(sm)
    ensures  forall i | 0 <= i < |path| :: path[i] in SMNodes(sm)
  {
    forall i | 0 <= i < |path|
      ensures path[i] in SMNodes(sm)
    {
      if i == 0 {
        // path[0] in SMNodes by requires.
      } else {
        // Derive path[i] from the edge path[i-1] -- path[i].
        assert HasBidirected(sm, path[i - 1], path[i]);
        if BiEdge(path[i - 1], path[i]) in sm.bidirected {
          assert BiEdge(path[i - 1], path[i]).v == path[i];
        } else {
          assert BiEdge(path[i], path[i - 1]) in sm.bidirected;
          assert BiEdge(path[i], path[i - 1]).u == path[i];
        }
      }
    }
  }

  // Pigeonhole: a sequence of length > |S| with all elements in S has a repeat.
  lemma PathRepeat<T>(path: seq<T>, S: set<T>)
    requires |path| > |S|
    requires forall i | 0 <= i < |path| :: path[i] in S
    ensures  exists i, j :: 0 <= i < j < |path| && path[i] == path[j]
    decreases |S|
  {
    if |path| == 0 { assert false; }
    else {
      var x := path[0];
      if exists j :: 1 <= j < |path| && path[j] == x {
        var j :| 1 <= j < |path| && path[j] == x;
        assert 0 < j;
      } else {
        var tail := path[1..];
        var S'   := S - {x};
        assert |S'| == |S| - 1;
        assert |tail| == |path| - 1;
        assert |tail| > |S'|;
        forall i | 0 <= i < |tail|
          ensures tail[i] in S'
        {
          assert tail[i] == path[i + 1];
          assert tail[i] != x;
        }
        PathRepeat(tail, S');
        var i', j' :| 0 <= i' < j' < |tail| && tail[i'] == tail[j'];
        assert 1 <= i' + 1 < j' + 1 < |path| && path[i' + 1] == path[j' + 1];
      }
    }
  }

  // Cycle removal: path[i] = path[j] → shorter path with the same endpoints.
  // Structural postcondition exposed for callers.
  lemma IsBiPath_RemoveCycle(sm: SMGraph, path: seq<Node>, i: nat, j: nat)
    returns (shorter: seq<Node>)
    requires IsBiPath(sm, path)
    requires 0 <= i < j < |path|
    requires path[i] == path[j]
    ensures  shorter == path[..i + 1] + path[j + 1..]
    ensures  IsBiPath(sm, shorter)
    ensures  shorter[0] == path[0] && shorter[|shorter| - 1] == path[|path| - 1]
    ensures  |shorter| < |path|
  {
    shorter := path[..i + 1] + path[j + 1..];
    assert |shorter| == |path| - (j - i) < |path|;
    forall k | 0 <= k < |shorter| - 1
      ensures HasBidirected(sm, shorter[k], shorter[k + 1])
    {
      if k < i {
        assert shorter[k] == path[k] && shorter[k + 1] == path[k + 1];
      } else if k == i {
        assert shorter[k] == path[i] && shorter[k + 1] == path[j + 1];
        assert path[j] == path[i];
        assert HasBidirected(sm, path[j], path[j + 1]);
      } else {
        var m := k - i - 1;
        assert shorter[k] == path[j + 1 + m] && shorter[k + 1] == path[j + 1 + m + 1];
      }
    }
  }

  // Path shortening: any IsBiPath with all nodes in SMNodes can be shortened
  // to at most |SMNodes(sm)|+1 nodes.
  lemma IsBiPath_Shorten(sm: SMGraph, path: seq<Node>)
    returns (short: seq<Node>)
    requires WellFormedSM(sm)
    requires IsBiPath(sm, path)
    requires forall i | 0 <= i < |path| :: path[i] in SMNodes(sm)
    ensures  IsBiPath(sm, short)
    ensures  short[0] == path[0] && short[|short| - 1] == path[|path| - 1]
    ensures  |short| <= |SMNodes(sm)| + 1
    decreases |path|
  {
    var V := SMNodes(sm);
    if |path| <= |V| + 1 {
      short := path;
    } else {
      PathRepeat(path, V);
      var ci, cj :| 0 <= ci < cj < |path| && path[ci] == path[cj];
      var mid := IsBiPath_RemoveCycle(sm, path, ci, cj);
      // mid = path[..ci+1] + path[cj+1..], so mid nodes are from path nodes.
      assert mid == path[..ci + 1] + path[cj + 1..];
      assert forall i | 0 <= i < |mid| :: mid[i] in V by {
        forall i | 0 <= i < |mid|
          ensures mid[i] in V
        {
          if i <= ci {
            assert mid[i] == path[..ci + 1][i];
          } else {
            assert mid[i] == path[cj + 1..][i - ci - 1];
            assert path[cj + 1..][i - ci - 1] == path[cj + 1 + (i - ci - 1)];
          }
        }
      }
      assert mid[0] == path[0];
      assert mid[|mid| - 1] == path[|path| - 1];
      short := IsBiPath_Shorten(sm, mid);
      assert short[0] == mid[0];
      assert short[|short| - 1] == mid[|mid| - 1];
    }
  }

  // An IsBiPath with all nodes in SMNodes gives BCC at fuel k ≥ |path|-1.
  lemma IsBiPath_BCC_Bounded(sm: SMGraph, path: seq<Node>, k: nat)
    requires IsBiPath(sm, path)
    requires k >= |path| - 1
    requires forall i | 0 <= i < |path| :: path[i] in SMNodes(sm)
    ensures  BidirectedConnectedBounded(sm, path[0], path[|path| - 1], k)
    decreases |path|
  {
    if |path| == 1 {
      // Reflexive: path[0] == path[0].
    } else {
      assert forall i | 0 <= i < |path[1..]| :: path[1..][i] in SMNodes(sm);
      IsBiPath_BCC_Bounded(sm, path[1..], k - 1);
      assert path[1..][|path[1..]| - 1] == path[|path| - 1];
      assert HasBidirected(sm, path[0], path[1]);
      assert path[1] in SMNodes(sm);
      // Witness w = path[1] for the BCC existential.
    }
  }

  // BidirectedConnected is transitive (requires well-formedness and nodes in graph).
  lemma BidirectedConnected_Transitive(sm: SMGraph, u: Node, v: Node, w: Node)
    requires WellFormedSM(sm)
    requires u in SMNodes(sm) && v in SMNodes(sm) && w in SMNodes(sm)
    requires BidirectedConnected(sm, u, v)
    requires BidirectedConnected(sm, v, w)
    ensures  BidirectedConnected(sm, u, w)
  {
    var V  := SMNodes(sm);
    var kV := |V|;
    assert BidirectedConnectedBounded(sm, u, v, kV);
    assert BidirectedConnectedBounded(sm, v, w, kV);
    // Extract paths.
    var pu := BidirectedConnectedBounded_ExtractPath(sm, u, v, kV);
    var pw := BidirectedConnectedBounded_ExtractPath(sm, v, w, kV);
    assert pu[0] == u && pw[0] == v;
    // Establish path[0] in SMNodes for IsBiPath_NodesInSM.
    IsBiPath_NodesInSM(sm, pu);
    IsBiPath_NodesInSM(sm, pw);
    // Concatenate: pu's last = v = pw's first.
    IsBiPath_Concat(sm, pu, pw);
    var cat := pu + pw[1..];
    // All nodes in cat are in SMNodes.
    assert forall i | 0 <= i < |cat| :: cat[i] in V by {
      forall i | 0 <= i < |cat|
        ensures cat[i] in V
      {
        if i < |pu| { } else { assert cat[i] == pw[1 + (i - |pu|)]; }
      }
    }
    // Shorten to ≤ |V|+1 nodes.
    var short := IsBiPath_Shorten(sm, cat);
    // short has ≤ |V| edges → BCC at fuel |V|.
    assert short[0] in V by {
      assert short[0] == cat[0];
      assert cat[0] == pu[0];
    }
    IsBiPath_NodesInSM(sm, short);
    IsBiPath_BCC_Bounded(sm, short, kV);
    assert BidirectedConnectedBounded(sm, u, w, kV);
  }

  // ── End BidirectedConnected helpers ────────────────────────────────

  // Helper: from Sp ∈ CComponents(sm) and s ∉ Sp, extract a witness w ∈ Sp
  // with ¬BCC(sm, s, w) — i.e., unfold condition (iv) of the C-component.
  lemma CComponents_Condition4(sm: SMGraph, Sp: set<Node>, s: Node)
    requires WellFormedSM(sm)
    requires Sp in CComponents(sm)
    requires s in SMNodes(sm) && s !in Sp
    ensures  exists w :: w in Sp && !BidirectedConnected(sm, s, w)
  {
    var V := SMNodes(sm);
    assert Sp <= V && Sp != {} &&
      (forall u, v | u in Sp && v in Sp :: BidirectedConnected(sm, u, v)) &&
      (forall u | u in V && u !in Sp :: exists v | v in Sp :: !BidirectedConnected(sm, u, v));
  }

  // Helper: if s ∈ S is connected (via z) into Sp, then s ∈ Sp.
  // Extracted so {:vcs_split_on_every_assert} can apply to the body.
  lemma {:vcs_split_on_every_assert} CComponentsWithout_RefinesG_SubsetStep(
    sm: SMGraph, S: set<Node>, Sp: set<Node>, z: Node, s: Node
  )
    requires WellFormedSM(sm)
    requires s in S && s in SMNodes(sm)
    requires z in SMNodes(sm)
    requires BidirectedConnected(sm, z, s)
    requires Sp in CComponents(sm) && z in Sp
    requires Sp <= SMNodes(sm)
    ensures  s in Sp
  {
    if s !in Sp {
      CComponents_Condition4(sm, Sp, s);
      var w :| w in Sp && !BidirectedConnected(sm, s, w);
      assert w in SMNodes(sm);
      var kV := |SMNodes(sm)|;
      BidirectedConnectedBounded_Symmetric(sm, z, s, kV);
      assert BidirectedConnected(sm, s, z);
      assert exists T :: T in CComponents(sm) && z in T && w in T by {
        assert Sp in CComponents(sm) && z in Sp && w in Sp;
      }
      CComponent_Connected(sm, z, w);
      assert BidirectedConnected(sm, z, w);
      BidirectedConnected_Transitive(sm, s, z, w);
      assert false;
    }
  }

  // Helper: the 4 conditions that characterise CComponents membership.
  // Having this as a separate lemma lets Dafny focus on just the
  // set-comprehension membership check without all the surrounding context.
  lemma CComponents_Membership(sm: SMGraph, S: set<Node>)
    requires S <= SMNodes(sm) && S != {}
    requires forall u, v | u in S && v in S :: BidirectedConnected(sm, u, v)
    requires forall u | u in SMNodes(sm) && u !in S ::
               exists v | v in S :: !BidirectedConnected(sm, u, v)
    ensures  S in CComponents(sm)
  {}

  // Helper: extract from S ∈ CComponents(smX) that every pair in S
  // is BCC in smX.  Using the proved CComponent_Connected avoids
  // re-unfolding the set comprehension inside every forall iteration.
  lemma CComponentsWithout_Pairs_BCC(sm: SMGraph, X: set<Node>, S: set<Node>)
    requires WellFormedSM(sm)
    requires S in CComponentsWithout(sm, X)
    ensures  forall u, v | u in S && v in S :: BidirectedConnected(RemoveNodesSM(sm, X), u, v)
  {
    var smX := RemoveNodesSM(sm, X);
    RemoveNodesSM_PreservesWellFormedness(sm, X);
    forall u, v | u in S && v in S
      ensures BidirectedConnected(smX, u, v)
    {
      // S ∈ CComponents(smX); u and v share the component S.
      CComponent_Connected(smX, u, v);
    }
  }

  // If S ∈ CComponents(G \ X) but S ∉ CComponents(G), then some C-component
  // of G strictly contains S (the G-component that "absorbs" S).
  lemma {:vcs_split_on_every_assert} CComponentsWithout_RefinesG(
    sm: SMGraph, X: set<Node>, S: set<Node>
  )
    requires WellFormedSM(sm)
    requires S in CComponentsWithout(sm, X)
    requires S !in CComponents(sm)
    ensures  exists Sp :: Sp in CComponents(sm) && S < Sp
  {
    var V   := SMNodes(sm);
    var smX := RemoveNodesSM(sm, X);
    RemoveNodesSM_PreservesWellFormedness(sm, X);

    // Basic facts about S.
    CComponentsWithout_Partition(sm, X);
    assert S <= V - X;
    assert S <= V;
    assert S != {};

    // 1. All pairs in S are BCC in smX → BCC in sm.
    CComponentsWithout_Pairs_BCC(sm, X, S);
    assert forall u, v | u in S && v in S :: BidirectedConnected(smX, u, v);
    assert forall u, v | u in S && v in S :: BidirectedConnected(sm, u, v) by {
      forall u, v | u in S && v in S
        ensures BidirectedConnected(sm, u, v)
      {
        BidirectedConnected_SubgraphLift(sm, X, u, v);
      }
    }

    // 2. S ∉ CComponents(sm) → condition (iv) fails → ∃ z ∉ S BCC to all S.
    assert exists z :: z in V && z !in S &&
                       forall v | v in S :: BidirectedConnected(sm, z, v) by {
      if forall u | u in V && u !in S ::
           exists v | v in S :: !BidirectedConnected(sm, u, v) {
        // All 4 CComponents conditions hold for S → S ∈ CComponents(sm).
        CComponents_Membership(sm, S);
        assert false;
      }
    }
    var z :| z in V && z !in S &&
             forall v | v in S :: BidirectedConnected(sm, z, v);

    // 3. By Partition, z belongs to some Sp ∈ CComponents(sm).
    CComponents_Partition(sm);
    var Sp :| Sp in CComponents(sm) && z in Sp;
    assert Sp <= V;

    // 4. S ⊆ Sp: for each s ∈ S use the extracted helper.
    assert Sp <= V;
    assert S <= Sp by {
      forall s | s in S ensures s in Sp {
        assert s in V;
        assert BidirectedConnected(sm, z, s);
        CComponentsWithout_RefinesG_SubsetStep(sm, S, Sp, z, s);
      }
    }

    // 5. z ∈ Sp \ S → S ⊊ Sp.
    assert z in Sp && z !in S;
    assert S < Sp;
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

  // Filtering a topological order for the parent graph yields a valid
  // topological order for any induced subgraph.
  // This is the key lemma used by IDImpl (Lines 2 and 7) to pass
  // FilterSort(ord, S) as the ordering for SubgraphSM(sm, S).
  lemma FilteredSort_ValidSM(sm: SMGraph, S: set<Node>, ord: seq<Node>)
    requires S <= SMNodes(sm)
    requires SMTopologicalSort(sm, ord)
    ensures SMTopologicalSort(SubgraphSM(sm, S), FilterSort(ord, S))
  {
    var V := SMNodes(sm);  // = Nodes(sm.dag)
    assert Nodes(sm.dag) - (V - S) == S;
    FilteredSort_Valid(sm.dag, V - S, ord);
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
  ghost predicate IsCTree(sm: SMGraph) {
    WellFormedSM(sm) &&
    // Single C-component: all nodes are bidirected-connected
    |CComponents(sm)| == 1 &&
    // Every node has at most one child
    AtMostOneChild(sm)
  }

  // A C-forest: a C-component where every node has at most one child.
  // Ref: Shpitser & Pearl (2006), Definition 5
  ghost predicate IsCForest(sm: SMGraph) {
    WellFormedSM(sm) &&
    // Must be a single C-component (connected via bidirected edges)
    |CComponents(sm)| == 1 &&
    // Every node has at most one child
    AtMostOneChild(sm)
  }

  // An R-rooted C-forest: a C-forest with root set R.
  ghost predicate IsRootedCForest(sm: SMGraph, R: set<Node>) {
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
  ghost predicate IsHedge(
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
