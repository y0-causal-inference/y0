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

  // Helper: for every v ∈ SMNodes(sm), the BCC-class of v is in CComponents(sm).
  // The witness is S := {u ∈ SMNodes(sm) | BCC(u, v)}.
  lemma {:vcs_split_on_every_assert} CComponent_Exists(sm: SMGraph, v: Node)
    requires WellFormedSM(sm)
    requires v in SMNodes(sm)
    ensures exists S :: S in CComponents(sm) && v in S
  {
    // Witness: all nodes bidirected-connected to v.
    var S := set u | u in SMNodes(sm) && BidirectedConnected(sm, u, v);
    // (i) S ⊆ SMNodes(sm): by construction.
    assert S <= SMNodes(sm);
    // (ii) v ∈ S: BCC is reflexive (u == v base case in the predicate).
    assert BidirectedConnected(sm, v, v);
    assert v in S;
    assert S != {};
    // (iii) Pairwise BCC within S.
    assert forall u, w | u in S && w in S :: BidirectedConnected(sm, u, w) by {
      forall u, w | u in S && w in S
        ensures BidirectedConnected(sm, u, w)
      {
        // u ∈ S ⇒ BCC(u, v); w ∈ S ⇒ BCC(w, v).
        assert BidirectedConnected(sm, u, v);
        assert BidirectedConnected(sm, w, v);
        // BCC(w, v) + symmetry → BCC(v, w).
        BCC_Symmetric(sm, w, v, |SMNodes(sm)|);
        assert BidirectedConnected(sm, v, w);
        // u ∈ S ⇒ u ∈ SMNodes(sm); BCC(u,v) + BCC(v,w) → BCC(u,w).
        assert u in SMNodes(sm);
        BidirectedConnected_Transitive(sm, u, v, w);
      }
    }
    // (iv) Maximality: u ∉ S ⇒ ∃ x ∈ S :: ¬BCC(u, x).
    //      u ∉ S means ¬BCC(u, v); witness x := v ∈ S.
    assert forall u | u in SMNodes(sm) && u !in S ::
        exists x | x in S :: !BidirectedConnected(sm, u, x) by {
      forall u | u in SMNodes(sm) && u !in S
        ensures exists x | x in S :: !BidirectedConnected(sm, u, x)
      {
        assert !BidirectedConnected(sm, u, v);
        assert v in S;
      }
    }
    // All four comprehension conditions hold → S ∈ CComponents(sm).
    assert S in CComponents(sm);
  }

  // Helper: two distinct C-components are disjoint.
  // Proof by contradiction: a shared node x forces S1 = S2 via BCC transitivity.
  lemma {:vcs_split_on_every_assert} {:timeLimitMultiplier 4} CComponents_Disjoint(
    sm: SMGraph, S1: set<Node>, S2: set<Node>
  )
    requires WellFormedSM(sm)
    requires S1 in CComponents(sm)
    requires S2 in CComponents(sm)
    requires S1 != S2
    ensures S1 * S2 == {}
  {
    // Extract the four comprehension conditions for each component.
    assert S1 <= SMNodes(sm) && S1 != {} &&
      (forall a, b | a in S1 && b in S1 :: BidirectedConnected(sm, a, b)) &&
      (forall a | a in SMNodes(sm) && a !in S1 ::
         exists b | b in S1 :: !BidirectedConnected(sm, a, b));
    assert S2 <= SMNodes(sm) && S2 != {} &&
      (forall a, b | a in S2 && b in S2 :: BidirectedConnected(sm, a, b)) &&
      (forall a | a in SMNodes(sm) && a !in S2 ::
         exists b | b in S2 :: !BidirectedConnected(sm, a, b));
    if S1 * S2 != {} {
      var x :| x in S1 * S2;
      // S2 ⊆ S1: x ∈ S1 and for every u ∈ S2, BCC(x, u) from S2's pairwise condition.
      forall u | u in S2 ensures u in S1 {
        assert x in S2 && u in S2;
        assert BidirectedConnected(sm, x, u);
        assert u in SMNodes(sm);
        BCC_InSameCComponent(sm, S1, x, u);
      }
      // S1 ⊆ S2: x ∈ S2 and for every u ∈ S1, BCC(x, u) from S1's pairwise condition.
      forall u | u in S1 ensures u in S2 {
        assert x in S1 && u in S1;
        assert BidirectedConnected(sm, x, u);
        assert u in SMNodes(sm);
        BCC_InSameCComponent(sm, S2, x, u);
      }
      // S2 ⊆ S1 and S1 ⊆ S2 → S1 = S2, contradicting S1 ≠ S2.
      assert S1 == S2;
      assert false;
    }
  }

  // C-components partition the node set.
  lemma CComponents_Partition(sm: SMGraph)
    requires WellFormedSM(sm)
    ensures
      (forall v :: v in SMNodes(sm) ==>
         exists S :: S in CComponents(sm) && v in S) &&
      (forall S1, S2 :: (S1 in CComponents(sm) && S2 in CComponents(sm)
         && S1 != S2) ==> S1 * S2 == {}) &&
      (forall S :: S in CComponents(sm) ==> S <= SMNodes(sm) && S != {})
  {
    // Existence: every node is covered by its BCC-class.
    forall v | v in SMNodes(sm)
      ensures exists S :: S in CComponents(sm) && v in S
    {
      CComponent_Exists(sm, v);
    }
    // Disjointness: distinct components are disjoint (proved via helper).
    forall S1, S2 | S1 in CComponents(sm) && S2 in CComponents(sm) && S1 != S2
      ensures S1 * S2 == {}
    {
      CComponents_Disjoint(sm, S1, S2);
    }
    // Subset/non-empty: immediate from the set-comprehension definition.
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

  // 1a. BidirectedNeighbors gives HasBidirected witnesses.
  lemma BidirectedNeighbors_HasBidirected(sm: SMGraph, u: Node, v: Node)
    requires v in BidirectedNeighbors(sm, u)
    ensures HasBidirected(sm, u, v)
  {
    // By definition of BidirectedNeighbors: BiEdge(u,v) or BiEdge(v,u) is in bidirected.
  }

  // 1b. Every IsBiPath can be shortened to a NoDup (simple) path of length ≤ |SMNodes(sm)|.
  // IsBiPath_Shorten stops at ≤ |SMNodes(sm)|+1 which may still have one duplicate.
  // We keep removing cycles until the path is strictly NoDup.
  ghost predicate NoDupSeq(s: seq<Node>)
  {
    forall i, j :: 0 <= i < j < |s| ==> s[i] != s[j]
  }

  lemma {:induction false} IsBiPath_ToNoDup(sm: SMGraph, path: seq<Node>)
    requires WellFormedSM(sm)
    requires IsBiPath(sm, path)
    requires path[0] in SMNodes(sm)
    ensures exists simple ::
      IsBiPath(sm, simple) &&
      simple[0] == path[0] &&
      simple[|simple|-1] == path[|path|-1] &&
      NoDupSeq(simple) &&
      |simple| <= |SMNodes(sm)|
    decreases |path|
  {
    if NoDupSeq(path) {
      // path is already NoDup.  Its nodes are all in SMNodes(sm) (by IsBiPath_AllInSMNodes)
      // and are distinct, so |path| ≤ |SMNodes(sm)|.
      IsBiPath_AllInSMNodes(sm, path);
      assert forall i :: 0 <= i < |path| ==> path[i] in SMNodes(sm);
      // Distinct elements of SMNodes(sm) → |path| ≤ |SMNodes(sm)|.
      if |path| > |SMNodes(sm)| {
        SeqRepeat(path, SMNodes(sm));
        var ii, jj :| 0 <= ii < jj < |path| && path[ii] == path[jj];
        assert !NoDupSeq(path);  // contradiction
        assert false;
      }
    } else {
      // Has a cycle: remove it, recurse on the shorter path.
      var ii, jj :| 0 <= ii < jj < |path| && path[ii] == path[jj];
      IsBiPath_RemoveCycle(sm, path, ii, jj);
      var shorter :| IsBiPath(sm, shorter) && shorter[0] == path[0] &&
                     shorter[|shorter|-1] == path[|path|-1] && |shorter| < |path|;
      // shorter[0] == path[0] ∈ SMNodes(sm).
      IsBiPath_ToNoDup(sm, shorter);
    }
  }

  // Phase 2 — Soundness: BFS stays within CComponent(sm, root).
  lemma {:induction false} BidirectedBFSLoop_Sound(
    sm: SMGraph,
    root: Node,
    frontier: set<Node>,
    visited: set<Node>,
    fuel: nat
  )
    requires WellFormedSM(sm)
    requires root in SMNodes(sm)
    requires frontier <= CComponent(sm, root)
    requires visited <= CComponent(sm, root)
    ensures BidirectedBFSLoop(sm, frontier, visited, fuel) <= CComponent(sm, root)
    decreases fuel
  {
    if frontier == {} || fuel == 0 {
      // BFS returns visited ⊆ CComponent(sm, root). ✓
    } else {
      var newVisited := visited + frontier;
      var nextFrontier :=
        (set u, v | u in frontier && v in BidirectedNeighbors(sm, u)
                  && v in SMNodes(sm) && v !in newVisited :: v);
      // Show nextFrontier ⊆ CComponent(sm, root).
      forall x | x in nextFrontier ensures x in CComponent(sm, root) {
        var u :| u in frontier && x in BidirectedNeighbors(sm, u)
               && x in SMNodes(sm) && x !in newVisited;
        // u ∈ frontier ⊆ CComponent(sm, root) → BCC(sm, u, root).
        assert u in CComponent(sm, root);
        assert BidirectedConnected(sm, u, root);
        // HasBidirected(sm, u, x) → BCC(sm, u, x, 1) → BCC_Symmetric → BCC(sm, x, u, 1).
        BidirectedNeighbors_HasBidirected(sm, u, x);
        assert HasBidirected(sm, u, x);
        assert BidirectedConnectedBounded(sm, u, x, 1);
        BCC_Symmetric(sm, u, x, 1);
        assert BidirectedConnectedBounded(sm, x, u, 1);
        BCC_FuelMono(sm, x, u, 1, |SMNodes(sm)|);
        assert BidirectedConnected(sm, x, u);
        // BCC(sm, x, u) + BCC(sm, u, root) → BCC(sm, x, root) → x ∈ CComponent.
        BidirectedConnected_Transitive(sm, x, u, root);
        assert BidirectedConnected(sm, x, root);
        assert x in SMNodes(sm);
      }
      assert newVisited <= CComponent(sm, root);
      BidirectedBFSLoop_Sound(sm, root, nextFrontier, newVisited, fuel - 1);
    }
  }

  // Phase 3 — Completeness: a fresh NoDup path leads to its endpoint in BFS.
  lemma {:induction false} {:vcs_split_on_every_assert}
      BidirectedBFSLoop_FollowsFreshNoDupPath(
    sm: SMGraph,
    path: seq<Node>,
    frontier: set<Node>,
    visited: set<Node>,
    fuel: nat
  )
    requires WellFormedSM(sm)
    requires IsBiPath(sm, path)
    requires NoDupSeq(path)
    requires path[0] in frontier
    requires forall i :: 0 <= i < |path| ==> path[i] !in visited
    requires fuel >= |path|
    ensures path[|path| - 1] in BidirectedBFSLoop(sm, frontier, visited, fuel)
    decreases |path|
  {
    if |path| == 1 {
      // path[0] ∈ frontier; fuel ≥ 1.
      assert fuel >= 1;
      BidirectedBFS_FrontierSubset(sm, frontier, visited, fuel);
    } else if exists j :: 0 < j < |path| && path[j] in frontier {
      // A later node is already in frontier; use the shorter suffix path[j..].
      var j :| 0 < j < |path| && path[j] in frontier;
      var suffix := path[j..];
      assert IsBiPath(sm, suffix) by {
        forall i | 0 <= i < |suffix| - 1
          ensures HasBidirected(sm, suffix[i], suffix[i+1]) && suffix[i+1] in SMNodes(sm)
        {
          IsBiPath_Step(sm, path, j + i);
        }
      }
      assert NoDupSeq(suffix) by {
        forall a, b | 0 <= a < b < |suffix| ensures suffix[a] != suffix[b]
        { assert path[j+a] != path[j+b]; }
      }
      assert suffix[0] in frontier by { assert suffix[0] == path[j]; }
      assert forall i :: 0 <= i < |suffix| ==> suffix[i] !in visited by {
        forall i | 0 <= i < |suffix| ensures suffix[i] !in visited
        { assert suffix[i] == path[j + i]; }
      }
      assert fuel >= |suffix| by { assert |suffix| == |path| - j; }
      assert suffix[|suffix| - 1] == path[|path| - 1];
      BidirectedBFSLoop_FollowsFreshNoDupPath(sm, suffix, frontier, visited, fuel);
    } else {
      // No later node is in frontier; path[1] goes into nextFrontier.
      assert fuel > 0;
      var newVisited := visited + frontier;
      var nextFrontier :=
        (set u, v | u in frontier && v in BidirectedNeighbors(sm, u)
                  && v in SMNodes(sm) && v !in newVisited :: v);
      // path[1] ∉ frontier (no later node in frontier) and path[1] ∉ visited.
      assert path[1] !in frontier by {
        if path[1] in frontier { assert 0 < 1 < |path| && path[1] in frontier; }
      }
      assert path[1] !in visited;
      assert path[1] !in newVisited;
      // path[1] ∈ SMNodes(sm): from IsBiPath.
      assert path[1] in SMNodes(sm) by { IsBiPath_Step(sm, path, 0); }
      // HasBidirected(sm, path[0], path[1]) → path[1] ∈ BidirectedNeighbors(sm, path[0]).
      assert HasBidirected(sm, path[0], path[1]) by { IsBiPath_Step(sm, path, 0); }
      assert path[1] in BidirectedNeighbors(sm, path[0]) by {
        assert BiEdge(path[0], path[1]) in sm.bidirected || BiEdge(path[1], path[0]) in sm.bidirected;
        if BiEdge(path[0], path[1]) in sm.bidirected {
          assert path[1] in (set e | e in sm.bidirected && e.u == path[0] :: e.v);
        } else {
          assert path[1] in (set e | e in sm.bidirected && e.v == path[0] :: e.u);
        }
      }
      assert path[1] in nextFrontier;
      // Recurse on path[1..] with (nextFrontier, newVisited, fuel-1).
      var suffix := path[1..];
      assert IsBiPath(sm, suffix) by {
        forall i | 0 <= i < |suffix| - 1
          ensures HasBidirected(sm, suffix[i], suffix[i+1]) && suffix[i+1] in SMNodes(sm)
        { IsBiPath_Step(sm, path, 1 + i); }
      }
      assert NoDupSeq(suffix) by {
        forall a, b | 0 <= a < b < |suffix| ensures suffix[a] != suffix[b]
        { assert path[1+a] != path[1+b]; }
      }
      assert suffix[0] in nextFrontier by { assert suffix[0] == path[1]; }
      assert forall i :: 0 <= i < |suffix| ==> suffix[i] !in newVisited by {
        forall i | 0 <= i < |suffix| ensures suffix[i] !in newVisited {
          assert suffix[i] == path[1 + i];
          assert path[1 + i] !in visited;
          assert path[1 + i] !in frontier by {
            if path[1 + i] in frontier { assert 0 < 1 + i < |path| && path[1 + i] in frontier; }
          }
        }
      }
      assert fuel - 1 >= |suffix| by { assert |suffix| == |path| - 1; }
      BidirectedBFSLoop_FollowsFreshNoDupPath(sm, suffix, nextFrontier, newVisited, fuel - 1);
      assert suffix[|suffix| - 1] == path[|path| - 1];
      assert frontier != {};
      assert BidirectedBFSLoop(sm, frontier, visited, fuel) ==
        BidirectedBFSLoop(sm, nextFrontier, newVisited, fuel - 1);
    }
  }

  lemma CComponentCompiled_Correct(sm: SMGraph, v: Node)
    requires WellFormedSM(sm)
    requires v in SMNodes(sm)
    ensures CComponentCompiled(sm, v) == CComponent(sm, v)
  {
    var fuel := |SMNodes(sm)|;
    // Soundness: BFS ⊆ CComponent(sm, v).
    BidirectedBFSLoop_Sound(sm, v, {v}, {}, fuel);
    // Completeness: CComponent(sm, v) ⊆ BFS.
    forall u | u in CComponent(sm, v) ensures u in CComponentCompiled(sm, v) {
      // u ∈ CComponent(sm, v) → BCC(sm, u, v).
      assert BidirectedConnected(sm, u, v);
      // BCC_Symmetric → BCC(sm, v, u).
      BCC_Symmetric(sm, u, v, fuel);
      assert BidirectedConnected(sm, v, u);
      // Extract a bipath from v to u.
      BCC_ExtractPath(sm, v, u, fuel);
      var path :| IsBiPath(sm, path) && path[0] == v && path[|path|-1] == u && |path| <= fuel + 1;
      // Shorten to a NoDup path of length ≤ fuel.
      IsBiPath_ToNoDup(sm, path);
      var simple :| IsBiPath(sm, simple) && simple[0] == path[0] &&
                    simple[|simple|-1] == path[|path|-1] && NoDupSeq(simple) &&
                    |simple| <= fuel;
      assert simple[0] == v;
      assert simple[|simple|-1] == u;
      // Apply completeness lemma (visited = {}, all nodes trivially fresh).
      BidirectedBFSLoop_FollowsFreshNoDupPath(sm, simple, {v}, {}, fuel);
      assert u in CComponentCompiled(sm, v);
    }
  }

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
  lemma {:timeLimitMultiplier 2} CComponentsWithout_SingletonCoversAll(sm: SMGraph, X: set<Node>, S: set<Node>)
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

  // Q[S] Markov-factorizes over the induced subgraph G_S.
  //
  // Q[S] = P_{V\S}(S) is the interventional distribution under do(V\S),
  // and the local conditionals of G_S are exactly the factors that
  // contribute to Q[S] (the non-intervened factors in TruncatePMF).
  //
  // Ref: Tian & Pearl (2002), Lemma 3 and its proof.
  //      Used in ID algorithm Line 7 to pass QValue(sm, p, Sprime, ord)
  //      as the observational distribution for the recursive call on G_{S'}.
  lemma {:axiom} MarkovFactorization_QValue(
    sm: SMGraph,
    p: Prob.PMF,
    S: set<Node>,
    ord: seq<Node>
  )
    requires WellFormedSM(sm)
    requires Prob.IsDistribution(p)
    requires MarkovFactorization(sm.dag, p)
    requires S <= SMNodes(sm)
    requires SMTopologicalSort(sm, ord)
    ensures MarkovFactorization(SubgraphSM(sm, S).dag, QValue(sm, p, S, ord))

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

  // ==================================================================
  // 9.  No-bidirected structural lemmas
  //
  //   When a graph has no bidirected edges, every C-component is a
  //   singleton.  This enables the MarkovianCompleteness proof.
  // ==================================================================

  /// In a graph with no bidirected edges, fuel-bounded BFS never finds
  /// a path between distinct nodes.
  lemma NoBidirected_BCC_Bounded(sm: SMGraph, u: Node, v: Node, fuel: nat)
    requires sm.bidirected == {}
    ensures BidirectedConnectedBounded(sm, u, v, fuel) <==> u == v
    decreases fuel
  {
    if fuel == 0 {
      // BidirectedConnectedBounded(sm, u, v, 0) = (u == v) by definition.
    } else {
      // The existential witness branch requires HasBidirected(sm, u, w) for some w.
      // Since sm.bidirected == {}, HasBidirected is always false.
      assert forall w :: !HasBidirected(sm, u, w) by {
        forall w {
          assert !(BiEdge(u, w) in sm.bidirected);
          assert !(BiEdge(w, u) in sm.bidirected);
        }
      }
      // So the existential branch is dead; result collapses to (u == v).
      NoBidirected_BCC_Bounded(sm, u, v, fuel - 1);
    }
  }

  /// In a graph with no bidirected edges, BidirectedConnected iff equal.
  lemma NoBidirected_BidirectedConnected(sm: SMGraph, u: Node, v: Node)
    requires sm.bidirected == {}
    ensures BidirectedConnected(sm, u, v) <==> u == v
  {
    NoBidirected_BCC_Bounded(sm, u, v, |SMNodes(sm)|);
  }

  /// Helper: extract BCC property from C-component membership.
  /// S ∈ CComponents(sm) means (among other things) every pair in S is BCC-related.
  lemma CComponents_BCC_Pair(sm: SMGraph, S: set<Node>, u: Node, v: Node)
    requires S in CComponents(sm)
    requires u in S && v in S
    ensures BidirectedConnected(sm, u, v)
  { }  // follows from set-comprehension membership condition

  // Generic: any two elements of a size-1 set are equal.
  lemma Size1SetUnique<T(==)>(A: set<T>, x: T, y: T)
    requires |A| == 1
    requires x in A
    requires y in A
    ensures x == y
  {
    if x != y {
      var B : set<T> := A - {x};
      assert y in B;          // y ∈ A and y ≠ x → y ∈ A - {x}
      assert x !in B;         // x ∉ A - {x}
      assert B + {x} == A;    // (A - {x}) + {x} == A when x ∈ A
      assert |B + {x}| == |B| + 1;   // x !in B
      assert |B| == 0;        // |A| == 1 == |B| + 1 → |B| == 0
      assert false;           // y ∈ B contradicts |B| == 0
    }
  }

  // Helper: two members of a size-1 set of C-components must be identical.
  lemma CComponents_UniqueComp(F: SMGraph, S: set<Node>, T: set<Node>)
    requires S in CComponents(F)
    requires T in CComponents(F)
    requires |CComponents(F)| == 1
    ensures S == T
  {
    Size1SetUnique(CComponents(F), S, T);
  }

  /// In a graph with no bidirected edges, every C-component is a singleton
  /// and hence a C-forest must have exactly one node.
  lemma {:timeLimitMultiplier 2} NoBidirected_IsCForest_OneNode(F: SMGraph)
    requires WellFormedSM(F)
    requires F.bidirected == {}
    requires IsCForest(F)
    ensures exists w: Node :: SMNodes(F) == {w}
  {
    CComponents_Partition(F);
    assert |CComponents(F)| == 1;
    var S :| S in CComponents(F);
    assert S != {};
    var w :| w in S;
    assert w in SMNodes(F);
    // Every node v ∈ SMNodes(F) equals w:
    //   v is in some T ∈ CComponents(F), T must equal S (unique component),
    //   so v ∈ S, so BCC(F,v,w), so v == w (no bidirected edges).
    forall v | v in SMNodes(F) ensures v == w {
      assert exists T :: T in CComponents(F) && v in T;
      var T :| T in CComponents(F) && v in T;
      CComponents_UniqueComp(F, S, T);
      assert v in S;
      CComponents_BCC_Pair(F, S, v, w);
      NoBidirected_BidirectedConnected(F, v, w);
    }
    assert SMNodes(F) == {w};
  }

  /// Subgraph inherits the no-bidirected-edge property.
  lemma Subgraph_NoBidirected(Fprime: SMGraph, F: SMGraph)
    requires IsSubgraphSM(Fprime, F)
    requires F.bidirected == {}
    ensures Fprime.bidirected == {}
  {
    // IsSubgraphSM requires Fprime.bidirected <= F.bidirected == {}.
    assert Fprime.bidirected <= F.bidirected;
  }

  // =====================================================================
  // BidirectedConnected fuel/graph monotonicity + extend + symmetry
  // =====================================================================

  lemma BCC_FuelMono(sm: SMGraph, u: Node, v: Node, k: nat, k': nat)
    requires k <= k'
    requires BidirectedConnectedBounded(sm, u, v, k)
    ensures BidirectedConnectedBounded(sm, u, v, k')
    decreases k
  {
    if k == k' || u == v {
    } else {
      var w :| w in SMNodes(sm) && HasBidirected(sm, u, w) &&
               BidirectedConnectedBounded(sm, w, v, k - 1);
      BCC_FuelMono(sm, w, v, k - 1, k' - 1);
    }
  }

  lemma BCC_Concat(sm: SMGraph, u: Node, v: Node, w: Node, n: nat, m: nat)
    requires BidirectedConnectedBounded(sm, u, v, n)
    requires BidirectedConnectedBounded(sm, v, w, m)
    ensures BidirectedConnectedBounded(sm, u, w, n + m)
    decreases n
  {
    if u == v {
      BCC_FuelMono(sm, v, w, m, n + m);
    } else {
      var w0 :| w0 in SMNodes(sm) && HasBidirected(sm, u, w0) &&
                BidirectedConnectedBounded(sm, w0, v, n - 1);
      BCC_Concat(sm, w0, v, w, n - 1, m);
    }
  }

  lemma BCC_Extend(sm: SMGraph, u: Node, v: Node, w: Node, k: nat)
    requires BidirectedConnectedBounded(sm, u, v, k)
    requires HasBidirected(sm, v, w)
    requires w in SMNodes(sm)
    ensures BidirectedConnectedBounded(sm, u, w, k + 1)
    decreases k
  {
    if u == v {
      // BCC(sm, v, w, 1) via direct step
    } else {
      var w0 :| w0 in SMNodes(sm) && HasBidirected(sm, u, w0) &&
                BidirectedConnectedBounded(sm, w0, v, k - 1);
      BCC_Extend(sm, w0, v, w, k - 1);
    }
  }

  lemma BCC_Symmetric(sm: SMGraph, u: Node, v: Node, k: nat)
    requires WellFormedSM(sm)
    requires BidirectedConnectedBounded(sm, u, v, k)
    ensures BidirectedConnectedBounded(sm, v, u, k)
    decreases k
  {
    if u == v || k == 0 {
    } else {
      var w :| w in SMNodes(sm) && HasBidirected(sm, u, w) &&
               BidirectedConnectedBounded(sm, w, v, k - 1);
      BCC_Symmetric(sm, w, v, k - 1);
      // BCC(sm, v, w, k-1) from IH
      assert HasBidirected(sm, w, u);
      assert u in SMNodes(sm) by {
        if BiEdge(u, w) in sm.bidirected {
          assert u in SMNodes(sm);
        } else {
          assert BiEdge(w, u) in sm.bidirected;
          assert u in SMNodes(sm);
        }
      }
      BCC_Extend(sm, v, w, u, k - 1);
    }
  }

  lemma BCC_GraphMono(sm: SMGraph, sm': SMGraph, u: Node, v: Node, k: nat)
    requires SMNodes(sm) <= SMNodes(sm')
    requires sm.bidirected <= sm'.bidirected
    requires BidirectedConnectedBounded(sm, u, v, k)
    ensures BidirectedConnectedBounded(sm', u, v, k)
    decreases k
  {
    if u == v || k == 0 {
    } else {
      var w :| w in SMNodes(sm) && HasBidirected(sm, u, w) &&
               BidirectedConnectedBounded(sm, w, v, k - 1);
      assert w in SMNodes(sm');
      assert HasBidirected(sm', u, w);
      BCC_GraphMono(sm, sm', w, v, k - 1);
    }
  }

  // =====================================================================
  // Bidirected path sequences for cycle removal and transitivity
  // =====================================================================

  ghost predicate IsBiPath(sm: SMGraph, path: seq<Node>)
  {
    |path| >= 1 &&
    forall i :: 0 <= i < |path| - 1 ==>
      HasBidirected(sm, path[i], path[i+1]) && path[i+1] in SMNodes(sm)
  }

  // Extract one step from a bipath.
  lemma IsBiPath_Step(sm: SMGraph, path: seq<Node>, i: nat)
    requires IsBiPath(sm, path)
    requires 0 <= i < |path| - 1
    ensures HasBidirected(sm, path[i], path[i+1]) && path[i+1] in SMNodes(sm)
  {}

  lemma IsBiPath_AllInSMNodes(sm: SMGraph, path: seq<Node>)
    requires IsBiPath(sm, path)
    requires path[0] in SMNodes(sm)
    ensures forall i :: 0 <= i < |path| ==> path[i] in SMNodes(sm)
  {
    forall i | 0 <= i < |path|
      ensures path[i] in SMNodes(sm)
    {
      if i > 0 {
        IsBiPath_Step(sm, path, i - 1);
      }
    }
  }

  lemma IsBiPath_BCC(sm: SMGraph, path: seq<Node>)
    requires IsBiPath(sm, path)
    requires path[0] in SMNodes(sm)
    ensures BidirectedConnectedBounded(sm, path[0], path[|path|-1], |path| - 1)
    decreases |path|
  {
    if |path| == 1 {
      // BCC(sm, path[0], path[0], 0) trivially
    } else {
      IsBiPath_Step(sm, path, 0);
      assert HasBidirected(sm, path[0], path[1]) && path[1] in SMNodes(sm);
      assert IsBiPath(sm, path[1..]) by {
        forall i | 0 <= i < |path[1..]| - 1
          ensures HasBidirected(sm, path[1..][i], path[1..][i+1]) && path[1..][i+1] in SMNodes(sm)
        {
          assert path[1..][i] == path[i+1];
          assert path[1..][i+1] == path[i+2];
          IsBiPath_Step(sm, path, i+1);
        }
      }
      assert path[1..][0] == path[1];
      IsBiPath_BCC(sm, path[1..]);
      assert path[1..][|path[1..]|-1] == path[|path|-1];
    }
  }

  lemma BCC_ExtractPath(sm: SMGraph, u: Node, v: Node, k: nat)
    requires BidirectedConnectedBounded(sm, u, v, k)
    ensures exists path ::
      IsBiPath(sm, path) && path[0] == u && path[|path|-1] == v && |path| <= k + 1
    decreases k
  {
    if u == v {
      var path := [u];
      assert IsBiPath(sm, path);
    } else {
      var w :| w in SMNodes(sm) && HasBidirected(sm, u, w) &&
               BidirectedConnectedBounded(sm, w, v, k - 1);
      BCC_ExtractPath(sm, w, v, k - 1);
      var tail :| IsBiPath(sm, tail) && tail[0] == w && tail[|tail|-1] == v && |tail| <= k;
      var path := [u] + tail;
      assert IsBiPath(sm, path) by {
        forall i | 0 <= i < |path| - 1
          ensures HasBidirected(sm, path[i], path[i+1]) && path[i+1] in SMNodes(sm)
        {
          if i == 0 {
            assert path[0] == u && path[1] == tail[0] == w;
          } else {
            assert path[i] == tail[i-1] && path[i+1] == tail[i];
          }
        }
      }
      assert path[0] == u;
      assert path[|path|-1] == tail[|tail|-1] == v by {
        assert |path| == |tail| + 1;
        assert path[|tail|] == tail[|tail|-1];
      }
      assert |path| == |tail| + 1 <= k + 1;
    }
  }

  lemma SeqRepeat(s: seq<Node>, S: set<Node>)
    requires |s| > |S|
    requires forall i :: 0 <= i < |s| ==> s[i] in S
    ensures exists i, j :: 0 <= i < j < |s| && s[i] == s[j]
    decreases |S|
  {
    if |S| == 0 {
      assert s[0] in S;
    } else if exists j :: 1 <= j < |s| && s[0] == s[j] {
      // i = 0, j = witness
    } else {
      var S' := S - {s[0]};
      assert |S'| == |S| - 1;
      assert |s[1..]| > |S'| by {
        assert |s| >= |S| + 1;
      }
      forall i | 0 <= i < |s[1..]|
        ensures s[1..][i] in S'
      {
        assert s[1..][i] == s[i+1];
        assert s[i+1] in S;
        assert s[i+1] != s[0] by {
          assert !(exists j :: 1 <= j < |s| && s[0] == s[j]);
        }
      }
      SeqRepeat(s[1..], S');
      var i', j' :| 0 <= i' < j' < |s[1..]| && s[1..][i'] == s[1..][j'];
      assert s[i'+1] == s[j'+1];
    }
  }

  lemma IsBiPath_Concat(sm: SMGraph, p1: seq<Node>, p2: seq<Node>)
    requires IsBiPath(sm, p1)
    requires IsBiPath(sm, p2)
    requires p1[|p1|-1] == p2[0]
    ensures
      var joined := p1 + p2[1..];
      IsBiPath(sm, joined) &&
      joined[0] == p1[0] &&
      joined[|joined|-1] == p2[|p2|-1]
  {
    var joined := p1 + p2[1..];
    assert joined[0] == p1[0];
    assert joined[|joined|-1] == p2[|p2|-1] by {
      if |p2| == 1 {
        assert p2[1..] == [];
        assert joined == p1;
        assert joined[|joined|-1] == p1[|p1|-1] == p2[0] == p2[|p2|-1];
      } else {
        assert |joined| == |p1| + |p2| - 1;
        assert joined[|joined|-1] == p2[1..][|p2|-2];
        assert p2[1..][|p2|-2] == p2[|p2|-1];
      }
    }
    forall i | 0 <= i < |joined| - 1
      ensures HasBidirected(sm, joined[i], joined[i+1]) && joined[i+1] in SMNodes(sm)
    {
      if i < |p1| - 1 {
        assert joined[i] == p1[i] && joined[i+1] == p1[i+1];
      } else if i == |p1| - 1 {
        // junction
        assert |p2| >= 2;
        assert joined[i] == p1[|p1|-1] == p2[0];
        assert joined[i+1] == p2[1];
        IsBiPath_Step(sm, p2, 0);
        assert HasBidirected(sm, p2[0], p2[1]) && p2[1] in SMNodes(sm);
        assert joined[i] == p2[0];
      } else {
        // i >= |p1|
        var j := i - |p1| + 1;
        assert j >= 1 && j < |p2| - 1;
        assert joined[i] == p2[j];
        assert joined[i+1] == p2[j+1];
        IsBiPath_Step(sm, p2, j);
        assert HasBidirected(sm, p2[j], p2[j+1]) && p2[j+1] in SMNodes(sm);
      }
    }
  }

  lemma IsBiPath_RemoveCycle(sm: SMGraph, path: seq<Node>, i: nat, j: nat)
    requires IsBiPath(sm, path)
    requires 0 <= i < j < |path|
    requires path[i] == path[j]
    ensures exists shorter ::
      IsBiPath(sm, shorter) &&
      shorter[0] == path[0] &&
      shorter[|shorter|-1] == path[|path|-1] &&
      |shorter| < |path|
  {
    var shorter := path[..i+1] + path[j+1..];
    assert |shorter| < |path|;
    assert shorter[0] == path[0];
    assert shorter[|shorter|-1] == path[|path|-1] by {
      if j + 1 == |path| {
        assert path[j+1..] == [];
        assert shorter == path[..i+1];
        assert shorter[|shorter|-1] == path[i] == path[j] == path[|path|-1];
      } else {
        assert shorter[|shorter|-1] == path[j+1..][|path[j+1..]|-1];
        assert path[j+1..][|path[j+1..]|-1] == path[|path|-1];
      }
    }
    forall k | 0 <= k < |shorter| - 1
      ensures HasBidirected(sm, shorter[k], shorter[k+1]) && shorter[k+1] in SMNodes(sm)
    {
      if k < i {
        assert shorter[k] == path[k] && shorter[k+1] == path[k+1];
      } else if k == i {
        // shorter[i] = path[i] = path[j], shorter[i+1] = path[j+1]
        // k < |shorter|-1 = i + (|path|-j-1), so j < |path|-1
        assert j < |path| - 1;
        assert shorter[k] == path[i];
        assert shorter[k+1] == path[j+1];
        assert path[i] == path[j];
        assert HasBidirected(sm, path[j], path[j+1]) && path[j+1] in SMNodes(sm);
      } else {
        // k > i: shorter[k] = path[j + (k - i)], shorter[k+1] = path[j + (k-i) + 1]
        var l := j + (k - i);
        assert shorter[k] == path[l];
        assert shorter[k+1] == path[l+1];
        assert HasBidirected(sm, path[l], path[l+1]) && path[l+1] in SMNodes(sm);
      }
    }
  }

  lemma IsBiPath_Shorten(sm: SMGraph, path: seq<Node>)
    requires WellFormedSM(sm)
    requires IsBiPath(sm, path)
    requires path[0] in SMNodes(sm)
    ensures exists short_path ::
      IsBiPath(sm, short_path) &&
      short_path[0] == path[0] &&
      short_path[|short_path|-1] == path[|path|-1] &&
      |short_path| <= |SMNodes(sm)| + 1
    decreases |path|
  {
    if |path| <= |SMNodes(sm)| + 1 {
      // path itself works
    } else {
      IsBiPath_AllInSMNodes(sm, path);
      SeqRepeat(path, SMNodes(sm));
      var idx_i, idx_j :| 0 <= idx_i < idx_j < |path| && path[idx_i] == path[idx_j];
      IsBiPath_RemoveCycle(sm, path, idx_i, idx_j);
      var shorter :| IsBiPath(sm, shorter) && shorter[0] == path[0] &&
                     shorter[|shorter|-1] == path[|path|-1] && |shorter| < |path|;
      IsBiPath_Shorten(sm, shorter);
    }
  }

  lemma BidirectedConnected_Transitive(sm: SMGraph, u: Node, v: Node, w: Node)
    requires WellFormedSM(sm)
    requires u in SMNodes(sm)
    requires BidirectedConnected(sm, u, v)
    requires BidirectedConnected(sm, v, w)
    ensures BidirectedConnected(sm, u, w)
  {
    BCC_ExtractPath(sm, u, v, |SMNodes(sm)|);
    var p1 :| IsBiPath(sm, p1) && p1[0] == u && p1[|p1|-1] == v && |p1| <= |SMNodes(sm)| + 1;
    BCC_ExtractPath(sm, v, w, |SMNodes(sm)|);
    var p2 :| IsBiPath(sm, p2) && p2[0] == v && p2[|p2|-1] == w && |p2| <= |SMNodes(sm)| + 1;
    IsBiPath_Concat(sm, p1, p2);
    var joined := p1 + p2[1..];
    assert joined[0] == u;
    IsBiPath_Shorten(sm, joined);
    var short_path :|
      IsBiPath(sm, short_path) && short_path[0] == joined[0] &&
      short_path[|short_path|-1] == joined[|joined|-1] &&
      |short_path| <= |SMNodes(sm)| + 1;
    assert short_path[0] == u;
    assert short_path[|short_path|-1] == w;
    IsBiPath_BCC(sm, short_path);
    BCC_FuelMono(sm, u, w, |short_path| - 1, |SMNodes(sm)|);
  }

  // Helper: if u and v are BCC-connected in sm, and u belongs to component Sp,
  // then v must also belong to Sp (maximality of C-components).
  lemma BCC_InSameCComponent(sm: SMGraph, Sp: set<Node>, u: Node, v: Node)
    requires WellFormedSM(sm)
    requires Sp in CComponents(sm)
    requires u in Sp
    requires v in SMNodes(sm)
    requires BidirectedConnected(sm, u, v)
    ensures v in Sp
  {
    if v !in Sp {
      var x :| x in Sp && !BidirectedConnected(sm, v, x);
      BCC_Symmetric(sm, u, v, |SMNodes(sm)|);
      CComponent_Connected(sm, u, x);
      BidirectedConnected_Transitive(sm, v, u, x);
      assert false;
    }
  }

  // Every C-component of the subgraph G\X is contained in some
  // C-component of the full graph G.
  lemma {:timeLimitMultiplier 2} CComponentsWithout_RefinesG(sm: SMGraph, X: set<Node>, S: set<Node>)
    requires WellFormedSM(sm)
    requires S in CComponents(RemoveNodesSM(sm, X))
    ensures exists Sp :: Sp in CComponents(sm) && S <= Sp
  {
    var smX := RemoveNodesSM(sm, X);
    RemoveNodesSM_PreservesWellFormedness(sm, X);
    assert WellFormedSM(smX);
    assert S <= SMNodes(smX);
    assert SMNodes(smX) == SMNodes(sm) - X;
    assert S <= SMNodes(sm);
    assert S != {} by { CComponents_Partition(smX); }
    var u :| u in S;
    assert u in SMNodes(sm);
    CComponents_Partition(sm);
    var Sp :| Sp in CComponents(sm) && u in Sp;
    forall v | v in S ensures v in Sp {
      if v != u {
        CComponent_Connected(smX, u, v);
        BCC_GraphMono(smX, sm, u, v, |SMNodes(smX)|);
        BCC_FuelMono(sm, u, v, |SMNodes(smX)|, |SMNodes(sm)|);
        BCC_InSameCComponent(sm, Sp, u, v);
      }
    }
    assert S <= Sp;
  }

}  // end module SemiMarkovian
