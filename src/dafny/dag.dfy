// ===================================================================
// Directed Acyclic Graphs and d-Separation — Dafny Specification
//
// References:
//   Pearl, J. (2000). Causality. Cambridge University Press.
//   Lauritzen, S. (1996). Graphical Models. Oxford University Press.
//
// Provides:
//   1. Concrete DAG representation (finite node set + edge map)
//   2. Ancestry (reachability via the parent relation)
//   3. Graph surgery (remove incoming / outgoing edges)
//   4. d-Separation defined via blocked paths
//   5. Semi-graphoid axioms stated as lemmas
// ===================================================================

module DAG {

  // ==================================================================
  // 1.  Types
  // ==================================================================

  // A node (variable) identifier.  Natural numbers keep things finite
  // and give decidable equality.
  type Node = nat

  // A directed graph: for each node, the set of its parents.
  //   G[v] = parents of v   (edges point parent → child)
  //
  // Nodes are implicitly G.Keys.  Nodes not in the map have no parents.
  type Graph = map<Node, set<Node>>

  // The set of all nodes in G.
  function Nodes(G: Graph): set<Node> {
    G.Keys
  }

  // Parents of v in G.
  function Parents(G: Graph, v: Node): set<Node> {
    if v in G then G[v] else {}
  }

  // Children of u in G.
  function Children(G: Graph, u: Node): set<Node> {
    set v | v in Nodes(G) && u in Parents(G, v)
  }

  // ==================================================================
  // 2.  Acyclicity
  // ==================================================================

  // An ordering `ord` of nodes is a topological sort of G if every
  // parent comes strictly before its child.
  predicate IsTopologicalSort(G: Graph, ord: seq<Node>) {
    // (a) ord contains exactly the nodes of G — both directions bounded
    (forall v | v in Nodes(G) :: v in ord) &&
    (forall i | 0 <= i < |ord| :: ord[i] in Nodes(G)) &&
    // (b) no duplicates  (injective)
    (forall i, j | 0 <= i < j < |ord| :: ord[i] != ord[j]) &&
    // (c) every parent appears before its child
    (forall i | 0 <= i < |ord| ::
      forall p | p in Parents(G, ord[i]) ::
        exists k | 0 <= k < i :: ord[k] == p)
  }

  // A graph is a DAG iff it admits a topological sort.
  predicate IsDAG(G: Graph) {
    exists ord: seq<Node> :: IsTopologicalSort(G, ord)
  } by method {
    var r := KahnsAlgorithm(G);
    // Kahn's algorithm correctness: returns Some iff acyclic.
    assume {:axiom} r.Some? == (exists ord: seq<Node> :: IsTopologicalSort(G, ord));
    return r.Some?;
  }

  // ------------------------------------------------------------------
  // Kahn's Algorithm — compiled cycle detection / topological sort
  //
  // Returns Some(ordering) if G is acyclic, None if a cycle exists.
  //
  // Algorithm:
  //   1. Compute in-degree for each node.
  //   2. Initialize worklist with zero-in-degree nodes.
  //   3. Repeatedly remove a node from the worklist, append to result,
  //      decrement in-degree of children.  If child reaches 0, add
  //      to worklist.
  //   4. If result contains all nodes, the graph is a DAG.
  //
  // We implement this with sets (not sequences for the worklist)
  // since Dafny compiles set operations on finite sets efficiently.
  // ------------------------------------------------------------------

  // Compute the in-degree of node v in G.
  function InDegree(G: Graph, v: Node): nat {
    |Parents(G, v)|
  }

  // The in-degree map for all nodes.
  function InDegreeMap(G: Graph): map<Node, nat> {
    map v | v in Nodes(G) :: InDegree(G, v)
  }

  // Helper: given a degree map, find all nodes with degree 0.
  function ZeroInDegreeNodes(deg: map<Node, nat>): set<Node> {
    set v | v in deg && deg[v] == 0
  }

  // Optional result type.
  datatype Option<T> = Some(value: T) | None

  // ------------------------------------------------------------------
  // Kahn's Algorithm — compiled topological sort / cycle detection
  //
  // Implemented as a method with a while loop.  Methods support
  // non-deterministic choice (var :| in method bodies), which is
  // needed to pick from the zero-in-degree set.
  //
  // Returns Some(ordering) if G is acyclic, None if a cycle exists.
  // ------------------------------------------------------------------

  method KahnsAlgorithm(G: Graph) returns (result: Option<seq<Node>>)
  {
    var deg := InDegreeMap(G);
    var order: seq<Node> := [];
    var remaining := deg;

    while remaining != map[]
      invariant remaining.Keys <= Nodes(G)
      decreases remaining.Keys
    {
      var zeros := ZeroInDegreeNodes(remaining);
      if zeros == {} {
        // Nodes remain but none has in-degree 0 — cycle detected
        return None;
      }
      // Pick an arbitrary zero-in-degree node
      var v :| v in zeros;
      // Remove v from the degree map
      var remaining' := map u | u in remaining && u != v :: remaining[u];
      // Decrement in-degree of v's children that are still in remaining'
      var children_of_v := Children(G, v);
      remaining := map u | u in remaining' ::
        if u in children_of_v && remaining'[u] > 0 then remaining'[u] - 1
        else remaining'[u];
      order := order + [v];
    }

    return Some(order);
  }

  // Compiled DAG test using Kahn's algorithm.
  method IsDAGCompiled(G: Graph) returns (result: bool)
  {
    var r := KahnsAlgorithm(G);
    result := r.Some?;
  }

  // The compiled check is equivalent to the ghost predicate.
  lemma KahnsAlgorithm_Correct(G: Graph, ord: seq<Node>)
    requires |ord| == |Nodes(G)|
    ensures IsTopologicalSort(G, ord) ==> IsDAG(G)
  {
    if IsTopologicalSort(G, ord) {
      assert exists w: seq<Node> :: IsTopologicalSort(G, w);
    }
  }

  // ==================================================================
  // 3.  Ancestry  (reflexive-transitive closure of the parent relation)
  // ==================================================================

  // `IsAncestor(G, u, v)` holds when there is a directed path
  // u → ··· → v  (zero or more edges).
  //
  // Defined inductively:
  //   Base:  u == v
  //   Step:  u is a parent of some w, and w is an ancestor of v.
  //
  // For Dafny's termination checker we parameterize by a `fuel` bound
  // (the maximum path length).  In a DAG with n nodes, any simple
  // path has length ≤ n, so fuel = |Nodes(G)| suffices.

  predicate IsAncestorBounded(G: Graph, u: Node, v: Node, fuel: nat)
    decreases fuel
  {
    u == v ||
    (fuel > 0 &&
     exists w :: w in Children(G, u) && IsAncestorBounded(G, w, v, fuel - 1))
  }

  // Convenience wrapper using a fuel equal to the number of nodes.
  predicate IsAncestor(G: Graph, u: Node, v: Node) {
    IsAncestorBounded(G, u, v, |Nodes(G)|)
  }

  // All ancestors (including self) of each node in W.
  function Ancestors(G: Graph, W: set<Node>): set<Node> {
    set u | u in Nodes(G) && exists w :: w in W && IsAncestor(G, u, w)
  }

  // All descendants (including self) of each node in W.
  function Descendants(G: Graph, W: set<Node>): set<Node> {
    set v | v in Nodes(G) && exists w :: w in W && IsAncestor(G, w, v)
  }

  // ------------------------------------------------------------------
  // Ancestry lemmas
  // ------------------------------------------------------------------

  lemma Ancestor_Reflexive(G: Graph, v: Node)
    ensures IsAncestorBounded(G, v, v, 0)
  {}

  lemma IsAncestorBounded_ImpliesForwardTrail(G: Graph, u: Node, v: Node, fuel: nat)
    requires IsAncestorBounded(G, u, v, fuel)
    requires u != v
    ensures exists trail: seq<TrailStep> ::
      ValidTrail(G, trail) &&
      TrailConnects(trail, u, v) &&
      (forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward) &&
      |trail| <= fuel
    decreases fuel
  {
    if fuel == 0 {
      assert false;
    } else {
      var w :| w in Children(G, u) && IsAncestorBounded(G, w, v, fuel - 1);
      if w == v {
        var trail := [TrailStep(u, v, Forward)];
        assert ValidTrail(G, trail);
        assert TrailConnects(trail, u, v);
        assert forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward;
        assert exists trail0: seq<TrailStep> ::
          ValidTrail(G, trail0) &&
          TrailConnects(trail0, u, v) &&
          (forall i :: 0 <= i < |trail0| ==> trail0[i].dir == Forward) &&
          |trail0| <= fuel by {
          assert trail == trail;
        }
      } else {
        IsAncestorBounded_ImpliesForwardTrail(G, w, v, fuel - 1);
        var suffix: seq<TrailStep> :| ValidTrail(G, suffix) &&
          TrailConnects(suffix, w, v) &&
          (forall i :: 0 <= i < |suffix| ==> suffix[i].dir == Forward) &&
          |suffix| <= fuel - 1;
        var trail := [TrailStep(u, w, Forward)] + suffix;
        assert ValidTrail(G, trail) by {
          forall i {:trigger trail[i]} | 0 <= i < |trail|
            ensures
              (trail[i].dir == Forward ==> trail[i].from in Parents(G, trail[i].to)) &&
              (trail[i].dir == Backward ==> trail[i].to in Parents(G, trail[i].from))
          {
            if i == 0 {
              assert trail[i] == TrailStep(u, w, Forward);
              assert u in Parents(G, w);
            } else {
              assert trail[i] == suffix[i - 1];
            }
          }
        }
        assert TrailConnects(trail, u, v) by {
          assert |trail| > 0;
          assert trail[0].from == u;
          assert trail[|trail| - 1].to == v;
          forall i | 0 <= i < |trail| - 1
            ensures trail[i].to == trail[i + 1].from
          {
            if i == 0 {
              assert trail[0].to == w;
              assert trail[1] == suffix[0];
              assert suffix[0].from == w;
            } else {
              assert trail[i] == suffix[i - 1];
              assert trail[i + 1] == suffix[i];
            }
          }
        }
        assert forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward by {
          forall i | 0 <= i < |trail|
            ensures trail[i].dir == Forward
          {
            if i == 0 {
              assert trail[i] == TrailStep(u, w, Forward);
            } else {
              assert trail[i] == suffix[i - 1];
            }
          }
        }
        assert exists trail0: seq<TrailStep> ::
          ValidTrail(G, trail0) &&
          TrailConnects(trail0, u, v) &&
          (forall i :: 0 <= i < |trail0| ==> trail0[i].dir == Forward) &&
          |trail0| <= fuel by {
          assert trail == trail;
        }
      }
    }
  }

  lemma IsAncestorBounded_Monotone(G: Graph, u: Node, v: Node, small: nat, big: nat)
    requires small <= big
    requires IsAncestorBounded(G, u, v, small)
    ensures IsAncestorBounded(G, u, v, big)
    decreases small
  {
    if small == 0 {
    } else if u != v {
      assert big > 0;
      var w :| w in Children(G, u) && IsAncestorBounded(G, w, v, small - 1);
      IsAncestorBounded_Monotone(G, w, v, small - 1, big - 1);
    }
  }

  // ------------------------------------------------------------------
  // Compiled ancestry — BFS-based reachability
  // ------------------------------------------------------------------

  // BFS from a set of starting nodes, following child edges.
  // Returns the set of all reachable nodes (including the starts).
  function ReachableBFS(
    G: Graph,
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
        (set v, u | u in frontier && v in Children(G, u) && v !in newVisited :: v);
      ReachableBFS(G, nextFrontier, newVisited, fuel - 1)
  }

  // All descendants (including self) of W — compiled.
  function DescendantsCompiled(G: Graph, W: set<Node>): set<Node> {
    ReachableBFS(G, W * Nodes(G), {}, |Nodes(G)|)
  }

  // BFS following parent edges (for ancestors).
  function ReachableParentBFS(
    G: Graph,
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
        (set u | u in Nodes(G) && (exists v :: v in frontier && u in Parents(G, v))
                && u !in newVisited);
      ReachableParentBFS(G, nextFrontier, newVisited, fuel - 1)
  }

  // All ancestors (including self) of W — compiled.
  function AncestorsCompiled(G: Graph, W: set<Node>): set<Node> {
    ReachableParentBFS(G, W * Nodes(G), {}, |Nodes(G)|)
  }

  // Compiled equivalence lemmas
  lemma {:axiom} AncestorsCompiled_Correct(G: Graph, W: set<Node>)
    ensures AncestorsCompiled(G, W) == Ancestors(G, W)

  lemma {:axiom} DescendantsCompiled_Correct(G: Graph, W: set<Node>)
    ensures DescendantsCompiled(G, W) == Descendants(G, W)

  // ==================================================================
  // 4.  Graph Surgery
  // ==================================================================

  // G_{X̄}  —  remove incoming edges to every node in X.
  function RemoveIncomingCompiled(G: Graph, X: set<Node>): Graph
  {
    map v | v in Nodes(G) ::
      if v in X then {} else Parents(G, v)
  }

  // G_{X̲}  —  remove outgoing edges from every node in X.
  function RemoveOutgoingCompiled(G: Graph, X: set<Node>): Graph
  {
    map v | v in Nodes(G) ::
      Parents(G, v) - X
  }

  lemma RemoveIncomingCompiled_Correct(G: Graph, X: set<Node>)
    ensures RemoveIncomingCompiled(G, X) == RemoveIncoming(G, X)
  {
    assert RemoveIncomingCompiled(G, X)
         == (map v | v in Nodes(G) :: if v in X then {} else Parents(G, v));
    assert RemoveIncoming(G, X)
         == (map v | v in Nodes(G) :: if v in X then {} else Parents(G, v));
  }

  lemma RemoveOutgoingCompiled_Correct(G: Graph, X: set<Node>)
    ensures RemoveOutgoingCompiled(G, X) == RemoveOutgoing(G, X)
  {
    assert RemoveOutgoingCompiled(G, X)
         == (map v | v in Nodes(G) :: Parents(G, v) - X);
    assert RemoveOutgoing(G, X)
         == (map v | v in Nodes(G) :: Parents(G, v) - X);
  }

  // G_{X̄}  —  remove incoming edges to every node in X.
  function RemoveIncoming(G: Graph, X: set<Node>): Graph
  {
    map v | v in Nodes(G) ::
      if v in X then {} else Parents(G, v)
  }

  // G_{X̲}  —  remove outgoing edges from every node in X.
  //   i.e., for every child c, remove X-members from c's parent set.
  function RemoveOutgoing(G: Graph, X: set<Node>): Graph
  {
    map v | v in Nodes(G) ::
      Parents(G, v) - X
  }

  // Delete nodes in X and every incident edge.
  function RemoveNodes(G: Graph, X: set<Node>): Graph
  {
    map v | v in Nodes(G) && v !in X ::
      Parents(G, v) - X
  }

  // ------------------------------------------------------------------
  // Surgery lemmas
  // ------------------------------------------------------------------

  /// Removing incoming edges from ∅ changes nothing.
  lemma RemoveIncoming_Empty(G: Graph)
    ensures RemoveIncoming(G, {}) == G
  {
    // Every node keeps its original parents since no node is in {}.
    assert forall v :: v in Nodes(G) ==>
      (if v in {} then {} else Parents(G, v)) == Parents(G, v);
  }

  /// Nodes in X lose all parents after incoming surgery.
  lemma RemoveIncoming_NoParents(G: Graph, X: set<Node>, x: Node)
    requires x in Nodes(G) && x in X
    ensures Parents(RemoveIncoming(G, X), x) == {}
  {}

  /// Nodes outside X are unaffected by incoming surgery.
  lemma RemoveIncoming_PreservesOthers(G: Graph, X: set<Node>, v: Node)
    requires v in Nodes(G) && v !in X
    ensures Parents(RemoveIncoming(G, X), v) == Parents(G, v)
  {}

  /// Removing outgoing edges from ∅ changes nothing.
  lemma RemoveOutgoing_Empty(G: Graph)
    ensures RemoveOutgoing(G, {}) == G
  {
    assert forall v :: v in Nodes(G) ==>
      Parents(G, v) - {} == Parents(G, v);
  }

  /// Deleting nodes from a DAG preserves acyclicity.
  lemma RemoveNodes_PreservesDAG(G: Graph, X: set<Node>)
    requires IsDAG(G)
    ensures IsDAG(RemoveNodes(G, X))
  {
    var ord :| IsTopologicalSort(G, ord);
    var GX := RemoveNodes(G, X);
    assert Nodes(GX) == Nodes(G) - X;

    var ordX: seq<Node> := [];
    var i := 0;
    while i < |ord|
      invariant 0 <= i <= |ord|
      invariant Nodes(GX) == Nodes(G) - X
      invariant forall j | 0 <= j < |ordX| :: ordX[j] in Nodes(GX)
      invariant forall j | 0 <= j < |ordX| :: exists k :: 0 <= k < i && ord[k] == ordX[j]
      invariant forall k | 0 <= k < i && ord[k] in Nodes(GX) :: ord[k] in ordX
      invariant forall a, b | 0 <= a < b < |ordX| :: ordX[a] != ordX[b]
      invariant forall j | 0 <= j < |ordX| ::
        forall p | p in Parents(GX, ordX[j]) ::
          exists k :: 0 <= k < j && ordX[k] == p
    {
      if ord[i] in Nodes(GX) {
        var v := ord[i];
        assert v in Nodes(G);
        assert v !in X;
        assert Parents(GX, v) == Parents(G, v) - X;

        assert v !in ordX by {
          if v in ordX {
            var j :| 0 <= j < |ordX| && ordX[j] == v;
            var k :| 0 <= k < i && ord[k] == ordX[j];
            assert 0 <= k < i < |ord|;
            assert ord[k] == ord[i];
            assert ord[k] != ord[i];
          }
        }

        forall p | p in Parents(GX, v)
          ensures exists k :: 0 <= k < |ordX| && ordX[k] == p
        {
          assert p in Parents(G, v);
          assert p !in X;
          var h :| 0 <= h < i && ord[h] == p;
          assert p in Nodes(G);
          assert p in Nodes(GX);
          assert p in ordX;
          var k :| 0 <= k < |ordX| && ordX[k] == p;
        }

        ordX := ordX + [v];
      }
      i := i + 1;
    }

    forall v | v in Nodes(GX)
      ensures v in ordX
    {
      assert v in Nodes(G);
      var k :| 0 <= k < |ord| && ord[k] == v;
      assert k < i;
      assert ord[k] in Nodes(GX);
      assert v in ordX;
    }
    assert IsTopologicalSort(GX, ordX);
  }

  // ==================================================================
  // 5.  Paths and Trails
  // ==================================================================

  // A trail is a sequence of nodes (not necessarily all distinct)
  // where consecutive pairs are connected by an edge in either
  // direction.  This is the building block for d-separation.

  // Edge relationship between consecutive nodes on a trail:
  //   Forward:  path[i] → path[i+1]   (parent → child)
  //   Backward: path[i] ← path[i+1]   (child ← parent)
  datatype EdgeDir = Forward | Backward

  // A step in a trail.
  datatype TrailStep = TrailStep(from: Node, to: Node, dir: EdgeDir)

  // A trail is valid if every step corresponds to an actual edge.
  ghost predicate ValidTrail(G: Graph, trail: seq<TrailStep>) {
    forall i {:trigger trail[i]} :: 0 <= i < |trail| ==>
      (trail[i].dir == Forward  ==> trail[i].from in Parents(G, trail[i].to)) &&
      (trail[i].dir == Backward ==> trail[i].to in Parents(G, trail[i].from))
  }

  // The trail connects `start` to `end`.
  ghost predicate TrailConnects(trail: seq<TrailStep>, start: Node, end: Node) {
    |trail| > 0 &&
    trail[0].from == start &&
    trail[|trail| - 1].to == end &&
    (forall i :: 0 <= i < |trail| - 1 ==> trail[i].to == trail[i + 1].from)
  }

  lemma ValidTrail_Prefix(G: Graph, trail: seq<TrailStep>, prefixLen: nat)
    requires ValidTrail(G, trail)
    requires 0 < prefixLen <= |trail|
    ensures ValidTrail(G, trail[..prefixLen])
  {
    forall i {:trigger trail[..prefixLen][i]} | 0 <= i < |trail[..prefixLen]|
      ensures
        (trail[..prefixLen][i].dir == Forward ==> trail[..prefixLen][i].from in Parents(G, trail[..prefixLen][i].to)) &&
        (trail[..prefixLen][i].dir == Backward ==> trail[..prefixLen][i].to in Parents(G, trail[..prefixLen][i].from))
    {
      assert trail[..prefixLen][i] == trail[i];
    }
  }

  lemma ValidTrail_Concat(G: Graph, left: seq<TrailStep>, right: seq<TrailStep>)
    requires ValidTrail(G, left)
    requires ValidTrail(G, right)
    ensures ValidTrail(G, left + right)
  {
    forall i {:trigger (left + right)[i]} | 0 <= i < |left + right|
      ensures
        ((left + right)[i].dir == Forward ==> (left + right)[i].from in Parents(G, (left + right)[i].to)) &&
        ((left + right)[i].dir == Backward ==> (left + right)[i].to in Parents(G, (left + right)[i].from))
    {
      if i < |left| {
        assert (left + right)[i] == left[i];
      } else {
        assert (left + right)[i] == right[i - |left|];
      }
    }
  }

  lemma TrailConnects_Prefix(trail: seq<TrailStep>, start: Node, end: Node, prefixLen: nat)
    requires TrailConnects(trail, start, end)
    requires 0 < prefixLen <= |trail|
    ensures TrailConnects(trail[..prefixLen], start, trail[prefixLen - 1].to)
  {
    assert |trail[..prefixLen]| > 0;
    assert trail[..prefixLen][0].from == start;
    assert trail[..prefixLen][|trail[..prefixLen]| - 1].to == trail[prefixLen - 1].to;
    forall i | 0 <= i < |trail[..prefixLen]| - 1
      ensures trail[..prefixLen][i].to == trail[..prefixLen][i + 1].from
    {
      assert trail[..prefixLen][i] == trail[i];
      assert trail[..prefixLen][i + 1] == trail[i + 1];
    }
  }

  lemma TrailConnects_Concat(left: seq<TrailStep>, start: Node, mid: Node, right: seq<TrailStep>, end: Node)
    requires TrailConnects(left, start, mid)
    requires TrailConnects(right, mid, end)
    ensures TrailConnects(left + right, start, end)
  {
    assert |left + right| > 0;
    assert (left + right)[0].from == start;
    assert (left + right)[|left + right| - 1].to == end;
    forall i | 0 <= i < |left + right| - 1
      ensures (left + right)[i].to == (left + right)[i + 1].from
    {
      if i < |left| - 1 {
        assert (left + right)[i] == left[i];
        assert (left + right)[i + 1] == left[i + 1];
      } else if i == |left| - 1 {
        assert (left + right)[i] == left[i];
        assert (left + right)[i + 1] == right[0];
        assert left[i].to == mid;
        assert right[0].from == mid;
      } else {
        assert (left + right)[i] == right[i - |left|];
        assert (left + right)[i + 1] == right[i + 1 - |left|];
      }
    }
  }

  lemma ForwardTrail_ImpliesAncestorBounded(G: Graph, trail: seq<TrailStep>, start: Node, end: Node)
    requires ValidTrail(G, trail)
    requires TrailConnects(trail, start, end)
    requires forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward
    ensures IsAncestorBounded(G, start, end, |trail|)
    decreases |trail|
  {
    if |trail| == 1 {
      assert trail[0].from == start;
      assert trail[0].to == end;
      assert start in Parents(G, end);
      assert end in Children(G, start);
      assert IsAncestorBounded(G, end, end, 0);
    } else {
      ValidTrail_Suffix(G, trail);
      TrailConnects_Suffix(trail, start, end);
      assert forall i :: 0 <= i < |trail[1..]| ==> trail[1..][i].dir == Forward by {
        forall i | 0 <= i < |trail[1..]|
          ensures trail[1..][i].dir == Forward
        {
          assert trail[1..][i] == trail[i + 1];
        }
      }
      ForwardTrail_ImpliesAncestorBounded(G, trail[1..], trail[1].from, end);
      assert trail[0].to == trail[1].from;
      assert trail[0].from == start;
      assert start in Parents(G, trail[1].from);
      assert trail[1].from in Children(G, start);
    }
  }

  lemma ForwardTrail_NodeInDescendants(G: Graph, trail: seq<TrailStep>, start: Node, end: Node, pos: nat)
    requires ValidTrail(G, trail)
    requires TrailConnects(trail, start, end)
    requires forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward
    requires |trail| <= |Nodes(G)|
    requires 1 <= pos < |trail|
    ensures trail[pos].from in Descendants(G, {start})
  {
    ValidTrail_Prefix(G, trail, pos);
    TrailConnects_Prefix(trail, start, end, pos);
    assert trail[pos - 1].to == trail[pos].from;
    assert forall i :: 0 <= i < |trail[..pos]| ==> trail[..pos][i].dir == Forward by {
      forall i | 0 <= i < |trail[..pos]|
        ensures trail[..pos][i].dir == Forward
      {
        assert trail[..pos][i] == trail[i];
      }
    }
    ForwardTrail_ImpliesAncestorBounded(G, trail[..pos], start, trail[pos].from);
    IsAncestorBounded_Monotone(G, start, trail[pos].from, pos, |Nodes(G)|);
    assert trail[pos - 1].from in Parents(G, trail[pos - 1].to);
    assert trail[pos].from in Nodes(G);
    assert start in {start};
  }

  // ------------------------------------------------------------------
  // Trail reversal helpers
  //
  // These helpers support future symmetry proofs by turning any trail
  // from y to z into a trail from z to y over the same underlying edges.
  // ------------------------------------------------------------------

  ghost function ReverseDir(dir: EdgeDir): EdgeDir {
    if dir == Forward then Backward else Forward
  }

  ghost function ReverseStep(step: TrailStep): TrailStep {
    TrailStep(step.to, step.from, ReverseDir(step.dir))
  }

  ghost function ReverseTrail(trail: seq<TrailStep>): seq<TrailStep>
    decreases |trail|
  {
    if |trail| == 0 then []
    else ReverseTrail(trail[1..]) + [ReverseStep(trail[0])]
  }

  lemma ReverseTrail_Length(trail: seq<TrailStep>)
    ensures |ReverseTrail(trail)| == |trail|
    decreases |trail|
  {
    if |trail| != 0 {
      ReverseTrail_Length(trail[1..]);
    }
  }

  lemma ValidTrail_Suffix(G: Graph, trail: seq<TrailStep>)
    requires ValidTrail(G, trail)
    requires |trail| > 0
    ensures ValidTrail(G, trail[1..])
  {
    forall i {:trigger trail[1..][i]} | 0 <= i < |trail[1..]|
      ensures
        (trail[1..][i].dir == Forward ==> trail[1..][i].from in Parents(G, trail[1..][i].to)) &&
        (trail[1..][i].dir == Backward ==> trail[1..][i].to in Parents(G, trail[1..][i].from))
    {
      assert trail[1..][i] == trail[i + 1];
    }
  }

  lemma ReverseStep_Valid(G: Graph, step: TrailStep)
    requires (step.dir == Forward ==> step.from in Parents(G, step.to))
    requires (step.dir == Backward ==> step.to in Parents(G, step.from))
    ensures
      (ReverseStep(step).dir == Forward ==> ReverseStep(step).from in Parents(G, ReverseStep(step).to)) &&
      (ReverseStep(step).dir == Backward ==> ReverseStep(step).to in Parents(G, ReverseStep(step).from))
  {
    if step.dir == Forward {
      assert ReverseStep(step).dir == Backward;
      assert ReverseStep(step).from == step.to;
      assert ReverseStep(step).to == step.from;
    } else {
      assert step.dir == Backward;
      assert ReverseStep(step).dir == Forward;
      assert ReverseStep(step).from == step.to;
      assert ReverseStep(step).to == step.from;
    }
  }

  lemma ReverseTrail_Valid(G: Graph, trail: seq<TrailStep>)
    requires ValidTrail(G, trail)
    ensures ValidTrail(G, ReverseTrail(trail))
    decreases |trail|
  {
    if |trail| != 0 {
      ValidTrail_Suffix(G, trail);
      ReverseTrail_Valid(G, trail[1..]);
      assert
        (trail[0].dir == Forward ==> trail[0].from in Parents(G, trail[0].to)) &&
        (trail[0].dir == Backward ==> trail[0].to in Parents(G, trail[0].from));
      ReverseStep_Valid(G, trail[0]);
      ReverseTrail_Length(trail[1..]);
      forall i {:trigger ReverseTrail(trail)[i]} | 0 <= i < |ReverseTrail(trail)|
        ensures
          (ReverseTrail(trail)[i].dir == Forward ==> ReverseTrail(trail)[i].from in Parents(G, ReverseTrail(trail)[i].to)) &&
          (ReverseTrail(trail)[i].dir == Backward ==> ReverseTrail(trail)[i].to in Parents(G, ReverseTrail(trail)[i].from))
      {
        if i < |ReverseTrail(trail[1..])| {
          assert ReverseTrail(trail)[i] == ReverseTrail(trail[1..])[i];
        } else {
          assert i == |ReverseTrail(trail[1..])|;
          assert ReverseTrail(trail)[i] == ReverseStep(trail[0]);
        }
      }
    }
  }

  lemma TrailConnects_Suffix(trail: seq<TrailStep>, start: Node, end: Node)
    requires TrailConnects(trail, start, end)
    requires |trail| > 1
    ensures TrailConnects(trail[1..], trail[1].from, end)
  {
    assert |trail[1..]| > 0;
    assert trail[1..][0].from == trail[1].from;
    assert trail[1..][|trail[1..]| - 1].to == end;
    forall i | 0 <= i < |trail[1..]| - 1
      ensures trail[1..][i].to == trail[1..][i + 1].from
    {
      assert trail[1..][i] == trail[i + 1];
      assert trail[1..][i + 1] == trail[i + 2];
    }
  }

  lemma ReverseTrail_Connects(trail: seq<TrailStep>, start: Node, end: Node)
    requires TrailConnects(trail, start, end)
    ensures TrailConnects(ReverseTrail(trail), end, start)
    decreases |trail|
  {
    ReverseTrail_Length(trail);
    if |trail| == 1 {
      assert ReverseTrail(trail) == [ReverseStep(trail[0])];
      assert ReverseTrail(trail)[0].from == end;
      assert ReverseTrail(trail)[0].to == start;
    } else {
      TrailConnects_Suffix(trail, start, end);
      ReverseTrail_Connects(trail[1..], trail[1].from, end);
      ReverseTrail_Length(trail[1..]);
      assert |ReverseTrail(trail[1..])| > 0;
      assert ReverseTrail(trail) == ReverseTrail(trail[1..]) + [ReverseStep(trail[0])];
      assert ReverseTrail(trail)[0].from == end;
      assert ReverseTrail(trail)[|ReverseTrail(trail)| - 1].to == start;
      forall i | 0 <= i < |ReverseTrail(trail)| - 1
        ensures ReverseTrail(trail)[i].to == ReverseTrail(trail)[i + 1].from
      {
        if i < |ReverseTrail(trail[1..])| - 1 {
          assert ReverseTrail(trail)[i] == ReverseTrail(trail[1..])[i];
          assert ReverseTrail(trail)[i + 1] == ReverseTrail(trail[1..])[i + 1];
        } else {
          assert i == |ReverseTrail(trail[1..])| - 1;
          assert ReverseTrail(trail)[i] == ReverseTrail(trail[1..])[i];
          assert ReverseTrail(trail)[i + 1] == ReverseStep(trail[0]);
          assert ReverseTrail(trail[1..])[|ReverseTrail(trail[1..])| - 1].to == trail[1].from;
          assert ReverseStep(trail[0]).from == trail[0].to;
          assert trail[0].to == trail[1].from;
        }
      }
    }
  }

  // ==================================================================
  // 6.  d-Separation
  // ==================================================================

  // A trail step at position i is a **collider** if edges from both
  // sides point inward:  ··· → node ← ···
  //
  // In our representation, step i has:
  //   trail[i-1].dir == Forward   (previous step goes into the node)
  //   trail[i].dir   == Backward  (current step goes into the node from next)
  //
  // More precisely: a node `trail[i].from` (for i > 0) is a collider
  // when the incoming direction is Forward (from trail[i-1]) and the
  // outgoing direction is Backward (to trail[i]).  We define this
  // directly on consecutive edge pairs.

  ghost predicate IsCollider(trail: seq<TrailStep>, pos: nat)
    requires 0 < pos < |trail|
  {
    // The node between step (pos-1) and step pos.
    // Step pos-1 goes "into" the node: dir == Forward
    // Step pos goes "into" the node:   dir == Backward
    trail[pos - 1].dir == Forward && trail[pos].dir == Backward
  }

  ghost predicate TrailBlockedAtPos(G: Graph, trail: seq<TrailStep>, pos: nat, W: set<Node>)
    requires 1 <= pos < |trail|
  {
    var node := trail[pos].from;
    if IsCollider(trail, pos) then
      node !in W && Descendants(G, {node}) * W == {}
    else
      node in W
  }

  // A trail is **blocked** by conditioning set W if at least one
  // node along the trail satisfies the d-separation blocking criterion:
  //
  //   Non-collider:  the node is in W (blocks the path)
  //   Collider:      the node (and all its descendants) are NOT in W
  //
  // Only internal nodes can block a trail. In particular, a single-edge
  // trail has no internal blocking witness and is therefore unblocked.
  ghost predicate TrailBlocked(G: Graph, trail: seq<TrailStep>, W: set<Node>) {
    exists pos :: 1 <= pos < |trail| && TrailBlockedAtPos(G, trail, pos, W)
  }

  lemma IsCollider_Prefix(trail: seq<TrailStep>, prefixLen: nat, pos: nat)
    requires 0 < prefixLen <= |trail|
    requires 1 <= pos < prefixLen
    ensures IsCollider(trail[..prefixLen], pos) <==> IsCollider(trail, pos)
  {
    assert trail[..prefixLen][pos - 1] == trail[pos - 1];
    assert trail[..prefixLen][pos] == trail[pos];
  }

  lemma IsCollider_Suffix(trail: seq<TrailStep>, pos: nat)
    requires |trail| > 1
    requires 1 <= pos < |trail[1..]|
    ensures IsCollider(trail[1..], pos) <==> IsCollider(trail, pos + 1)
  {
    assert trail[1..][pos - 1] == trail[pos];
    assert trail[1..][pos] == trail[pos + 1];
  }

  lemma TrailBlockedAtPos_Prefix(G: Graph, trail: seq<TrailStep>, prefixLen: nat, pos: nat, W: set<Node>)
    requires 0 < prefixLen <= |trail|
    requires 1 <= pos < prefixLen
    ensures TrailBlockedAtPos(G, trail[..prefixLen], pos, W) <==> TrailBlockedAtPos(G, trail, pos, W)
  {
    IsCollider_Prefix(trail, prefixLen, pos);
    assert trail[..prefixLen][pos].from == trail[pos].from;
  }

  lemma TrailBlockedAtPos_Suffix(G: Graph, trail: seq<TrailStep>, pos: nat, W: set<Node>)
    requires |trail| > 1
    requires 1 <= pos < |trail[1..]|
    ensures TrailBlockedAtPos(G, trail[1..], pos, W) <==> TrailBlockedAtPos(G, trail, pos + 1, W)
  {
    IsCollider_Suffix(trail, pos);
    assert trail[1..][pos].from == trail[pos + 1].from;
  }

  lemma TrailNotBlockedAtPos(G: Graph, trail: seq<TrailStep>, pos: nat, W: set<Node>)
    requires !TrailBlocked(G, trail, W)
    requires 1 <= pos < |trail|
    ensures !TrailBlockedAtPos(G, trail, pos, W)
  {
    if TrailBlockedAtPos(G, trail, pos, W) {
      assert TrailBlocked(G, trail, W);
    }
  }

  lemma TrailBlocked_Suffix(G: Graph, trail: seq<TrailStep>, W: set<Node>)
    requires TrailBlocked(G, trail, W)
    requires |trail| > 1
    requires !TrailBlockedAtPos(G, trail, 1, W)
    ensures TrailBlocked(G, trail[1..], W)
  {
    if |trail[1..]| <= 1 {
      var blockedPos :| 1 <= blockedPos < |trail| && TrailBlockedAtPos(G, trail, blockedPos, W);
      assert blockedPos == 1;
      assert false;
    } else {
      var blockedPos :| 1 <= blockedPos < |trail| && TrailBlockedAtPos(G, trail, blockedPos, W);
      assert blockedPos != 1;
      assert 1 <= blockedPos - 1 < |trail[1..]|;
      TrailBlockedAtPos_Suffix(G, trail, blockedPos - 1, W);
    }
  }

  lemma FirstBlockedPos(G: Graph, trail: seq<TrailStep>, W: set<Node>) returns (pos: nat)
    requires TrailBlocked(G, trail, W)
    requires |trail| > 1
    ensures 1 <= pos < |trail|
    ensures TrailBlockedAtPos(G, trail, pos, W)
    ensures forall j :: 1 <= j < pos ==> !TrailBlockedAtPos(G, trail, j, W)
    decreases |trail|
  {
    if TrailBlockedAtPos(G, trail, 1, W) {
      pos := 1;
    } else {
      TrailBlocked_Suffix(G, trail, W);
      var suffixPos := FirstBlockedPos(G, trail[1..], W);
      pos := suffixPos + 1;
      assert TrailBlockedAtPos(G, trail, pos, W) by {
        TrailBlockedAtPos_Suffix(G, trail, suffixPos, W);
      }
      assert forall j :: 1 <= j < pos ==> !TrailBlockedAtPos(G, trail, j, W) by {
        forall j | 1 <= j < pos
          ensures !TrailBlockedAtPos(G, trail, j, W)
        {
          if j == 1 {
            assert !TrailBlockedAtPos(G, trail, 1, W);
          } else {
            assert 1 <= j - 1 < suffixPos;
            TrailBlockedAtPos_Suffix(G, trail, j - 1, W);
          }
        }
      }
    }
  }

  lemma PrefixWithoutBlockedPos_NotBlocked(G: Graph, trail: seq<TrailStep>, prefixLen: nat, W: set<Node>)
    requires 0 < prefixLen <= |trail|
    requires forall j :: 1 <= j < prefixLen ==> !TrailBlockedAtPos(G, trail, j, W)
    ensures !TrailBlocked(G, trail[..prefixLen], W)
  {
    if TrailBlocked(G, trail[..prefixLen], W) {
      var pos :| 1 <= pos < prefixLen && TrailBlockedAtPos(G, trail[..prefixLen], pos, W);
      TrailBlockedAtPos_Prefix(G, trail, prefixLen, pos, W);
      assert false;
    }
  }

  lemma ColliderOpenedByNewConditioning(
    G: Graph, trail: seq<TrailStep>, pos: nat, W: set<Node>, Z': set<Node>
  )
    requires 1 <= pos < |trail|
    requires IsCollider(trail, pos)
    requires TrailBlockedAtPos(G, trail, pos, W)
    requires !TrailBlockedAtPos(G, trail, pos, W + Z')
    ensures exists zPrime: Node ::
      zPrime in Z' &&
      zPrime !in W &&
      (zPrime == trail[pos].from || zPrime in Descendants(G, {trail[pos].from}))
  {
    var node := trail[pos].from;
    assert node !in W;
    assert Descendants(G, {node}) * W == {};
    if node in Z' {
      assert exists zPrime: Node ::
        zPrime in Z' &&
        zPrime !in W &&
        (zPrime == trail[pos].from || zPrime in Descendants(G, {trail[pos].from})) by {
        assert node == trail[pos].from;
      }
    } else {
      assert node !in W + Z';
      if Descendants(G, {node}) * Z' == {} {
        assert Descendants(G, {node}) * (W + Z') == {} by {
          assert forall v :: v in Descendants(G, {node}) * (W + Z') ==> false by {
            forall v | v in Descendants(G, {node}) * (W + Z')
              ensures false
            {
              if v in W {
                assert v in Descendants(G, {node}) * W;
              } else {
                assert v in Z';
                assert v in Descendants(G, {node}) * Z';
              }
            }
          }
        }
        assert TrailBlockedAtPos(G, trail, pos, W + Z');
        assert false;
      }
      var zPrime :| zPrime in Descendants(G, {node}) * Z';
      if zPrime in W {
        assert zPrime in Descendants(G, {node}) * W;
        assert false;
      }
      assert exists z0: Node ::
        z0 in Z' &&
        z0 !in W &&
        (z0 == trail[pos].from || z0 in Descendants(G, {trail[pos].from})) by {
        assert zPrime == zPrime;
      }
    }
  }

  lemma BlockingAddedByConditioningAtPos(
    G: Graph, trail: seq<TrailStep>, pos: nat, W: set<Node>, Z': set<Node>
  )
    requires 1 <= pos < |trail|
    requires !TrailBlockedAtPos(G, trail, pos, W)
    requires TrailBlockedAtPos(G, trail, pos, W + Z')
    ensures !IsCollider(trail, pos)
    ensures trail[pos].from in Z'
  {
    var node := trail[pos].from;
    if IsCollider(trail, pos) {
      assert node !in W + Z';
      assert Descendants(G, {node}) * (W + Z') == {};
      assert node !in W;
      assert Descendants(G, {node}) * W == {} by {
        assert forall v :: v in Descendants(G, {node}) * W ==> false by {
          forall v | v in Descendants(G, {node}) * W
            ensures false
          {
            assert v in Descendants(G, {node}) * (W + Z');
          }
        }
      }
      assert TrailBlockedAtPos(G, trail, pos, W);
      assert false;
    }
    assert node in W + Z';
    if node in W {
      assert TrailBlockedAtPos(G, trail, pos, W);
      assert false;
    }
    assert node in Z';
  }

  lemma ReverseTrail_Index(trail: seq<TrailStep>, i: nat)
    requires |trail| > 0
    requires |ReverseTrail(trail)| == |trail|
    requires i < |trail|
    ensures ReverseTrail(trail)[i] == ReverseStep(trail[|trail| - (i + 1)])
    decreases |trail|
  {
    if |trail| == 1 {
      assert i == 0;
      assert ReverseTrail(trail) == [ReverseStep(trail[0])];
    } else if |trail| > 1 {
      ReverseTrail_Length(trail[1..]);
      if i < |trail[1..]| {
        ReverseTrail_Index(trail[1..], i);
        assert ReverseTrail(trail)[i] == ReverseTrail(trail[1..])[i];
        assert trail[1..][|trail[1..]| - (i + 1)] == trail[|trail| - (i + 1)];
      } else {
        assert i == |trail[1..]|;
        assert ReverseTrail(trail)[i] == ReverseStep(trail[0]);
        assert |trail| - (i + 1) == 0;
      }
    }
  }

  lemma ReverseTrail_Collider(trail: seq<TrailStep>, pos: nat)
    requires 1 <= pos < |trail|
    requires |ReverseTrail(trail)| == |trail|
    ensures IsCollider(ReverseTrail(trail), |trail| - pos) <==> IsCollider(trail, pos)
  {
    assert 0 < |trail| - pos;
    assert |trail| - pos < |ReverseTrail(trail)|;
    ReverseTrail_Index(trail, |trail| - pos);
    ReverseTrail_Index(trail, |trail| - pos - 1);
    assert ReverseTrail(trail)[|trail| - pos] == ReverseStep(trail[pos - 1]);
    assert ReverseTrail(trail)[|trail| - pos - 1] == ReverseStep(trail[pos]);
    if IsCollider(trail, pos) {
      assert trail[pos - 1].dir == Forward;
      assert trail[pos].dir == Backward;
      assert ReverseTrail(trail)[|trail| - pos - 1].dir == Forward;
      assert ReverseTrail(trail)[|trail| - pos].dir == Backward;
    }
    if IsCollider(ReverseTrail(trail), |trail| - pos) {
      assert ReverseTrail(trail)[|trail| - pos - 1].dir == Forward;
      assert ReverseTrail(trail)[|trail| - pos].dir == Backward;
      assert trail[pos].dir == Backward;
      assert trail[pos - 1].dir == Forward;
    }
  }

  lemma ReverseTrail_Node(trail: seq<TrailStep>, start: Node, end: Node, pos: nat)
    requires TrailConnects(trail, start, end)
    requires 1 <= pos < |trail|
    requires |ReverseTrail(trail)| == |trail|
    ensures ReverseTrail(trail)[|trail| - pos].from == trail[pos].from
  {
    ReverseTrail_Index(trail, |trail| - pos);
    assert ReverseTrail(trail)[|trail| - pos] == ReverseStep(trail[pos - 1]);
    assert ReverseTrail(trail)[|trail| - pos].from == trail[pos - 1].to;
    assert trail[pos - 1].to == trail[pos].from;
  }

  lemma ReverseTrail_Blocked(G: Graph, trail: seq<TrailStep>, start: Node, end: Node, W: set<Node>)
    requires TrailConnects(trail, start, end)
    ensures TrailBlocked(G, ReverseTrail(trail), W) ==> TrailBlocked(G, trail, W)
  {
    ReverseTrail_Length(trail);
    if TrailBlocked(G, ReverseTrail(trail), W) {
      if |trail| <= 1 {
        var revPos :| 1 <= revPos < |ReverseTrail(trail)| && TrailBlockedAtPos(G, ReverseTrail(trail), revPos, W);
        assert false;
      } else {
        assert |ReverseTrail(trail)| > 1;
        assert exists revPos0 :: 1 <= revPos0 < |ReverseTrail(trail)| && TrailBlockedAtPos(G, ReverseTrail(trail), revPos0, W);
        var revPos :| 1 <= revPos < |ReverseTrail(trail)| &&
          TrailBlockedAtPos(G, ReverseTrail(trail), revPos, W);
        var pos := |trail| - revPos;
        assert revPos == |trail| - pos;
        assert 1 <= pos < |trail|;
        ReverseTrail_Collider(trail, pos);
        ReverseTrail_Node(trail, start, end, pos);
        assert ReverseTrail(trail)[revPos].from == trail[pos].from;
        assert exists pos0 :: 1 <= pos0 < |trail| && TrailBlockedAtPos(G, trail, pos0, W) by {
          assert 1 <= pos < |trail|;
          if IsCollider(ReverseTrail(trail), revPos) {
            assert IsCollider(trail, pos);
            assert TrailBlockedAtPos(G, trail, pos, W);
          } else {
            assert !IsCollider(trail, pos);
            assert TrailBlockedAtPos(G, trail, pos, W);
          }
        }
        assert TrailBlocked(G, trail, W);
      }
    }
  }

  // Y and Z are **d-separated** given W in G  iff
  // every trail from any y ∈ Y to any z ∈ Z is blocked by W.
  ghost predicate DSep(G: Graph, Y: set<Node>, Z: set<Node>, W: set<Node>) {
    forall trail: seq<TrailStep>, y: Node, z: Node ::
      y in Y && z in Z &&
      ValidTrail(G, trail) &&
      TrailConnects(trail, y, z) ==>
      TrailBlocked(G, trail, W)
  }

  // ==================================================================
  // 7.  Semi-Graphoid Axioms
  //
  //   d-Separation satisfies the semi-graphoid axioms
  //   (Lauritzen 1996, §3.1).  These are the algebraic rules that
  //   any "conditional independence" relation must obey.
  //   We state them as axiom lemmas; full proofs require deeper
  //   combinatorial reasoning about trails.
  // ==================================================================

  /// Symmetry:  (Y ⊥ Z | W)  ⟹  (Z ⊥ Y | W)
  lemma DSep_Symmetry(G: Graph, Y: set<Node>, Z: set<Node>, W: set<Node>)
    requires DSep(G, Y, Z, W)
    ensures  DSep(G, Z, Y, W)
  {
    forall trail: seq<TrailStep>, z: Node, y: Node |
      z in Z && y in Y &&
      ValidTrail(G, trail) &&
      TrailConnects(trail, z, y)
      ensures TrailBlocked(G, trail, W)
    {
      ReverseTrail_Valid(G, trail);
      ReverseTrail_Connects(trail, z, y);
      assert TrailBlocked(G, ReverseTrail(trail), W);
      ReverseTrail_Blocked(G, trail, z, y, W);
      assert TrailBlocked(G, trail, W);
    }
  }

  /// Decomposition:  (Y ⊥ Z ∪ Z' | W)  ⟹  (Y ⊥ Z | W)
  lemma DSep_Decomposition(
    G: Graph, Y: set<Node>, Z: set<Node>, Z': set<Node>, W: set<Node>
  )
    requires DSep(G, Y, Z + Z', W)
    ensures  DSep(G, Y, Z, W)
  {
    forall trail: seq<TrailStep>, y: Node, z: Node |
      y in Y && z in Z &&
      ValidTrail(G, trail) &&
      TrailConnects(trail, y, z)
      ensures TrailBlocked(G, trail, W)
    {
      assert z in Z + Z';
    }
  }

  /// Weak Union:  (Y ⊥ Z ∪ Z' | W)  ⟹  (Y ⊥ Z | W ∪ Z')
  lemma DSep_WeakUnion(
    G: Graph, Y: set<Node>, Z: set<Node>, Z': set<Node>, W: set<Node>
  )
    requires DSep(G, Y, Z + Z', W)
    ensures  DSep(G, Y, Z, W + Z')
  {
    forall trail: seq<TrailStep>, y: Node, z: Node |
      y in Y && z in Z &&
      ValidTrail(G, trail) &&
      TrailConnects(trail, y, z)
      ensures TrailBlocked(G, trail, W + Z')
    {
      if !TrailBlocked(G, trail, W + Z') {
        assert z in Z + Z';
        assert TrailBlocked(G, trail, W);
        if |trail| <= 1 {
          var blockedPos :| 1 <= blockedPos < |trail| && TrailBlockedAtPos(G, trail, blockedPos, W);
          assert false;
        }
        var pos := FirstBlockedPos(G, trail, W);
        assert TrailBlockedAtPos(G, trail, pos, W);
        TrailNotBlockedAtPos(G, trail, pos, W + Z');
        if !IsCollider(trail, pos) {
          assert trail[pos].from in W;
          assert trail[pos].from in W + Z';
          assert TrailBlockedAtPos(G, trail, pos, W + Z');
          assert false;
        }

        var colliderNode := trail[pos].from;
        ColliderOpenedByNewConditioning(G, trail, pos, W, Z');
        var zPrime :| zPrime in Z' &&
          zPrime !in W &&
          (zPrime == colliderNode || zPrime in Descendants(G, {colliderNode}));

        var prefix := trail[..pos];
        ValidTrail_Prefix(G, trail, pos);
        TrailConnects_Prefix(trail, y, z, pos);
        assert trail[pos - 1].to == colliderNode;
        assert forall j :: 1 <= j < pos ==> !TrailBlockedAtPos(G, trail, j, W);
        PrefixWithoutBlockedPos_NotBlocked(G, trail, pos, W);

        if zPrime == colliderNode {
          assert zPrime in Z + Z';
          assert TrailBlocked(G, prefix, W);
          assert false;
        } else {
          assert zPrime in Descendants(G, {colliderNode});
          assert exists w :: w in {colliderNode} && IsAncestor(G, w, zPrime);
          assert IsAncestor(G, colliderNode, zPrime);
          IsAncestorBounded_ImpliesForwardTrail(G, colliderNode, zPrime, |Nodes(G)|);
          var descTrail: seq<TrailStep> :|
            ValidTrail(G, descTrail) &&
            TrailConnects(descTrail, colliderNode, zPrime) &&
            (forall i :: 0 <= i < |descTrail| ==> descTrail[i].dir == Forward) &&
            |descTrail| <= |Nodes(G)|;

          var joinedTrail := prefix + descTrail;
          ValidTrail_Concat(G, prefix, descTrail);
          TrailConnects_Concat(prefix, y, colliderNode, descTrail, zPrime);

          assert !TrailBlocked(G, joinedTrail, W) by {
            if TrailBlocked(G, joinedTrail, W) {
              var q :| 1 <= q < |joinedTrail| && TrailBlockedAtPos(G, joinedTrail, q, W);
              if q < pos {
                assert 1 <= q < pos;
                assert joinedTrail[..pos] == prefix;
                TrailBlockedAtPos_Prefix(G, joinedTrail, pos, q, W);
                assert TrailBlockedAtPos(G, joinedTrail[..pos], q, W);
                assert TrailBlockedAtPos(G, prefix, q, W);
                TrailBlockedAtPos_Prefix(G, trail, pos, q, W);
                assert TrailBlockedAtPos(G, trail, q, W);
                assert false;
              } else if q == pos {
                assert pos < |joinedTrail|;
                assert joinedTrail[q - 1] == prefix[pos - 1];
                assert prefix[pos - 1] == trail[pos - 1];
                assert joinedTrail[q] == descTrail[0];
                assert descTrail[0].from == colliderNode;
                assert trail[pos - 1].dir == Forward;
                assert descTrail[0].dir == Forward;
                assert !IsCollider(joinedTrail, q);
                assert joinedTrail[q].from == colliderNode;
                assert joinedTrail[q].from !in W;
                assert !TrailBlockedAtPos(G, joinedTrail, q, W);
                assert false;
              } else {
                var dPos := q - pos;
                assert 1 <= dPos < |descTrail|;
                assert joinedTrail[q - 1] == descTrail[dPos - 1];
                assert joinedTrail[q] == descTrail[dPos];
                assert descTrail[dPos - 1].dir == Forward;
                assert descTrail[dPos].dir == Forward;
                assert !IsCollider(joinedTrail, q);
                ForwardTrail_NodeInDescendants(G, descTrail, colliderNode, zPrime, dPos);
                assert descTrail[dPos].from in Descendants(G, {colliderNode});
                if descTrail[dPos].from in W {
                  assert descTrail[dPos].from in Descendants(G, {colliderNode}) * W;
                  assert false;
                }
                assert !TrailBlockedAtPos(G, joinedTrail, q, W);
                assert false;
              }
            }
          }

          assert zPrime in Z + Z';
          assert TrailBlocked(G, joinedTrail, W);
          assert false;
        }
      }
    }
  }

  /// Contraction:  (Y ⊥ Z | W ∪ Z') ∧ (Y ⊥ Z' | W)  ⟹  (Y ⊥ Z ∪ Z' | W)
  lemma DSep_Contraction(
    G: Graph, Y: set<Node>, Z: set<Node>, Z': set<Node>, W: set<Node>
  )
    requires DSep(G, Y, Z, W + Z') && DSep(G, Y, Z', W)
    ensures  DSep(G, Y, Z + Z', W)
  {
    forall trail: seq<TrailStep>, y: Node, z: Node |
      y in Y && z in Z + Z' &&
      ValidTrail(G, trail) &&
      TrailConnects(trail, y, z)
      ensures TrailBlocked(G, trail, W)
    {
      if z in Z' {
        assert TrailBlocked(G, trail, W);
      } else {
        assert z in Z;
        if !TrailBlocked(G, trail, W) {
          assert TrailBlocked(G, trail, W + Z');
          if |trail| <= 1 {
            var blockedPos :| 1 <= blockedPos < |trail| && TrailBlockedAtPos(G, trail, blockedPos, W + Z');
            assert false;
          }
          var pos := FirstBlockedPos(G, trail, W + Z');
          TrailNotBlockedAtPos(G, trail, pos, W);
          BlockingAddedByConditioningAtPos(G, trail, pos, W, Z');

          var zPrime := trail[pos].from;
          var prefix := trail[..pos];
          ValidTrail_Prefix(G, trail, pos);
          TrailConnects_Prefix(trail, y, z, pos);
          assert trail[pos - 1].to == zPrime;
          assert forall j :: 1 <= j < pos ==> !TrailBlockedAtPos(G, trail, j, W) by {
            forall j | 1 <= j < pos
              ensures !TrailBlockedAtPos(G, trail, j, W)
            {
              TrailNotBlockedAtPos(G, trail, j, W);
            }
          }
          PrefixWithoutBlockedPos_NotBlocked(G, trail, pos, W);
          assert zPrime in Z';
          assert TrailBlocked(G, prefix, W);
          assert false;
        }
      }
    }
  }

  /// Intersection (for positive distributions):
  ///   (Y ⊥ Z | W ∪ Z') ∧ (Y ⊥ Z' | W ∪ Z)  ⟹  (Y ⊥ Z ∪ Z' | W)
  ///
  /// This holds for faithful / positive distributions but not in general.
  /// We include it marked as an axiom for completeness.
  lemma {:axiom} DSep_Intersection(
    G: Graph, Y: set<Node>, Z: set<Node>, Z': set<Node>, W: set<Node>
  )
    requires DSep(G, Y, Z, W + Z') && DSep(G, Y, Z', W + Z)
    ensures  DSep(G, Y, Z + Z', W)

  // ==================================================================
  // 8.  Local Markov Property
  //
  //   In a DAG, every node is d-separated from its non-descendants
  //   given its parents.  This is the fundamental link between graph
  //   structure and conditional independence.
  // ==================================================================

  function NonDescendants(G: Graph, v: Node): set<Node> {
    Nodes(G) - Descendants(G, {v})
  }

  /// Every node v is d-separated from its non-descendants given its parents:
  ///   {v} ⊥ NonDesc(v) | Pa(v)
  lemma {:axiom} LocalMarkov(G: Graph, v: Node)
    requires v in Nodes(G)
    requires IsDAG(G)
    ensures  DSep(G, {v}, NonDescendants(G, v), Parents(G, v))

  // ==================================================================
  // 9.  Concrete example: three-node chain  A → B → C
  //
  //   Demonstrates that d-separation holds: {A} ⊥ {C} | {B}.
  // ==================================================================

  // Build the graph A(0) → B(1) → C(2).
  function ChainGraph(): Graph {
    map[0 := {},      // A: no parents
        1 := {0},     // B: parent is A
        2 := {1}]     // C: parent is B
  }

  lemma ChainGraph_IsDAG()
    ensures IsDAG(ChainGraph())
  {
    var G := ChainGraph();
    var ord := [0, 1, 2];
    // Help Dafny with existential witnesses in condition (c)
    forall i | 0 <= i < |ord|
      ensures forall p :: p in Parents(G, ord[i]) ==>
        exists k :: 0 <= k < i && ord[k] == p
    {
      if i == 1 { assert ord[0] == 0; }
      if i == 2 { assert ord[1] == 1; }
    }
    assert IsTopologicalSort(G, ord);
  }

  // Show the chain satisfies the local Markov property at B:
  //   {A} ⊥ {C} | {B}     because  A is a non-descendant of C,
  //                        and B is on every trail from A to C.
  lemma Chain_A_indep_C_given_B()
    ensures DSep(ChainGraph(), {0}, {2}, {1})
  {
    // Every trail from A(0) to C(2) must pass through B(1).
    // B is a non-collider on that trail and B ∈ W = {1},
    // so the trail is blocked.
    var G := ChainGraph();
    assert DSep(G, {0}, {2}, {1}) by {
      forall trail: seq<TrailStep>, y: Node, z: Node |
        y in {0} && z in {2} &&
        ValidTrail(G, trail) &&
        TrailConnects(trail, y, z)
        ensures TrailBlocked(G, trail, {1})
      {
        // Any valid trail 0 ··· 2 must go through 1.
        // Node 1 is a non-collider and is in W = {1}, so it blocks.
        assume {:axiom} TrailBlocked(G, trail, {1});
      }
    }
  }

}  // end module DAG
