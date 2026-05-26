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

  // ==================================================================
  // 2a. Ghost infrastructure for KahnsAlgorithm correctness
  // ==================================================================

  // A partial topological sort: for each element order[i], all its
  // parents appear somewhere before index i in order.
  ghost predicate IsPartialTopoSort(G: Graph, order: seq<Node>) {
    forall i | 0 <= i < |order| ::
      forall p | p in Parents(G, order[i]) ::
        exists k | 0 <= k < i :: order[k] == p
  }

  // The set of elements in a sequence.
  ghost function OrderSet(order: seq<Node>): set<Node> {
    set i | 0 <= i < |order| :: order[i]
  }

  // Appending v extends OrderSet by exactly {v}.
  lemma OrderSet_Append(order: seq<Node>, v: Node)
    ensures OrderSet(order + [v]) == OrderSet(order) + {v}
  {
    var ext := order + [v];
    assert forall x | x in OrderSet(ext) :: x in OrderSet(order) + {v} by {
      forall x | x in OrderSet(ext)
        ensures x in OrderSet(order) + {v}
      {
        var i :| 0 <= i < |ext| && ext[i] == x;
        if i < |order| {
          assert order[i] == ext[i];
        } else {
          assert i == |order| && ext[i] == v;
        }
      }
    }
    assert forall x | x in OrderSet(order) + {v} :: x in OrderSet(ext) by {
      forall x | x in OrderSet(order) + {v}
        ensures x in OrderSet(ext)
      {
        if x in OrderSet(order) {
          var i :| 0 <= i < |order| && order[i] == x;
          assert ext[i] == x;
        } else {
          assert x == v;
          assert ext[|order|] == v;
        }
      }
    }
  }

  // Extending a partial sort: if v's parents are all already in order,
  // then order + [v] is still a partial topological sort.
  lemma IsPartialTopoSort_Extend(G: Graph, order: seq<Node>, v: Node)
    requires IsPartialTopoSort(G, order)
    requires forall p | p in Parents(G, v) ::
               exists k | 0 <= k < |order| :: order[k] == p
    ensures IsPartialTopoSort(G, order + [v])
  {
    var ext := order + [v];
    forall i | 0 <= i < |ext|
      ensures forall p | p in Parents(G, ext[i]) ::
                exists k | 0 <= k < i :: ext[k] == p
    {
      if i < |order| {
        assert ext[i] == order[i];
        forall p | p in Parents(G, order[i])
          ensures exists k | 0 <= k < i :: ext[k] == p
        {
          var k :| 0 <= k < i && order[k] == p;
          assert ext[k] == order[k] == p;
        }
      } else {
        assert i == |order| && ext[i] == v;
        forall p | p in Parents(G, v)
          ensures exists k | 0 <= k < i :: ext[k] == p
        {
          var k :| 0 <= k < |order| && order[k] == p;
          assert ext[k] == order[k] == p;
          assert k < i;
        }
      }
    }
  }

  // L2: A non-empty subset of a DAG's nodes has some node with no
  // predecessor still in that subset (given the degree invariant).
  lemma DagImpliesZeroInDegree(G: Graph, remaining: map<Node, nat>)
    requires IsDAG(G)
    requires remaining.Keys != {}
    requires remaining.Keys <= Nodes(G)
    requires forall u | u in remaining.Keys ::
      remaining[u] == |Parents(G, u) * remaining.Keys| + |Parents(G, u) - Nodes(G)|
    ensures ZeroInDegreeNodes(remaining) != {}
  {
    var ord :| IsTopologicalSort(G, ord);
    // Walk ord to find the first element that is still in remaining.Keys.
    var i := 0;
    while i < |ord| && ord[i] !in remaining.Keys
      invariant 0 <= i <= |ord|
      invariant forall j | 0 <= j < i :: ord[j] !in remaining.Keys
      decreases |ord| - i
    {
      i := i + 1;
    }
    // The loop must terminate before |ord|: remaining.Keys is non-empty,
    // all its elements are in Nodes(G), and ord enumerates all of Nodes(G).
    assert i < |ord| by {
      var w :| w in remaining.Keys;
      assert w in Nodes(G);
      assert w in ord;
    }
    var v := ord[i];
    assert v in remaining.Keys;
    // All parents of v appear before index i in ord (topological order).
    // By the loop invariant, none of them are in remaining.Keys.
    assert Parents(G, v) * remaining.Keys == {} by {
      forall p | p in Parents(G, v)
        ensures p !in remaining.Keys
      {
        var k :| 0 <= k < i && ord[k] == p;
        assert ord[k] !in remaining.Keys;
      }
    }
    // All parents of v are in Nodes(G): they appear in ord.
    assert Parents(G, v) - Nodes(G) == {} by {
      forall p | p in Parents(G, v)
        ensures p in Nodes(G)
      {
        var k :| 0 <= k < i && ord[k] == p;
        assert ord[k] in Nodes(G);
      }
    }
    assert remaining[v] == 0;
    assert v in ZeroInDegreeNodes(remaining);
  }

  // L3: The initial in-degree map satisfies the degree invariant.
  lemma DegreeInvariant_Init(G: Graph)
    ensures forall u | u in InDegreeMap(G) ::
      InDegreeMap(G)[u] == |Parents(G, u) * Nodes(G)| + |Parents(G, u) - Nodes(G)|
  {
    forall u | u in InDegreeMap(G)
      ensures InDegreeMap(G)[u] == |Parents(G, u) * Nodes(G)| + |Parents(G, u) - Nodes(G)|
    {
      // Parents(G, u) partitions into (∩ Nodes(G)) and (- Nodes(G)).
      assert Parents(G, u) == (Parents(G, u) * Nodes(G)) + (Parents(G, u) - Nodes(G));
      assert (Parents(G, u) * Nodes(G)) * (Parents(G, u) - Nodes(G)) == {};
    }
  }

  // L4: The degree invariant is preserved after one Kahn step.
  // Precondition: v has degree 0 in old_rem (all its predecessors already processed).
  // Effect: remove v from remaining, decrement children's degree by 1.
  lemma DegreeInvariant_Update(G: Graph, old_rem: map<Node, nat>, v: Node)
    requires v in old_rem.Keys
    requires old_rem.Keys <= Nodes(G)
    requires old_rem[v] == 0
    requires forall u | u in old_rem.Keys ::
      old_rem[u] == |Parents(G, u) * old_rem.Keys| + |Parents(G, u) - Nodes(G)|
    ensures
      var K' := old_rem.Keys - {v};
      var new_rem := map u | u in old_rem && u != v ::
        if u in Children(G, v) && old_rem[u] > 0 then old_rem[u] - 1
        else old_rem[u];
      forall u | u in K' ::
        new_rem[u] == |Parents(G, u) * K'| + |Parents(G, u) - Nodes(G)|
  {
    var K' := old_rem.Keys - {v};
    var new_rem := map u | u in old_rem && u != v ::
      if u in Children(G, v) && old_rem[u] > 0 then old_rem[u] - 1
      else old_rem[u];
    forall u | u in K'
      ensures new_rem[u] == |Parents(G, u) * K'| + |Parents(G, u) - Nodes(G)|
    {
      assert u in old_rem && u != v;
      if v in Parents(G, u) {
        // u is a child of v.
        assert u in Nodes(G);
        assert u in Children(G, v);
        // v ∈ Parents(G, u) ∩ old_rem.Keys → old_rem[u] ≥ 1.
        assert v in Parents(G, u) * old_rem.Keys;
        assert |Parents(G, u) * old_rem.Keys| >= 1;
        assert old_rem[u] >= 1;
        assert new_rem[u] == old_rem[u] - 1;
        // Removing v from old_rem.Keys shrinks the intersection by exactly 1.
        assert Parents(G, u) * K' == (Parents(G, u) * old_rem.Keys) - {v};
        assert |Parents(G, u) * K'| == |Parents(G, u) * old_rem.Keys| - 1;
      } else {
        // v ∉ Parents(G, u): no decrement, and intersection unchanged.
        assert new_rem[u] == old_rem[u];
        assert v !in Parents(G, u);
        assert Parents(G, u) * K' == Parents(G, u) * old_rem.Keys;
      }
    }
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

  method {:vcs_split_on_every_assert} KahnsAlgorithm(G: Graph) returns (result: Option<seq<Node>>)
    ensures result.Some? ==> IsTopologicalSort(G, result.value)
    ensures result.None? ==> !IsDAG(G)
  {
    var deg := InDegreeMap(G);
    var order: seq<Node> := [];
    var remaining := deg;
    // Establish the degree invariant for the initial state (remaining = InDegreeMap(G)).
    DegreeInvariant_Init(G);

    while remaining != map[]
      invariant remaining.Keys <= Nodes(G)
      invariant OrderSet(order) + remaining.Keys == Nodes(G)
      invariant OrderSet(order) * remaining.Keys == {}
      invariant forall i, j | 0 <= i < j < |order| :: order[i] != order[j]
      invariant forall u | u in remaining.Keys ::
        remaining[u] == |Parents(G, u) * remaining.Keys| + |Parents(G, u) - Nodes(G)|
      invariant IsPartialTopoSort(G, order)
      decreases remaining.Keys
    {
      var zeros := ZeroInDegreeNodes(remaining);
      if zeros == {} {
        // Remaining nodes form a cycle — no zero-in-degree node exists.
        assert !IsDAG(G) by {
          if IsDAG(G) {
            assert remaining.Keys != {} by {
              if remaining.Keys == {} { assert remaining == map[]; assert false; }
            }
            DagImpliesZeroInDegree(G, remaining);
            assert false;
          }
        }
        return None;
      }

      var v :| v in zeros;
      ghost var old_order := order;
      var old_rem := remaining;

      // v has in-degree 0 in old_rem.
      assert old_rem[v] == 0;

      // Single-step update over the pre-state snapshot old_rem.
      remaining := map u | u in old_rem && u != v ::
        if u in Children(G, v) && old_rem[u] > 0 then old_rem[u] - 1
        else old_rem[u];
      order := order + [v];

      // --- I1: remaining.Keys <= Nodes(G) ---
      assert remaining.Keys == old_rem.Keys - {v};
      assert remaining.Keys <= Nodes(G);

      // --- I2 & I3: partition / disjoint ---
      // v was in old_rem.Keys (came from zeros ⊆ remaining = old_rem).
      assert v in old_rem.Keys;
      // v was NOT in OrderSet(old_order): v ∈ old_rem.Keys and I3 said they are disjoint.
      assert v !in OrderSet(old_order);
      OrderSet_Append(old_order, v);
      assert OrderSet(order) == OrderSet(old_order) + {v};
      assert OrderSet(order) + remaining.Keys == Nodes(G);
      assert OrderSet(order) * remaining.Keys == {};

      // --- I4: no duplicates ---
      assert forall i, j | 0 <= i < j < |order| :: order[i] != order[j] by {
        forall i, j | 0 <= i < j < |order|
          ensures order[i] != order[j]
        {
          if j < |old_order| {
            assert order[i] == old_order[i];
            assert order[j] == old_order[j];
          } else {
            // j == |old_order|; order[j] == v.
            assert j == |old_order|;
            assert order[j] == v;
            assert order[i] == old_order[i];
            assert old_order[i] in OrderSet(old_order);
            assert v !in OrderSet(old_order);
          }
        }
      }

      // --- I6: degree invariant ---
      // DegreeInvariant_Update produces the same map as our remaining.
      DegreeInvariant_Update(G, old_rem, v);
      assert forall u | u in remaining.Keys ::
        remaining[u] == |Parents(G, u) * remaining.Keys| + |Parents(G, u) - Nodes(G)|
      by {
        forall u | u in remaining.Keys
          ensures remaining[u] == |Parents(G, u) * remaining.Keys| + |Parents(G, u) - Nodes(G)|
        {
          assert remaining.Keys == old_rem.Keys - {v};
          assert u in old_rem.Keys && u != v;
        }
      }

      // --- I7: IsPartialTopoSort ---
      // v ∈ zeros → old_rem[v] == 0 → all parents of v are in OrderSet(old_order).
      assert forall p | p in Parents(G, v) ::
        exists k | 0 <= k < |old_order| :: old_order[k] == p
      by {
        assert |Parents(G, v) * old_rem.Keys| + |Parents(G, v) - Nodes(G)| == 0;
        assert |Parents(G, v) * old_rem.Keys| == 0;
        assert |Parents(G, v) - Nodes(G)| == 0;
        assert Parents(G, v) * old_rem.Keys == {};
        assert Parents(G, v) - Nodes(G) == {};
        forall p | p in Parents(G, v)
          ensures exists k | 0 <= k < |old_order| :: old_order[k] == p
        {
          assert p in Nodes(G);
          assert p !in old_rem.Keys;
          assert p in OrderSet(old_order) + old_rem.Keys;
          assert p in OrderSet(old_order);
          var k :| 0 <= k < |old_order| && old_order[k] == p;
        }
      }
      IsPartialTopoSort_Extend(G, old_order, v);
      assert IsPartialTopoSort(G, order);
    }

    // remaining == map[] → remaining.Keys == {} → OrderSet(order) == Nodes(G).
    assert remaining.Keys == {} by {
      if remaining.Keys != {} { assert remaining != map[]; assert false; }
    }
    assert OrderSet(order) == Nodes(G);
    assert IsTopologicalSort(G, order) by {
      assert forall v | v in Nodes(G) :: v in order by {
        forall v | v in Nodes(G) ensures v in order {
          assert v in OrderSet(order);
          var k :| 0 <= k < |order| && order[k] == v;
        }
      }
      assert forall i | 0 <= i < |order| :: order[i] in Nodes(G) by {
        forall i | 0 <= i < |order| ensures order[i] in Nodes(G) {
          assert order[i] in OrderSet(order);
        }
      }
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

  lemma TopologicalSort_Length(G: Graph, ord: seq<Node>)
    requires IsTopologicalSort(G, ord)
    ensures |ord| == |Nodes(G)|
  {
    var seen: set<Node> := {};
    var i := 0;
    while i < |ord|
      invariant 0 <= i <= |ord|
      invariant seen <= Nodes(G)
      invariant forall j :: 0 <= j < i ==> ord[j] in seen
      invariant forall v :: v in seen ==> exists j :: 0 <= j < i && ord[j] == v
      invariant |seen| == i
    {
      assert ord[i] in Nodes(G);
      if ord[i] in seen {
        var j :| 0 <= j < i && ord[j] == ord[i];
        assert ord[j] != ord[i];
        assert false;
      }
      seen := seen + {ord[i]};
      assert |seen| == i + 1;
      i := i + 1;
    }
    assert seen == Nodes(G) by {
      assert forall v :: v in seen ==> v in Nodes(G);
      assert forall v :: v in Nodes(G) ==> v in seen by {
        forall v | v in Nodes(G)
          ensures v in seen
        {
          assert v in ord;
          var j :| 0 <= j < |ord| && ord[j] == v;
          assert ord[j] in seen;
        }
      }
    }
    assert |seen| == |Nodes(G)|;
  }

  // ==================================================================
  // 2b. Filtered Topological Sort
  //
  // FilterSort(ord, nodes) is the subsequence of `ord` keeping only
  // elements in `nodes`.  If `ord` is a topological sort of G and
  // `nodes == Nodes(G) - X`, then FilterSort(ord, nodes) is a valid
  // topological sort of the induced subgraph RemoveNodes(G, X).
  //
  // This is the formal counterpart of the Python filtering step:
  //   ordering = [v for v in ordering if v in nodes]
  // that _identify() performs before each recursive call.
  // ==================================================================

  // Filter a sequence to elements in `nodes`, preserving order.
  ghost function FilterSort(ord: seq<Node>, nodes: set<Node>): seq<Node>
    decreases |ord|
  {
    if ord == [] then []
    else if ord[0] in nodes then [ord[0]] + FilterSort(ord[1..], nodes)
    else FilterSort(ord[1..], nodes)
  }

  // v appears in FilterSort(ord, nodes) iff v is in both ord and nodes.
  lemma FilterSort_Contains(ord: seq<Node>, nodes: set<Node>, v: Node)
    ensures v in FilterSort(ord, nodes) <==> (v in nodes && v in ord)
    decreases |ord|
  {
    if ord == [] {
    } else {
      FilterSort_Contains(ord[1..], nodes, v);
      if ord[0] !in nodes {
        // FilterSort(ord, nodes) == FilterSort(ord[1..], nodes)
        // Need backward direction: v in nodes && v in ord → v in FilterSort(tail)
        // Only concern: v == ord[0], but ord[0] !in nodes contradicts v in nodes.
        if v in nodes && v == ord[0] { assert false; }
      }
      // Case ord[0] in nodes: FilterSort(ord) = [ord[0]] + FilterSort(tail)
      // IH covers tail; v == ord[0] ∈ nodes covers the head.  Automatic.
    }
  }

  // Every element of FilterSort(ord, nodes) is in nodes.
  lemma FilterSort_Sound(ord: seq<Node>, nodes: set<Node>)
    ensures forall i | 0 <= i < |FilterSort(ord, nodes)| ::
              FilterSort(ord, nodes)[i] in nodes
    decreases |ord|
  {
    if ord != [] { FilterSort_Sound(ord[1..], nodes); }
  }

  // FilterSort inherits the no-duplicates property from ord.
  lemma FilterSort_NoDup(ord: seq<Node>, nodes: set<Node>)
    requires forall i, j | 0 <= i < j < |ord| :: ord[i] != ord[j]
    ensures forall i, j | 0 <= i < j < |FilterSort(ord, nodes)| ::
              FilterSort(ord, nodes)[i] != FilterSort(ord, nodes)[j]
    decreases |ord|
  {
    if ord == [] {
    } else {
      assert forall i, j | 0 <= i < j < |ord[1..]| :: ord[1..][i] != ord[1..][j];
      FilterSort_NoDup(ord[1..], nodes);
      if ord[0] in nodes {
        // FilterSort(ord) == [ord[0]] + FilterSort(tail)
        // Need ord[0] ∉ FilterSort(tail): it's not in ord[1..] (no-dups), so
        // FilterSort_Contains says it's not in FilterSort(tail) either.
        assert ord[0] !in ord[1..] by {
          forall k | 0 <= k < |ord[1..]| ensures ord[1..][k] != ord[0] {
            assert ord[k + 1] != ord[0]; // no-dups at (0, k+1)
          }
        }
        FilterSort_Contains(ord[1..], nodes, ord[0]);
        assert ord[0] !in FilterSort(ord[1..], nodes);
      }
    }
  }

  // FilterSort distributes over a single-step extension of the prefix:
  //   FilterSort(ord[..i+1], nodes)
  //     == FilterSort(ord[..i], nodes) ++ (if ord[i] ∈ nodes then [ord[i]] else [])
  lemma {:vcs_split_on_every_assert} FilterSort_Append(
    ord: seq<Node>, nodes: set<Node>, i: nat
  )
    requires i < |ord|
    ensures FilterSort(ord[..i + 1], nodes) ==
            FilterSort(ord[..i], nodes) + (if ord[i] in nodes then [ord[i]] else [])
    decreases i
  {
    if i == 0 {
      assert ord[..1] == [ord[0]];
      assert ord[..0] == [];
      // FilterSort([ord[0]], nodes):
      //   ord[0] in nodes  → [ord[0]] + FilterSort([], nodes) = [ord[0]] = [] + [ord[0]] ✓
      //   ord[0] !in nodes → FilterSort([], nodes)            = []       = [] + []       ✓
    } else {
      // Establish key sequence-slice equalities once, use them throughout.
      assert ord[..i + 1][1..] == ord[1..i + 1] by {
        forall k {:trigger ord[..i + 1][1..][k]} | 0 <= k < i
          ensures ord[..i + 1][1..][k] == ord[1..i + 1][k] {}
      }
      assert ord[..i][1..] == ord[1..i] by {
        forall k {:trigger ord[..i][1..][k]} | 0 <= k < i - 1
          ensures ord[..i][1..][k] == ord[1..i][k] {}
      }
      assert ord[1..][..i] == ord[1..i + 1] by {
        forall k {:trigger ord[1..][..i][k]} | 0 <= k < i
          ensures ord[1..][..i][k] == ord[1..i + 1][k] {}
      }
      assert ord[1..][..i - 1] == ord[1..i] by {
        forall k {:trigger ord[1..][..i - 1][k]} | 0 <= k < i - 1
          ensures ord[1..][..i - 1][k] == ord[1..i][k] {}
      }
      assert ord[1..][i - 1] == ord[i];

      // IH on ord[1..] at index i-1:
      FilterSort_Append(ord[1..], nodes, i - 1);
      // → FilterSort(ord[1..i+1], nodes)
      //     == FilterSort(ord[1..i], nodes) + (if ord[i] in nodes then [ord[i]] else [])

      if ord[0] in nodes {
        // FilterSort(ord[..i+1]) = [ord[0]] + FilterSort(ord[1..i+1])
        assert FilterSort(ord[..i + 1], nodes) == [ord[0]] + FilterSort(ord[1..i + 1], nodes);
        // FilterSort(ord[..i])   = [ord[0]] + FilterSort(ord[1..i])
        assert FilterSort(ord[..i], nodes) == [ord[0]] + FilterSort(ord[1..i], nodes);
      } else {
        // FilterSort(ord[..i+1]) = FilterSort(ord[1..i+1])
        assert FilterSort(ord[..i + 1], nodes) == FilterSort(ord[1..i + 1], nodes);
        // FilterSort(ord[..i])   = FilterSort(ord[1..i])
        assert FilterSort(ord[..i], nodes) == FilterSort(ord[1..i], nodes);
      }
    }
  }

  // FilterSort(ord, Nodes(G)−X) is a valid topological sort of RemoveNodes(G, X).
  // This is the formal counterpart of the Python ordering-filter in _identify().
  lemma {:vcs_split_on_every_assert} FilteredSort_Valid(G: Graph, X: set<Node>, ord: seq<Node>)
    requires IsTopologicalSort(G, ord)
    ensures IsTopologicalSort(RemoveNodes(G, X), FilterSort(ord, Nodes(G) - X))
  {
    var GX    := RemoveNodes(G, X);
    var nodes := Nodes(G) - X;
    var ordX  := FilterSort(ord, nodes);

    // (a1) Every node of GX appears in ordX.
    forall v | v in Nodes(GX) ensures v in ordX {
      assert v in nodes;
      assert v in ord; // IsTopologicalSort(G, ord): all G-nodes appear in ord
      FilterSort_Contains(ord, nodes, v);
    }

    // (a2) Every element of ordX is in Nodes(GX) = nodes.
    FilterSort_Sound(ord, nodes);

    // (b) No duplicates in ordX.
    FilterSort_NoDup(ord, nodes);

    // (d) Every parent in GX appears before its child in ordX.
    // Proved by rebuilding ordX in a loop while tracking the invariant
    // that the partial ordX equals FilterSort(ord[..i], nodes).
    var ordX2: seq<Node> := [];
    var i := 0;
    while i < |ord|
      invariant 0 <= i <= |ord|
      invariant ordX2 == FilterSort(ord[..i], nodes)
      invariant forall j | 0 <= j < |ordX2| ::
                  forall p | p in Parents(GX, ordX2[j]) ::
                    exists k | 0 <= k < j :: ordX2[k] == p
    {
      FilterSort_Append(ord, nodes, i);

      if ord[i] in nodes {
        var v := ord[i];
        assert Parents(GX, v) == Parents(G, v) - X;

        // Prove all GX-parents of v sit in ordX2 (with their indices).
        assert forall p | p in Parents(GX, v) ::
                 exists kk | 0 <= kk < |ordX2| :: ordX2[kk] == p by {
          forall p | p in Parents(GX, v) ensures
            exists kk | 0 <= kk < |ordX2| :: ordX2[kk] == p
          {
            assert p in Parents(G, v);
            assert p !in X;
            assert p in nodes;
            var h :| 0 <= h < i && ord[h] == p;
            assert p in ord[..i];
            FilterSort_Contains(ord[..i], nodes, p);
            // p in FilterSort(ord[..i], nodes) = ordX2
          }
        }

        // Prove the invariant for (ordX2 + [v]) before the assignment.
        var newOrdX2 := ordX2 + [v];
        assert forall j | 0 <= j < |newOrdX2| ::
                 forall p | p in Parents(GX, newOrdX2[j]) ::
                   exists k | 0 <= k < j :: newOrdX2[k] == p by {
          forall j | 0 <= j < |newOrdX2|
            ensures forall p | p in Parents(GX, newOrdX2[j]) ::
                      exists k | 0 <= k < j :: newOrdX2[k] == p
          {
            forall p | p in Parents(GX, newOrdX2[j])
              ensures exists k | 0 <= k < j :: newOrdX2[k] == p
            {
              if j < |ordX2| {
                // Existing position: use the loop invariant.
                assert newOrdX2[j] == ordX2[j];
                assert p in Parents(GX, ordX2[j]);
                assert exists k0 | 0 <= k0 < j :: ordX2[k0] == p;
                var k :| 0 <= k < j && ordX2[k] == p;
                assert 0 <= k < j;
                assert newOrdX2[k] == ordX2[k];
                assert newOrdX2[k] == p;
              } else {
                assert j == |ordX2|;
                assert newOrdX2[j] == v;
                assert p in Parents(GX, v);
                assert exists kk :: 0 <= kk < |ordX2| && ordX2[kk] == p;
                var kk :| 0 <= kk < |ordX2| && ordX2[kk] == p;
                assert 0 <= kk < j;
                assert newOrdX2[kk] == ordX2[kk];
                assert newOrdX2[kk] == p;
              }
            }
          }
        }

        ordX2 := newOrdX2;
        assert ordX2 == FilterSort(ord[..i + 1], nodes);
      } else {
        assert ordX2 == FilterSort(ord[..i + 1], nodes);
      }
      i := i + 1;
    }

    assert ord[..|ord|] == ord;
    assert ordX2 == ordX;
    // The loop invariant at exit gives the parent-ordering property for ordX2 = ordX.
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

  lemma IsAncestorBounded_Transitive(
    G: Graph, u: Node, v: Node, w: Node, uvFuel: nat, vwFuel: nat
  )
    requires IsAncestorBounded(G, u, v, uvFuel)
    requires IsAncestorBounded(G, v, w, vwFuel)
    ensures IsAncestorBounded(G, u, w, uvFuel + vwFuel)
    decreases uvFuel
  {
    if u == v {
      IsAncestorBounded_Monotone(G, v, w, vwFuel, uvFuel + vwFuel);
    } else {
      assert uvFuel > 0;
      var x :| x in Children(G, u) && IsAncestorBounded(G, x, v, uvFuel - 1);
      IsAncestorBounded_Transitive(G, x, v, w, uvFuel - 1, vwFuel);
      assert uvFuel + vwFuel > 0;
      assert (uvFuel + vwFuel) - 1 == (uvFuel - 1) + vwFuel;
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

  lemma ForwardTrail_EndInDescendants(G: Graph, trail: seq<TrailStep>, start: Node, end: Node)
    requires ValidTrail(G, trail)
    requires TrailConnects(trail, start, end)
    requires forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward
    requires |trail| <= |Nodes(G)|
    ensures end in Descendants(G, {start})
  {
    ForwardTrail_ImpliesAncestorBounded(G, trail, start, end);
    IsAncestorBounded_Monotone(G, start, end, |trail|, |Nodes(G)|);
    assert IsAncestor(G, start, end);
    assert trail[|trail| - 1].dir == Forward;
    assert trail[|trail| - 1].to == end;
    assert trail[|trail| - 1].from in Parents(G, end);
    assert end in Nodes(G);
    assert start in {start};
  }

  lemma ForwardTrail_StartAtOrBeforeEndInTopologicalOrder(
    G: Graph, ord: seq<Node>, trail: seq<TrailStep>, start: Node, end: Node
  )
    requires IsTopologicalSort(G, ord)
    requires ValidTrail(G, trail)
    requires TrailConnects(trail, start, end)
    requires forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward
    ensures exists i, j :: 0 <= i <= j < |ord| && ord[i] == start && ord[j] == end
    decreases |trail|
  {
    if |trail| == 1 {
      var endIdx :| 0 <= endIdx < |ord| && ord[endIdx] == end;
      assert trail[0].from == start;
      assert trail[0].to == end;
      assert start in Parents(G, end);
      var startIdx :| 0 <= startIdx < endIdx && ord[startIdx] == start;
      assert exists i0, j0 :: 0 <= i0 <= j0 < |ord| && ord[i0] == start && ord[j0] == end by {
        assert startIdx == startIdx;
        assert endIdx == endIdx;
      }
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
      ForwardTrail_StartAtOrBeforeEndInTopologicalOrder(G, ord, trail[1..], trail[1].from, end);
      var midIdx, endIdx :| 0 <= midIdx <= endIdx < |ord| && ord[midIdx] == trail[1].from && ord[endIdx] == end;
      assert trail[0].to == trail[1].from;
      assert trail[0].from == start;
      assert start in Parents(G, trail[1].from);
      var startIdx :| 0 <= startIdx < midIdx && ord[startIdx] == start;
      assert exists i0, j0 :: 0 <= i0 <= j0 < |ord| && ord[i0] == start && ord[j0] == end by {
        assert startIdx <= endIdx;
      }
    }
  }

  lemma ForwardTrail_StartBeforeEndInTopologicalOrder(
    G: Graph, ord: seq<Node>, trail: seq<TrailStep>, start: Node, end: Node
  )
    requires IsTopologicalSort(G, ord)
    requires ValidTrail(G, trail)
    requires TrailConnects(trail, start, end)
    requires forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward
    ensures exists i, j :: 0 <= i < j < |ord| && ord[i] == start && ord[j] == end
  {
    if |trail| == 1 {
      var endIdx :| 0 <= endIdx < |ord| && ord[endIdx] == end;
      assert trail[0].from == start;
      assert trail[0].to == end;
      assert start in Parents(G, end);
      var startIdx :| 0 <= startIdx < endIdx && ord[startIdx] == start;
      assert exists i0, j0 :: 0 <= i0 < j0 < |ord| && ord[i0] == start && ord[j0] == end by {
        assert startIdx == startIdx;
        assert endIdx == endIdx;
      }
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
      ForwardTrail_StartAtOrBeforeEndInTopologicalOrder(G, ord, trail[1..], trail[1].from, end);
      var midIdx, endIdx :| 0 <= midIdx <= endIdx < |ord| && ord[midIdx] == trail[1].from && ord[endIdx] == end;
      assert trail[0].to == trail[1].from;
      assert trail[0].from == start;
      assert start in Parents(G, trail[1].from);
      var startIdx :| 0 <= startIdx < midIdx && ord[startIdx] == start;
      assert exists i0, j0 :: 0 <= i0 < j0 < |ord| && ord[i0] == start && ord[j0] == end by {
        assert startIdx < endIdx;
      }
    }
  }

  lemma ForwardTrail_ImpliesAncestorBoundedByOrderDistance(
    G: Graph,
    ord: seq<Node>,
    trail: seq<TrailStep>,
    start: Node,
    end: Node,
    startIdx: nat,
    endIdx: nat
  )
    requires IsTopologicalSort(G, ord)
    requires ValidTrail(G, trail)
    requires TrailConnects(trail, start, end)
    requires forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward
    requires 0 <= startIdx < endIdx < |ord|
    requires ord[startIdx] == start
    requires ord[endIdx] == end
    ensures IsAncestorBounded(G, start, end, endIdx - startIdx)
    decreases endIdx - startIdx
  {
    if |trail| == 1 {
      assert endIdx - startIdx > 0;
      assert trail[0].from == start;
      assert trail[0].to == end;
      assert start in Parents(G, end);
      assert end in Children(G, start);
      Ancestor_Reflexive(G, end);
      IsAncestorBounded_Monotone(G, end, end, 0, endIdx - startIdx - 1);
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
      ForwardTrail_StartBeforeEndInTopologicalOrder(G, ord, trail[1..], trail[1].from, end);
      var midIdx0, endIdx0 :| 0 <= midIdx0 < endIdx0 < |ord| && ord[midIdx0] == trail[1].from && ord[endIdx0] == end;
      assert endIdx0 == endIdx by {
        if endIdx0 < endIdx {
          assert ord[endIdx0] != ord[endIdx];
        } else if endIdx < endIdx0 {
          assert ord[endIdx] != ord[endIdx0];
        }
      }
      ForwardTrail_ImpliesAncestorBoundedByOrderDistance(G, ord, trail[1..], trail[1].from, end, midIdx0, endIdx);
      assert trail[0].to == trail[1].from;
      assert trail[0].from == start;
      assert start in Parents(G, trail[1].from);
      var parentIdx :| 0 <= parentIdx < midIdx0 && ord[parentIdx] == start;
      assert parentIdx == startIdx by {
        if parentIdx < startIdx {
          assert ord[parentIdx] != ord[startIdx];
        } else if startIdx < parentIdx {
          assert ord[startIdx] != ord[parentIdx];
        }
      }
      assert endIdx - midIdx0 <= endIdx - startIdx - 1;
      IsAncestorBounded_Monotone(G, trail[1].from, end, endIdx - midIdx0, endIdx - startIdx - 1);
      assert IsAncestorBounded(G, trail[1].from, end, endIdx - startIdx - 1);
      assert trail[1].from in Children(G, trail[0].from);
    }
  }

  lemma ForwardTrail_EndInDescendants_DAG(G: Graph, trail: seq<TrailStep>, start: Node, end: Node)
    requires IsDAG(G)
    requires ValidTrail(G, trail)
    requires TrailConnects(trail, start, end)
    requires forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward
    ensures end in Descendants(G, {start})
  {
    var ord :| IsTopologicalSort(G, ord);
    ForwardTrail_StartBeforeEndInTopologicalOrder(G, ord, trail, start, end);
    var startIdx, endIdx :| 0 <= startIdx < endIdx < |ord| && ord[startIdx] == start && ord[endIdx] == end;
    ForwardTrail_ImpliesAncestorBoundedByOrderDistance(G, ord, trail, start, end, startIdx, endIdx);
    TopologicalSort_Length(G, ord);
    assert endIdx - startIdx <= |Nodes(G)|;
    IsAncestorBounded_Monotone(G, start, end, endIdx - startIdx, |Nodes(G)|);
    assert IsAncestor(G, start, end);
    assert end in Nodes(G);
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

  lemma BackwardFirstStep_BlockedByParents(G: Graph, trail: seq<TrailStep>, start: Node, end: Node)
    requires ValidTrail(G, trail)
    requires TrailConnects(trail, start, end)
    requires |trail| > 1
    requires trail[0].dir == Backward
    ensures TrailBlockedAtPos(G, trail, 1, Parents(G, start))
  {
    assert trail[0].from == start;
    assert trail[0].to in Parents(G, trail[0].from);
    assert trail[0].to in Parents(G, start);
    assert trail[1].from == trail[0].to;
    assert !IsCollider(trail, 1);
    assert trail[1].from in Parents(G, start);
  }

  lemma FirstBackwardPos(trail: seq<TrailStep>) returns (pos: nat)
    requires exists i :: 0 <= i < |trail| && trail[i].dir == Backward
    ensures 0 <= pos < |trail|
    ensures trail[pos].dir == Backward
    ensures forall j :: 0 <= j < pos ==> trail[j].dir == Forward
    decreases |trail|
  {
    if trail[0].dir == Backward {
      pos := 0;
    } else {
      assert exists i :: 0 <= i < |trail[1..]| && trail[1..][i].dir == Backward by {
        var i :| 0 <= i < |trail| && trail[i].dir == Backward;
        assert i != 0;
        assert trail[1..][i - 1] == trail[i];
      }
      var suffixPos := FirstBackwardPos(trail[1..]);
      pos := suffixPos + 1;
      assert trail[pos].dir == Backward;
      assert forall j :: 0 <= j < pos ==> trail[j].dir == Forward by {
        forall j | 0 <= j < pos
          ensures trail[j].dir == Forward
        {
          if j == 0 {
            assert trail[j].dir == Forward;
          } else {
            assert trail[1..][j - 1] == trail[j];
          }
        }
      }
    }
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

  lemma DescendantsOfDescendant_DisjointParents(G: Graph, v: Node, u: Node)
    requires v in Nodes(G)
    requires u in Descendants(G, {v})
    requires IsDAG(G)
    ensures Descendants(G, {u}) * Parents(G, v) == {}
  {
    if Descendants(G, {u}) * Parents(G, v) != {} {
      var p :| p in Descendants(G, {u}) * Parents(G, v);
      var ord :| IsTopologicalSort(G, ord);
      if p == v {
        var vIdx :| 0 <= vIdx < |ord| && ord[vIdx] == v;
        var pIdx :| 0 <= pIdx < vIdx && ord[pIdx] == p;
        assert pIdx < vIdx;
        assert ord[pIdx] == ord[vIdx];
        assert false;
      } else {
        assert exists w :: w in {v} && IsAncestor(G, w, u);
        assert IsAncestor(G, v, u);
        assert exists w :: w in {u} && IsAncestor(G, w, p);
        assert IsAncestor(G, u, p);
        IsAncestorBounded_Transitive(G, v, u, p, |Nodes(G)|, |Nodes(G)|);
        IsAncestorBounded_ImpliesForwardTrail(G, v, p, |Nodes(G)| + |Nodes(G)|);
        var trail: seq<TrailStep> :| ValidTrail(G, trail) &&
          TrailConnects(trail, v, p) &&
          (forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward) &&
          |trail| <= |Nodes(G)| + |Nodes(G)|;
        ForwardTrail_StartBeforeEndInTopologicalOrder(G, ord, trail, v, p);
        var vIdx, pIdx0 :| 0 <= vIdx < pIdx0 < |ord| && ord[vIdx] == v && ord[pIdx0] == p;
        var pIdx :| 0 <= pIdx < vIdx && ord[pIdx] == p;
        assert pIdx < pIdx0;
        assert ord[pIdx] == ord[pIdx0];
        assert false;
      }
    }
  }

  lemma FirstForwardBackwardPivot_BlockedByParents(
    G: Graph, trail: seq<TrailStep>, start: Node, end: Node, pos: nat
  )
    requires ValidTrail(G, trail)
    requires TrailConnects(trail, start, end)
    requires IsDAG(G)
    requires 1 <= pos < |trail|
    requires trail[pos].dir == Backward
    requires forall i :: 0 <= i < pos ==> trail[i].dir == Forward
    ensures TrailBlockedAtPos(G, trail, pos, Parents(G, start))
  {
    var node := trail[pos].from;
    ValidTrail_Prefix(G, trail, pos);
    TrailConnects_Prefix(trail, start, end, pos);
    assert forall i :: 0 <= i < |trail[..pos]| ==> trail[..pos][i].dir == Forward by {
      forall i | 0 <= i < |trail[..pos]|
        ensures trail[..pos][i].dir == Forward
      {
        assert trail[..pos][i] == trail[i];
      }
    }
    assert trail[pos - 1].to == node;
    assert trail[0].from == start;
    assert start in Parents(G, trail[0].to);
    assert start in Nodes(G);
    ForwardTrail_EndInDescendants_DAG(G, trail[..pos], start, node);
    DescendantsOfDescendant_DisjointParents(G, start, node);
    assert node in Descendants(G, {node});
    if node in Parents(G, start) {
      assert node in Descendants(G, {node}) * Parents(G, start);
      assert false;
    }
    assert IsCollider(trail, pos);
    assert Descendants(G, {node}) * Parents(G, start) == {};
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

  lemma ShorterUnblockedPrefixIntoConditioningSet(
    G: Graph,
    trail: seq<TrailStep>,
    start: Node,
    end: Node,
    W: set<Node>,
    Added: set<Node>
  ) returns (prefix: seq<TrailStep>, mid: Node)
    requires ValidTrail(G, trail)
    requires TrailConnects(trail, start, end)
    requires !TrailBlocked(G, trail, W)
    requires TrailBlocked(G, trail, W + Added)
    ensures 0 < |prefix| < |trail|
    ensures ValidTrail(G, prefix)
    ensures TrailConnects(prefix, start, mid)
    ensures mid in Added
    ensures !TrailBlocked(G, prefix, W)
  {
    if |trail| <= 1 {
      var blockedPos :| 1 <= blockedPos < |trail| && TrailBlockedAtPos(G, trail, blockedPos, W + Added);
      assert false;
    }

    var pos := FirstBlockedPos(G, trail, W + Added);
    TrailNotBlockedAtPos(G, trail, pos, W);
    BlockingAddedByConditioningAtPos(G, trail, pos, W, Added);

    prefix := trail[..pos];
    mid := trail[pos - 1].to;
    ValidTrail_Prefix(G, trail, pos);
    TrailConnects_Prefix(trail, start, end, pos);
    assert trail[pos - 1].to == trail[pos].from;
    assert mid == trail[pos].from;
    assert mid in Added;

    assert forall j :: 1 <= j < pos ==> !TrailBlockedAtPos(G, trail, j, W) by {
      forall j | 1 <= j < pos
        ensures !TrailBlockedAtPos(G, trail, j, W)
      {
        TrailNotBlockedAtPos(G, trail, j, W);
      }
    }
    PrefixWithoutBlockedPos_NotBlocked(G, trail, pos, W);
  }

  lemma DSep_Intersection_Descend(
    G: Graph,
    Y: set<Node>,
    A: set<Node>,
    B: set<Node>,
    W: set<Node>,
    trail: seq<TrailStep>,
    y: Node,
    endpoint: Node
  )
    requires y in Y
    requires endpoint in A
    requires ValidTrail(G, trail)
    requires TrailConnects(trail, y, endpoint)
    requires DSep(G, Y, A, W + B)
    requires DSep(G, Y, B, W + A)
    ensures TrailBlocked(G, trail, W)
    decreases |trail|
  {
    if !TrailBlocked(G, trail, W) {
      assert TrailBlocked(G, trail, W + B);
      var shorter, mid := ShorterUnblockedPrefixIntoConditioningSet(G, trail, y, endpoint, W, B);
      DSep_Intersection_Descend(G, Y, B, A, W, shorter, y, mid);
      assert false;
    }
  }

  /// Intersection:
  ///   (Y ⊥ Z | W ∪ Z') ∧ (Y ⊥ Z' | W ∪ Z)  ⟹  (Y ⊥ Z ∪ Z' | W)
  ///
  /// This is stated here as a graph-level property of the `DSep` predicate.
  /// The earlier positivity/faithfulness caveat belongs to probabilistic
  /// conditional independence semantics, not to this DAG d-separation layer.
  lemma DSep_Intersection(
    G: Graph, Y: set<Node>, Z: set<Node>, Z': set<Node>, W: set<Node>
  )
    requires DSep(G, Y, Z, W + Z') && DSep(G, Y, Z', W + Z)
    ensures  DSep(G, Y, Z + Z', W)
  {
    forall trail: seq<TrailStep>, y: Node, z: Node |
      y in Y && z in Z + Z' &&
      ValidTrail(G, trail) &&
      TrailConnects(trail, y, z)
      ensures TrailBlocked(G, trail, W)
    {
      if z in Z {
        DSep_Intersection_Descend(G, Y, Z, Z', W, trail, y, z);
      } else {
        assert z in Z';
        DSep_Intersection_Descend(G, Y, Z', Z, W, trail, y, z);
      }
    }
  }

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

  /// Every node v is d-separated from its non-descendants excluding parents,
  /// given its parents:
  ///   {v} ⊥ (NonDesc(v) \ Pa(v)) | Pa(v)
  lemma LocalMarkov(G: Graph, v: Node)
    requires v in Nodes(G)
    requires IsDAG(G)
    ensures  DSep(G, {v}, NonDescendants(G, v) - Parents(G, v), Parents(G, v))
  {
    forall trail: seq<TrailStep>, y: Node, z: Node |
      y in {v} && z in NonDescendants(G, v) - Parents(G, v) &&
      ValidTrail(G, trail) &&
      TrailConnects(trail, y, z)
      ensures TrailBlocked(G, trail, Parents(G, v))
    {
      assert y == v;
      assert z !in Parents(G, v);
      if |trail| == 1 {
        if trail[0].dir == Backward {
          assert trail[0].from == v;
          assert trail[0].to == z;
          assert z in Parents(G, v);
          assert false;
        } else {
          assert forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward;
          ForwardTrail_EndInDescendants_DAG(G, trail, v, z);
          assert z in NonDescendants(G, v);
          assert false;
        }
      } else if trail[0].dir == Backward {
        BackwardFirstStep_BlockedByParents(G, trail, v, z);
        assert TrailBlocked(G, trail, Parents(G, v));
      } else if forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward {
        ForwardTrail_EndInDescendants_DAG(G, trail, v, z);
        assert z in NonDescendants(G, v);
        assert false;
      } else {
        assert exists i :: 0 <= i < |trail| && trail[i].dir == Backward by {
          if !(exists i :: 0 <= i < |trail| && trail[i].dir == Backward) {
            assert forall i :: 0 <= i < |trail| ==> trail[i].dir == Forward by {
              forall i | 0 <= i < |trail|
                ensures trail[i].dir == Forward
              {
                if trail[i].dir != Forward {
                  assert trail[i].dir == Backward;
                  assert false;
                }
              }
            }
            assert false;
          }
        }
        var pos := FirstBackwardPos(trail);
        assert pos != 0;
        assert 1 <= pos < |trail|;
        FirstForwardBackwardPivot_BlockedByParents(G, trail, v, z, pos);
        assert TrailBlocked(G, trail, Parents(G, v));
      }
    }
  }

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
        // Any valid trail 0 ··· 2 must pass through 1.
        // The last step ends at 2; it must be Forward (Backward would need
        // 2 ∈ Parents(G, ·), impossible in ChainGraph).
        // Forward step to 2 forces from=1 (Parents(G,2)={1}).
        // A single-step trail is impossible (0 ∉ Parents(G,2)).
        // So pos=|trail|-1 is internal, from=1, not a collider, 1∈W={1} → blocked.
        assert y == 0 && z == 2;
        var lastPos := |trail| - 1;
        assert trail[lastPos].to == 2;
        // Last step cannot be Backward: 2 ∈ Parents(G, ·) impossible.
        assert trail[lastPos].dir == Forward by {
          if trail[lastPos].dir == Backward {
            assert trail[lastPos].to in Parents(G, trail[lastPos].from);
            var f := trail[lastPos].from;
            if f == 0      { assert G[0] == {};  assert false; }
            else if f == 1 { assert G[1] == {0}; assert false; }
            else if f == 2 { assert G[2] == {1}; assert false; }
            else           { assert f !in G; assert Parents(G, f) == {}; assert false; }
          }
        }
        // Forward to 2: from ∈ Parents(G, 2) = {1}.
        assert trail[lastPos].from in Parents(G, 2);
        assert Parents(G, 2) == {1};
        assert trail[lastPos].from == 1;
        // Single-step trail impossible: 0 ∉ Parents(G, 2) = {1}.
        assert |trail| > 1 by {
          if |trail| == 1 {
            assert trail[0].from == 0;
            assert trail[0].from in Parents(G, 2);
            assert false;
          }
        }
        // pos=lastPos is internal; non-collider (dir=Forward); 1∈W={1}.
        var pos := lastPos;
        assert 1 <= pos < |trail|;
        assert !IsCollider(trail, pos) by { assert trail[pos].dir == Forward; }
        assert trail[pos].from == 1;
        assert trail[pos].from in {1};
        assert TrailBlockedAtPos(G, trail, pos, {1});
        assert TrailBlocked(G, trail, {1});
      }
    }
  }

}  // end module DAG
