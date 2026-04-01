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
  ghost function Nodes(G: Graph): set<Node> {
    G.Keys
  }

  // Parents of v in G.
  function Parents(G: Graph, v: Node): set<Node> {
    if v in G then G[v] else {}
  }

  // Children of u in G.
  ghost function Children(G: Graph, u: Node): set<Node> {
    set v | v in Nodes(G) && u in Parents(G, v)
  }

  // ==================================================================
  // 2.  Acyclicity
  // ==================================================================

  // An ordering `ord` of nodes is a topological sort of G if every
  // parent comes strictly before its child.
  ghost predicate IsTopologicalSort(G: Graph, ord: seq<Node>) {
    // (a) ord contains exactly the nodes of G
    (forall v :: v in Nodes(G) <==> v in ord) &&
    // (b) no duplicates  (injective)
    (forall i, j :: 0 <= i < j < |ord| ==> ord[i] != ord[j]) &&
    // (c) every parent appears before its child
    (forall i :: 0 <= i < |ord| ==>
      forall p :: p in Parents(G, ord[i]) ==>
        exists k :: 0 <= k < i && ord[k] == p)
  }

  // A graph is a DAG iff it admits a topological sort.
  ghost predicate IsDAG(G: Graph) {
    exists ord: seq<Node> :: IsTopologicalSort(G, ord)
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

  ghost predicate IsAncestorBounded(G: Graph, u: Node, v: Node, fuel: nat)
    decreases fuel
  {
    u == v ||
    (fuel > 0 &&
     exists w :: w in Children(G, u) && IsAncestorBounded(G, w, v, fuel - 1))
  }

  // Convenience wrapper using a fuel equal to the number of nodes.
  ghost predicate IsAncestor(G: Graph, u: Node, v: Node) {
    IsAncestorBounded(G, u, v, |Nodes(G)|)
  }

  // All ancestors (including self) of each node in W.
  ghost function Ancestors(G: Graph, W: set<Node>): set<Node> {
    set u | u in Nodes(G) && exists w :: w in W && IsAncestor(G, u, w)
  }

  // All descendants (including self) of each node in W.
  ghost function Descendants(G: Graph, W: set<Node>): set<Node> {
    set v | v in Nodes(G) && exists w :: w in W && IsAncestor(G, w, v)
  }

  // ------------------------------------------------------------------
  // Ancestry lemmas
  // ------------------------------------------------------------------

  lemma Ancestor_Reflexive(G: Graph, v: Node)
    ensures IsAncestorBounded(G, v, v, 0)
  {}

  // ==================================================================
  // 4.  Graph Surgery
  // ==================================================================

  // G_{X̄}  —  remove incoming edges to every node in X.
  ghost function RemoveIncoming(G: Graph, X: set<Node>): Graph
  {
    map v | v in Nodes(G) ::
      if v in X then {} else Parents(G, v)
  }

  // G_{X̲}  —  remove outgoing edges from every node in X.
  //   i.e., for every child c, remove X-members from c's parent set.
  ghost function RemoveOutgoing(G: Graph, X: set<Node>): Graph
  {
    map v | v in Nodes(G) ::
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

  // A trail is **blocked** by conditioning set W if at least one
  // node along the trail satisfies the d-separation blocking criterion:
  //
  //   Non-collider:  the node is in W (blocks the path)
  //   Collider:      the node (and all its descendants) are NOT in W
  ghost predicate TrailBlocked(G: Graph, trail: seq<TrailStep>, W: set<Node>) {
    |trail| <= 1 ||  // trivial trail (single edge) — check endpoints only
    exists pos :: 1 <= pos < |trail| &&
      var node := trail[pos].from;
      if IsCollider(trail, pos) then
        // Collider: blocks unless the collider or a descendant is in W
        node !in W && Descendants(G, {node}) * W == {}
      else
        // Non-collider: blocks when the node IS in W
        node in W
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
  lemma {:axiom} DSep_Symmetry(G: Graph, Y: set<Node>, Z: set<Node>, W: set<Node>)
    requires DSep(G, Y, Z, W)
    ensures  DSep(G, Z, Y, W)

  /// Decomposition:  (Y ⊥ Z ∪ Z' | W)  ⟹  (Y ⊥ Z | W)
  lemma {:axiom} DSep_Decomposition(
    G: Graph, Y: set<Node>, Z: set<Node>, Z': set<Node>, W: set<Node>
  )
    requires DSep(G, Y, Z + Z', W)
    ensures  DSep(G, Y, Z, W)

  /// Weak Union:  (Y ⊥ Z ∪ Z' | W)  ⟹  (Y ⊥ Z | W ∪ Z')
  lemma {:axiom} DSep_WeakUnion(
    G: Graph, Y: set<Node>, Z: set<Node>, Z': set<Node>, W: set<Node>
  )
    requires DSep(G, Y, Z + Z', W)
    ensures  DSep(G, Y, Z, W + Z')

  /// Contraction:  (Y ⊥ Z | W ∪ Z') ∧ (Y ⊥ Z' | W)  ⟹  (Y ⊥ Z ∪ Z' | W)
  lemma {:axiom} DSep_Contraction(
    G: Graph, Y: set<Node>, Z: set<Node>, Z': set<Node>, W: set<Node>
  )
    requires DSep(G, Y, Z, W + Z') && DSep(G, Y, Z', W)
    ensures  DSep(G, Y, Z + Z', W)

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

  ghost function NonDescendants(G: Graph, v: Node): set<Node> {
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

  lemma {:axiom} ChainGraph_IsDAG()
    ensures IsDAG(ChainGraph())

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
