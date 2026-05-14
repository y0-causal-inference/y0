// Executable extraction-oriented implementation of the modified Tian (2002)
// identification algorithm shown as Figure 5 in Shpitser & Pearl (2006).
//
// This module is standalone and focuses on executable control flow shape:
//  - identify(y, x, P, G)
//  - c-identify(C, T, Q[T])
//
// It emits the same v1 IR payload style used by other extracted ID runtimes.

module IDTian2002Extracted {

  datatype IRNode =
    | IRSum(over: seq<string>, body: IRNode)
    | IRProduct(factors: seq<IRNode>)
    | IRProb(vars: seq<string>, given: seq<string>, intervened: seq<string>)
    | IRFrac(numer: IRNode, denom: IRNode)
    | IRFailHedge(F_nodes: seq<string>, Fprime_nodes: seq<string>)

  datatype IRQuery = IRQuery(
    graph_id: string,
    outcomes: seq<string>,
    treatments: seq<string>,
    ordering: seq<string>
  )

  datatype IRDoc = IRDoc(
    version: string,
    engine: string,
    query: IRQuery,
    result: IRNode
  )

  datatype Edge = Edge(src: string, tgt: string)

  method FilterByOrdering(ordering: seq<string>, members: set<string>) returns (values: seq<string>)
  {
    values := [];
    var i := 0;
    while i < |ordering|
      invariant 0 <= i <= |ordering|
      invariant forall j :: 0 <= j < |values| ==> values[j] in members
    {
      if ordering[i] in members {
        values := values + [ordering[i]];
      }
      i := i + 1;
    }
  }

  method ComplementByOrdering(ordering: seq<string>, members: set<string>) returns (values: seq<string>)
  {
    values := [];
    var i := 0;
    while i < |ordering|
      invariant 0 <= i <= |ordering|
      invariant forall j :: 0 <= j < |values| ==> values[j] !in members
    {
      if ordering[i] !in members {
        values := values + [ordering[i]];
      }
      i := i + 1;
    }
  }

  method PrefixBefore(ordering: seq<string>, target: string) returns (values: seq<string>)
  {
    values := [];
    var i := 0;
    while i < |ordering|
      invariant 0 <= i <= |ordering|
    {
      if ordering[i] == target {
        return;
      }
      values := values + [ordering[i]];
      i := i + 1;
    }
  }

  method ComputeAncestorsRestricted(
    edges: seq<Edge>,
    targets: set<string>,
    allowed_nodes: set<string>
  ) returns (ancestors: set<string>)
  {
    ancestors := targets * allowed_nodes;
    var frontier := ancestors;
    var iteration := 0;
    var max_iterations := |allowed_nodes| + 1;

    while iteration < max_iterations && |frontier| > 0
      invariant frontier <= ancestors
      invariant iteration <= max_iterations
    {
      var new_parents: set<string> := {};
      var i := 0;
      while i < |edges|
        invariant 0 <= i <= |edges|
      {
        var e := edges[i];
        if e.tgt in frontier && e.src in allowed_nodes {
          new_parents := new_parents + {e.src};
        }
        i := i + 1;
      }

      frontier := (new_parents - ancestors) * allowed_nodes;
      ancestors := ancestors + new_parents;
      iteration := iteration + 1;
    }
  }

  method ComputeAncestorsWithIncomingRemovedToTreatments(
    edges: seq<Edge>,
    targets: set<string>,
    treatments: set<string>,
    all_nodes: set<string>
  ) returns (ancestors: set<string>)
  {
    ancestors := targets;
    var frontier := targets;
    var iteration := 0;
    var max_iterations := |all_nodes| + 1;

    while iteration < max_iterations && |frontier| > 0
      invariant targets <= ancestors
      invariant frontier <= ancestors
      invariant iteration <= max_iterations
    {
      var new_parents: set<string> := {};
      var i := 0;
      while i < |edges|
        invariant 0 <= i <= |edges|
      {
        var e := edges[i];
        if e.tgt in frontier && e.tgt !in treatments {
          new_parents := new_parents + {e.src};
        }
        i := i + 1;
      }

      frontier := new_parents - ancestors;
      ancestors := ancestors + new_parents;
      iteration := iteration + 1;
    }
  }

  method ReachableUndirected(
    edges: seq<Edge>,
    start: string,
    allowed_nodes: set<string>
  ) returns (reached: set<string>)
    requires start in allowed_nodes
    ensures start in reached
  {
    reached := {start};
    var frontier := {start};
    var iteration := 0;
    var max_iterations := |allowed_nodes| + 1;

    while iteration < max_iterations && |frontier| > 0
      invariant frontier <= reached
      invariant start in reached
      invariant iteration <= max_iterations
    {
      var next: set<string> := {};
      var i := 0;
      while i < |edges|
        invariant 0 <= i <= |edges|
      {
        var e := edges[i];
        if e.src in frontier && e.tgt in allowed_nodes {
          next := next + {e.tgt};
        }
        if e.tgt in frontier && e.src in allowed_nodes {
          next := next + {e.src};
        }
        i := i + 1;
      }

      frontier := next - reached;
      reached := reached + next;
      iteration := iteration + 1;
    }
  }

  method ComponentsOfSubset(
    undirected_edges: seq<Edge>,
    ordering: seq<string>,
    nodes: set<string>
  ) returns (components: seq<set<string>>)
  {
    components := [];
    var visited: set<string> := {};
    var i := 0;

    while i < |ordering|
      invariant 0 <= i <= |ordering|
    {
      var node := ordering[i];
      if node in nodes && node !in visited {
        var c := ReachableUndirected(undirected_edges, node, nodes);
        components := components + [c];
        visited := visited + c;
      }
      i := i + 1;
    }
  }

  method FindContainingComponent(
    components: seq<set<string>>,
    needle: set<string>
  ) returns (ok: bool, component: set<string>)
  {
    ok := false;
    component := {};
    var i := 0;
    while i < |components|
      invariant 0 <= i <= |components|
    {
      if needle <= components[i] {
        ok := true;
        component := components[i];
        return;
      }
      i := i + 1;
    }
  }

  method BuildQNode(ordering: seq<string>, nodes: set<string>) returns (q: IRNode)
  {
    var node_seq := FilterByOrdering(ordering, nodes);

    if |node_seq| == 0 {
      q := IRFailHedge([], []);
      return;
    }

    var factors: seq<IRNode> := [];

    var i := 0;
    while i < |node_seq|
      invariant 0 <= i <= |node_seq|
    {
      var v := node_seq[i];
      var given := PrefixBefore(ordering, v);
      factors := factors + [IRProb([v], given, [])];
      i := i + 1;
    }

    if |factors| == 1 {
      q := factors[0];
    } else {
      q := IRProduct(factors);
    }
  }

  method CIdentify(
    directed_edges: seq<Edge>,
    undirected_edges: seq<Edge>,
    ordering: seq<string>,
    C: set<string>,
    T: set<string>,
    qT: IRNode,
    fuel: nat
  ) returns (ok: bool, value: IRNode, failF: set<string>, failFprime: set<string>)
    decreases fuel
  {
    failF := {};
    failFprime := {};

    if C == {} || T == {} || !(C <= T) {
      ok := false;
      value := IRFailHedge([], []);
      failF := T;
      failFprime := C;
      return;
    }

    if fuel == 0 {
      ok := false;
      value := IRFailHedge([], []);
      failF := T;
      failFprime := C;
      return;
    }

    // Figure 5 c-identify: A = An(C)_T
    var A := ComputeAncestorsRestricted(directed_edges, C, T);

    // 1. if A = C, return sum_{T\C} Q[T]
    if A == C {
      var over := FilterByOrdering(ordering, T - C);
      if |over| == 0 {
        ok := true;
        value := qT;
      } else {
        ok := true;
        value := IRSum(over, qT);
      }
      return;
    }

    // 2. if A = T, return FAIL
    if A == T {
      ok := false;
      value := IRFailHedge([], []);
      failF := T;
      failFprime := C;
      return;
    }

    // 3. if C subset A subset T, recurse on T' where C subset T' subset A.
    var compsA := ComponentsOfSubset(undirected_edges, ordering, A);
    var found, Tprime := FindContainingComponent(compsA, C);
    if !found || Tprime == A || Tprime == {} {
      ok := false;
      value := IRFailHedge([], []);
      failF := T;
      failFprime := C;
      return;
    }

    var qTprime := BuildQNode(ordering, Tprime);
    ok, value, failF, failFprime := CIdentify(
      directed_edges,
      undirected_edges,
      ordering,
      C,
      Tprime,
      qTprime,
      fuel - 1
    );
  }

  method IDTianToIR(
    graph_id: string,
    directed_edges: seq<Edge>,
    undirected_edges: seq<Edge>,
    all_nodes: seq<string>,
    outcomes: set<string>,
    treatments: set<string>,
    ordering: seq<string>
  ) returns (ok: bool, doc: IRDoc)
    requires |all_nodes| > 0
  {
    var all_nodes_set: set<string> := set x | x in all_nodes;
    var outcome_seq := FilterByOrdering(ordering, outcomes);
    var treatment_seq := FilterByOrdering(ordering, treatments);
    var query := IRQuery(graph_id, outcome_seq, treatment_seq, ordering);

    // Figure 5 identify line 1: D = An(Y)_{G_xbar}
    var D := ComputeAncestorsWithIncomingRemovedToTreatments(
      directed_edges,
      outcomes,
      treatments,
      all_nodes_set
    );

    if D == {} {
      ok := false;
      doc := IRDoc("1", "id-tian", query, IRFailHedge([], []));
      return;
    }

    // C(D) and C(G)
    var compsD := ComponentsOfSubset(undirected_edges, ordering, D);
    var compsG := ComponentsOfSubset(undirected_edges, ordering, all_nodes_set);

    var factors: seq<IRNode> := [];
    var i := 0;
    while i < |compsD|
      invariant 0 <= i <= |compsD|
    {
      var Di := compsD[i];

      // Find C_{Di} in C(G) with Di subset C_{Di}
      var found, Cdi := FindContainingComponent(compsG, Di);
      if !found || Di == {} || Cdi == {} {
        ok := false;
        doc := IRDoc("1", "id-tian", query, IRFailHedge([], []));
        return;
      }

      var qCdi := BuildQNode(ordering, Cdi);
      var local_ok, local_value, local_failF, local_failFprime := CIdentify(
        directed_edges,
        undirected_edges,
        ordering,
        Di,
        Cdi,
        qCdi,
        |all_nodes| + 1
      );

      if !local_ok {
        var f_nodes := FilterByOrdering(ordering, local_failF);
        var fprime_nodes := FilterByOrdering(ordering, local_failFprime);
        ok := false;
        doc := IRDoc("1", "id-tian", query, IRFailHedge(f_nodes, fprime_nodes));
        return;
      }

      factors := factors + [local_value];
      i := i + 1;
    }

    var body: IRNode;
    if |factors| == 1 {
      body := factors[0];
    } else {
      body := IRProduct(factors);
    }

    // Figure 5 identify line 3: sum_{D\Y} prod_i ...
    var over := FilterByOrdering(ordering, D - outcomes);
    if |over| == 0 {
      ok := true;
      doc := IRDoc("1", "id-tian", query, body);
    } else {
      ok := true;
      doc := IRDoc("1", "id-tian", query, IRSum(over, body));
    }
  }
}
