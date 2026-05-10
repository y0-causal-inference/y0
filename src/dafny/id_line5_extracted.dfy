// Executable Dafny slice for ID algorithm Line 5 IR emission.
//
// Line 5: Hedge witness failure
// if C(G) = {G} and C(G\X) = {S}, return fail(hedge)
//
// This module is intentionally standalone and avoids theorem/axiom layers so
// it can be translated to Python runtime code.

module IDLine5Extracted {

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

  method ReachableUndirected(
    edges: seq<Edge>,
    start: string,
    allowed_nodes: set<string>
  ) returns (reached: set<string>)
    requires start in allowed_nodes
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
        var edge := edges[i];
        if edge.src in frontier && edge.tgt in allowed_nodes {
          next := next + {edge.tgt};
        }
        if edge.tgt in frontier && edge.src in allowed_nodes {
          next := next + {edge.src};
        }
        i := i + 1;
      }

      frontier := next - reached;
      reached := reached + next;
      iteration := iteration + 1;
    }
  }

  method CountComponents(
    edges: seq<Edge>,
    ordered_nodes: seq<string>,
    nodes: set<string>
  ) returns (count: nat, first_component: set<string>)
  {
    count := 0;
    first_component := {};
    var visited: set<string> := {};
    var i := 0;

    while i < |ordered_nodes|
      invariant 0 <= i <= |ordered_nodes|
    {
      var node := ordered_nodes[i];
      if node in nodes && node !in visited {
        var component := ReachableUndirected(edges, node, nodes);

        if count == 0 {
          first_component := component;
        }

        count := count + 1;
        visited := visited + component;
      }

      i := i + 1;
    }
  }

  method IDLine5ToIR(
    graph_id: string,
    undirected_edges: seq<Edge>,
    all_nodes: seq<string>,
    outcomes: set<string>,
    treatments: set<string>,
    ordering: seq<string>
  ) returns (ok: bool, doc: IRDoc)
  {
    var all_nodes_set: set<string> := set x | x in all_nodes;
    var outcome_seq := FilterByOrdering(ordering, outcomes);
    var treatment_seq := FilterByOrdering(ordering, treatments);
    var query := IRQuery(graph_id, outcome_seq, treatment_seq, ordering);

    // C(G) = {G} iff there is exactly one district over all nodes.
    var full_count, _ := CountComponents(undirected_edges, all_nodes, all_nodes_set);

    // C(G\X) = {S} iff there is exactly one district after removing X nodes.
    var reduced_nodes := all_nodes_set - treatments;
    var reduced_count, reduced_component := CountComponents(undirected_edges, all_nodes, reduced_nodes);

    if full_count != 1 || reduced_count != 1 {
      ok := false;
      doc := IRDoc("5", "id", query, IRFailHedge([], []));
      return;
    }

    var f_nodes := FilterByOrdering(ordering, all_nodes_set);
    var fprime_nodes := FilterByOrdering(ordering, reduced_component);
    ok := true;
    doc := IRDoc("5", "id", query, IRFailHedge(f_nodes, fprime_nodes));
  }
}