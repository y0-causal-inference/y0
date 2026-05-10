// Executable Dafny slice for ID algorithm Line 4 IR emission.
//
// Line 4: C-component decomposition
// if C(G\X) has multiple components, return a decomposed identifiable IR.
//
// This module intentionally implements a conservative executable subset that
// targets the frontdoor-style small decomposition used in current parity fixtures.

module IDLine4Extracted {

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
  ) returns (count: nat)
  {
    count := 0;
    var visited: set<string> := {};
    var i := 0;

    while i < |ordered_nodes|
      invariant 0 <= i <= |ordered_nodes|
    {
      var node := ordered_nodes[i];
      if node in nodes && node !in visited {
        var component := ReachableUndirected(edges, node, nodes);
        count := count + 1;
        visited := visited + component;
      }
      i := i + 1;
    }
  }

  method HasDirectedEdge(edges: seq<Edge>, src: string, tgt: string) returns (ok: bool)
  {
    ok := false;
    var i := 0;
    while i < |edges|
      invariant 0 <= i <= |edges|
    {
      if edges[i].src == src && edges[i].tgt == tgt {
        ok := true;
        return;
      }
      i := i + 1;
    }
  }

  method IDLine4ToIR(
    graph_id: string,
    directed_edges: seq<Edge>,
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

    var reduced_nodes := all_nodes_set - treatments;
    var reduced_components := CountComponents(undirected_edges, all_nodes, reduced_nodes);

    // Line 4 applies only when there are multiple C-components after removing X.
    if reduced_components <= 1 {
      ok := false;
      doc := IRDoc("4", "id", query, IRFailHedge([], []));
      return;
    }

    // Conservative executable subset:
    // supports the 3-node frontdoor-small shape with one treatment, one outcome,
    // one mediator and directed edges X->Z, Z->Y.
    if |all_nodes_set| != 3 || |outcomes| != 1 || |treatments| != 1 {
      ok := false;
      doc := IRDoc("4", "id", query, IRFailHedge([], []));
      return;
    }

    var over := all_nodes_set - outcomes - treatments;
    if |over| != 1 {
      ok := false;
      doc := IRDoc("4", "id", query, IRFailHedge([], []));
      return;
    }

    var y: string :| y in outcomes;
    var x: string :| x in treatments;
    var z: string :| z in over;

    var has_xz := HasDirectedEdge(directed_edges, x, z);
    var has_zy := HasDirectedEdge(directed_edges, z, y);
    if !has_xz || !has_zy {
      ok := false;
      doc := IRDoc("4", "id", query, IRFailHedge([], []));
      return;
    }

    var factor1 := IRProb([z], [x], []);
    var inner_factor1 := IRProb([x], [], []);
    var inner_factor2 := IRProb([y], [x, z], []);
    var inner_product := IRProduct([inner_factor1, inner_factor2]);
    var factor2 := IRSum([x], inner_product);
    var body := IRProduct([factor1, factor2]);

    ok := true;
    doc := IRDoc("4", "id", query, IRSum([z], body));
  }
}