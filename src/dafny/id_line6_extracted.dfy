// Executable Dafny slice for ID algorithm Line 6 IR emission.
//
// Line 6: Direct computation when S in C(G)
// if C(G\X) = {S} and S is a district in G (with line-5 already excluded),
// return sum_{S\Y} product_{v in S} P(v | prefix(ordering, v)).

module IDLine6Extracted {

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

  method IDLine6ToIR(
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

    var full_count := CountComponents(undirected_edges, all_nodes, all_nodes_set);
    if full_count <= 1 {
      ok := false;
      doc := IRDoc("6", "id", query, IRFailHedge([], []));
      return;
    }

    var reduced_nodes := all_nodes_set - treatments;
    if reduced_nodes == {} {
      ok := false;
      doc := IRDoc("6", "id", query, IRFailHedge([], []));
      return;
    }

    var reduced_count := CountComponents(undirected_edges, all_nodes, reduced_nodes);
    if reduced_count != 1 {
      ok := false;
      doc := IRDoc("6", "id", query, IRFailHedge([], []));
      return;
    }

    var s0: string :| s0 in reduced_nodes;
    var reduced_component := ReachableUndirected(undirected_edges, s0, reduced_nodes);
    var full_component := ReachableUndirected(undirected_edges, s0, all_nodes_set);
    if full_component != reduced_component {
      ok := false;
      doc := IRDoc("6", "id", query, IRFailHedge([], []));
      return;
    }

    var district_seq := FilterByOrdering(ordering, reduced_component);
    var factors: seq<IRNode> := [];

    var i := 0;
    while i < |district_seq|
      invariant 0 <= i <= |district_seq|
    {
      var v := district_seq[i];
      var given := PrefixBefore(ordering, v);
      factors := factors + [IRProb([v], given, [])];
      i := i + 1;
    }

    var body := IRProduct(factors);
    var ranges := FilterByOrdering(ordering, reduced_component - outcomes);
    ok := true;
    if |ranges| == 0 {
      doc := IRDoc("6", "id", query, body);
    } else {
      doc := IRDoc("6", "id", query, IRSum(ranges, body));
    }
  }
}