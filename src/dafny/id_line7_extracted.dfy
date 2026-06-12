// Executable Dafny slice for ID algorithm Line 7 transformation.
//
// Line 7 computes a reduced subproblem when the single district in G\X is a
// strict subset of some district in G.

module IDLine7Extracted {

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

  method IDLine7Transform(
    undirected_edges: seq<Edge>,
    all_nodes: seq<string>,
    treatments: set<string>,
    ordering: seq<string>
  ) returns (
    ok: bool,
    district_nodes: seq<string>,
    reduced_treatments: seq<string>
  )
  {
    var all_nodes_set: set<string> := set x | x in all_nodes;
    district_nodes := [];
    reduced_treatments := [];

    var reduced_nodes := all_nodes_set - treatments;
    if reduced_nodes == {} {
      ok := false;
      return;
    }

    // Line 7 precondition requires one district in G\X.
    var reduced_count := CountComponents(undirected_edges, all_nodes, reduced_nodes);
    if reduced_count != 1 {
      ok := false;
      return;
    }

    var s0: string :| s0 in reduced_nodes;
    var reduced_component := ReachableUndirected(undirected_edges, s0, reduced_nodes);
    var full_component := ReachableUndirected(undirected_edges, s0, all_nodes_set);

    // Need strict subset S < S'.
    if full_component == reduced_component {
      ok := false;
      return;
    }

    district_nodes := FilterByOrdering(ordering, full_component);
    reduced_treatments := FilterByOrdering(ordering, treatments * full_component);
    ok := true;
  }
}