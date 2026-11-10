// Executable Dafny slice for ID algorithm Line 3 transformation.
//
// Line 3: let W = (V \ X) \ An(Y)_{G_{X̄}}. If W != {}, recurse with X + W.

module IDLine3Extracted {

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
      invariant forall n :: n in frontier ==> n in ancestors
      invariant iteration <= max_iterations
    {
      var new_parents: set<string> := {};
      var i := 0;
      while i < |edges|
        invariant 0 <= i <= |edges|
      {
        var edge := edges[i];
        // In G_{X̄}, incoming arrows into X are removed.
        if edge.tgt in frontier && edge.tgt !in treatments {
          new_parents := new_parents + {edge.src};
        }
        i := i + 1;
      }

      frontier := new_parents - ancestors;
      ancestors := ancestors + new_parents;
      iteration := iteration + 1;
    }
  }

  method IDLine3Transform(
    edges: seq<Edge>,
    all_nodes: seq<string>,
    outcomes: set<string>,
    treatments: set<string>,
    ordering: seq<string>
  ) returns (
    ok: bool,
    no_effect_nodes: seq<string>,
    expanded_treatments: seq<string>
  )
  {
    var all_nodes_set: set<string> := set x | x in all_nodes;
    no_effect_nodes := [];
    expanded_treatments := [];

    var ancestors_in_gx := ComputeAncestorsWithIncomingRemovedToTreatments(
      edges,
      outcomes,
      treatments,
      all_nodes_set
    );
    var W := (all_nodes_set - treatments) - ancestors_in_gx;

    if W == {} {
      ok := false;
      return;
    }

    no_effect_nodes := FilterByOrdering(ordering, W);
    expanded_treatments := FilterByOrdering(ordering, treatments + W);
    ok := true;
  }
}