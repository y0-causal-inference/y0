// Executable Dafny slice for ID algorithm Line 2 IR emission.
//
// Line 2: Ancestral graph reduction
// if V ≠ An(Y)_G, return ID(y, x ∩ An(Y), P(An(Y)), G_{An(Y)})
//
// This module is intentionally standalone and avoids theorem/axiom layers so
// it can be translated to Python runtime code.

module IDLine2Extracted {

  datatype IRNode =
    | IRSum(over: seq<string>, body: IRNode)
    | IRProduct(factors: seq<IRNode>)
    | IRProb(vars: seq<string>, given: seq<string>, intervened: seq<string>)
    | IRFrac(numer: IRNode, denom: IRNode)
    | IRFailHedge(F_nodes: seq<string>, Fprime_nodes: seq<string>)
    | IRNotApplicable  // precondition for this line was not met; try the next line
    | IRRecursive(reduced_nodes: seq<string>, reduced_treatments: seq<string>)

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

  method ComputeAncestors(
    edges: seq<Edge>,
    targets: set<string>,
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
        if edge.tgt in frontier {
          new_parents := new_parents + {edge.src};
        }
        i := i + 1;
      }

      frontier := new_parents - ancestors;
      ancestors := ancestors + new_parents;
      iteration := iteration + 1;
    }
  }

  method IDLine2ToIR(
    graph_id: string,
    edges: seq<Edge>,
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

    // Compute ancestors of Y
    var ancestors := ComputeAncestors(edges, outcomes, all_nodes_set);

    // Check if V ≠ An(Y): if there are nodes in all_nodes_set but not in ancestors
    var non_ancestral: set<string> := all_nodes_set - ancestors;

    if non_ancestral == {} {
      // Line 2 does not apply — return not applicable
      ok := false;
      doc := IRDoc("2", "id", query, IRNotApplicable);
      return;
    }

    // Line 2 applies: compute reduced treatment set X ∩ An(Y)
    var reduced_treatments := treatments * ancestors;
    var reduced_treatment_seq := FilterByOrdering(ordering, reduced_treatments);

    // Emit IR for recursive call on ancestral subgraph
    var reduced_nodes_seq := FilterByOrdering(ordering, ancestors);

    ok := true;
    doc := IRDoc("2", "id", query, IRRecursive(reduced_nodes_seq, reduced_treatment_seq));
  }
}
