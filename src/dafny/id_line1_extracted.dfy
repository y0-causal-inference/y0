// Executable Dafny slice for ID algorithm Line 1 IR emission.
//
// This module is intentionally standalone and avoids theorem/axiom layers so
// it can be translated to Python runtime code.

module IDLine1Extracted {

  datatype IRNode =
    | IRSum(over: seq<string>, body: IRNode)
    | IRProduct(factors: seq<IRNode>)
    | IRProb(vars: seq<string>, given: seq<string>, intervened: seq<string>)
    | IRFrac(numer: IRNode, denom: IRNode)
    | IRFailHedge(F_nodes: seq<string>, Fprime_nodes: seq<string>)
    | IRNotApplicable  // precondition for this line was not met; try the next line

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

  // Compute An(Y)_{G_{bar_x}}: ancestors of `outcomes` in the manipulated graph
  // G_{bar_x}, which is G with all incoming edges to nodes in `treatments` removed.
  //
  // Algorithm: reverse-topological propagation. Since `ordering` is a topological
  // order (parents before children), iterating from last to first ensures that
  // when we process a node, all its potential children have already been checked.
  // A node is an ancestor of Y if it has an outgoing edge (in G_{bar_x}) to an
  // already-marked ancestor. Edge (n, m) exists in G_{bar_x} iff m ∉ treatments.
  method AncestorsInManipulatedGraph(
    outcomes: set<string>,
    treatments: set<string>,
    edges: seq<(string, string)>,
    ordering: seq<string>
  ) returns (ancestors: set<string>)
  {
    ancestors := outcomes;
    var i := |ordering|;
    while i > 0
      invariant 0 <= i <= |ordering|
      decreases i
    {
      i := i - 1;
      var node := ordering[i];
      if node !in ancestors {
        var found := false;
        var j := 0;
        while j < |edges| && !found
          invariant 0 <= j <= |edges|
        {
          // Edge (node, child) exists in G_{bar_x} iff child ∉ treatments
          if edges[j].0 == node && edges[j].1 !in treatments && edges[j].1 in ancestors {
            found := true;
          }
          j := j + 1;
        }
        if found {
          ancestors := ancestors + {node};
        }
      }
    }
  }

  // Line 1 of the ID algorithm:
  //   if X ∩ An(Y)_{G_{bar_x}} = ∅, return Σ_{v\Y} P(V | ∅)
  //   otherwise: this line does not apply (ok = false)
  //
  // `edges` encodes the directed graph G as a sequence of (parent, child) pairs.
  method IDLine1ToIR(
    graph_id: string,
    outcomes: set<string>,
    treatments: set<string>,
    edges: seq<(string, string)>,
    ordering: seq<string>
  ) returns (ok: bool, doc: IRDoc)
  {
    var outcome_seq := FilterByOrdering(ordering, outcomes);
    var treatment_seq := FilterByOrdering(ordering, treatments);
    var query := IRQuery(graph_id, outcome_seq, treatment_seq, ordering);

    // Compute X ∩ An(Y)_{G_{bar_x}}
    var ancestors := AncestorsInManipulatedGraph(outcomes, treatments, edges, ordering);
    var x_intersects_ancestors := treatments * ancestors != {};

    if x_intersects_ancestors {
      // Line 1 precondition not satisfied; this line does not apply.
      // IRNotApplicable signals the caller to try subsequent lines.
      ok := false;
      doc := IRDoc("1", "id", query, IRNotApplicable);
      return;
    }

    // X ∩ An(Y)_{G_{bar_x}} = ∅: return Σ_{v\Y} P(V | ∅)
    var over := [];
    var i := 0;
    while i < |ordering|
      invariant 0 <= i <= |ordering|
      invariant forall j :: 0 <= j < |over| ==> over[j] !in outcomes
    {
      if ordering[i] !in outcomes {
        over := over + [ordering[i]];
      }
      i := i + 1;
    }

    var body := IRProb(ordering, [], []);
    ok := true;
    doc := IRDoc("1", "id", query, IRSum(over, body));
  }
}