// ===================================================================
// Identification Executable Core (Axiom-Free)
//
// This module contains only executable, extraction-oriented ID surface
// definitions and avoids theorem-level axioms/assumptions.
// ===================================================================

module IdentificationExecutableCore {

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

  predicate HasNoDuplicates(values: seq<string>) {
    forall i, j :: 0 <= i < j < |values| ==> values[i] != values[j]
  }

  predicate DisjointSeq(a: seq<string>, b: seq<string>) {
    forall x :: x in a ==> x !in b
  }

  predicate IsCanonicalIRNode(node: IRNode, allowFail: bool)
    decreases node
  {
    match node
    case IRProb(vars, given, intervened) =>
      |vars| > 0 && DisjointSeq(given, intervened)
    case IRSum(over, body) =>
      |over| > 0 && IsCanonicalIRNode(body, false)
    case IRProduct(factors) =>
      |factors| > 0 &&
      forall i :: 0 <= i < |factors| ==> IsCanonicalIRNode(factors[i], false)
    case IRFrac(numer, denom) =>
      IsCanonicalIRNode(numer, false) && IsCanonicalIRNode(denom, false)
    case IRFailHedge(fNodes, fprimeNodes) =>
      allowFail
  }

  predicate IsCanonicalIRDoc(doc: IRDoc) {
    doc.version != "" &&
    doc.engine == "id" &&
    IsCanonicalIRNode(doc.result, true)
  }

  method FilterByOrdering(ordering: seq<string>, members: set<string>) returns (values: seq<string>)
    ensures forall i :: 0 <= i < |values| ==> values[i] in members
    ensures forall x :: x in members && x in ordering ==> x in values
  {
    values := [];
    var i := 0;
    while i < |ordering|
      invariant 0 <= i <= |ordering|
      invariant forall j :: 0 <= j < |values| ==> values[j] in members
      invariant forall k :: 0 <= k < i && ordering[k] in members ==> ordering[k] in values
    {
      if ordering[i] in members {
        values := values + [ordering[i]];
      }
      i := i + 1;
    }
  }

  method ComplementByOrdering(ordering: seq<string>, members: set<string>) returns (values: seq<string>)
    ensures forall i :: 0 <= i < |values| ==> values[i] !in members
    ensures forall x :: x in ordering && x !in members ==> x in values
  {
    values := [];
    var i := 0;
    while i < |ordering|
      invariant 0 <= i <= |ordering|
      invariant forall j :: 0 <= j < |values| ==> values[j] !in members
      invariant forall k :: 0 <= k < i && ordering[k] !in members ==> ordering[k] in values
    {
      if ordering[i] !in members {
        values := values + [ordering[i]];
      }
      i := i + 1;
    }
  }

  method IDToIR(
    graph_id: string,
    outcomes: set<string>,
    treatments: set<string>,
    ordering: seq<string>
  ) returns (doc: IRDoc)
    requires graph_id != ""
    requires |ordering| > 0
    ensures doc.version == "1"
    ensures doc.engine == "id"
    ensures doc.query.graph_id == graph_id
    ensures IsCanonicalIRDoc(doc)
  {
    var outcome_seq := FilterByOrdering(ordering, outcomes);
    var treatment_seq := FilterByOrdering(ordering, treatments);
    var query := IRQuery(graph_id, outcome_seq, treatment_seq, ordering);

    // Line 1 shape: if no interventions are present, emit
    // sum_{V\Y} P(V) in IR form.
    if treatments == {} {
      var over := ComplementByOrdering(ordering, outcomes);
      var body := IRProb(ordering, [], []);
      if |over| == 0 {
        doc := IRDoc("1", "id", query, body);
      } else {
        doc := IRDoc("1", "id", query, IRSum(over, body));
      }
      return;
    }

    // Conservative fallback: unresolved recursive cases represented as
    // hedge-style root failure in the executable IR surface.
    var all_nodes := FilterByOrdering(ordering, set x | x in ordering);
    var reduced_nodes := ComplementByOrdering(ordering, treatments);
    doc := IRDoc("1", "id", query, IRFailHedge(all_nodes, reduced_nodes));
  }
}
