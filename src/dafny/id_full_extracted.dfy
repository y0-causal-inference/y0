// Executable consolidated ID runtime for extraction-only workflows.
//
// This module is intentionally standalone and avoids importing theorem-heavy
// modules. It provides a single IDToIR entrypoint that emits v1 IR docs.

module IDFullExtracted {

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

  method DistrictNodesForSingleReducedComponent(
    undirected_edges: seq<Edge>,
    all_nodes: seq<string>,
    treatments: set<string>
  ) returns (ok: bool, reduced_component: set<string>, full_component: set<string>)
  {
    var all_nodes_set: set<string> := set x | x in all_nodes;
    var reduced_nodes := all_nodes_set - treatments;
    reduced_component := {};
    full_component := {};

    if reduced_nodes == {} {
      ok := false;
      return;
    }

    var reduced_count := CountComponents(undirected_edges, all_nodes, reduced_nodes);
    if reduced_count != 1 {
      ok := false;
      return;
    }

    var s0: string :| s0 in reduced_nodes;
    reduced_component := ReachableUndirected(undirected_edges, s0, reduced_nodes);
    full_component := ReachableUndirected(undirected_edges, s0, all_nodes_set);
    ok := true;
  }

  method IDToIRWithFuel(
    graph_id: string,
    directed_edges: seq<Edge>,
    undirected_edges: seq<Edge>,
    all_nodes: seq<string>,
    outcomes: set<string>,
    treatments: set<string>,
    ordering: seq<string>,
    fuel: nat
  ) returns (ok: bool, result: IRNode)
    decreases fuel
  {
    if fuel == 0 {
      ok := false;
      result := IRFailHedge([], []);
      return;
    }

    var all_nodes_set: set<string> := set x | x in all_nodes;

    // Line 1: if X ∩ An(Y)_{G_{bar_x}} = ∅, return Σ_{v\Y} P(V)
    // G_{bar_x} is G with all incoming edges to treatments removed.
    // When treatments = {}, this condition is trivially satisfied (∅ ∩ anything = ∅).
    var anc_gx_l1 := ComputeAncestorsWithIncomingRemovedToTreatments(
      directed_edges,
      outcomes,
      treatments,
      all_nodes_set
    );
    if treatments * anc_gx_l1 == {} {
      var over := ComplementByOrdering(ordering, outcomes);
      var body := IRProb(ordering, [], []);
      if |over| == 0 {
        ok := true;
        result := body;
      } else {
        ok := true;
        result := IRSum(over, body);
      }
      return;
    }

    // Line 2: ancestral reduction shape.
    var ancestors := ComputeAncestors(directed_edges, outcomes, all_nodes_set);
    var non_ancestral := all_nodes_set - ancestors;
    if non_ancestral != {} {
      var reduced_treatments := treatments * ancestors;
      var reduced_treatment_seq := FilterByOrdering(ordering, reduced_treatments);
      var reduced_nodes_seq := FilterByOrdering(ordering, ancestors);
      var over2 := ComplementByOrdering(reduced_nodes_seq, outcomes);
      var body2 := IRProb(reduced_nodes_seq, [], reduced_treatment_seq);
      if |over2| == 0 {
        ok := true;
        result := body2;
      } else {
        ok := true;
        result := IRSum(over2, body2);
      }
      return;
    }

    // Line 3: no-effect expansion shape.
    var anc_gx := ComputeAncestorsWithIncomingRemovedToTreatments(
      directed_edges,
      outcomes,
      treatments,
      all_nodes_set
    );
    var W := (all_nodes_set - treatments) - anc_gx;
    if W != {} {
      var expanded_treatments := treatments + W;
      var expanded_seq := FilterByOrdering(ordering, expanded_treatments);
      var over3 := ComplementByOrdering(ordering, outcomes);
      var body3 := IRProb(ordering, [], expanded_seq);
      if |over3| == 0 {
        ok := true;
        result := body3;
      } else {
        ok := true;
        result := IRSum(over3, body3);
      }
      return;
    }

    // Line 4: recursive C-component decomposition.
    var reduced_nodes_l4 := all_nodes_set - treatments;
    var reduced_components := CountComponents(undirected_edges, all_nodes, reduced_nodes_l4);
    if reduced_components > 1 {
      var components: seq<set<string>> := [];
      var visited: set<string> := {};
      var i := 0;
      while i < |all_nodes|
        invariant 0 <= i <= |all_nodes|
      {
        var node := all_nodes[i];
        if node in reduced_nodes_l4 && node !in visited {
          var component := ReachableUndirected(undirected_edges, node, reduced_nodes_l4);
          visited := visited + component;
          components := components + [component];
        }
        i := i + 1;
      }

      var factors: seq<IRNode> := [];
      var all_ok := true;
      i := 0;
      while i < |components|
        invariant 0 <= i <= |components|
      {
        var s := components[i];
        var sub_outcomes := s;
        var sub_treatments := all_nodes_set - s;
        var sub_ok, sub_node := IDToIRWithFuel(
          graph_id,
          directed_edges,
          undirected_edges,
          all_nodes,
          sub_outcomes,
          sub_treatments,
          ordering,
          fuel - 1
        );
        if !sub_ok {
          all_ok := false;
        } else {
          factors := factors + [sub_node];
        }
        i := i + 1;
      }

      if !all_ok {
        ok := false;
        result := IRFailHedge([], []);
        return;
      }

      var over4 := ComplementByOrdering(ordering, outcomes + treatments);
      var body4 := IRProduct(factors);
      if |over4| == 0 {
        ok := true;
        result := body4;
      } else {
        ok := true;
        result := IRSum(over4, body4);
      }
      return;
    }

    // Shared single-component analysis for Lines 5/6/7.
    var ok_comp, reduced_component, full_component := DistrictNodesForSingleReducedComponent(
      undirected_edges,
      all_nodes,
      treatments
    );

    if ok_comp {
      var full_count := CountComponents(undirected_edges, all_nodes, all_nodes_set);

      // Line 5 hedge fail: C(G)={G} and C(G\X)={S}.
      if full_count == 1 {
        var f_nodes := FilterByOrdering(ordering, all_nodes_set);
        var fprime_nodes := FilterByOrdering(ordering, reduced_component);
        ok := true;
        result := IRFailHedge(f_nodes, fprime_nodes);
        return;
      }

      // Line 6: direct district computation when reduced_component is also a district in G.
      if full_component == reduced_component {
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

        var body6 := IRProduct(factors);
        var ranges := FilterByOrdering(ordering, reduced_component - outcomes);
        if |ranges| == 0 {
          ok := true;
          result := body6;
        } else {
          ok := true;
          result := IRSum(ranges, body6);
        }
        return;
      }

      // Line 7: strict superset district reduction shape.
      var district_nodes := FilterByOrdering(ordering, full_component);
      var reduced_treatments := FilterByOrdering(ordering, treatments * full_component);
      var ranges7 := ComplementByOrdering(district_nodes, outcomes);
      var body7 := IRProb(district_nodes, [], reduced_treatments);
      if |ranges7| == 0 {
        ok := true;
        result := body7;
      } else {
        ok := true;
        result := IRSum(ranges7, body7);
      }
      return;
    }

    // Not applicable / unsupported shape.
    ok := false;
    result := IRFailHedge([], []);
  }

  method IDToIR(
    graph_id: string,
    directed_edges: seq<Edge>,
    undirected_edges: seq<Edge>,
    all_nodes: seq<string>,
    outcomes: set<string>,
    treatments: set<string>,
    ordering: seq<string>
  ) returns (ok: bool, doc: IRDoc)
  {
    var outcome_seq := FilterByOrdering(ordering, outcomes);
    var treatment_seq := FilterByOrdering(ordering, treatments);
    var query := IRQuery(graph_id, outcome_seq, treatment_seq, ordering);

    var ok_result, result := IDToIRWithFuel(
      graph_id,
      directed_edges,
      undirected_edges,
      all_nodes,
      outcomes,
      treatments,
      ordering,
      16
    );

    ok := ok_result;
    doc := IRDoc("1", "id", query, result);
  }
}
