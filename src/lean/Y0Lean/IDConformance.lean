/-
  IDConformance.lean — Computable #guard conformance tests for the ID algorithm.
  Driven by: tests/data/generated/dafny_oracle/id_cases.v1.json (10 active cases)
  Phase L2-006/L2-007.

  Graph conventions:
  - `dag` is a parents map: [(v, [parents of v])]
  - `biEdges` is a list of undirected (symmetric) edge pairs

  Test oracle: each graph ID below is sourced from the fixture.
  We check identifiable cases return a non-IRFailHedge result and
  exception cases return IRFailHedge.
-/
import Y0Lean.IDAlgorithm

namespace Y0Lean.Conformance

open Y0Lean

-- ======================================================================
-- Helper: check whether an IRNode is a failure (hedge)
-- ======================================================================

def isHedge : IRNode → Bool
  | IRNode.IRFailHedge _ _ => true
  | _                      => false

def isIdentifiable (r : IRNode) : Bool := !isHedge r

-- ======================================================================
-- Graph definitions
-- Each graph ID corresponds to a case in id_cases.v1.json
-- ======================================================================

-- smoke_line1: X → Y, no bidirected edges
-- Case: id.line1.extracted.identifiable
-- Query: X=[], Y=[Y], ordering=[X,Y]
-- Expected: identifiable (sum over X of P(X,Y))
def g_smoke_line1 : StrSMGraph := {
  dag     := [("X", []), ("Y", ["X"])]
  biEdges := []
}

-- frontdoor_small: X → Z → Y, with latent X ↔ Y
-- Case: id.line4.frontdoor_small.identifiable
-- Query: X=[X], Y=[Y], ordering=[X,Z,Y]
-- Expected: identifiable (frontdoor formula)
def g_frontdoor_small : StrSMGraph := {
  dag     := [("X", []), ("Z", ["X"]), ("Y", ["Z"])]
  biEdges := [("X", "Y")]
}

-- Same graph, different case anchor
-- Case: id.line4.extracted.frontdoor_small.identifiable
def g_line4_frontdoor_small : StrSMGraph := g_frontdoor_small

-- figure1a_like: X → Y, X ↔ Y (bow-arc) — NOT identifiable
-- Case: id.line5.figure1a.hedge
-- Query: X=[X], Y=[Y], ordering=[X,Y]
-- Expected: hedge / exception
def g_figure1a_like : StrSMGraph := {
  dag     := [("X", []), ("Y", ["X"])]
  biEdges := [("X", "Y")]
}

-- line5_bowarc: Same bow-arc structure, different anchor
-- Case: id.line5.extracted.bowarc.hedge
-- Expected: fail (hedge IRFailHedge)
def g_line5_bowarc : StrSMGraph := g_figure1a_like

-- chain_isolated: X → Y with isolated Z
-- Case: id.line2.chain_with_isolated.reduction
-- Query: X=[X], Y=[Y], ordering=[X,Y,Z]
-- Expected: identifiable (line 2 reduction removes Z)
def g_chain_isolated : StrSMGraph := {
  dag     := [("X", []), ("Y", ["X"]), ("Z", [])]
  biEdges := []
}

-- multipath_irrelevant: X→Z, X→Y, Z→Y, irrelevant W
-- Case: id.line2.multipath_with_irrelevant.reduction
-- Query: X=[X], Y=[Y], ordering=[X,Z,Y,W]
-- Expected: identifiable (W is irrelevant, line 2 reduces)
def g_multipath_irrelevant : StrSMGraph := {
  dag     := [("X", []), ("Z", ["X"]), ("Y", ["X", "Z"]), ("W", [])]
  biEdges := []
}

-- line6_single_district: X→Z, Z→Y, X→Y, with Z↔Y only (X is NOT in the district)
-- C(G) = {{X}, {Z,Y}} — two districts; C(G\X)={{Z,Y}} — line 6 fires with S={Z,Y}
-- Case: id.line6.extracted.single_district_formula
-- Query: X=[X], Y=[Y], ordering=[X,Z,Y]
-- Expected: identifiable (Q-formula sum_Z[P(Z|X)*P(Y|X,Z)])
def g_line6_single_district : StrSMGraph := {
  dag     := [("X", []), ("Z", ["X"]), ("Y", ["X", "Z"])]
  biEdges := [("Z", "Y")]
}

-- full.line3.recursive_like: Z→X→Y (chain, intervention on X)
-- Case: id.full.line3.recursive_like
-- Query: X=[X], Y=[Y], ordering=[Z,X,Y]
-- Expected: identifiable (P(Y|X,Z) marginalised over Z)
def g_full_line3 : StrSMGraph := {
  dag     := [("Z", []), ("X", ["Z"]), ("Y", ["X"])]
  biEdges := []
}

-- full.line7.recursive_like: X→W→Y, X↔W, W↔Y — NOT identifiable
-- Case: id.full.line7.recursive_like
-- Query: X=[X], Y=[Y], ordering=[X,W,Y]
-- Expected: hedge / exception
def g_full_line7 : StrSMGraph := {
  dag     := [("X", []), ("W", ["X"]), ("Y", ["W"])]
  biEdges := [("X", "W"), ("W", "Y")]
}

-- ======================================================================
-- Conformance guard tests
-- ======================================================================

-- Case: id.line1.extracted.identifiable → identifiable
#guard isIdentifiable (runID g_smoke_line1 [] ["Y"] ["X", "Y"])

-- Case: id.line4.frontdoor_small.identifiable → identifiable
#guard isIdentifiable (runID g_frontdoor_small ["X"] ["Y"] ["X", "Z", "Y"])

-- Case: id.line4.extracted.frontdoor_small.identifiable → identifiable
#guard isIdentifiable (runID g_line4_frontdoor_small ["X"] ["Y"] ["X", "Z", "Y"])

-- Case: id.line5.figure1a.hedge → hedge (not identifiable)
#guard isHedge (runID g_figure1a_like ["X"] ["Y"] ["X", "Y"])

-- Case: id.line5.extracted.bowarc.hedge → hedge
#guard isHedge (runID g_line5_bowarc ["X"] ["Y"] ["X", "Y"])

-- Case: id.line2.chain_with_isolated.reduction → identifiable (line 2 removes Z)
#guard isIdentifiable (runID g_chain_isolated ["X"] ["Y"] ["X", "Y", "Z"])

-- Case: id.line2.multipath_with_irrelevant.reduction → identifiable (line 2 removes W)
#guard isIdentifiable (runID g_multipath_irrelevant ["X"] ["Y"] ["X", "Z", "Y", "W"])

-- Case: id.line6.extracted.single_district_formula → identifiable
#guard isIdentifiable (runID g_line6_single_district ["X"] ["Y"] ["X", "Z", "Y"])

-- Case: id.full.line3.recursive_like → identifiable
#guard isIdentifiable (runID g_full_line3 ["X"] ["Y"] ["Z", "X", "Y"])

-- Case: id.full.line7.recursive_like → hedge (not identifiable)
#guard isHedge (runID g_full_line7 ["X"] ["Y"] ["X", "W", "Y"])

end Y0Lean.Conformance
