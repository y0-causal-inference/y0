/-
  IDAlgorithm.lean — Computable implementation of the ID algorithm (Shpitser & Pearl 2006).
  Port of: identification.dfy (IDImpl) + identification_executable_core.dfy
  Phase L2-005: computable IDImpl returning IRNode, using String-based graphs to
  match the id_cases.v1.json fixture format.
-/
import Y0Lean.Graph
import Y0Lean.SemiMarkovian
import Y0Lean.Traversal
import Y0Lean.IR

namespace Y0Lean

-- ======================================================================
-- String-keyed graph for conformance tests
-- ======================================================================

/-- A directed graph keyed by String node labels (parents map).
    Used for conformance tests that use the `id_cases.v1.json` fixture. -/
abbrev StrGraph := List (String × List String)

/-- Look up parents of a node in a `StrGraph`. -/
def strParents (G : StrGraph) (v : String) : List String :=
  (G.find? (fun p => p.1 = v)).map Prod.snd |>.getD []

/-- All nodes in a `StrGraph` (domain). -/
def strNodes (G : StrGraph) : List String :=
  G.map Prod.fst

/-- Ancestors of `targets` in `G` (BFS over parent edges). -/
def strAncestors (G : StrGraph) (targets : List String) : List String :=
  let allNodes := strNodes G
  let fuel := allNodes.length
  let rec go (frontier visited : List String) (f : ℕ) : List String :=
    match f with
    | 0 => visited ++ frontier
    | f' + 1 =>
      let newNodes := (frontier.flatMap (strParents G)).filter (fun v => v ∉ visited ∧ v ∉ frontier)
      if newNodes.isEmpty then visited ++ frontier
      else go newNodes (visited ++ frontier) f'
  termination_by f
  go (targets.filter (fun t => t ∈ allNodes)) [] fuel

/-- Remove all incoming edges to nodes in `X` from `G`. -/
def strRemoveIncoming (G : StrGraph) (X : List String) : StrGraph :=
  G.map (fun (v, ps) => if v ∈ X then (v, []) else (v, ps))

/-- Restrict `G` to node set `S`. -/
def strSubgraph (G : StrGraph) (S : List String) : StrGraph :=
  G.filterMap (fun (v, ps) =>
    if v ∈ S then some (v, ps.filter (· ∈ S)) else none)

/-- C-component of `v` in undirected-edge representation `biEdges`.
    `biEdges` is a list of symmetric pairs (u, v). -/
def strCComponent (biEdges : List (String × String)) (allNodes : List String) (v : String) : List String :=
  let fuel := allNodes.length
  let neighbors (u : String) : List String :=
    (biEdges.filterMap (fun (a, b) =>
      if a = u then some b
      else if b = u then some a
      else none))
  let rec bfs (frontier visited : List String) (f : ℕ) : List String :=
    match f with
    | 0 => visited ++ frontier
    | f' + 1 =>
      let newN := (frontier.flatMap neighbors).filter (fun w => w ∉ visited ∧ w ∉ frontier)
      if newN.isEmpty then visited ++ frontier
      else bfs newN (visited ++ frontier) f'
  termination_by f
  bfs [v] [] fuel

/-- All C-components of `nodes` given undirected edges, preserving topological order. -/
def strCComponents (biEdges : List (String × String)) (allNodes : List String)
    (nodes : List String) (ord : List String) : List (List String) :=
  let (comps, _) : List (List String) × List String :=
    ord.foldl
      (fun (acc : List (List String) × List String) v =>
        let (cs, vis) := acc
        if v ∈ nodes ∧ v ∉ vis then
          let c := (strCComponent biEdges allNodes v).filter (· ∈ nodes)
          (cs ++ [c], vis ++ c)
        else acc)
      ([], [])
  comps

-- ======================================================================
-- Prefix-before helper (for QValue IR construction)
-- ======================================================================

/-- All elements of `ord` before `target`. -/
def prefixBefore (ord : List String) (target : String) : List String :=
  ord.takeWhile (· ≠ target)

/-- Build Q[S] as a product of conditional probabilities in topological order. -/
def buildQNode (ord : List String) (S : List String) : IRNode :=
  let nodeSeq := ord.filter (· ∈ S)
  let factors := nodeSeq.map (fun v => IRNode.IRProb [v] (prefixBefore ord v) [])
  match factors with
  | []  => IRNode.IRProduct []  -- degenerate; shouldn't happen for non-empty S
  | [f] => f
  | _   => IRNode.IRProduct factors

/-- Filter `ord` to only elements in `S`. -/
def filterByOrdering (ord : List String) (S : List String) : List String :=
  ord.filter (· ∈ S)

-- ======================================================================
-- L2-005: Computable IDImpl (returns IRNode, uses fuel)
-- ======================================================================

/-- A string-keyed semi-Markovian graph for conformance tests. -/
structure StrSMGraph : Type where
  dag       : StrGraph                       -- directed edges (parents map)
  biEdges   : List (String × String)         -- bidirected (undirected) edges

/-- Computable ID algorithm.  Returns an `IRNode` (either a probability expression
    or `IRFailHedge`).  Corresponds to `IDImpl` from identification.dfy but uses
    `String` node labels and emits IR instead of a PMF value.

    Parameters:
    - `sm`  : the causal graph
    - `X`   : intervention variables
    - `Y`   : outcome variables
    - `ord` : topological ordering of all variables in the original graph
    - `fuel`: recursion bound (n² suffices by Shpitser & Pearl 2006 Lemma 3)
-/
def idImpl (sm : StrSMGraph) (X Y : List String) (ord : List String) (fuel : ℕ) : IRNode :=
  let V := strNodes sm.dag
  match fuel with
  | 0 =>
    -- Out of fuel (shouldn't happen with fuel = |V|²)
    IRNode.IRFailHedge V V
  | fuel' + 1 =>
    -- Line 1: if X = ∅, return Σ_{V\Y} P(V)
    if X.isEmpty then
      let over := filterByOrdering ord (V.filter (· ∉ Y))
      let body := IRNode.IRProb (filterByOrdering ord V) [] []
      if over.isEmpty then body
      else IRNode.IRSum over body

    -- Line 2: if V ≠ An(Y)_G, recurse on subgraph of An(Y)
    else
      let ancY := strAncestors sm.dag Y
      if V.any (· ∉ ancY) then
        let smAncY : StrSMGraph := {
          dag     := strSubgraph sm.dag ancY
          biEdges := sm.biEdges.filter (fun (u, v) => u ∈ ancY ∧ v ∈ ancY)
        }
        let X' := X.filter (· ∈ ancY)
        let ord' := filterByOrdering ord ancY
        idImpl smAncY X' Y ord' fuel'

      -- Line 3: W = (V\X) \ An(Y)_{G_{X̄}}; if W ≠ ∅, recurse with X ∪ W
      else
        let Gxbar := strRemoveIncoming sm.dag X
        let smGxbar : StrSMGraph := { dag := Gxbar, biEdges := sm.biEdges }
        let ancYGxbar := strAncestors smGxbar.dag Y
        let W := V.filter (fun v => v ∉ X ∧ v ∉ ancYGxbar)
        if !W.isEmpty then
          let X' := X ++ W
          idImpl sm X' Y ord fuel'

        -- Lines 4-7: C(G \ X) decomposition
        else
          let VminusX := V.filter (· ∉ X)
          let smGX : StrSMGraph := {
            dag     := strSubgraph sm.dag VminusX
            biEdges := sm.biEdges.filter (fun (u, v) => u ∈ VminusX ∧ v ∈ VminusX)
          }
          let ccompsGX := strCComponents smGX.biEdges (strNodes smGX.dag) VminusX ord

          -- Line 4: |C(G\X)| > 1, decompose into products
          if ccompsGX.length > 1 then
            let factors := ccompsGX.map (fun Si =>
              let ord' := filterByOrdering ord Si
              idImpl sm (V.filter (fun v => v ∉ Si)) Si ord' fuel')
            -- sum over V \ (Y ∪ X) of the product
            let over := filterByOrdering ord (V.filter (fun v => v ∉ Y ∧ v ∉ X))
            let body :=
              match factors with
              | [f] => f
              | _   => IRNode.IRProduct factors
            if over.isEmpty then body
            else IRNode.IRSum over body

          -- C(G\X) is a singleton {S}
          else
            match ccompsGX with
            | [] =>
              -- empty: V \ X was empty; shouldn't happen for valid queries but guard
              IRNode.IRFailHedge [] []
            | [S] =>
              let ccompsG := strCComponents sm.biEdges V V ord

              -- Line 5: C(G) = {G} (whole graph is one component) → FAIL
              if ccompsG.length == 1 then
                IRNode.IRFailHedge V S

              -- Line 6: S ∈ C(G) → compute Q[S] directly
              else if ccompsG.any (fun comp => comp.toFinset = S.toFinset) then
                let qS  := buildQNode ord S
                let over := filterByOrdering ord (S.filter (· ∉ Y))
                if over.isEmpty then qS
                else IRNode.IRSum over qS

              -- Line 7: S ⊊ S' ∈ C(G) → recurse on G_{S'}
              else
                match ccompsG.find? (fun comp => S.all (· ∈ comp) ∧ comp.length > S.length) with
                | none =>
                  -- No strictly-larger component contains S (shouldn't happen)
                  IRNode.IRFailHedge V S
                | some Sprime =>
                  let smSp : StrSMGraph := {
                    dag     := strSubgraph sm.dag Sprime
                    biEdges := sm.biEdges.filter (fun (u, v) => u ∈ Sprime ∧ v ∈ Sprime)
                  }
                  -- Q[Sprime] is used implicitly in the recursive call which builds it fresh
                  let X'    := X.filter (· ∈ Sprime)
                  let ord'  := filterByOrdering ord Sprime
                  idImpl smSp X' Y ord' fuel'
            | _ :: _ :: _ =>
              -- More than one component (handled above), unreachable
              IRNode.IRFailHedge [] []
termination_by fuel

/-- Top-level ID entry point.  Wraps `idImpl` with fuel = |V|². -/
def runID (sm : StrSMGraph) (X Y ord : List String) : IRNode :=
  let n := (strNodes sm.dag).length
  idImpl sm X Y ord (n * n)

end Y0Lean
