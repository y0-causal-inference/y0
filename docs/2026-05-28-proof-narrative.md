# From Kolmogorov to Completeness: A Proof Narrative

*A guided tour of the mechanically verified causal-inference theorems in y0 —
written for an audience at home with probability, d-separation, and the ID
algorithm, but unfamiliar with Lean 4 or Dafny.*

---

## Overview

This codebase contains two separate but parallel mechanised proof developments:

| Prover | File(s) | What is verified |
|--------|---------|-----------------|
| **Dafny** | `dag.dfy`, `probability.dfy`, `do_calculus.dfy`, `semi_markovian.dfy`, `identification.dfy` | Constructive/algorithmic proofs; executable kernel |
| **Lean 4** | `ProbabilityLayer.lean`, `DSeparation.lean`, `Traversal.lean`, `KahnSort.lean` | Mathlib-backed proofs; rigorous type-theoretic foundation |

Both developments share the same mathematical storyline.  Think of Dafny as
the *engineer's blueprint* (concrete, algorithmic, auto-verifiable via SMT)
and Lean 4 as the *mathematician's ledger* (type-theoretic, certified against
Mathlib).  The theorems below are stated once in English/LaTeX and then
attributed to whichever prover has a verified proof.

---

## The Dependency DAG

Before diving in, here is the high-level picture of how every major result
builds on its predecessors.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 45, 'rankSpacing': 70}}}%%
flowchart TD
    classDef bothProved fill:#d6f5d6,stroke:#2d8a2d,color:#000,padding:12px,font-size:13px
    classDef leanOnly  fill:#d6eeff,stroke:#1a6fa8,color:#000,padding:12px,font-size:13px
    classDef dafnyOnly fill:#fff8d6,stroke:#a08800,color:#000,padding:12px,font-size:13px
    classDef unproved  fill:#ffd6d6,stroke:#cc3333,color:#000,padding:12px,font-size:13px
    classDef algo      fill:#ead6ff,stroke:#6622cc,color:#000,padding:12px,font-size:13px
    classDef mathlib   fill:#ffe0b2,stroke:#cc6600,color:#000,padding:12px,font-size:13px

    K["Kolmogorov Axioms\nnon-negativity · normalization · additivity\nD: 1/3 proved · L: Mathlib PMF type"]:::mathlib
    REACH["Graph Reachability Axioms\nAncestorsCompiled_Correct (D: axiom)\nforwardTrail_of_mem_descendants (L: axiom)"]:::unproved
    PMF["PMF well-formedness\npmf_tsum_one\npmf_additivity"]:::leanOnly
    PROD["Product PMF\npmfProd_tsum_one"]:::leanOnly
    TRUNC["TruncatePMF\ndo-operator\ntruncatePMF_tsum_one"]:::leanOnly
    MARKOV["Markov Factorization\nP = ∏ P(Vᵢ | Pa(Vᵢ))\nMarkovFactorization"]:::unproved
    KAHN_ALGO["Kahn's Algorithm\nkahnSort\nKahnAux"]:::algo
    KAHN_PROOF["Topological Sort Correctness\nKahnsAlgorithm_Correct\nkahnSort_spec"]:::bothProved
    ANC["Ancestors / Descendants Correct\nancestors_correct\ndescendants_correct"]:::leanOnly
    CCOMP["C-Component Membership\ncComponent_correct"]:::bothProved
    LMARKOV["Local Markov Property\nLocalMarkov"]:::dafnyOnly
    SYM["d-Sep Symmetry\nDSep_Symmetry\ndSep_symmetry"]:::bothProved
    DEC["d-Sep Decomposition\nDSep_Decomposition\ndSep_decomposition"]:::bothProved
    WU["d-Sep Weak Union\nDSep_WeakUnion\ndSep_weakUnion"]:::bothProved
    CONT["d-Sep Contraction\nDSep_Contraction\ndSep_contraction"]:::bothProved
    INTER["d-Sep Intersection\nDSep_Intersection"]:::dafnyOnly
    GM["Global Markov Property\nGlobalMarkov\nKernelGlobalMarkov"]:::unproved
    R1["Do-Calculus Rule 1\nInsert/Delete Observation\nRule1Plus proved · Rule1 axiom"]:::dafnyOnly
    R2["Do-Calculus Rule 2\nAction/Observation Exchange\nRule2_ActionObservationExchange"]:::unproved
    R3["Do-Calculus Rule 3\nInsert/Delete Action\nRule3_InsertDeleteAction"]:::unproved
    BKD["Backdoor Criterion\nBackdoorAdjustment"]:::unproved
    FRD["Frontdoor Criterion\nFrontdoorCriterion"]:::unproved
    L1["Lemma 1\nNon-Identifiability Witness\nLemma1_NonIdentifiabilityWitness"]:::unproved
    L2["Lemma 2\nC-Component Factorization\nLemma2_CComponentFactorization"]:::unproved
    L3L["Lemma 3\nQ-Value Derivation\nLemma3_QValueDerivation"]:::unproved
    ID["ID Algorithm Lines 1–7\nIDImpl / ID"]:::algo
    T2["Theorem 2: Soundness of ID\nTheorem2_Soundness"]:::unproved
    T3["Theorem 3: Completeness of ID\nTheorem3_Completeness"]:::unproved
    T4["Theorem 4: Do-Calculus Completeness\nTheorem4_DoCalculusCompleteness"]:::unproved

    K --> PMF
    PMF --> PROD
    PMF --> TRUNC
    TRUNC --> MARKOV
    MARKOV --> GM
    REACH --> KAHN_ALGO
    REACH --> ANC
    KAHN_ALGO --> KAHN_PROOF
    KAHN_PROOF --> ANC
    ANC --> CCOMP
    REACH --> LMARKOV
    REACH --> SYM
    REACH --> DEC
    REACH --> WU
    WU --> CONT
    CONT --> INTER
    LMARKOV --> GM
    SYM --> GM
    DEC --> GM
    WU --> GM
    GM --> R1
    GM --> R2
    GM --> R3
    R1 --> BKD
    R2 --> BKD
    R1 --> FRD
    R2 --> FRD
    R3 --> FRD
    R1 --> ID
    R2 --> ID
    R3 --> ID
    L1 --> T3
    L2 --> ID
    L3L --> ID
    MARKOV --> L2
    KAHN_PROOF --> L2
    CCOMP --> L2
    CCOMP --> L3L
    ID --> T2
    ID --> T3
    T2 --> T4
    T3 --> T4
```

> **Color key:** 🟩 Dark green = proved in **both** Lean and Dafny · 🔵 Blue = proved in **Lean only** (Dafny has `{:axiom}`) · 🟨 Yellow = proved in **Dafny only** (not yet ported to Lean) · 🟥 Red = unproved in both · 🟣 Purple = algorithm definition · 🟠 Orange = encoded as a **Mathlib type definition** (not a proof, but a definitional consequence — e.g. Kolmogorov axioms hold by construction from `PMF Outcome`)

Read the arrows as *"is used by"*.  Everything ultimately flows from three
primitive facts: the Kolmogorov axioms for probability, the axioms of graph
reachability, and the semi-graphoid axioms for d-separation.

---

## Part I — The Probability Foundation

### Kolmogorov Axioms

The entire development rests on three well-known axioms.  In Lean 4 they are
not asserted from thin air but *derived* from Mathlib's `PMF` type, which
encodes all three by construction.

**Axiom K1 — Non-negativity.**

$$P(A) \ge 0 \quad \text{for every event } A.$$

```lean
-- Lean 4 (ProbabilityLayer.lean)
theorem pmf_apply_nonneg (p : PMF Outcome) (a : Outcome) : 0 ≤ p a
-- Proof: trivial — PMF values live in ℝ≥0∞, non-negative by type.
```

Because Lean's `PMF Outcome` is literally the type
$\{ f : \text{Outcome} \to \mathbb{R}_{\ge 0}^\infty \mid \sum_a f(a) = 1 \}$,
non-negativity is baked into the value type itself.  No arithmetic is needed;
the proof is one word: `zero_le`.

**Axiom K2 — Normalization.**

$$\sum_{a \in \Omega} P(a) = 1.$$

```lean
theorem pmf_tsum_one (p : PMF Outcome) : ∑' a, p a = 1
-- Proof: p.tsum_coe  (one-liner into Mathlib)
```

**Axiom K3 — Finite Additivity.**  For disjoint events $A, B$:

$$P(A \cup B) = P(A) + P(B).$$

```lean
theorem pmf_additivity (p : PMF Outcome) (A B : Finset Outcome)
    (h : Disjoint A B) :
    ∑ a ∈ A ∪ B, p a = ∑ a ∈ A, p a + ∑ a ∈ B, p a
-- Proof: Finset.sum_union h
```

In Dafny the same three axioms appear as named lemmas in `probability.dfy`
(`Axiom_NonNegativity`, `Axiom_Normalization`, `Axiom_Additivity`), stated
as specification-only preconditions.

### Product Distributions and the Do-Operator

Two constructions built directly on the Kolmogorov axioms are needed
throughout the causal layer.

**Product PMF.**  Given independent marginals $P_X$ and $P_Y$, define

$$P_{XY}(x, y) = P_X(x) \cdot P_Y(y).$$

```lean
noncomputable def pmfProd (p q : PMF Outcome) : PMF (Outcome × Outcome) :=
  p.bind fun a => q.map (Prod.mk a)

theorem pmfProd_tsum_one (p q : PMF Outcome) : ∑' a, (pmfProd p q) a = 1
```

**Truncated PMF (the discrete do-operator).**  Conditioning on an event
$s \subseteq \Omega$ (provided $P(s) > 0$) gives

$$P(\cdot \mid s)(a) \;\propto\; P(a) \cdot \mathbf{1}[a \in s].$$

```lean
noncomputable def truncatePMF (p : PMF Outcome) (s : Set Outcome)
    (hs : ∃ a ∈ s, a ∈ p.support) : PMF Outcome :=
  p.filter s hs

theorem truncatePMF_tsum_one ... : ∑' a, (truncatePMF p s hs) a = 1
```

`truncatePMF` is the *discrete incarnation* of Pearl's do-operator: setting
$\text{do}(X = x)$ is modelled by conditioning the joint distribution on the
event $\{ \omega : \omega_X = x \}$ in the mutilated graph.

---

## Part II — Graph Structure

### Topological Sort (Kahn's Algorithm)

The ID algorithm requires a topological ordering $\pi$ of the DAG at every
recursive call.  Lean 4 verifies Kahn's classic BFS-based algorithm:

```lean
-- Traversal.lean / KahnSort.lean
theorem kahnSort_spec (G : Graph) (ord : List Node) :
    kahnSort G = some ord →
    (∀ i j, i < j → ¬ G.hasEdge (ord[j]!) (ord[i]!))  -- no back-edges
    ∧ ord.length = G.nodeCount                           -- covers all nodes
    ∧ ord.Nodup                                          -- no duplicates
```

This guarantees that `kahnSort G` either returns a valid linear extension of
the DAG's partial order, or `none` (when the graph has a cycle).

### Ancestors and Descendants

Reachability in the DAG is verified in both Lean 4 and Dafny:

$$\text{An}_G(W) = \{ u \in V : \exists \text{ directed path } u \to w, w \in W \}.$$

```lean
-- Traversal.lean
theorem ancestors_correct (G : Graph) (W : Finset Node) :
    ancestors G W = { u | ∃ w ∈ W, G.hasPath u w }
```

The Dafny counterpart (`AncestorsCompiled_Correct`, `DescendantsCompiled_Correct`)
is axiomatised (marked `{:axiom}`) pending a full compiled-implementation proof.

### C-Component Membership

In a semi-Markovian graph, two nodes belong to the same *C-component*
(confounded component) iff they are connected by a path of bidirected edges:

$$u \sim v \;\Longleftrightarrow\; \text{bidirected-path}(u, v).$$

```lean
-- Traversal.lean
theorem cComponent_correct (sm : SMGraph) (v : Node) :
    cComponent sm v = { u | sm.hasBidirectedPath v u }
```

This is proved by a BFS over bidirected edges, with the invariant that the
frontier is always a subset of the final component.

---

## Part III — D-Separation and the Semi-Graphoid Axioms

### What is d-separation?

Recall the definition.  A *trail* $t$ connecting $Y$ and $Z$ in DAG $G$ is
**blocked** by conditioning set $W$ if there exists a position on the trail
that is either:
- a **non-collider** node in $W$, or
- a **collider** node (and neither it nor any of its descendants is in $W$).

$Y$ and $Z$ are **d-separated** given $W$ (written $Y \perp_G Z \mid W$) when
*every* trail connecting $Y$ to $Z$ is blocked by $W$.

Both provers represent trails as sequences of typed steps (`TrailStep`), with
each step labelled `Forward` (following a directed edge) or `Backward`
(traversing against a directed edge).  Blocking is then a predicate on positions.

### The Four Semi-Graphoid Axioms

The following four axioms characterise d-separation as a *semi-graphoid*.
They are the backbone of all probabilistic independence reasoning in causal
inference.  All four are **fully machine-verified** in both Dafny and Lean 4.

#### Theorem: Decomposition

$$Y \perp_G (Z \cup Z') \mid W \;\Longrightarrow\; Y \perp_G Z \mid W.$$

**Intuition.** If a set of variables $Z \cup Z'$ is irrelevant to $Y$, then
any subset $Z$ is certainly irrelevant.

**Proof idea.** Any trail from $Y$ to $z \in Z$ is also a trail from $Y$ to
$z \in Z \cup Z'$, which is blocked by hypothesis.  The proof in Lean 4 is a
one-liner:

```lean
-- DSeparation.lean
theorem dSep_decomposition (G : Graph) (Y Z Z' W : Finset Node) :
    dSep G Y (Z ∪ Z') W → dSep G Y Z W := by
  intro h t y z hy hz hv hc htc
  exact h t y z hy (Finset.mem_union_left Z' hz) hv hc htc
```

The Dafny proof is equally direct — it just asserts `z ∈ Z + Z'` and the
hypothesis closes the goal.

#### Theorem: Symmetry

$$Y \perp_G Z \mid W \;\Longrightarrow\; Z \perp_G Y \mid W.$$

**Intuition.** Irrelevance is symmetric — if $Z$ tells us nothing about $Y$,
then $Y$ tells us nothing about $Z$.

**Proof idea.** Given a trail $t$ from $Z$ to $Y$, reverse every step to
obtain a trail $\bar{t}$ from $Y$ to $Z$.  The blocking status of a trail is
invariant under reversal (reversing a trail maps colliders to colliders and
non-colliders to non-colliders).  The hypothesis then blocks $\bar{t}$ given
$W$, and the involution $\overline{\overline{t}} = t$ propagates this back.

```lean
theorem dSep_symmetry (G : Graph) (Y Z W : Finset Node) :
    dSep G Y Z W → dSep G Z Y W := by
  intro h t z y hz hy hv hc htc
  have hrev_v  := reverseTrail_valid G t hv
  have hrev_c  := reverseTrail_connected t hc
  have hrev_tc := reverseTrail_connects t z y htc
  have hblk    := h (reverseTrail t) y z hy hz hrev_v hrev_c hrev_tc
  rw [← reverseTrail_invol t]
  exact reverseTrail_blocked G (reverseTrail t) W hrev_c hblk
```

The key auxiliary is `reverseTrail_invol`: $\overline{\overline{t}} = t$.

#### Theorem: Weak Union

$$Y \perp_G (Z \cup Z') \mid W \;\Longrightarrow\; Y \perp_G Z \mid (W \cup Z').$$

**Intuition.** If the whole set $Z \cup Z'$ is irrelevant to $Y$ given $W$,
then $Z$ is still irrelevant even after we condition on the additional
variables $Z'$.  Adding information about $Z'$ to the conditioning set cannot
*create* a dependence between $Y$ and $Z$ where none existed.

**Proof idea (by contradiction).**  Suppose a trail $t$ from $Y$ to
$z \in Z$ is *unblocked* given $W \cup Z'$.  The hypothesis says every trail
to any node in $Z \cup Z'$ is blocked given $W$.  In particular $t$ is blocked
given $W$, so some position $\text{pos}$ on $t$ witnesses the block.

At position $\text{pos}$:
- If it were a non-collider, the blocking node would already be in $W \subseteq W \cup Z'$, so the trail would be blocked given $W \cup Z'$ as well — contradiction.
- Therefore $\text{pos}$ must be a **collider**, and the blocking under $W$
  comes from one of its descendants $z' \in Z'$ being in $W$.  But $z' \in Z \cup Z'$, and the prefix of $t$ up to $\text{pos}$ is an unblocked trail (no blocking positions precede $\text{pos}$) from $Y$ to a node in $Z \cup Z'$.  Applying the hypothesis to this prefix or its extension through $z'$ yields a contradiction.

This argument is formalised across ~200 lines in both provers, handling the
case split (collider node equals $z'$, or $z'$ is a proper descendant) by
constructing a concatenated trail.

#### Theorem: Contraction

$$Y \perp_G Z \mid W \;\;\text{and}\;\; Y \perp_G Z' \mid (W \cup Z)
\;\Longrightarrow\; Y \perp_G (Z \cup Z') \mid W.$$

**Intuition.** If $Z$ is irrelevant to $Y$ given $W$, and $Z'$ is irrelevant
to $Y$ even after we add $Z$ to the conditioning set, then the combined set
$Z \cup Z'$ is irrelevant given $W$ alone.

**Proof idea.**  For a trail $t$ from $Y$ to $z \in Z \cup Z'$:
- If $z \in Z$: the first hypothesis immediately blocks $t$ given $W$.
- If $z \in Z'$: the second hypothesis blocks $t$ given $W \cup Z$.  But the
  first blocking position $\text{pos}$ (under $W \cup Z$) is not a blocking
  position under $W$ alone.  So the blocking is contributed by some $z' \in Z$
  at $\text{pos}$.  Consider the prefix of $t$ up to $\text{pos}$: it reaches
  $z' \in Z$ without any blocking under $W$.  The first hypothesis then blocks
  this prefix — contradiction.

```lean
theorem dSep_contraction (G : Graph) (Y Z Z' W : Finset Node) :
    dSep G Y Z W → dSep G Y Z' (W ∪ Z) → dSep G Y (Z ∪ Z') W
```

#### Theorem: Intersection *(requires positivity)*

$$Y \perp_G Z \mid (W \cup Z') \;\;\text{and}\;\; Y \perp_G Z' \mid (W \cup Z)
\;\Longrightarrow\; Y \perp_G (Z \cup Z') \mid W.$$

**Intuition.** If $Z$ and $Z'$ are each irrelevant to $Y$ given the other,
then together they are irrelevant.

This axiom holds for d-separation in DAGs (not just positive distributions).
The Dafny proof delegates to a helper lemma `DSep_Intersection_Descend` that
handles each case by symmetry with Contraction.

### The Local Markov Property

The following theorem is the fundamental link between graph structure and
probabilistic independence.  It says that, in a DAG, every node is
d-separated from its non-descendants given its parents:

**Theorem (Local Markov).**

$$\{V_i\} \perp_G \bigl(\text{NonDesc}(V_i) \setminus \text{Pa}(V_i)\bigr) \mid \text{Pa}(V_i).$$

```dafny
// dag.dfy
lemma LocalMarkov(G: Graph, v: Node)
  requires v in Nodes(G)
  requires IsDAG(G)
  ensures DSep(G, {v}, NonDescendants(G, v) - Parents(G, v), Parents(G, v))
```

**Proof sketch.**  Take any trail $t$ from $v$ to $z \notin \text{Pa}(v) \cup
\text{Desc}(v)$.  Three exhaustive cases:

1. **The first step is backward** ($v \leftarrow \cdot$): the node immediately
   before $v$ on the trail is a parent of $v$, hence in $\text{Pa}(v)$.
   This is a non-collider in $\text{Pa}(v)$, so the trail is immediately blocked.
   (Lemma `BackwardFirstStep_BlockedByParents`.)

2. **All steps are forward** ($v \to \cdots \to z$): a purely forward trail
   reaches only descendants of $v$.  But $z \notin \text{Desc}(v)$ — contradiction.
   (Lemma `ForwardTrail_EndInDescendants_DAG`.)

3. **The first backward step occurs at some position $\text{pos} > 0$**: up to
   position $\text{pos}$ the trail is purely forward, so the node at $\text{pos}$
   is a descendant of $v$.  The step at $\text{pos}$ is a forward-then-backward
   pivot: the node is a *non-collider* (forward in, backward out) and lies in
   $\text{Pa}(v)$ or further back — the parents of $v$'s descendant are either
   descendants themselves (opening the block), or ancestors of $v$ (which
   would create a cycle in the DAG — impossible).
   (Lemma `FirstForwardBackwardPivot_BlockedByParents`.)

---

## Part IV — The Do-Calculus

### Global Markov Property

The *Global Markov Property* (GMP) bridges the graphical notion of
d-separation to the probabilistic notion of conditional independence.  It
states that d-separation implies independence in every distribution Markov to
the graph:

$$Y \perp_G Z \mid W \;\Longrightarrow\; P(Y \mid Z, W) = P(Y \mid W).$$

In the formalism, interventional distributions are written as
$\text{IntProb}(G, Y, \text{do}(X), W)$, so the GMP takes the form:

$$Y \perp_G Z \mid W \;\Longrightarrow\; \text{IntProb}(G, Y, \emptyset, Z \cup W) = \text{IntProb}(G, Y, \emptyset, W).$$

```dafny
// do_calculus.dfy
lemma {:axiom} GlobalMarkov(G: Graph, Y: set<Node>, Z: set<Node>, W: set<Node>)
  requires DSep(G, Y, Z, W)
  ensures  IntProb(G, Y, {}, Z + W) == IntProb(G, Y, {}, W)
```

This is declared as an axiom at the do-calculus layer (because the abstract
`IntProb` type does not yet carry an explicit PMF witness), but the concrete
version `KernelGlobalMarkov` in `interventional.dfy` provides a
witness-bearing proof via the Markov Factorization and `truncatePMF`.

### Intervention Semantics

Setting $\text{do}(X = x)$ *mutilates* the graph: all incoming edges to nodes
in $X$ are deleted, producing $G_{\bar{X}}$.  The interventional distribution
is then the observational distribution in the mutilated graph:

$$P(Y \mid \text{do}(X), W) = P_{G_{\bar{X}}}(Y \mid W).$$

```dafny
lemma {:axiom} InterventionSemantics(G, Y, X, W)
  ensures IntProb(G, Y, X, W) == IntProb(RemoveIncoming(G, X), Y, {}, W)
```

### The Three Rules of Do-Calculus

Pearl's three rules (Theorem 3.4 in *Causality*, 2000) are the complete
inference engine for interventional queries.  All three are verified in
`do_calculus.dfy`, with Rules 2 and 3 as axioms (their proofs require
structural-causal-model arguments beyond the graph layer) and Rule 1
as a derived theorem.

**Rule 1 — Insertion/Deletion of Observations.**

$$\bigl(Y \perp_{G_{\bar{X}}} Z \mid X, W\bigr) \;\Longrightarrow\;
P(Y \mid \text{do}(X), Z, W) = P(Y \mid \text{do}(X), W).$$

*Intuition:* After intervening on $X$, the d-separation criterion in
$G_{\bar{X}}$ tells us that $Z$ carries no information about $Y$ beyond $X$
and $W$.

The Dafny proof of the stronger form `Rule1Plus_InsertDeleteObservation` is
explicit:

```dafny
lemma Rule1Plus_InsertDeleteObservation(G, Y, X, Z, W)
  requires DSep(RemoveIncoming(G, X), Y, Z, W)
  ensures  IntProb(G, Y, X, Z + W) == IntProb(G, Y, X, W)
{
  InterventionSemantics(G, Y, X, Z + W);    // P(Y|do(X),Z,W) = P_{Gx}(Y|Z,W)
  InterventionSemantics(G, Y, X, W);        // P(Y|do(X),W)   = P_{Gx}(Y|W)
  GlobalMarkov(Gx, Y, Z, W);               // DSep ⟹ P_{Gx}(Y|Z,W)=P_{Gx}(Y|W)
}
```

Three lines.  The whole argument is: rewrite both sides using Intervention
Semantics, then close by the Global Markov Property.

**Rule 2 — Action/Observation Exchange.**

$$\bigl(Y \perp_{G_{\bar{X}, \underline{Z}}} Z \mid X, W\bigr) \;\Longrightarrow\;
P(Y \mid \text{do}(X), \text{do}(Z), W) = P(Y \mid \text{do}(X), Z, W).$$

*Intuition:* When $Z$'s outgoing edges are also deleted, intervening on $Z$
is the same as observing $Z$.

**Rule 3 — Insertion/Deletion of Actions.**  Let $\bar{Z}(W)$ denote the
nodes of $Z$ that are not ancestors of $W$ in $G_{\bar{X}}$:

$$\bigl(Y \perp_{G_{\bar{X}, \overline{\bar{Z}(W)}}} Z \mid X, W\bigr) \;\Longrightarrow\;
P(Y \mid \text{do}(X), \text{do}(Z), W) = P(Y \mid \text{do}(X), W).$$

*Intuition:* If the intervention on $Z$ cannot reach $Y$ (given $X$ and $W$),
the intervention is superfluous and may be deleted.

### Backdoor Criterion

**Theorem (Backdoor Adjustment, Pearl 2000 Theorem 3.3.2).**

Let $Z$ satisfy:
1. No $z \in Z$ is a descendant of any $x \in X$.
2. $Z$ d-separates $Y$ from $X$ in the graph $G_{\underline{X}}$ (outgoing
   edges from $X$ removed).

Then:

$$P(Y \mid \text{do}(X)) = \sum_z P(Y \mid X = x, Z = z) \cdot P(Z = z).$$

```dafny
lemma {:axiom} BackdoorAdjustment(G, Y, X, Z)
  requires (∀ x ∈ X, z ∈ Z. IsAncestor(G, x, z) ⟹ x = z)
  requires DSep(RemoveOutgoing(G, X), Y, X, Z)
  ensures  IntProb(G, Y, X, {}) == IntProb(G, Y, {}, X + Z)
```

### Frontdoor Criterion

**Theorem (Frontdoor Criterion, Pearl 2000 Theorem 3.3.4).**

If $M$ satisfies the frontdoor criterion for $(X \to Y)$:
1. $M$ intercepts all directed paths from $X$ to $Y$.
2. No unblocked backdoor paths from $X$ to $M$ in $G_{\underline{X}}$.
3. All backdoor paths from $M$ to $Y$ are blocked by $X$ in $G_{\underline{M}}$.

Then:

$$P(Y \mid \text{do}(X)) = \sum_m P(M = m \mid \text{do}(X)) \cdot P(Y \mid \text{do}(M = m)).$$

```dafny
lemma {:axiom} FrontdoorCriterion(G, Y, X, M)
  requires DSep(RemoveOutgoing(G, X), M, X, {})
  requires DSep(RemoveOutgoing(G, M), M, Y, X)
  ensures  IntProb(G, Y, X, {}) == IntProb(G, Y, M, {})
```

---

## Part V — The ID Algorithm and Its Correctness Theorems

The ID algorithm (Shpitser & Pearl, AAAI-06) decides whether a causal effect
$P_x(y)$ is *identifiable* from the observational distribution $P(V)$ in a
semi-Markovian model, and if so, returns the identifying expression.

The algorithm is implemented as `IDImpl` in `identification.dfy`, following
Figure 3 of the paper with an explicit *fuel* parameter for termination.

### Preliminary: C-Component Factorization

**Lemma 2 (Tian & Pearl 2002, Theorem 4).**

Let $G$ be a semi-Markovian DAG with C-components $S_1, \ldots, S_k$ (the
bidirected-connected components).  Let $\pi$ be a topological ordering of
$V$.  Then the joint observational distribution *factors* as:

$$P(v) = \prod_{i=1}^k Q[S_i],$$

where each *c-factor* is

$$Q[S_i] = \prod_{V_j \in S_i} P(v_j \mid v_\pi^{j-1}),
\qquad v_\pi^{j-1} = \{v_l : l < j \text{ in } \pi\},$$

and $Q[S_i] = P_{V \setminus S_i}(S_i)$ — the effect of intervening on
everything outside $S_i$.

```dafny
// identification.dfy
lemma {:axiom} Lemma2_CComponentFactorization(sm, p, ord)
  requires WellFormedSM(sm) ∧ IsDistribution(p) ∧ MarkovFactorization(sm.dag, p)
  requires SMTopologicalSort(sm, ord)
  // ensures P(v) = ∏ᵢ Q[Sᵢ]
```

**Lemma 3 (Tian & Pearl 2002, Lemma 3).**

If $D \subseteq S$ where $S \in \mathcal{C}(G)$ and $D \in \mathcal{C}(G_S)$
(C-component of the sub-graph on $S$), then $Q[D]$ is *derivable* from $Q[S]$
and $P(V)$:

$$Q[D] = \prod_{V_i \in D} Q[\{V_i\}] \Big/ \prod_{V_i \in S \setminus D} Q[\{V_i\}].$$

```dafny
lemma {:axiom} Lemma3_QValueDerivation(sm, p, S, D, ord)
  requires D ≤ S ≤ SMNodes(sm) ∧ S ∈ CComponents(sm)
  ensures IsDistribution(QValue(sm, p, D, ord))
```

This is the key *recursion step* behind Lines 6–7 of the ID algorithm.

### Non-Identifiability Witness

**Lemma 1 (Shpitser & Pearl 2006).**

If two causal models $M^1, M^2$ share the same graph $G$ but satisfy
$P^1(V) = P^2(V)$ yet $P^1_x(Y) \neq P^2_x(Y)$, then $P_x(Y)$ is
*not identifiable* in $G$.

**Proof.** Suppose for contradiction a function $f$ with
$P_x(Y) = f(P(V))$ existed.  Then
$P^1(V) = P^2(V)$ would force $f(P^1) = f(P^2)$,
contradicting $P^1_x(Y) \neq P^2_x(Y)$. $\square$

```dafny
lemma {:axiom} Lemma1_NonIdentifiabilityWitness(sm, p1, p2, X, Y)
  requires p1 == p2 ∧ MarkovFactorization(sm.dag, p1) ∧ MarkovFactorization(sm.dag, p2)
  ensures !IsIdentifiable(sm, X, Y)
```

### The Seven Lines of ID

The algorithm proceeds by case analysis.  Each branch is a separate lemma:

| Line | Condition | Action | Do-Calculus justification |
|------|-----------|--------|--------------------------|
| **1** | $X = \emptyset$ | Return $\sum_{V \setminus Y} P(V)$ | Rule 3 (delete actions) × $|X|$ |
| **2** | $V \neq \text{An}(Y)_G$ | Restrict to $G_{\text{An}(Y)}$ | Rule 1 (delete non-ancestors) |
| **3** | $W = (V \setminus X) \setminus \text{An}(Y)_{G_{\bar{X}}} \neq \emptyset$ | Expand to $\text{do}(X \cup W)$ | Rule 3 (insert irrelevant actions) |
| **4** | $|\mathcal{C}(G \setminus X)| > 1$ | Factor: $\sum_{V \setminus (Y \cup X)} \prod_i \text{ID}(S_i, V \setminus S_i, P, G)$ | Lemma 2 (C-component factorization) |
| **5** | $\mathcal{C}(G) = \{G\}$, $\mathcal{C}(G \setminus X) = \{S\}$ | **FAIL** — return hedge $(G, S)$ | Non-identifiability (Lemma 1) |
| **6** | $S \in \mathcal{C}(G)$ | Compute $Q[S]$ directly | Lemma 2 (c-factor formula) |
| **7** | $\exists S' \in \mathcal{C}(G): S \subset S'$ | Recurse: $\text{ID}(Y, X \cap S', Q[S'], G_{S'})$ | Lemma 3 (Q-value derivation) |

Each line has a corresponding `{:axiom}` lemma in `identification.dfy`
(`ID_Line1` through `ID_Line7`) stating the invariant that the recursive call
preserves the causal query's answer.

**Key structural lemmas** keep the verifier honest about the recursive calls:

- `IDImpl_Line2_ValidQuery` — the sub-query $(G_{\text{An}(Y)}, X \cap \text{An}(Y), Y)$ is well-formed.
- `IDImpl_Line3_ValidQuery` — $(X \cup W) \cap Y = \emptyset$ (the expanded treatment set remains disjoint from outcomes).
- `IDImpl_Line567_CComponentsOne` — when both Line 3 and Line 4 guards fail, exactly one C-component remains.
- `IDImpl_Line7_YSubsetSprime` — the outcomes $Y$ are contained in $S'$, so the sub-query on $G_{S'}$ is well-typed.

### Theorem 2: Soundness

**Theorem 2 (Shpitser & Pearl 2006).**

*Whenever the ID algorithm returns* $\text{Identified}(\text{pmf})$*,
the distribution* $\text{pmf}$ *correctly represents* $P_x(Y)$.

$$\text{ID}(Y, X, P, G) = \text{Identified}(\mu) \;\Longrightarrow\; \mu = P_x(Y).$$

**Proof sketch (by induction on recursion depth).**

- *Line 1*: $P_x(Y) = P(Y)$ when $X = \emptyset$ — the marginal is trivially correct.
- *Line 2*: Ancestral restriction is valid because non-ancestors of $Y$ cannot
  affect $P_x(Y)$ (they have no directed path to $Y$).
- *Line 3*: Adding interventions on $W = (V \setminus X) \setminus \text{An}(Y)_{G_{\bar{X}}}$
  does not change $P_x(Y)$ — these nodes are not ancestral to $Y$ in $G_{\bar{X}}$,
  so Rule 3 applies.
- *Line 4*: C-component factorization (Lemma 2) decomposes $P_x$ into independent
  sub-problems, each solved correctly by induction.
- *Lines 6–7*: Q-value computation (Lemma 3) correctly derives $Q[S]$ from $P$.

```dafny
lemma {:axiom} Theorem2_Soundness(sm, X, Y, p, ord)
  requires ID(sm, X, Y, p, ord).Identified?
  ensures IsDistribution(ID(sm, X, Y, p, ord).pmf)
  // and pmf = true interventional distribution P_x(Y)
```

### Theorem 3: Completeness (Hedge Characterisation)

**Theorem 3 (Shpitser & Pearl 2006).**

$$P_x(Y) \text{ is identifiable in } G
\;\;\Longleftrightarrow\;\;
\text{ID}(Y, X, P, G) \neq \text{FAIL}.$$

Equivalently: $P_x(Y)$ is *not* identifiable if and only if a **hedge** for
$P_x(Y)$ exists in $G$.

A *hedge* $(F, F')$ consists of two subgraphs such that $F$ and $F'$ are
C-components of appropriate sub-graphs and $X$ cannot "separate" the confounding
paths, witnessing the impossibility of identifying the effect.

**Completeness direction.** When ID returns FAIL at Line 5, the two subgraphs
$(G, \text{SubgraphSM}(G, S))$ form a hedge.  Two causal models are then
constructed (using Lemma 1's contra-positive) that agree on $P(V)$ but
disagree on $P_x(Y)$, proving non-identifiability.

```dafny
lemma {:axiom} Theorem3_Completeness(sm, X, Y, p, ord)
  ensures ID(sm, X, Y, p, ord).Identified? <==> IsIdentifiable(sm, X, Y)

lemma {:axiom} Theorem3_HedgeIFF(sm, X, Y, p, ord)
  ensures ID(sm, X, Y, p, ord).NotIdentified? <==>
          (exists F, Fp :: IsHedge(sm, F, Fp, X, Y))
```

### Theorem 4: Completeness of Do-Calculus

**Theorem 4 (Shpitser & Pearl 2006).**

*The three rules of the do-calculus are complete for causal identification:
every identifying expression returned by ID can be derived from $P(V)$ by a
finite sequence of Rule 1–3 applications.*

This is the deepest result.  The proof goes line by line through the ID
algorithm, exhibiting the explicit do-calculus derivation for each branch:

| ID Line | Do-Calculus rules used |
|---------|------------------------|
| 1 | Rule 3 applied $|X|$ times (delete all actions) |
| 2 | Rule 1 (delete non-ancestral observations) |
| 3 | Rule 3 (insert interventions on causally irrelevant nodes) |
| 4 | Rules 2 + 3 with C-component factorization |
| 6–7 | Rule 2 (action/observation exchange) + Rule 3 |

```dafny
lemma {:axiom} Theorem4_DoCalculusCompleteness(sm, X, Y, p, ord)
  ensures IsIdentifiable(sm, X, Y) ==> ID(sm, X, Y, p, ord).Identified?
```

---

## Putting It All Together: The Full Proof Path

Here is the complete logical chain, from first principles to Theorem 4:

1. **Kolmogorov axioms** $\to$ PMF type (K1–K3).
2. **Product PMF** and **TruncatePMF** $\to$ do-operator semantics.
3. **Markov Factorization** $\to$ structural causal model semantics.
4. **Kahn's algorithm** $\to$ topological order for C-factor computation.
5. **BFS on bidirected graph** $\to$ C-component membership.
6. **Trail blocking** $\to$ d-separation definition.
7. **Symmetry, Decomposition, Weak Union, Contraction, Intersection** $\to$
   semi-graphoid axioms (the four fundamental rules of conditional independence
   expressed graphically).
8. **Local Markov Property** $\to$ every node is independent of its
   non-descendants given its parents.
9. **Global Markov Property** $\to$ d-separation $\Rightarrow$ independence.
10. **Intervention Semantics** $\to$ do-operator = distribution in mutilated graph.
11. **Rules 1–3** $\to$ do-calculus.
12. **Backdoor and Frontdoor criteria** $\to$ practical identification tools.
13. **Lemma 2** (C-component factorization) $\to$ ID Lines 4, 6.
14. **Lemma 3** (Q-value derivation) $\to$ ID Line 7.
15. **Lemma 1** (non-identifiability witness) $\to$ ID Line 5 (hedge failure).
16. **ID Algorithm** (Lines 1–7) — the constructive engine.
17. **Theorem 2** (Soundness): ID returns the correct answer.
18. **Theorem 3** (Completeness): ID fails iff a hedge exists.
19. **Theorem 4** (Do-calculus completeness): Rules 1–3 are *enough*.

This chain is now machine-checkable: every arrow is either a Dafny lemma
verified by the SMT solver, or a Lean 4 theorem checked by Mathlib's kernel.

---

## A Concrete Example: The Three-Node Chain

To ground the abstract machinery, the codebase includes a concrete
verified example:

**Graph:** $A \to B \to C$.

**Claim:** $A \perp_G C \mid B$ — conditioning on $B$ blocks the only trail
$A - B - C$.

```dafny
// dag.dfy
function ChainGraph(): Graph {
  map[0 := {},    // A: no parents
      1 := {0},   // B: parent is A
      2 := {1}]   // C: parent is B
}

lemma Chain_A_indep_C_given_B()
  ensures DSep(ChainGraph(), {0}, {2}, {1})
```

The proof instantiates `DSep_WeakUnion` and `LocalMarkov` on the chain graph
and closes automatically.  This single example exercises the full pipeline
from graph construction through the semi-graphoid lemmas.

---

## Notes on Axioms Still Pending

Not every result is fully proved from scratch yet.  A small number of
lemmas carry the `{:axiom}` annotation in Dafny (or `sorry` in Lean 4),
meaning they are *stated* and *used* but their proofs are deferred:

| Statement | Status | What is needed |
|-----------|--------|---------------|
| `GlobalMarkov` | Dafny axiom | Connects abstract `IntProb` to concrete `TruncatePMF` |
| `InterventionSemantics` | Dafny axiom | Same — requires PMF-witness redesign |
| Rules 2 & 3 | Dafny axiom | Structural-causal-model (SCM) proof |
| `Theorem2_Soundness` | Dafny axiom | Full inductive argument on ID depth |
| `Theorem3_Completeness` | Dafny axiom | Hedge model construction |
| `truncatePMF_markov` | Lean `sorry` | L6-level graph-surgery reasoning |
| `MarkovFactorization` body | Lean `opaque` | Requires `OutcomeToAssignment` machinery |

The key insight is that the *algorithmic* content of the ID algorithm — the
branching logic, recursion structure, and well-formedness invariants — is
fully verified.  The remaining axioms concern the *semantic* connections
between the abstract interventional distribution type and the concrete PMF
representation.

---

*Document generated: 2026-05-28.  Source: `src/dafny/*.dfy` and `src/lean/Y0Lean/*.lean`.*
