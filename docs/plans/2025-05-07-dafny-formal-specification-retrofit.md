# Dafny Formal Specification Retrofit

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retrofit `graph.py` (ADMG) and `dsl.py` (probability DSL) so that their operations are formally specified by the Dafny code in `src/dafny/`.

**Architecture:** Dafny serves as a *verified oracle* — its specs define the ground truth for probability axioms (`probability.dfy`), DAG structure + d-separation + semi-graphoid axioms (`dag.dfy`), and Pearl's do-calculus rules (`do_calculus.dfy`). Rather than manually translating each Dafny lemma into Python tests, we use an **auto-generated conformance test suite** that parses the Dafny `.dfy` files and emits a pytest module. The implementation work then follows a strict Red→Green→Refactor loop: run the generated tests, observe failures, implement the minimal Python code to make them pass.

**Tech Stack:** Python 3.11+, pytest, networkx, y0 DSL

## Verified Oracle Approach

```
┌──────────────────┐      parse       ┌────────────────────────────┐
│  src/dafny/*.dfy │ ──────────────►  │  scripts/generate_dafny_   │
│  (verified specs)│                  │  conformance_tests.py      │
└──────────────────┘                  └────────────┬───────────────┘
                                                   │ emit
                                                   ▼
                                      ┌────────────────────────────┐
                                      │  tests/test_dafny_         │
                                      │  correspondence.py         │
                                      │  (auto-generated, ≈43     │
                                      │   test methods)            │
                                      └────────────┬───────────────┘
                                                   │ pytest
                                                   ▼
                                      ┌────────────────────────────┐
                                      │  src/y0/graph.py           │
                                      │  src/y0/dsl.py             │
                                      │  src/y0/algorithm/         │
                                      │    do_calculus.py           │
                                      │  src/y0/mutate/chain.py    │
                                      └────────────────────────────┘
```

### Key Files

| File | Role |
|------|------|
| `scripts/generate_dafny_conformance_tests.py` | Generator — parses `.dfy`, emits test module |
| `tests/test_dafny_correspondence.py` | **Auto-generated** — do not edit manually |
| `src/dafny/dag.dfy` | Dafny spec: DAG, surgery, ancestry, d-separation, semi-graphoid, Local Markov |
| `src/dafny/do_calculus.dfy` | Dafny spec: Rules 1–3, backdoor, frontdoor criteria |
| `src/dafny/probability.dfy` | Dafny spec: Kolmogorov axioms, chain rule, Bayes' theorem |

The generator produces two kinds of tests from `probability.dfy`:
1. **Algebraic identity tests** — derived from axioms (`Axiom_Normalization` → `One()`, `EmptyEventZero` → `Zero()`, `Independent_Symmetric` → joint commutativity, `CondIndep_Symmetric` → conditional commutativity)
2. **Transformation tests** — derived from lemmas (`ChainRule` → `chain_expand()`, `BayesTheorem` → `bayes_expand()`)

### Regenerating Tests

When a Dafny spec changes, regenerate:

```bash
python scripts/generate_dafny_conformance_tests.py
```

This overwrites `tests/test_dafny_correspondence.py`. All test logic flows from the Dafny source.

## Dafny ↔ Python Correspondence Table

| Dafny construct | Python equivalent | File |
|---|---|---|
| `Graph = map<Node, set<Node>>` | `NxMixedGraph.directed: nx.DiGraph` | `src/y0/graph.py` |
| `Nodes(G)` | `graph.nodes()` | `src/y0/graph.py` |
| `Parents(G, v)` | `set(graph.directed.predecessors(v))` | `src/y0/graph.py` |
| `Children(G, u)` | `set(graph.directed.successors(u))` | `src/y0/graph.py` |
| `IsDAG(G)` | `graph.is_acyclic()` | `src/y0/graph.py` |
| `IsTopologicalSort(G, ord)` | `graph.topological_sort()` | `src/y0/graph.py` |
| `Ancestors(G, W)` | `graph.ancestors_inclusive(W)` | `src/y0/graph.py` |
| `Descendants(G, W)` | `graph.descendants_inclusive(W)` | `src/y0/graph.py` |
| `NonDescendants(G, v)` | `graph.non_descendants(v)` | `src/y0/graph.py` |
| `RemoveIncoming(G, X)` | `graph.remove_in_edges(X)` | `src/y0/graph.py` |
| `RemoveOutgoing(G, X)` | `graph.remove_out_edges(X)` | `src/y0/graph.py` |
| `DSep(G, Y, Z, W)` | `are_d_separated(graph, y, z, conditions=W)` | `src/y0/algorithm/conditional_independencies.py` |
| `Rule1` | `rule_1_of_do_calculus_applies(...)` | `src/y0/algorithm/do_calculus.py` |
| `Rule2` | `rule_2_of_do_calculus_applies(...)` | `src/y0/algorithm/do_calculus.py` |
| `Rule3` | `rule_3_of_do_calculus_applies(...)` | `src/y0/algorithm/do_calculus.py` |
| `BackdoorAdjustment` | `satisfies_backdoor(...)` | `src/y0/algorithm/do_calculus.py` |
| `FrontdoorCriterion` | `satisfies_frontdoor(...)` | `src/y0/algorithm/do_calculus.py` |
| `ChainRule` | `chain_expand()` | `src/y0/mutate/chain.py` |
| `BayesTheorem` | `bayes_expand()` | `src/y0/mutate/chain.py` |
| `CondIndep` | `DSeparationJudgement` | `src/y0/struct.py` |
| `Axiom_Normalization` | `One() * expr == expr` (algebraic) | `src/y0/dsl.py` |
| `EmptyEventZero` | `Zero() * expr == Zero()` (algebraic) | `src/y0/dsl.py` |
| `Independent_Symmetric` | `P(A, B) == P(B, A)` (algebraic) | `src/y0/dsl.py` |
| `CondIndep_Symmetric` | `P(A & B \| C) == P(B & A \| C)` (algebraic) | `src/y0/dsl.py` |

---

## Phase 0: Generate Conformance Test Suite

### Task 0.1: Run the generator to produce `tests/test_dafny_correspondence.py`

**Status:** `[x]` done

```bash
python scripts/generate_dafny_conformance_tests.py
```

The generator parses all three `.dfy` files and emits 8 test classes covering 43 test methods:

| Test class | Dafny section | # tests |
|---|---|---|
| `TestSurgeryLemmas` | dag.dfy §4 — Graph Surgery | 7 |
| `TestAncestryLemmas` | dag.dfy §2–3 — Acyclicity & Ancestry | 8 |
| `TestDSeparation` | dag.dfy §6 — D-Separation | 3 |
| `TestSemiGraphoidAxioms` | dag.dfy §7 — Semi-Graphoid | 3 |
| `TestLocalMarkov` | dag.dfy §8 — Local Markov Property | 1 |
| `TestDoCalculusRules` | do_calculus.dfy §4–7 — Rules 1–3, Backdoor, Frontdoor | 8 |
| `TestProbabilityAxioms` | probability.dfy — Chain Rule, Bayes | 3 |
| `TestAlgebraicIdentities` | probability.dfy §§2–5, 9 — Kolmogorov algebraic consequences | 8 |

The `TestAlgebraicIdentities` class tests symbolic consequences of the Kolmogorov axioms:
- `One()` as multiplicative identity (from `Axiom_Normalization`)
- `Zero()` as multiplicative absorber (from `EmptyEventZero`)
- Joint distribution commutativity (from `Independent_Symmetric`)
- Conditional distribution commutativity (from `CondIndep_Symmetric`)

### Task 0.2: Run tests, triage failures

**Status:** `[x]` done

```bash
pytest tests/test_dafny_correspondence.py -v 2>&1 | tail -40
```

**Expected triage:**

| Category | Expected result | Reason |
|---|---|---|
| Surgery tests (7) | PASS | Existing `remove_in_edges` / `remove_out_edges` |
| Ancestry tests (6 of 8) | PASS | Existing `ancestors_inclusive` / `descendants_inclusive` / `topological_sort` |
| `test_chain_is_dag` | **FAIL** | `NxMixedGraph.is_acyclic()` not yet implemented |
| `test_non_descendants*` (2) | **FAIL** | `NxMixedGraph.non_descendants()` not yet implemented |
| D-Separation tests (3) | PASS | Existing `are_d_separated` |
| Semi-Graphoid tests (3) | PASS | Existing `are_d_separated` |
| Local Markov test (1) | **FAIL** | Depends on `non_descendants()` |
| Rule 1 tests (2) | **FAIL** | `rule_1_of_do_calculus_applies` not yet implemented |
| Rule 2 test (1) | PASS | Existing `rule_2_of_do_calculus_applies` |
| Rule 3 test (1) | **FAIL** | `rule_3_of_do_calculus_applies` not yet implemented |
| Backdoor tests (2) | **FAIL** | `satisfies_backdoor` not yet implemented |
| Frontdoor tests (2) | **FAIL** | `satisfies_frontdoor` not yet implemented |
| Probability tests (3) | PASS | Existing `chain_expand` / `bayes_expand` |
| Algebraic identity tests (8) | PASS | Existing `One`, `Zero`, `Distribution` |

**Expected: ~27 PASS, ~11 FAIL**

---

## Phase 1: Graph Predicates (make ancestry/DAG tests green)

### Task 1.1: Add `NxMixedGraph.is_acyclic()`

**Status:** `[x]` done

**Dafny predicate:** `IsDAG(G)` (dag.dfy)
**Fixes test:** `TestAncestryLemmas::test_chain_is_dag`

Add to `src/y0/graph.py`:

```python
def is_acyclic(self) -> bool:
    """Check if the directed component is acyclic. Ref: dag.dfy IsDAG."""
    return nx.is_directed_acyclic_graph(self.directed)
```

**Verify:** `pytest tests/test_dafny_correspondence.py::TestAncestryLemmas::test_chain_is_dag -v`

### Task 1.2: Add `NxMixedGraph.non_descendants()`

**Status:** `[x]` done

**Dafny function:** `NonDescendants(G, v) = Nodes(G) - Descendants(G, {v})` (dag.dfy)
**Fixes tests:** `test_non_descendants`, `test_non_descendants_of_source`, `test_local_markov_chain`

Add to `src/y0/graph.py`:

```python
def non_descendants(self, node: Variable) -> set[Variable]:
    """Get non-descendants of a node. Ref: dag.dfy NonDescendants."""
    return set(self.nodes()) - self.descendants_inclusive(node)
```

**Verify:** `pytest tests/test_dafny_correspondence.py -k "non_descendants or local_markov" -v`

**Commit after both:**
```bash
git add src/y0/graph.py
git commit -m "feat: add is_acyclic() and non_descendants() per dag.dfy IsDAG/NonDescendants"
```

---

## Phase 2: Do-Calculus Rules

Implement the missing Rules 1 and 3 plus backdoor/frontdoor criteria. All tests already exist in the auto-generated suite.

### Task 2.1: `rule_1_of_do_calculus_applies` — Rule 1 Insertion/Deletion of Observations

**Status:** `[x]` done

**Dafny lemma:** `Rule1_InsertDeleteObservation` (do_calculus.dfy)
**Fixes tests:** `test_rule_1_does_not_apply_chain`, `test_rule_1_applies_isolated_node`

Add to `src/y0/algorithm/do_calculus.py`:

```python
def rule_1_of_do_calculus_applies(
    graph: NxMixedGraph,
    *,
    treatments: set[Variable],
    outcomes: set[Variable],
    conditions: set[Variable],
    observation: Variable,
) -> bool:
    """Check if Rule 1 of the Do-Calculus applies.

    Condition: (Y ⊥ Z | X, W) in G_{X̄}
    Ref: do_calculus.dfy Rule1_InsertDeleteObservation
    """
    mutilated = graph.remove_in_edges(treatments)
    return all(
        are_d_separated(mutilated, outcome, observation, conditions=treatments | conditions)
        for outcome in outcomes
    )
```

**Verify:** `pytest tests/test_dafny_correspondence.py -k rule_1 -v`

### Task 2.2: `rule_3_of_do_calculus_applies` — Rule 3 Insertion/Deletion of Actions

**Status:** `[x]` done

**Dafny lemma:** `Rule3_InsertDeleteAction` (do_calculus.dfy)
**Fixes test:** `test_rule_3_isolated_action`

Add to `src/y0/algorithm/do_calculus.py`:

```python
def rule_3_of_do_calculus_applies(
    graph: NxMixedGraph,
    *,
    treatments: set[Variable],
    outcomes: set[Variable],
    conditions: set[Variable],
    action: Variable,
) -> bool:
    """Check if Rule 3 of the Do-Calculus applies.

    Let Z̄(W) = Z \ An_{G_{X̄}}(W).
    Condition: (Y ⊥ Z | X, W) in G_{X̄, Z̄(W)_bar}
    Ref: do_calculus.dfy Rule3_InsertDeleteAction
    """
    gx = graph.remove_in_edges(treatments)
    ancestors_of_w = gx.ancestors_inclusive(conditions) if conditions else set()
    z_not_anc = {action} - ancestors_of_w
    gxz = gx.remove_in_edges(z_not_anc)
    return all(
        are_d_separated(gxz, outcome, action, conditions=treatments | conditions)
        for outcome in outcomes
    )
```

**Verify:** `pytest tests/test_dafny_correspondence.py -k rule_3 -v`

### Task 2.3: `satisfies_backdoor` — Backdoor Criterion

**Status:** `[x]` done

**Dafny lemma:** `BackdoorAdjustment` (do_calculus.dfy)
**Fixes tests:** `test_backdoor_simple`, `test_backdoor_fails_descendant`

> **Spec change:** The original Dafny spec used `DSep(RemoveIncoming(G, X), Y, X, Z)`, which only worked accidentally because `TrailBlocked` has a `|trail| <= 1` short-circuit that silently ignores direct single-step trails. This was changed to `DSep(RemoveOutgoing(G, X), Y, X, Z)` — physically cutting causal edges from X — which is the standard textbook formulation and maps directly to the Python implementation without relying on that quirk.

Implemented in `src/y0/algorithm/do_calculus.py`:

```python
def satisfies_backdoor(
    graph: NxMixedGraph,
    *,
    outcomes: set[Variable],
    treatments: set[Variable],
    adjustment: set[Variable],
) -> bool:
    """Check if adjustment set Z satisfies the backdoor criterion.

    (i) No z in Z is a descendant of any x in X (except x itself).
    (ii) Z d-separates Y from X in G_{X̲}  (X's outgoing edges removed).
    Ref: do_calculus.dfy BackdoorAdjustment
    """
    descendants_of_x = graph.descendants_inclusive(treatments)
    if adjustment & (descendants_of_x - treatments):
        return False
    g_xout = graph.remove_out_edges(treatments)
    return all(
        are_d_separated(g_xout, outcome, treatment, conditions=adjustment)
        for outcome in outcomes
        for treatment in treatments
    )
```

### Task 2.4: `satisfies_frontdoor` — Frontdoor Criterion

**Status:** `[-]` in progress

**Dafny lemma:** `FrontdoorCriterion` (do_calculus.dfy)
**Fixes tests:** `test_frontdoor_classic`, `test_frontdoor_fails_no_mediator`

Add to `src/y0/algorithm/do_calculus.py`:

```python
def satisfies_frontdoor(
    graph: NxMixedGraph,
    *,
    outcomes: set[Variable],
    treatments: set[Variable],
    mediators: set[Variable],
) -> bool:
    """Check if mediator set M satisfies the frontdoor criterion.

    Condition 1: DSep(RemoveIncoming(G, X), M, X, {})
    Condition 2: DSep(RemoveIncoming(RemoveOutgoing(G, X), M), Y, X, M)
    Ref: do_calculus.dfy FrontdoorCriterion
    """
    gx = graph.remove_in_edges(treatments)
    cond1 = all(
        are_d_separated(gx, mediator, treatment, conditions=set())
        for mediator in mediators
        for treatment in treatments
    )
    if not cond1:
        return False
    gx_out = graph.remove_out_edges(treatments)
    gx_out_m = gx_out.remove_in_edges(mediators)
    return all(
        are_d_separated(gx_out_m, outcome, treatment, conditions=mediators)
        for outcome in outcomes
        for treatment in treatments
    )
```

**Verify:** `pytest tests/test_dafny_correspondence.py -k frontdoor -v`

**Commit after Phase 2:**
```bash
git add src/y0/algorithm/do_calculus.py
git commit -m "feat: implement Rules 1, 3, backdoor, frontdoor per do_calculus.dfy"
```

---

## Phase 3: Final Verification

### Task 3.1: Full conformance suite — all green

**Status:** `[ ]` blocked on Task 2.4

```bash
pytest tests/test_dafny_correspondence.py -v
```

Expected: **43/43 PASS**

### Task 3.2: Existing test suite — no regressions

**Status:** `[ ]` not started

```bash
pytest tests/ -v --tb=short
```

Expected: No new failures.

### Task 3.3: Update this plan — mark all tasks complete

**Status:** `[ ]` not started

---

## Phase 4: Numerical Kolmogorov Axiom Verification (Future Work)

> **Status:** Not yet planned for this branch. Added for completeness.

The symbolic conformance tests (Phase 0 `TestAlgebraicIdentities`) verify *algebraic consequences* of the Kolmogorov axioms — commutativity, identity elements, absorbing elements. However, three core axioms are inherently *numerical* and cannot be tested symbolically:

| Dafny axiom | Statement | Why symbolic DSL can't test it |
|---|---|---|
| `Axiom_NonNegativity` | P(A) ≥ 0 | DSL doesn't evaluate to numbers |
| `Axiom_Normalization` | P(Ω) = 1 | DSL doesn't know about sample spaces |
| `Axiom_Additivity` | P(A ∪ B) = P(A) + P(B) when A ∩ B = ∅ | DSL has no set-union operation |

### Approach: Concrete PMF Evaluation

Build a small evaluator that takes a y0 `Expression` and a concrete joint distribution (e.g., a pandas DataFrame or numpy array), evaluates the expression numerically, and checks the axioms hold:

```python
# Sketch — not yet implemented
class ConcreteEvaluator:
    """Evaluate y0 Expressions against a concrete joint distribution."""
    def __init__(self, joint: pd.DataFrame):
        # joint is a DataFrame with columns for each variable and a 'prob' column
        self.joint = joint

    def evaluate(self, expr: Expression) -> float:
        """Recursively evaluate a symbolic expression to a number."""
        ...

# Tests would look like:
def test_axiom_nonnegativity(evaluator, event):
    assert evaluator.evaluate(P(event)) >= 0.0

def test_axiom_normalization(evaluator, all_vars):
    assert abs(evaluator.evaluate(Sum[all_vars](P(*all_vars))) - 1.0) < 1e-10

def test_axiom_additivity(evaluator, A_event, B_event):
    # given A ∩ B = ∅
    p_union = evaluator.evaluate(P(A_event) + P(B_event))
    p_a = evaluator.evaluate(P(A_event))
    p_b = evaluator.evaluate(P(B_event))
    assert abs(p_union - p_a - p_b) < 1e-10
```

### Derived laws also testable numerically

- `ComplementRule`: P(Aᶜ) = 1 − P(A)
- `Monotonicity`: A ⊆ B ⟹ P(A) ≤ P(B)
- `ProbAtMostOne`: P(A) ≤ 1
- `InclusionExclusion`: P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
- `TotalProbability`: P(A) = P(A|B₁)P(B₁) + P(A|B₂)P(B₂)

### Prerequisites

- A concrete expression evaluator (may leverage `y0.simulation` or build a new `y0.evaluate` module)
- Parametric test fixtures generating random valid PMFs
- Integration with the generator script to emit parametrized pytest tests

---

## Progress Tracker

| Phase | Task | Description | Status |
|---|---|---|---|
| 0 | 0.1 | Run generator → emit `tests/test_dafny_correspondence.py` | `[x]` |
| 0 | 0.2 | Run tests, triage pass/fail | `[x]` |
| 1 | 1.1 | Add `NxMixedGraph.is_acyclic()` | `[x]` |
| 1 | 1.2 | Add `NxMixedGraph.non_descendants()` | `[x]` |
| 2 | 2.1 | Implement `rule_1_of_do_calculus_applies` | `[x]` |
| 2 | 2.2 | Implement `rule_3_of_do_calculus_applies` | `[x]` |
| 2 | 2.3 | Implement `satisfies_backdoor` + fix Dafny spec | `[x]` |
| 2 | 2.4 | Implement `satisfies_frontdoor` | `[-]` |
| 3 | 3.1 | Full conformance suite — all green | `[ ]` |
| 3 | 3.2 | Existing test suite — no regressions | `[ ]` |
| 3 | 3.3 | Update plan — mark complete | `[ ]` |
| 4 | — | Numerical Kolmogorov axiom verification | *future work* |
