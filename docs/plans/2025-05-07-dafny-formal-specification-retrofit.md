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
                                      │  (auto-generated, ≈30     │
                                      │   test methods)            │
                                      └────────────┬───────────────┘
                                                   │ pytest
                                                   ▼
                                      ┌────────────────────────────┐
                                      │  src/y0/graph.py           │
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

---

## Phase 0: Generate Conformance Test Suite

### Task 0.1: Run the generator to produce `tests/test_dafny_correspondence.py`

**Status:** `[x]` done

```bash
python scripts/generate_dafny_conformance_tests.py
```

The generator parses all three `.dfy` files and emits 8 test classes covering 30 test methods:

| Test class | Dafny section | # tests |
|---|---|---|
| `TestSurgeryLemmas` | dag.dfy §4 — Graph Surgery | 7 |
| `TestAncestryLemmas` | dag.dfy §2–3 — Acyclicity & Ancestry | 8 |
| `TestDSeparation` | dag.dfy §6 — D-Separation | 3 |
| `TestSemiGraphoidAxioms` | dag.dfy §7 — Semi-Graphoid | 3 |
| `TestLocalMarkov` | dag.dfy §8 — Local Markov Property | 1 |
| `TestDoCalculusRules` | do_calculus.dfy §4–7 — Rules 1–3, Backdoor, Frontdoor | 8 |
| `TestProbabilityAxioms` | probability.dfy — Chain Rule, Bayes | 3 |

### Task 0.2: Run tests, triage failures

**Status:** `[ ]` not started

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

**Expected: ~19 PASS, ~11 FAIL**

---

## Phase 1: Graph Predicates (make ancestry/DAG tests green)

### Task 1.1: Add `NxMixedGraph.is_acyclic()`

**Status:** `[ ]` not started

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

**Status:** `[ ]` not started

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

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dafny_correspondence.py::TestAncestryLemmas -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_dafny_correspondence.py src/y0/graph.py
git commit -m "feat: add NxMixedGraph.is_acyclic() per dag.dfy IsDAG"
```

---

## Phase 2: Do-Calculus Rules

Implement the missing Rules 1 and 3 plus backdoor/frontdoor criteria. All tests already exist in the auto-generated suite.

### Task 2.1: `rule_1_of_do_calculus_applies` — Rule 1 Insertion/Deletion of Observations

**Status:** `[ ]` not started

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
    r"""Check if Rule 1 of the Do-Calculus applies.

    Condition: (Y ⊥ Z | X, W) in G_{X̄}

    If true, the observation Z can be inserted/deleted:
    P(Y | do(X), Z, W) = P(Y | do(X), W)

    Ref: do_calculus.dfy Rule1_InsertDeleteObservation

    :param graph: The causal graph
    :param treatments: The do-variables X
    :param outcomes: The outcome variables Y
    :param conditions: The conditioning variables W
    :param observation: The observation Z to test for insertion/deletion
    :returns: True if Rule 1 applies
    """
    mutilated = graph.remove_in_edges(treatments)
    return all(
        are_d_separated(
            mutilated, outcome, observation, conditions=treatments | conditions
        )
        for outcome in outcomes
    )
```

**Step 4: Run tests**

Run: `pytest tests/test_dafny_correspondence.py::TestDoCalculusRules -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/y0/algorithm/do_calculus.py tests/test_dafny_correspondence.py
git commit -m "feat: implement Rule 1 of do-calculus per do_calculus.dfy Rule1_InsertDeleteObservation"
```

---

### Task 4.2: Rule 3 — Insertion/Deletion of Actions

**Status:** `[ ]` not started

**Dafny lemma:** `Rule3_InsertDeleteAction` (do_calculus.dfy lines 118–130)

**Files:**
- Modify: `tests/test_dafny_correspondence.py`
- Modify: `src/y0/algorithm/do_calculus.py`

**Step 1: Write the failing test**

```python
from y0.algorithm.do_calculus import rule_3_of_do_calculus_applies


class TestDoCalculusRule3(unittest.TestCase):
    """Tests for Rule 3 of do-calculus. Ref: do_calculus.dfy Rule3_InsertDeleteAction."""

    def test_rule_3_applies_when_action_has_no_effect(self):
        """Rule 3: do(Z) can be deleted when Z has no causal effect on Y.

        Graph: X->Y, Z (isolated node). do(X), do(Z).
        G_{X̄} = same (X is source). Z̄(W) = Z (Z not ancestor of W={}).
        G_{X̄, Z̄} = remove incoming to Z in G_{X̄} = same (Z is source).
        (Y ⊥ Z | X) in that graph => True (Z is isolated).
        """
        X, Y, Z = Variable("X"), Variable("Y"), Variable("Z")
        graph = NxMixedGraph.from_edges(nodes=[Z], directed=[(X, Y)])
        self.assertTrue(
            rule_3_of_do_calculus_applies(
                graph,
                treatments={X},
                outcomes={Y},
                conditions=set(),
                action=Z,
            )
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dafny_correspondence.py::TestDoCalculusRule3 -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
def rule_3_of_do_calculus_applies(
    graph: NxMixedGraph,
    *,
    treatments: set[Variable],
    outcomes: set[Variable],
    conditions: set[Variable],
    action: Variable,
) -> bool:
    r"""Check if Rule 3 of the Do-Calculus applies.

    Let Z̄(W) = Z \ An_{G_{X̄}}(W).
    Condition: (Y ⊥ Z | X, W) in G_{X̄, Z̄(W)_bar}

    If true, the action do(Z) can be inserted/deleted:
    P(Y | do(X), do(Z), W) = P(Y | do(X), W)

    Ref: do_calculus.dfy Rule3_InsertDeleteAction

    :param graph: The causal graph
    :param treatments: The do-variables X
    :param outcomes: The outcome variables Y
    :param conditions: The conditioning variables W
    :param action: The action variable Z to test for insertion/deletion
    :returns: True if Rule 3 applies
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

**Step 4: Run tests**

Run: `pytest tests/test_dafny_correspondence.py::TestDoCalculusRule3 -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/y0/algorithm/do_calculus.py tests/test_dafny_correspondence.py
git commit -m "feat: implement Rule 3 of do-calculus per do_calculus.dfy Rule3_InsertDeleteAction"
```

---

## Phase 5: Backdoor and Frontdoor Criteria

### Task 5.1: `satisfies_backdoor` predicate

**Status:** `[ ]` not started

**Dafny lemma:** `BackdoorAdjustment` (do_calculus.dfy lines 140–153)

**Files:**
- Modify: `tests/test_dafny_correspondence.py`
- Modify: `src/y0/algorithm/do_calculus.py`

**Step 1: Write the failing test**

```python
from y0.algorithm.do_calculus import satisfies_backdoor


class TestBackdoorCriterion(unittest.TestCase):
    """Tests for Backdoor Criterion. Ref: do_calculus.dfy §6."""

    def test_backdoor_simple(self):
        """Z satisfies backdoor for X->Y when Z blocks backdoor paths.

        Graph: Z->X->Y, Z->Y. Z is a valid backdoor set.
        (i) Z is not a descendant of X.
        (ii) Z d-separates Y from X in G_{X̄}.
        """
        X, Y, Z = Variable("X"), Variable("Y"), Variable("Z")
        graph = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        self.assertTrue(satisfies_backdoor(graph, outcomes={Y}, treatments={X}, adjustment={Z}))

    def test_backdoor_fails_for_descendant(self):
        """M does NOT satisfy backdoor for X->M->Y because M is a descendant of X."""
        X, M, Y = Variable("X"), Variable("M"), Variable("Y")
        graph = NxMixedGraph.from_edges(directed=[(X, M), (M, Y)])
        self.assertFalse(satisfies_backdoor(graph, outcomes={Y}, treatments={X}, adjustment={M}))
```

**Step 2: Run test to verify it fails**

Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
def satisfies_backdoor(
    graph: NxMixedGraph,
    *,
    outcomes: set[Variable],
    treatments: set[Variable],
    adjustment: set[Variable],
) -> bool:
    r"""Check if adjustment set Z satisfies the backdoor criterion for (X -> Y).

    (i) No z in Z is a descendant of any x in X (except x itself).
    (ii) Z d-separates Y from X in G_{X̄}.

    Ref: do_calculus.dfy BackdoorAdjustment

    :param graph: The causal graph
    :param outcomes: Outcome variables Y
    :param treatments: Treatment variables X
    :param adjustment: Adjustment set Z
    :returns: True if the backdoor criterion is satisfied
    """
    # (i) No descendant of X in Z
    descendants_of_x = graph.descendants_inclusive(treatments)
    if adjustment & (descendants_of_x - treatments):
        return False
    # (ii) Z d-separates Y from X in G_{X̄}
    mutilated = graph.remove_in_edges(treatments)
    return all(
        are_d_separated(mutilated, outcome, treatment, conditions=adjustment)
        for outcome in outcomes
        for treatment in treatments
    )
```

**Step 4: Run tests**

Run: `pytest tests/test_dafny_correspondence.py::TestBackdoorCriterion -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/y0/algorithm/do_calculus.py tests/test_dafny_correspondence.py
git commit -m "feat: add satisfies_backdoor() per do_calculus.dfy BackdoorAdjustment"
```

---

### Task 5.2: `satisfies_frontdoor` predicate

**Status:** `[ ]` not started

**Dafny lemma:** `FrontdoorCriterion` (do_calculus.dfy lines 160–175)

**Files:**
- Modify: `tests/test_dafny_correspondence.py`
- Modify: `src/y0/algorithm/do_calculus.py`

**Step 1: Write the failing test**

```python
from y0.algorithm.do_calculus import satisfies_frontdoor


class TestFrontdoorCriterion(unittest.TestCase):
    """Tests for Frontdoor Criterion. Ref: do_calculus.dfy §7."""

    def test_frontdoor_classic(self):
        """M satisfies frontdoor for X->Y with confounding X<->Y.

        Graph: X->M->Y, X<->Y (bidirected). M is the frontdoor set.
        """
        X, M, Y = Variable("X"), Variable("M"), Variable("Y")
        graph = NxMixedGraph.from_edges(
            directed=[(X, M), (M, Y)],
            undirected=[(X, Y)],
        )
        self.assertTrue(
            satisfies_frontdoor(graph, outcomes={Y}, treatments={X}, mediators={M})
        )

    def test_frontdoor_fails_without_mediator(self):
        """Direct path X->Y with confounding: no valid frontdoor set."""
        X, Y = Variable("X"), Variable("Y")
        graph = NxMixedGraph.from_edges(
            directed=[(X, Y)],
            undirected=[(X, Y)],
        )
        # Empty set doesn't satisfy frontdoor
        self.assertFalse(
            satisfies_frontdoor(graph, outcomes={Y}, treatments={X}, mediators=set())
        )
```

**Step 2: Run test to verify it fails**

Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
def satisfies_frontdoor(
    graph: NxMixedGraph,
    *,
    outcomes: set[Variable],
    treatments: set[Variable],
    mediators: set[Variable],
) -> bool:
    r"""Check if mediator set M satisfies the frontdoor criterion for (X -> Y).

    (i) M intercepts all directed paths from X to Y.
    (ii) No unblocked back-door paths from X to M.
    (iii) All back-door paths from M to Y are blocked by X.

    Ref: do_calculus.dfy FrontdoorCriterion

    :param graph: The causal graph
    :param outcomes: Outcome variables Y
    :param treatments: Treatment variables X
    :param mediators: Mediator set M
    :returns: True if the frontdoor criterion is satisfied
    """
    # Condition from Dafny: DSep(RemoveIncoming(G, X), M, X, {})
    gx = graph.remove_in_edges(treatments)
    cond1 = all(
        are_d_separated(gx, mediator, treatment, conditions=set())
        for mediator in mediators
        for treatment in treatments
    )
    if not cond1:
        return False
    # Condition from Dafny: DSep(RemoveIncoming(RemoveOutgoing(G, X), M), Y, X, M)
    gx_out = graph.remove_out_edges(treatments)
    gx_out_m = gx_out.remove_in_edges(mediators)
    cond2 = all(
        are_d_separated(gx_out_m, outcome, treatment, conditions=mediators)
        for outcome in outcomes
        for treatment in treatments
    )
    return cond2
```

**Step 4: Run tests**

Run: `pytest tests/test_dafny_correspondence.py::TestFrontdoorCriterion -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/y0/algorithm/do_calculus.py tests/test_dafny_correspondence.py
git commit -m "feat: add satisfies_frontdoor() per do_calculus.dfy FrontdoorCriterion"
```

---

## Phase 6: Probability DSL Alignment

### Task 6.1: Verify `chain_expand` matches Dafny `ChainRule`

**Status:** `[ ]` not started

**Dafny lemma:** `ChainRule` (probability.dfy lines 126–133)

**Files:**
- Modify: `tests/test_dafny_correspondence.py`

**Step 1: Write the tests**

```python
from y0.dsl import P, A, B, C
from y0.mutate.chain import chain_expand


class TestProbabilityAxioms(unittest.TestCase):
    """Tests aligning y0 DSL with probability.dfy axioms."""

    def test_chain_rule_two_vars(self):
        """P(A, B) = P(A | B) * P(B). Ref: probability.dfy ChainRule."""
        result = chain_expand(P(A, B))
        expected = P(A | B) * P(B)
        self.assertEqual(expected, result)

    def test_chain_rule_three_vars(self):
        """P(A, B, C) = P(A | B, C) * P(B | C) * P(C). Ref: probability.dfy ChainRule (iterated)."""
        result = chain_expand(P(A, B, C))
        expected = P(A | B, C) * P(B | C) * P(C)
        self.assertEqual(expected, result)
```

**Step 2: Run tests**

Run: `pytest tests/test_dafny_correspondence.py::TestProbabilityAxioms -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_dafny_correspondence.py
git commit -m "test: verify chain_expand matches probability.dfy ChainRule"
```

---

### Task 6.2: Verify `bayes_expand` matches Dafny `BayesTheorem`

**Status:** `[ ]` not started

**Dafny lemma:** `BayesTheorem` (probability.dfy lines 138–153)

**Files:**
- Modify: `tests/test_dafny_correspondence.py`

**Step 1: Write the tests**

```python
from y0.mutate.chain import bayes_expand
from y0.dsl import Fraction


class TestBayesTheorem(unittest.TestCase):
    """Tests aligning bayes_expand with probability.dfy BayesTheorem."""

    def test_bayes_two_vars(self):
        """P(A | B) = P(B | A) * P(A) / P(B). Ref: probability.dfy BayesTheorem."""
        result = bayes_expand(P(A | B))
        expected = Fraction(P(B | A) * P(A), P(B))
        self.assertEqual(expected, result)
```

**Step 2: Run tests**

Run: `pytest tests/test_dafny_correspondence.py::TestBayesTheorem -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_dafny_correspondence.py
git commit -m "test: verify bayes_expand matches probability.dfy BayesTheorem"
```

---

## Progress Tracker

| Phase | Task | Description | Status |
|---|---|---|---|
| 1 | 1.1 | Surgery identity tests (`RemoveIncoming/Outgoing_Empty`) | `[ ]` |
| 1 | 1.2 | Surgery preservation tests (`NoParents`, `PreservesOthers`) | `[ ]` |
| 1 | 1.3 | Surgery on ADMG with bidirected edges | `[ ]` |
| 2 | 2.1 | Ancestry reflexivity and transitivity | `[ ]` |
| 2 | 2.2 | `is_acyclic` predicate (`IsDAG`) | `[ ]` |
| 2 | 2.3 | `non_descendants` method (`NonDescendants`) | `[ ]` |
| 3 | 3.1 | Chain graph d-separation | `[ ]` |
| 3 | 3.2 | Semi-graphoid: Symmetry | `[ ]` |
| 3 | 3.3 | Semi-graphoid: Decomposition, Weak Union, Contraction | `[ ]` |
| 3 | 3.4 | Local Markov Property | `[ ]` |
| 4 | 4.1 | Rule 1 — Insertion/Deletion of Observations | `[ ]` |
| 4 | 4.2 | Rule 3 — Insertion/Deletion of Actions | `[ ]` |
| 5 | 5.1 | `satisfies_backdoor` predicate | `[ ]` |
| 5 | 5.2 | `satisfies_frontdoor` predicate | `[ ]` |
| 6 | 6.1 | `chain_expand` ↔ `ChainRule` | `[ ]` |
| 6 | 6.2 | `bayes_expand` ↔ `BayesTheorem` | `[ ]` |
