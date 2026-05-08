#!/usr/bin/env python
"""Generate Python conformance tests from Dafny formal specifications.

Parses .dfy files to extract lemma signatures (requires/ensures), concrete
examples, and ghost function definitions. Maps them to y0 Python constructs
and emits a pytest test module that verifies the Python implementation
conforms to the Dafny spec.

Usage:
    python scripts/generate_dafny_conformance_tests.py

Output:
    tests/test_dafny_correspondence.py
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAFNY_DIR = ROOT / "src" / "dafny"
OUTPUT = ROOT / "tests" / "test_dafny_correspondence.py"


# ── Dafny AST (lightweight) ─────────────────────────────────────────


@dataclass
class DafnyLemma:
    """A parsed Dafny lemma."""

    name: str
    module: str
    params: list[tuple[str, str]]  # (name, type)
    requires: list[str]
    ensures: list[str]
    is_axiom: bool = False
    doc: str = ""
    line: int = 0


@dataclass
class DafnyFunction:
    """A parsed Dafny ghost function or concrete function."""

    name: str
    module: str
    params: list[tuple[str, str]]
    return_type: str
    body: str = ""
    is_ghost: bool = False
    line: int = 0


@dataclass
class DafnySpec:
    """All extracted declarations from one .dfy file."""

    path: Path
    module: str = ""
    lemmas: list[DafnyLemma] = field(default_factory=list)
    functions: list[DafnyFunction] = field(default_factory=list)


# ── Parser ───────────────────────────────────────────────────────────

_LEMMA_RE = re.compile(
    r"^\s*lemma\s+(\{[^}]*\}\s*)?(\w+)\s*\(",
    re.MULTILINE,
)
_FUNCTION_RE = re.compile(
    r"^\s*(ghost\s+)?function\s+(\{[^}]*\}\s*)?(\w+)\s*\(",
    re.MULTILINE,
)
_PREDICATE_RE = re.compile(
    r"^\s*(ghost\s+)?predicate\s+(\{[^}]*\}\s*)?(\w+)\s*\(",
    re.MULTILINE,
)
_MODULE_RE = re.compile(r"^\s*module\s+(\w+)", re.MULTILINE)
_REQUIRES_RE = re.compile(r"^\s*requires\s+(.*)", re.MULTILINE)
_ENSURES_RE = re.compile(r"^\s*ensures\s+(.*)", re.MULTILINE)


def _extract_block_after(text: str, start: int) -> str:
    """Extract text from start until the next blank line or next declaration."""
    lines = text[start:].split("\n")
    result = []
    for line in lines:
        result.append(line)
        if line.strip() == "" or line.strip().startswith("//"):
            continue
    return "\n".join(result)


def _extract_params(text: str, paren_start: int) -> tuple[list[tuple[str, str]], int]:
    """Extract parameter list from opening paren position."""
    depth = 0
    i = paren_start
    start = paren_start + 1
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                param_str = text[start:i].strip()
                params = _parse_params(param_str)
                return params, i + 1
        i += 1
    return [], len(text)


def _parse_params(param_str: str) -> list[tuple[str, str]]:
    """Parse 'G: Graph, X: set<Node>, x: Node' into list of tuples."""
    if not param_str.strip():
        return []
    params = []
    for part in param_str.split(","):
        part = part.strip()
        if ":" in part:
            name, typ = part.split(":", 1)
            params.append((name.strip(), typ.strip()))
        elif part:
            params.append((part, "unknown"))
    return params


def _collect_clauses(text: str, start: int) -> tuple[list[str], list[str], int]:
    """Collect requires and ensures clauses starting from a position."""
    requires = []
    ensures = []
    lines = text[start:].split("\n")
    end = start
    for line in lines:
        end += len(line) + 1
        stripped = line.strip()
        if stripped.startswith("requires"):
            clause = stripped[len("requires"):].strip()
            requires.append(clause)
        elif stripped.startswith("ensures"):
            clause = stripped[len("ensures"):].strip()
            ensures.append(clause)
        elif stripped.startswith("{") or stripped.startswith("//") or stripped == "":
            if stripped.startswith("{"):
                break
            continue
        elif requires or ensures:
            # continuation of a multi-line clause
            if ensures:
                ensures[-1] += " " + stripped
            elif requires:
                requires[-1] += " " + stripped
    return requires, ensures, end


def _get_doc_comment(text: str, pos: int) -> str:
    """Extract /// doc comments immediately preceding pos."""
    lines = text[:pos].rstrip().split("\n")
    doc_lines = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("///"):
            doc_lines.insert(0, stripped[3:].strip())
        elif stripped.startswith("//"):
            continue
        else:
            break
    return " ".join(doc_lines)


def parse_dafny_file(path: Path) -> DafnySpec:
    """Parse a .dfy file and extract lemmas and functions."""
    text = path.read_text()
    spec = DafnySpec(path=path)

    # Module name
    mod_match = _MODULE_RE.search(text)
    if mod_match:
        spec.module = mod_match.group(1)

    # Lemmas
    for match in _LEMMA_RE.finditer(text):
        attrs = match.group(1) or ""
        name = match.group(2)
        paren_pos = match.end() - 1
        params, after_params = _extract_params(text, paren_pos)
        requires, ensures, _ = _collect_clauses(text, after_params)
        doc = _get_doc_comment(text, match.start())
        line_no = text[: match.start()].count("\n") + 1

        lemma = DafnyLemma(
            name=name,
            module=spec.module,
            params=params,
            requires=requires,
            ensures=ensures,
            is_axiom="{:axiom}" in attrs,
            doc=doc,
            line=line_no,
        )
        spec.lemmas.append(lemma)

    # Functions and predicates
    for pattern in [_FUNCTION_RE, _PREDICATE_RE]:
        for match in pattern.finditer(text):
            ghost = bool(match.group(1))
            name = match.group(3)
            paren_pos = match.end() - 1
            params, _ = _extract_params(text, paren_pos)
            line_no = text[: match.start()].count("\n") + 1

            func = DafnyFunction(
                name=name,
                module=spec.module,
                params=params,
                return_type="",
                is_ghost=ghost,
                line=line_no,
            )
            spec.functions.append(func)

    return spec


# ── Test generation ──────────────────────────────────────────────────

# The chain graph A(0) -> B(1) -> C(2) from dag.dfy §9
CHAIN_GRAPH = """
        A = Variable("A")
        B = Variable("B")
        C = Variable("C")
        chain = NxMixedGraph.from_edges(directed=[(A, B), (B, C)])"""

# A four-node chain A->B->C->D for semi-graphoid tests
CHAIN4_GRAPH = """
        A = Variable("A")
        B = Variable("B")
        C = Variable("C")
        D = Variable("D")
        graph = NxMixedGraph.from_edges(
            directed=[(A, B), (B, C), (C, D)]
        )"""

# Maps from Dafny lemma names -> Python test code generators
# Each value is a function(lemma) -> (test_name, test_body, needs_impl)
# needs_impl is a list of (function_name, implementation_code)


def _gen_surgery_tests(lemmas: list[DafnyLemma]) -> list[tuple[str, str, list]]:
    """Generate tests for surgery lemmas from dag.dfy."""
    tests = []
    for lemma in lemmas:
        if lemma.name == "RemoveIncoming_Empty":
            tests.append((
                "test_remove_incoming_empty_is_identity",
                f'''    def test_remove_incoming_empty_is_identity(self):
        """RemoveIncoming(G, {{}}) == G.

        Ref: dag.dfy:{lemma.line} {lemma.name}
        """
        self.assertEqual(self.chain, self.chain.remove_in_edges(set()))''',
                [],
            ))
        elif lemma.name == "RemoveOutgoing_Empty":
            tests.append((
                "test_remove_outgoing_empty_is_identity",
                f'''    def test_remove_outgoing_empty_is_identity(self):
        """RemoveOutgoing(G, {{}}) == G.

        Ref: dag.dfy:{lemma.line} {lemma.name}
        """
        self.assertEqual(self.chain, self.chain.remove_out_edges(set()))''',
                [],
            ))
        elif lemma.name == "RemoveIncoming_NoParents":
            tests.append((
                "test_remove_incoming_no_parents_for_target",
                f'''    def test_remove_incoming_no_parents_for_target(self):
        """Nodes in X lose all parents after incoming surgery.

        Ref: dag.dfy:{lemma.line} {lemma.name}
        """
        mutilated = self.chain.remove_in_edges({{self.B}})
        self.assertEqual(set(), set(mutilated.directed.predecessors(self.B)))''',
                [],
            ))
        elif lemma.name == "RemoveIncoming_PreservesOthers":
            tests.append((
                "test_remove_incoming_preserves_others",
                f'''    def test_remove_incoming_preserves_others(self):
        """Nodes outside X keep their parents after incoming surgery.

        Ref: dag.dfy:{lemma.line} {lemma.name}
        """
        mutilated = self.chain.remove_in_edges({{self.B}})
        self.assertEqual({{self.B}}, set(mutilated.directed.predecessors(self.C)))''',
                [],
            ))

    # Dual tests for RemoveOutgoing (not explicit lemmas but follow from definition)
    tests.append((
        "test_remove_outgoing_removes_children",
        '''    def test_remove_outgoing_removes_children(self):
        """Nodes in X lose all children after outgoing surgery.

        Ref: dag.dfy RemoveOutgoing definition (dual of RemoveIncoming_NoParents)
        """
        mutilated = self.chain.remove_out_edges({self.B})
        self.assertEqual(set(), set(mutilated.directed.successors(self.B)))''',
        [],
    ))
    tests.append((
        "test_remove_outgoing_preserves_others",
        '''    def test_remove_outgoing_preserves_others(self):
        """Nodes outside X keep their children after outgoing surgery.

        Ref: dag.dfy RemoveOutgoing definition (dual of RemoveIncoming_PreservesOthers)
        """
        mutilated = self.chain.remove_out_edges({self.B})
        self.assertEqual({self.B}, set(mutilated.directed.successors(self.A)))''',
        [],
    ))

    # ADMG extension: bidirected edge removal
    tests.append((
        "test_remove_incoming_also_removes_bidirected",
        '''    def test_remove_incoming_also_removes_bidirected(self):
        """RemoveIncoming on ADMG also removes bidirected edges to X.

        Ref: ADMG extension of dag.dfy RemoveIncoming (Dafny models directed-only)
        """
        graph = NxMixedGraph.from_edges(
            directed=[(self.A, self.B), (self.B, self.C)],
            undirected=[(self.A, self.B)],
        )
        mutilated = graph.remove_in_edges({self.B})
        self.assertEqual(set(), set(mutilated.directed.predecessors(self.B)))
        self.assertEqual(set(), set(mutilated.undirected.neighbors(self.B)))''',
        [],
    ))

    return tests


def _gen_ancestry_tests(
    functions: list[DafnyFunction], lemmas: list[DafnyLemma]
) -> list[tuple[str, str, list]]:
    """Generate tests for ancestry functions and IsDAG from dag.dfy."""
    tests = []

    # Ancestor_Reflexive
    tests.append((
        "test_ancestor_reflexive",
        '''    def test_ancestor_reflexive(self):
        """Every node is its own ancestor.

        Ref: dag.dfy Ancestor_Reflexive
        """
        for node in [self.A, self.B, self.C]:
            self.assertIn(node, self.chain.ancestors_inclusive(node))''',
        [],
    ))
    tests.append((
        "test_ancestor_transitive",
        '''    def test_ancestor_transitive(self):
        """Ancestry is transitive: A ancestor of B, B ancestor of C => A ancestor of C.

        Ref: dag.dfy IsAncestor (reflexive-transitive closure)
        """
        self.assertIn(self.A, self.chain.ancestors_inclusive(self.C))''',
        [],
    ))
    tests.append((
        "test_descendant_reflexive",
        '''    def test_descendant_reflexive(self):
        """Every node is its own descendant.

        Ref: dag.dfy Descendants definition
        """
        for node in [self.A, self.B, self.C]:
            self.assertIn(node, self.chain.descendants_inclusive(node))''',
        [],
    ))
    tests.append((
        "test_descendant_transitive",
        '''    def test_descendant_transitive(self):
        """Descendancy is transitive.

        Ref: dag.dfy Descendants definition
        """
        self.assertIn(self.C, self.chain.descendants_inclusive(self.A))''',
        [],
    ))
    tests.append((
        "test_ancestors_of_chain_endpoint",
        '''    def test_ancestors_of_chain_endpoint(self):
        """Ancestors(G, {C}) = {A, B, C} in the chain graph.

        Ref: dag.dfy Ancestors ghost function
        """
        self.assertEqual(
            {self.A, self.B, self.C},
            self.chain.ancestors_inclusive(self.C),
        )''',
        [],
    ))
    tests.append((
        "test_descendants_of_chain_start",
        '''    def test_descendants_of_chain_start(self):
        """Descendants(G, {A}) = {A, B, C} in the chain graph.

        Ref: dag.dfy Descendants ghost function
        """
        self.assertEqual(
            {self.A, self.B, self.C},
            self.chain.descendants_inclusive(self.A),
        )''',
        [],
    ))

    # IsDAG
    tests.append((
        "test_chain_is_dag",
        '''    def test_chain_is_dag(self):
        """The chain graph is a DAG.

        Ref: dag.dfy IsDAG, ChainGraph_IsDAG
        """
        self.assertTrue(self.chain.is_acyclic())''',
        [("is_acyclic", '''    def is_acyclic(self) -> bool:
        """Check if the directed component is acyclic.

        Ref: dag.dfy IsDAG
        """
        return nx.is_directed_acyclic_graph(self.directed)''')],
    ))

    # Topological sort
    tests.append((
        "test_topological_sort_valid",
        '''    def test_topological_sort_valid(self):
        """A DAG admits a topological sort where parents precede children.

        Ref: dag.dfy IsTopologicalSort
        """
        order = self.chain.topological_sort()
        self.assertEqual(3, len(order))
        self.assertLess(order.index(self.A), order.index(self.B))
        self.assertLess(order.index(self.B), order.index(self.C))''',
        [],
    ))

    # NonDescendants
    tests.append((
        "test_non_descendants",
        '''    def test_non_descendants(self):
        """NonDescendants(G, B) = {A}.

        Ref: dag.dfy NonDescendants
        """
        result = self.chain.non_descendants(self.B)
        self.assertEqual({self.A}, result)''',
        [("non_descendants", '''    def non_descendants(self, node: Variable) -> set[Variable]:
        """Get non-descendants of a node.

        Ref: dag.dfy NonDescendants
        """
        return set(self.nodes()) - self.descendants_inclusive(node)''')],
    ))
    tests.append((
        "test_non_descendants_of_source",
        '''    def test_non_descendants_of_source(self):
        """NonDescendants(G, A) = {} since A is the root.

        Ref: dag.dfy NonDescendants
        """
        result = self.chain.non_descendants(self.A)
        self.assertEqual(set(), result)''',
        [],
    ))

    return tests


def _gen_dsep_tests(lemmas: list[DafnyLemma]) -> list[tuple[str, str, list]]:
    """Generate d-separation and semi-graphoid tests from dag.dfy."""
    tests = []

    # Chain d-sep example
    tests.append((
        "test_chain_a_indep_c_given_b",
        '''    def test_chain_a_indep_c_given_b(self):
        """A ⊥ C | {B} in chain A->B->C.

        Ref: dag.dfy Chain_A_indep_C_given_B
        """
        judgement = are_d_separated(self.chain, self.A, self.C, conditions=[self.B])
        self.assertTrue(judgement.separated)''',
        [],
    ))
    tests.append((
        "test_chain_a_not_indep_c_unconditional",
        '''    def test_chain_a_not_indep_c_unconditional(self):
        """A is NOT d-separated from C unconditionally in A->B->C.

        Ref: dag.dfy (negative case — no conditioning blocks the path)
        """
        judgement = are_d_separated(self.chain, self.A, self.C)
        self.assertFalse(judgement.separated)''',
        [],
    ))

    # Semi-graphoid axioms
    for lemma in lemmas:
        if lemma.name == "DSep_Symmetry":
            tests.append((
                "test_dsep_symmetry",
                f'''    def test_dsep_symmetry(self):
        """(Y ⊥ Z | W) => (Z ⊥ Y | W).

        Ref: dag.dfy:{lemma.line} {lemma.name}
        """
        j1 = are_d_separated(self.chain, self.A, self.C, conditions=[self.B])
        j2 = are_d_separated(self.chain, self.C, self.A, conditions=[self.B])
        self.assertEqual(j1.separated, j2.separated)''',
                [],
            ))

    return tests


def _gen_semigraphoid_tests(lemmas: list[DafnyLemma]) -> list[tuple[str, str, list]]:
    """Generate semi-graphoid axiom tests on a 4-node chain."""
    tests = []

    for lemma in lemmas:
        if lemma.name == "DSep_Decomposition":
            tests.append((
                "test_decomposition",
                f'''    def test_decomposition(self):
        """(Y ⊥ Z ∪ Z' | W) => (Y ⊥ Z | W).

        Chain A->B->C->D: B blocks all paths from A to C and D.
        Ref: dag.dfy:{lemma.line} {lemma.name}
        """
        j_c = are_d_separated(self.chain4, self.A, self.C, conditions=[self.B])
        j_d = are_d_separated(self.chain4, self.A, self.D, conditions=[self.B])
        self.assertTrue(j_c.separated, "Decomposition: A ⊥ C | {{B}}")
        self.assertTrue(j_d.separated, "Decomposition: A ⊥ D | {{B}}")''',
                [],
            ))
        elif lemma.name == "DSep_WeakUnion":
            tests.append((
                "test_weak_union",
                f'''    def test_weak_union(self):
        """(Y ⊥ Z ∪ Z' | W) => (Y ⊥ Z | W ∪ Z').

        Chain A->B->C->D: A ⊥ C | {{B}} => A ⊥ C | {{B, D}}.
        Ref: dag.dfy:{lemma.line} {lemma.name}
        """
        j_base = are_d_separated(self.chain4, self.A, self.C, conditions=[self.B])
        self.assertTrue(j_base.separated)
        j_weak = are_d_separated(self.chain4, self.A, self.C, conditions=[self.B, self.D])
        self.assertTrue(j_weak.separated, "Weak union: A ⊥ C | {{B, D}}")''',
                [],
            ))
        elif lemma.name == "DSep_Contraction":
            tests.append((
                "test_contraction",
                f'''    def test_contraction(self):
        """(Y ⊥ Z | W ∪ Z') ∧ (Y ⊥ Z' | W) => (Y ⊥ Z ∪ Z' | W).

        Chain A->B->C->D: premises A ⊥ D | {{B,C}} and A ⊥ C | {{B}}.
        Ref: dag.dfy:{lemma.line} {lemma.name}
        """
        j1 = are_d_separated(self.chain4, self.A, self.D, conditions=[self.B, self.C])
        j2 = are_d_separated(self.chain4, self.A, self.C, conditions=[self.B])
        self.assertTrue(j1.separated, "Premise 1: A ⊥ D | {{B, C}}")
        self.assertTrue(j2.separated, "Premise 2: A ⊥ C | {{B}}")
        j3 = are_d_separated(self.chain4, self.A, self.D, conditions=[self.B])
        self.assertTrue(j3.separated, "Contraction: A ⊥ D | {{B}}")''',
                [],
            ))

    return tests


def _gen_local_markov_tests(lemmas: list[DafnyLemma]) -> list[tuple[str, str, list]]:
    """Generate Local Markov Property tests."""
    tests = []
    for lemma in lemmas:
        if lemma.name == "LocalMarkov":
            tests.append((
                "test_local_markov_chain",
                f'''    def test_local_markov_chain(self):
        """Every node v: {{v}} ⊥ NonDesc(v) | Pa(v).

        Ref: dag.dfy:{lemma.line} {lemma.name}
        """
        A, B, C = Variable("A"), Variable("B"), Variable("C")
        chain = NxMixedGraph.from_edges(directed=[(A, B), (B, C)])
        for node in [A, B, C]:
            non_desc = chain.non_descendants(node)
            parents = set(chain.directed.predecessors(node))
            for nd in non_desc:
                judgement = are_d_separated(chain, node, nd, conditions=parents)
                self.assertTrue(
                    judgement.separated,
                    f"Local Markov violated: {{node}} not d-sep from {{nd}} given {{parents}}",
                )''',
                [],
            ))
    return tests


def _gen_do_calculus_tests(lemmas: list[DafnyLemma]) -> list[tuple[str, str, list]]:
    """Generate do-calculus rule tests from do_calculus.dfy."""
    tests = []
    impls: list[tuple[str, str]] = []

    for lemma in lemmas:
        if lemma.name == "Rule1_InsertDeleteObservation":
            tests.append((
                "test_rule_1_does_not_apply_chain",
                f'''    def test_rule_1_does_not_apply_chain(self):
        """Rule 1 does NOT apply: M is not d-sep from Y given X in G_{{X̄}}.

        Graph: X->M->Y with do(X). G_{{X̄}} same (X is source).
        (Y ⊥ M | {{X}}) fails because M->Y is active.
        Ref: do_calculus.dfy:{lemma.line} {lemma.name}
        """
        X, M, Y = Variable("X"), Variable("M"), Variable("Y")
        graph = NxMixedGraph.from_edges(directed=[(X, M), (M, Y)])
        self.assertFalse(
            rule_1_of_do_calculus_applies(
                graph, treatments={{X}}, outcomes={{Y}}, conditions={{M}}, observation=M,
            )
        )''',
                [],
            ))
            tests.append((
                "test_rule_1_applies_isolated_node",
                f'''    def test_rule_1_applies_isolated_node(self):
        """Rule 1 applies: isolated Z is trivially d-sep from Y.

        Graph: X->Y, Z (isolated). Z d-sep from Y in G_{{X̄}}.
        Ref: do_calculus.dfy:{lemma.line} {lemma.name}
        """
        X, Y, Z = Variable("X"), Variable("Y"), Variable("Z")
        graph = NxMixedGraph.from_edges(nodes=[Z], directed=[(X, Y)])
        self.assertTrue(
            rule_1_of_do_calculus_applies(
                graph, treatments={{X}}, outcomes={{Y}}, conditions=set(), observation=Z,
            )
        )''',
                [],
            ))
            impls.append(("rule_1_of_do_calculus_applies", ""))  # marker

        elif lemma.name == "Rule2_ActionObservationExchange":
            tests.append((
                "test_rule_2_chain",
                f'''    def test_rule_2_chain(self):
        """Rule 2: action/observation exchange on a chain.

        Graph: X->Z->Y. Check condition in G_{{X̄, Z̲}}.
        Ref: do_calculus.dfy:{lemma.line} {lemma.name}
        """
        X, Z, Y = Variable("X"), Variable("Z"), Variable("Y")
        graph = NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)])
        # G_{{X̄}} = same (X is source). G_{{X̄, Z̲}} = remove Z's outgoing.
        # In G_{{X̄, Z̲}}: X->Z, Y isolated. (Y ⊥ Z | {{X}}) holds.
        self.assertTrue(
            rule_2_of_do_calculus_applies(
                graph, treatments={{X}}, outcomes={{Y}}, conditions={{Z}}, condition=Z,
            )
        )''',
                [],
            ))

        elif lemma.name == "Rule3_InsertDeleteAction":
            tests.append((
                "test_rule_3_isolated_action",
                f'''    def test_rule_3_isolated_action(self):
        """Rule 3: do(Z) deletable when Z is isolated (no effect on Y).

        Graph: X->Y, Z (isolated). Z not ancestor of W={{}}.
        G_{{X̄, Z̄}} = same. (Y ⊥ Z | X) holds.
        Ref: do_calculus.dfy:{lemma.line} {lemma.name}
        """
        X, Y, Z = Variable("X"), Variable("Y"), Variable("Z")
        graph = NxMixedGraph.from_edges(nodes=[Z], directed=[(X, Y)])
        self.assertTrue(
            rule_3_of_do_calculus_applies(
                graph, treatments={{X}}, outcomes={{Y}}, conditions=set(), action=Z,
            )
        )''',
                [],
            ))
            impls.append(("rule_3_of_do_calculus_applies", ""))  # marker

        elif lemma.name == "BackdoorAdjustment":
            tests.append((
                "test_backdoor_simple",
                f'''    def test_backdoor_simple(self):
        """Z satisfies backdoor when it blocks backdoor paths and is not a descendant of X.

        Graph: Z->X->Y, Z->Y.
        Ref: do_calculus.dfy:{lemma.line} {lemma.name}
        """
        X, Y, Z = Variable("X"), Variable("Y"), Variable("Z")
        graph = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y), (Z, Y)])
        self.assertTrue(satisfies_backdoor(graph, outcomes={{Y}}, treatments={{X}}, adjustment={{Z}}))''',
                [],
            ))
            tests.append((
                "test_backdoor_fails_descendant",
                f'''    def test_backdoor_fails_descendant(self):
        """Backdoor fails when adjustment set contains a descendant of X.

        Graph: X->M->Y. M is a descendant of X.
        Ref: do_calculus.dfy:{lemma.line} {lemma.name}
        """
        X, M, Y = Variable("X"), Variable("M"), Variable("Y")
        graph = NxMixedGraph.from_edges(directed=[(X, M), (M, Y)])
        self.assertFalse(satisfies_backdoor(graph, outcomes={{Y}}, treatments={{X}}, adjustment={{M}}))''',
                [],
            ))
            impls.append(("satisfies_backdoor", ""))

        elif lemma.name == "FrontdoorCriterion":
            tests.append((
                "test_frontdoor_classic",
                f'''    def test_frontdoor_classic(self):
        """M satisfies frontdoor for X->Y with confounding X<->Y.

        Graph: X->M->Y, X<->Y (bidirected).
        Ref: do_calculus.dfy:{lemma.line} {lemma.name}
        """
        X, M, Y = Variable("X"), Variable("M"), Variable("Y")
        graph = NxMixedGraph.from_edges(
            directed=[(X, M), (M, Y)],
            undirected=[(X, Y)],
        )
        self.assertTrue(
            satisfies_frontdoor(graph, outcomes={{Y}}, treatments={{X}}, mediators={{M}})
        )''',
                [],
            ))
            tests.append((
                "test_frontdoor_fails_no_mediator",
                f'''    def test_frontdoor_fails_no_mediator(self):
        """Frontdoor fails with empty mediator set and direct confounding.

        Graph: X->Y, X<->Y.
        Ref: do_calculus.dfy:{lemma.line} {lemma.name}
        """
        X, Y = Variable("X"), Variable("Y")
        graph = NxMixedGraph.from_edges(
            directed=[(X, Y)],
            undirected=[(X, Y)],
        )
        self.assertFalse(
            satisfies_frontdoor(graph, outcomes={{Y}}, treatments={{X}}, mediators=set())
        )''',
                [],
            ))
            impls.append(("satisfies_frontdoor", ""))

    return tests


def _gen_probability_tests(lemmas: list[DafnyLemma]) -> list[tuple[str, str, list]]:
    """Generate probability axiom tests from probability.dfy."""
    tests = []

    for lemma in lemmas:
        if lemma.name == "ChainRule":
            tests.append((
                "test_chain_rule_two_vars",
                f'''    def test_chain_rule_two_vars(self):
        """P(A, B) = P(A | B) * P(B).

        Ref: probability.dfy:{lemma.line} {lemma.name}
        """
        from y0.dsl import A, B, P
        from y0.mutate.chain import chain_expand

        result = chain_expand(P(A, B))
        expected = P(A | B) * P(B)
        self.assertEqual(expected, result)''',
                [],
            ))
            tests.append((
                "test_chain_rule_three_vars",
                f'''    def test_chain_rule_three_vars(self):
        """P(A, B, C) = P(A | B, C) * P(B | C) * P(C).

        Ref: probability.dfy:{lemma.line} {lemma.name} (iterated)
        """
        from y0.dsl import A, B, C, P
        from y0.mutate.chain import chain_expand

        result = chain_expand(P(A, B, C))
        expected = P(A | B, C) * P(B | C) * P(C)
        self.assertEqual(expected, result)''',
                [],
            ))
        elif lemma.name == "BayesTheorem":
            tests.append((
                "test_bayes_theorem",
                f'''    def test_bayes_theorem(self):
        """P(A | B) = P(B | A) * P(A) / P(B).

        Ref: probability.dfy:{lemma.line} {lemma.name}
        """
        from y0.dsl import A, B, Fraction, P
        from y0.mutate.chain import bayes_expand

        result = bayes_expand(P(A | B))
        expected = Fraction(P(B | A) * P(A), P(B))
        self.assertEqual(expected, result)''',
                [],
            ))

    return tests


def generate_test_file(dag_spec: DafnySpec, dc_spec: DafnySpec, prob_spec: DafnySpec) -> str:
    """Generate the full test file from parsed specs."""
    surgery_tests = _gen_surgery_tests(dag_spec.lemmas)
    ancestry_tests = _gen_ancestry_tests(dag_spec.functions, dag_spec.lemmas)
    dsep_tests = _gen_dsep_tests(dag_spec.lemmas)
    semigraphoid_tests = _gen_semigraphoid_tests(dag_spec.lemmas)
    local_markov_tests = _gen_local_markov_tests(dag_spec.lemmas)
    do_calc_tests = _gen_do_calculus_tests(dc_spec.lemmas)
    prob_tests = _gen_probability_tests(prob_spec.lemmas)

    # Collect needed implementations
    needed_impls: dict[str, str] = {}
    for group in [surgery_tests, ancestry_tests, dsep_tests, semigraphoid_tests,
                  local_markov_tests, do_calc_tests, prob_tests]:
        for _, _, impls in group:
            for name, code in impls:
                needed_impls[name] = code

    parts = []
    parts.append(f'''"""Tests verifying Python graph/DSL operations match Dafny formal specifications.

Auto-generated by scripts/generate_dafny_conformance_tests.py from:
  - {dag_spec.path.relative_to(ROOT)}
  - {dc_spec.path.relative_to(ROOT)}
  - {prob_spec.path.relative_to(ROOT)}

Do not edit manually. Re-run the generator to update.
"""

import unittest

from y0.algorithm.conditional_independencies import are_d_separated
from y0.algorithm.do_calculus import (
    rule_1_of_do_calculus_applies,
    rule_2_of_do_calculus_applies,
    rule_3_of_do_calculus_applies,
    satisfies_backdoor,
    satisfies_frontdoor,
)
from y0.dsl import Variable
from y0.graph import NxMixedGraph
''')

    # ── Surgery tests
    parts.append('''
class TestSurgeryLemmas(unittest.TestCase):
    """Tests corresponding to dag.dfy §4 Graph Surgery lemmas."""

    def setUp(self) -> None:
        """Build the three-node chain A -> B -> C from dag.dfy §9."""
        self.A = Variable("A")
        self.B = Variable("B")
        self.C = Variable("C")
        self.chain = NxMixedGraph.from_edges(
            directed=[(self.A, self.B), (self.B, self.C)]
        )
''')
    for _, body, _ in surgery_tests:
        parts.append(body)
        parts.append("")

    # ── Ancestry tests
    parts.append('''
class TestAncestryLemmas(unittest.TestCase):
    """Tests corresponding to dag.dfy §2-3 Acyclicity and Ancestry."""

    def setUp(self) -> None:
        """Build the three-node chain A -> B -> C from dag.dfy §9."""
        self.A = Variable("A")
        self.B = Variable("B")
        self.C = Variable("C")
        self.chain = NxMixedGraph.from_edges(
            directed=[(self.A, self.B), (self.B, self.C)]
        )
''')
    for _, body, _ in ancestry_tests:
        parts.append(body)
        parts.append("")

    # ── D-Separation tests
    parts.append('''
class TestDSeparation(unittest.TestCase):
    """Tests corresponding to dag.dfy §6 d-Separation."""

    def setUp(self) -> None:
        """Build the three-node chain A -> B -> C from dag.dfy §9."""
        self.A = Variable("A")
        self.B = Variable("B")
        self.C = Variable("C")
        self.chain = NxMixedGraph.from_edges(
            directed=[(self.A, self.B), (self.B, self.C)]
        )
''')
    for _, body, _ in dsep_tests:
        parts.append(body)
        parts.append("")

    # ── Semi-graphoid tests
    parts.append('''
class TestSemiGraphoidAxioms(unittest.TestCase):
    """Tests for semi-graphoid axioms. Ref: dag.dfy §7."""

    def setUp(self) -> None:
        """Build a 4-node chain A->B->C->D for richer d-separation testing."""
        self.A = Variable("A")
        self.B = Variable("B")
        self.C = Variable("C")
        self.D = Variable("D")
        self.chain4 = NxMixedGraph.from_edges(
            directed=[(self.A, self.B), (self.B, self.C), (self.C, self.D)]
        )
''')
    for _, body, _ in semigraphoid_tests:
        parts.append(body)
        parts.append("")

    # ── Local Markov tests
    parts.append('''
class TestLocalMarkov(unittest.TestCase):
    """Tests for the Local Markov Property. Ref: dag.dfy §8."""
''')
    for _, body, _ in local_markov_tests:
        parts.append(body)
        parts.append("")

    # ── Do-Calculus tests
    parts.append('''
class TestDoCalculusRules(unittest.TestCase):
    """Tests corresponding to do_calculus.dfy §4-7."""
''')
    for _, body, _ in do_calc_tests:
        parts.append(body)
        parts.append("")

    # ── Probability tests
    parts.append('''
class TestProbabilityAxioms(unittest.TestCase):
    """Tests aligning y0 DSL with probability.dfy axioms."""
''')
    for _, body, _ in prob_tests:
        parts.append(body)
        parts.append("")

    # Collect implementation stubs info
    if needed_impls:
        parts.append("")
        parts.append("# ── Implementation stubs needed ──")
        parts.append(f"# The following methods/functions must exist for these tests to pass:")
        for name in needed_impls:
            parts.append(f"#   - {name}")
        parts.append("")

    return "\n".join(parts)


def main() -> None:
    """Parse Dafny specs and generate conformance tests."""
    dag_spec = parse_dafny_file(DAFNY_DIR / "dag.dfy")
    dc_spec = parse_dafny_file(DAFNY_DIR / "do_calculus.dfy")
    prob_spec = parse_dafny_file(DAFNY_DIR / "probability.dfy")

    print(f"Parsed {dag_spec.path.name}: {len(dag_spec.lemmas)} lemmas, {len(dag_spec.functions)} functions")
    print(f"Parsed {dc_spec.path.name}: {len(dc_spec.lemmas)} lemmas, {len(dc_spec.functions)} functions")
    print(f"Parsed {prob_spec.path.name}: {len(prob_spec.lemmas)} lemmas, {len(prob_spec.functions)} functions")

    test_code = generate_test_file(dag_spec, dc_spec, prob_spec)
    OUTPUT.write_text(test_code)
    print(f"\nGenerated {OUTPUT.relative_to(ROOT)}")

    # Count tests
    test_count = test_code.count("    def test_")
    print(f"  {test_count} test methods across {test_code.count('class Test')} test classes")


if __name__ == "__main__":
    main()
