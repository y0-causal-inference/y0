"""Tests for scripts/generate_proof_dag.py."""

import sys
import textwrap
from pathlib import Path

import pytest

# Make the scripts/ directory importable without installing it
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_proof_dag import (  # noqa: E402
    _parse_minimal_yaml,
    build_mermaid,
    collapse_to_concepts,
    cycles_in,
    load_concepts,
    parse_dafny,
    parse_lean,
    sanitize,
)


# ---------------------------------------------------------------------------
# parse_dafny — attribute annotation regression
# ---------------------------------------------------------------------------


class TestParseDafny:
    """Regression tests for DAFNY_DECL regex handling {:attr} annotations."""

    def _write(self, tmp_path, content):
        p = tmp_path / "test.dfy"
        p.write_text(content)
        return p

    def test_plain_lemma_parsed(self, tmp_path):
        p = self._write(tmp_path, "  lemma MyLemma(x: int)\n  ensures true\n  {}\n")
        nodes, _ = parse_dafny(p)
        assert "MyLemma" in nodes

    def test_axiom_attribute_inline_parsed(self, tmp_path):
        # Regression: `lemma {:axiom} FrontdoorCriterion(...)` was previously
        # skipped entirely because {:axiom} sat between `lemma` and the name.
        p = self._write(
            tmp_path,
            "  lemma {:axiom} FrontdoorCriterion(x: int)\n  ensures true\n",
        )
        nodes, _ = parse_dafny(p)
        assert "FrontdoorCriterion" in nodes

    def test_axiom_attribute_inline_marked_as_axiom(self, tmp_path):
        # The {:axiom} attribute must be reflected in the node's axiom flag.
        p = self._write(
            tmp_path,
            "  lemma {:axiom} BackdoorAdjustment(x: int)\n  ensures true\n",
        )
        nodes, _ = parse_dafny(p)
        assert nodes["BackdoorAdjustment"]["axiom"] is True

    def test_plain_lemma_not_marked_as_axiom(self, tmp_path):
        p = self._write(tmp_path, "  lemma Proved(x: int)\n  ensures true\n  {}\n")
        nodes, _ = parse_dafny(p)
        assert nodes["Proved"]["axiom"] is False

    def test_multiple_attributes_parsed(self, tmp_path):
        # e.g. lemma {:axiom} {:verify false} SomeLemma(...)
        p = self._write(
            tmp_path,
            "  lemma {:axiom} {:verify false} MultiAttr(x: int)\n  ensures true\n",
        )
        nodes, _ = parse_dafny(p)
        assert "MultiAttr" in nodes
        assert nodes["MultiAttr"]["axiom"] is True

    def test_reachability_axiom_pattern(self, tmp_path):
        # Regression: AncestorsCompiled_Correct uses {:axiom} inline — the exact
        # pattern that previously caused the node to be silently dropped.
        p = self._write(
            tmp_path,
            "  lemma {:axiom} AncestorsCompiled_Correct(G: Graph, W: set<Node>)\n"
            "    ensures true\n",
        )
        nodes, _ = parse_dafny(p)
        assert "AncestorsCompiled_Correct" in nodes
        assert nodes["AncestorsCompiled_Correct"]["axiom"] is True


# ---------------------------------------------------------------------------
# parse_lean — axiom keyword
# ---------------------------------------------------------------------------


class TestParseLean:
    """Tests for parse_lean, including bare `axiom` keyword declarations."""

    def _write(self, tmp_path, content):
        p = tmp_path / "test.lean"
        p.write_text(content)
        return p

    def test_theorem_parsed(self, tmp_path):
        p = self._write(tmp_path, "theorem myThm : True := trivial\n")
        nodes, _ = parse_lean(p)
        assert "myThm" in nodes

    def test_theorem_without_sorry_not_axiom(self, tmp_path):
        p = self._write(tmp_path, "theorem myThm : True := trivial\n")
        nodes, _ = parse_lean(p)
        assert nodes["myThm"]["axiom"] is False

    def test_theorem_with_sorry_marked_axiom(self, tmp_path):
        p = self._write(tmp_path, "theorem myThm : True := by sorry\n")
        nodes, _ = parse_lean(p)
        assert nodes["myThm"]["axiom"] is True

    def test_bare_axiom_keyword_parsed(self, tmp_path):
        # Regression: `axiom forwardTrail_of_mem_descendants ...` was previously
        # missed entirely because LEAN_DECL only matched theorem/lemma/def.
        p = self._write(
            tmp_path,
            "axiom forwardTrail_of_mem_descendants (G : Graph) (n : Node) : True\n",
        )
        nodes, _ = parse_lean(p)
        assert "forwardTrail_of_mem_descendants" in nodes

    def test_bare_axiom_keyword_marked_as_axiom(self, tmp_path):
        # A bare `axiom` declaration must be flagged as unproved.
        p = self._write(
            tmp_path,
            "axiom forwardTrail_of_mem_descendants (G : Graph) (n : Node) : True\n",
        )
        nodes, _ = parse_lean(p)
        assert nodes["forwardTrail_of_mem_descendants"]["axiom"] is True

    def test_axiom_and_theorem_in_same_file(self, tmp_path):
        # Both declarations must be found; only the axiom is marked unproved.
        src = (
            "axiom myAxiom : True\n"
            "theorem myThm : True := trivial\n"
        )
        p = self._write(tmp_path, src)
        nodes, _ = parse_lean(p)
        assert "myAxiom" in nodes
        assert "myThm" in nodes
        assert nodes["myAxiom"]["axiom"] is True
        assert nodes["myThm"]["axiom"] is False


# ---------------------------------------------------------------------------
# sanitize
# ---------------------------------------------------------------------------


class TestSanitize:
    def test_prefix_prevents_reserved_word_graph(self):
        assert sanitize("graph") == "n_graph"

    def test_prefix_prevents_reserved_word_end(self):
        assert sanitize("end") == "n_end"

    def test_plain_name(self):
        assert sanitize("DSep_Symmetry") == "n_DSep_Symmetry"

    def test_special_chars_replaced(self):
        assert sanitize("foo-bar.baz") == "n_foo_bar_baz"

    def test_already_prefixed_not_double_prefixed(self):
        # sanitize is idempotent in the sense that calling it twice produces
        # a valid (if ugly) ID rather than crashing
        result = sanitize(sanitize("x"))
        assert result.startswith("n_")


# ---------------------------------------------------------------------------
# _parse_minimal_yaml
# ---------------------------------------------------------------------------


SIMPLE_YAML = textwrap.dedent("""\
    concepts:
      SYM:
        label: "d-Sep Symmetry"
        class: bothProved
        members:
          - DSep_Symmetry
          - dSep_symmetry
      UNPROVED_THING:
        label: "Something unproved"
        members:
          - foo
    extra_edges:
      - SYM --> UNPROVED_THING
      - UNPROVED_THING --> SYM
""")


class TestParseMinimalYaml:
    def test_concepts_parsed(self):
        data = _parse_minimal_yaml(SIMPLE_YAML)
        assert "SYM" in data["concepts"]
        assert "UNPROVED_THING" in data["concepts"]

    def test_label_parsed(self):
        data = _parse_minimal_yaml(SIMPLE_YAML)
        assert data["concepts"]["SYM"]["label"] == "d-Sep Symmetry"

    def test_class_parsed(self):
        data = _parse_minimal_yaml(SIMPLE_YAML)
        assert data["concepts"]["SYM"]["class"] == "bothProved"

    def test_members_parsed(self):
        data = _parse_minimal_yaml(SIMPLE_YAML)
        assert data["concepts"]["SYM"]["members"] == ["DSep_Symmetry", "dSep_symmetry"]

    def test_extra_edges_parsed(self):
        data = _parse_minimal_yaml(SIMPLE_YAML)
        edges = data["extra_edges"]
        assert {"from": "SYM", "to": "UNPROVED_THING"} in edges
        assert {"from": "UNPROVED_THING", "to": "SYM"} in edges

    def test_comments_ignored(self):
        yaml = textwrap.dedent("""\
            # top comment
            concepts:
              # inner comment
              A:
                label: "A label"
                members:
                  - a1  # inline comment not supported but shouldn't crash
        """)
        data = _parse_minimal_yaml(yaml)
        assert "A" in data["concepts"]

    def test_no_class_key_absent(self):
        data = _parse_minimal_yaml(SIMPLE_YAML)
        assert "class" not in data["concepts"]["UNPROVED_THING"]

    def test_empty_extra_edges(self):
        yaml = textwrap.dedent("""\
            concepts:
              X:
                label: "X"
                members:
                  - x1
        """)
        data = _parse_minimal_yaml(yaml)
        assert data["extra_edges"] == []


# ---------------------------------------------------------------------------
# load_concepts (via tmp file)
# ---------------------------------------------------------------------------


class TestLoadConcepts:
    def test_returns_member_to_concept_mapping(self, tmp_path):
        p = tmp_path / "concepts.yaml"
        p.write_text(SIMPLE_YAML)
        mapping = load_concepts(p)
        assert mapping["DSep_Symmetry"] == "SYM"
        assert mapping["dSep_symmetry"] == "SYM"
        assert mapping["foo"] == "UNPROVED_THING"

    def test_unknown_member_not_in_mapping(self, tmp_path):
        p = tmp_path / "concepts.yaml"
        p.write_text(SIMPLE_YAML)
        mapping = load_concepts(p)
        assert "some_random_lemma" not in mapping


# ---------------------------------------------------------------------------
# collapse_to_concepts
# ---------------------------------------------------------------------------


def _make_concepts_file(tmp_path: Path, yaml: str) -> Path:
    p = tmp_path / "concepts.yaml"
    p.write_text(yaml)
    return p


CONCEPTS_YAML = textwrap.dedent("""\
    concepts:
      GROUP_A:
        label: "Group A"
        members:
          - lemma_a1
          - lemma_a2
      GROUP_B:
        label: "Group B"
        members:
          - lemma_b1
""")


class TestCollapseToConceptsProverCounts:
    def _run(self, tmp_path, nodes, edges):
        p = _make_concepts_file(tmp_path, CONCEPTS_YAML)
        labels = {"GROUP_A": "Group A", "GROUP_B": "Group B"}
        return collapse_to_concepts(nodes, edges, p, labels)

    def test_all_dafny_proved(self, tmp_path):
        nodes = {
            "lemma_a1": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
            "lemma_a2": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
        }
        c_nodes, _ = self._run(tmp_path, nodes, [])
        a = c_nodes["GROUP_A"]
        assert a["dafny_proved"] == 2
        assert a["dafny_total"] == 2
        assert a["lean_total"] == 0
        assert a["axiom"] is False

    def test_mixed_dafny_lean(self, tmp_path):
        nodes = {
            "lemma_a1": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
            "lemma_a2": {"axiom": False, "prover": "lean", "file": "f.lean"},
        }
        c_nodes, _ = self._run(tmp_path, nodes, [])
        a = c_nodes["GROUP_A"]
        assert a["dafny_proved"] == 1
        assert a["dafny_total"] == 1
        assert a["lean_proved"] == 1
        assert a["lean_total"] == 1
        assert a["prover"] == "both"

    def test_one_axiom_poisons_concept(self, tmp_path):
        nodes = {
            "lemma_a1": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
            "lemma_a2": {"axiom": True, "prover": "dafny", "file": "f.dfy"},
        }
        c_nodes, _ = self._run(tmp_path, nodes, [])
        assert c_nodes["GROUP_A"]["axiom"] is True
        assert c_nodes["GROUP_A"]["dafny_proved"] == 1
        assert c_nodes["GROUP_A"]["dafny_total"] == 2

    def test_unmapped_lemmas_dropped_by_default(self, tmp_path):
        nodes = {
            "lemma_a1": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
            "some_helper": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
        }
        c_nodes, _ = self._run(tmp_path, nodes, [])
        assert "some_helper" not in c_nodes

    def test_concept_with_no_matched_members_still_appears(self, tmp_path):
        # GROUP_B has lemma_b1 but it's not in nodes — concept should still be present
        nodes = {
            "lemma_a1": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
        }
        c_nodes, _ = self._run(tmp_path, nodes, [])
        assert "GROUP_B" in c_nodes
        assert c_nodes["GROUP_B"]["dafny_total"] == 0


class TestCollapseToConceptsEdges:
    def _run(self, tmp_path, nodes, edges):
        p = _make_concepts_file(tmp_path, CONCEPTS_YAML)
        labels = {"GROUP_A": "Group A", "GROUP_B": "Group B"}
        return collapse_to_concepts(nodes, edges, p, labels)

    def test_intra_concept_edges_dropped(self, tmp_path):
        nodes = {
            "lemma_a1": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
            "lemma_a2": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
        }
        # edge within GROUP_A — should be dropped
        edges = [("lemma_a1", "lemma_a2")]
        _, c_edges = self._run(tmp_path, nodes, edges)
        assert c_edges == []

    def test_cross_concept_edges_aggregated(self, tmp_path):
        nodes = {
            "lemma_a1": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
            "lemma_b1": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
        }
        edges = [("lemma_a1", "lemma_b1"), ("lemma_a1", "lemma_b1")]  # duplicate
        _, c_edges = self._run(tmp_path, nodes, edges)
        assert c_edges.count(("GROUP_A", "GROUP_B")) == 1

    def test_unmapped_edge_endpoints_dropped(self, tmp_path):
        nodes = {
            "lemma_a1": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
            "helper": {"axiom": False, "prover": "dafny", "file": "f.dfy"},
        }
        edges = [("lemma_a1", "helper")]
        _, c_edges = self._run(tmp_path, nodes, edges)
        assert c_edges == []


# ---------------------------------------------------------------------------
# build_mermaid
# ---------------------------------------------------------------------------


class TestBuildMermaid:
    def _proved_node(self, prover="dafny", dp=1, dt=1, lp=0, lt=0):
        return {
            "label": "My Node",
            "prover": prover,
            "axiom": False,
            "dafny_proved": dp,
            "dafny_total": dt,
            "lean_proved": lp,
            "lean_total": lt,
        }

    def test_output_is_fenced_mermaid(self):
        out = build_mermaid({}, [])
        assert out.startswith("```mermaid")
        assert out.endswith("```")

    def test_dafny_proved_gets_dafny_only_class(self):
        nodes = {"X": self._proved_node(prover="dafny")}
        out = build_mermaid(nodes, [])
        assert ":::dafnyOnly" in out

    def test_lean_proved_gets_lean_only_class(self):
        nodes = {"X": self._proved_node(prover="lean", dp=0, dt=0, lp=1, lt=1)}
        out = build_mermaid(nodes, [])
        assert ":::leanOnly" in out

    def test_both_proved_gets_both_class(self):
        nodes = {"X": self._proved_node(prover="both", dp=1, dt=1, lp=1, lt=1)}
        out = build_mermaid(nodes, [])
        assert ":::bothProved" in out

    def test_axiom_gets_unproved_class(self):
        node = self._proved_node()
        node["axiom"] = True
        node["dafny_proved"] = 0
        nodes = {"X": node}
        out = build_mermaid(nodes, [])
        assert ":::unproved" in out

    def test_partial_gets_partial_class(self):
        # 1/2 Dafny proved, 0 Lean — partial within a single prover
        nodes = {"X": self._proved_node(prover="dafny", dp=1, dt=2)}
        out = build_mermaid(nodes, [])
        assert ":::partial" in out

    def test_lean_proved_dafny_axiom_gets_lean_only(self):
        # Lean fully proved the concept; Dafny has it as {:axiom}.
        # Regression: previously classified as :::partial instead of :::leanOnly.
        node = {
            "label": "My Concept",
            "prover": "both",
            "axiom": True,       # one Dafny member is axiom
            "dafny_proved": 0,
            "dafny_total": 1,
            "lean_proved": 1,
            "lean_total": 1,
        }
        out = build_mermaid({"X": node}, [])
        assert ":::leanOnly" in out, f"expected leanOnly, got: {out}"
        assert ":::partial" not in out

    def test_dafny_proved_lean_sorry_gets_dafny_only(self):
        # Dafny fully proved the concept; Lean has it as sorry.
        # Regression: previously classified as :::partial instead of :::dafnyOnly.
        node = {
            "label": "My Concept",
            "prover": "both",
            "axiom": True,       # one Lean member has sorry
            "dafny_proved": 1,
            "dafny_total": 1,
            "lean_proved": 0,
            "lean_total": 1,
        }
        out = build_mermaid({"X": node}, [])
        assert ":::dafnyOnly" in out, f"expected dafnyOnly, got: {out}"
        assert ":::partial" not in out

    def test_class_override_wins_over_axiom(self):
        node = self._proved_node()
        node["axiom"] = True
        node["class"] = "mathlib"
        node["dafny_total"] = 0  # suppress fraction
        nodes = {"X": node}
        out = build_mermaid(nodes, [])
        assert ":::mathlib" in out
        assert ":::unproved" not in out

    def test_fraction_label_dafny_only(self):
        nodes = {"X": self._proved_node(prover="dafny", dp=3, dt=5)}
        out = build_mermaid(nodes, [])
        assert "D: 3/5" in out
        assert "L:" not in out

    def test_fraction_label_both(self):
        nodes = {"X": self._proved_node(prover="both", dp=2, dt=3, lp=1, lt=1)}
        out = build_mermaid(nodes, [])
        assert "D: 2/3" in out
        assert "L: 1/1" in out

    def test_no_fraction_when_class_override(self):
        node = self._proved_node(dp=3, dt=5)
        node["class"] = "algo"
        nodes = {"X": node}
        out = build_mermaid(nodes, [])
        assert "proved)" not in out

    def test_no_fraction_when_no_members(self):
        node = {
            "label": "Empty concept",
            "prover": "dafny",
            "axiom": False,
            "dafny_proved": 0,
            "dafny_total": 0,
            "lean_proved": 0,
            "lean_total": 0,
        }
        nodes = {"X": node}
        out = build_mermaid(nodes, [])
        assert "proved)" not in out

    def test_edge_rendered(self):
        nodes = {
            "A": self._proved_node(),
            "B": self._proved_node(),
        }
        out = build_mermaid(nodes, [("A", "B")])
        assert "n_A --> n_B" in out

    def test_filter_excludes_unmatched_nodes(self):
        import re

        nodes = {
            "DSep_Sym": self._proved_node(),
            "Kahn": self._proved_node(),
        }
        out = build_mermaid(nodes, [], filter_pat=re.compile("DSep"))
        assert "DSep_Sym" in out
        assert "Kahn" not in out

    def test_reserved_word_node_id_prefixed(self):
        nodes = {"graph": self._proved_node()}
        out = build_mermaid(nodes, [])
        assert "n_graph" in out
        # bare "graph[" would be a Mermaid parse error
        assert '\n    graph["' not in out


# ---------------------------------------------------------------------------
# cycles_in — cycle detection utility
# ---------------------------------------------------------------------------


class TestCyclesIn:
    def test_empty_graph_is_acyclic(self):
        assert cycles_in([]) == []

    def test_linear_chain_is_acyclic(self):
        assert cycles_in([("A", "B"), ("B", "C"), ("A", "C")]) == []

    def test_self_loop_detected(self):
        result = cycles_in([("A", "A")])
        assert result != []

    def test_two_cycle_detected(self):
        result = cycles_in([("A", "B"), ("B", "A")])
        assert result != []
        # The cycle must mention both nodes
        flat = [n for cycle in result for n in cycle]
        assert "A" in flat and "B" in flat

    def test_three_cycle_detected(self):
        result = cycles_in([("A", "B"), ("B", "C"), ("C", "A")])
        assert result != []

    def test_cycle_in_larger_dag(self):
        # Pure DAG segment plus one isolated cycle
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("E", "F"), ("F", "E")]
        result = cycles_in(edges)
        assert result != []
        flat = [n for cycle in result for n in cycle]
        # The cycle is between E and F, not in the DAG segment
        assert "E" in flat or "F" in flat

    def test_returns_cycle_node_sequence(self):
        # The returned list is a closed walk: first == last
        result = cycles_in([("X", "Y"), ("Y", "X")])
        assert len(result) >= 1
        cycle = result[0]
        assert cycle[0] == cycle[-1]


# ---------------------------------------------------------------------------
# Concept DAG acyclicity — integration test against real sources + YAML
# ---------------------------------------------------------------------------


class TestConceptDAGAcyclicity:
    """The full concept graph built from actual Dafny/Lean sources + YAML
    extra_edges must be a DAG.  If this test fails it prints every cycle
    found so you can pinpoint the conflicting edges.
    """

    def test_concept_graph_is_acyclic(self):
        repo_root = Path(__file__).parent.parent
        src_root = repo_root / "src"
        yaml_path = repo_root / "scripts" / "proof_dag_concepts.yaml"

        # Parse all Dafny and Lean source files
        all_nodes: dict = {}
        all_edges: list = []

        for dfy in sorted((src_root / "dafny").glob("*.dfy")):
            n, e = parse_dafny(dfy)
            all_nodes.update(n)
            all_edges.extend(e)

        for lean in sorted((src_root / "lean").rglob("*.lean")):
            n, e = parse_lean(lean)
            all_nodes.update(n)
            all_edges.extend(e)

        # Collapse to concepts
        text = yaml_path.read_text(encoding="utf-8")
        data = _parse_minimal_yaml(text)
        concept_labels = {
            cid: info.get("label", cid)
            for cid, info in data.get("concepts", {}).items()
        }
        concept_nodes, concept_edges = collapse_to_concepts(
            all_nodes, all_edges, yaml_path, concept_labels
        )

        # Append hand-specified extra_edges (same logic as main())
        seen: set = set(concept_edges)
        for entry in data.get("extra_edges", []):
            src, dst = entry["from"], entry["to"]
            if (src, dst) not in seen:
                concept_edges.append((src, dst))
                seen.add((src, dst))

        found = cycles_in(concept_edges)
        assert found == [], (
            f"Concept DAG has {len(found)} cycle(s) — each line shows one cycle:\n"
            + "\n".join("  " + " → ".join(c) for c in found)
            + "\n\nRoot cause: extra_edges in proof_dag_concepts.yaml use "
            "prerequisite-first direction (A→B = 'B needs A') while auto-generated "
            "proof-call edges use caller-first direction (A→B = 'A calls B'). "
            "Any pair captured by both directions creates a cycle. "
            "Remove the conflicting extra_edges entry to fix each cycle."
        )
