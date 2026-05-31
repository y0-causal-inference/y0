"""
generate_proof_dag.py
---------------------
Automatically generate a Mermaid dependency DAG from Dafny and Lean 4 source files.

For Dafny:
  - Discovers top-level lemma/theorem/function declarations
  - Marks {:axiom} nodes as unproved
  - Infers edges by scanning proof bodies for calls to other declared lemmas

For Lean 4:
  - Discovers top-level theorem/lemma/def declarations
  - Marks `sorry` as unproved
  - Infers edges from `apply`, `exact`, `have`, and bare references to declared names

Output: a Mermaid flowchart snippet (TD layout) to stdout or a file.

Usage:
    python scripts/generate_proof_dag.py              # scan src/dafny + src/lean
    python scripts/generate_proof_dag.py --dafny-only
    python scripts/generate_proof_dag.py --lean-only
    python scripts/generate_proof_dag.py --output docs/dag.md
    python scripts/generate_proof_dag.py --filter "DSep|Markov|ID|Lemma|Theorem"

    # Collapse hundreds of lemmas into ~25 high-level concept nodes:
    python scripts/generate_proof_dag.py --concepts scripts/proof_dag_concepts.yaml
    python scripts/generate_proof_dag.py --concepts scripts/proof_dag_concepts.yaml --output docs/high-level-dag.md
"""

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dafny parsing
# ---------------------------------------------------------------------------

DAFNY_DECL = re.compile(
    r"^\s{0,4}"
    r"(?:lemma|ghost lemma|theorem"
    r"|function|ghost function"
    r"|method|ghost method"
    r"|predicate|ghost predicate"
    r")\s+"
    r"(?:\{[^}]*\}\s+)*"   # skip optional {:axiom} / {:verify false} / etc.
    r"(\w+)",
    re.MULTILINE,
)
DAFNY_AXIOM = re.compile(r"\{:axiom\}")


def parse_dafny(path: Path) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Return (nodes, edges) extracted from a Dafny file.

    nodes: {name: {"axiom": bool, "file": str}}
    edges: [(callee, caller)]  — prerequisite-first, matching extra_edges convention
    """
    src = path.read_text(encoding="utf-8")
    decls = [(m.group(1), m.start()) for m in DAFNY_DECL.finditer(src)]
    if not decls:
        return {}, []

    decls_with_end: list[tuple[str, int, int]] = []
    for i, (name, start) in enumerate(decls):
        end = decls[i + 1][1] if i + 1 < len(decls) else len(src)
        decls_with_end.append((name, start, end))

    nodes: dict[str, dict] = {}
    for name, start, end in decls_with_end:
        header = src[start : src.index("\n", start) + 200 if "\n" in src[start:] else end]
        is_axiom = bool(DAFNY_AXIOM.search(header[:500]))
        nodes[name] = {"axiom": is_axiom, "file": path.name, "prover": "dafny"}

    lemma_names = set(nodes)
    call_pat = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in lemma_names) + r")\s*[(<(]"
    )

    edges: list[tuple[str, str]] = []
    for name, start, end in decls_with_end:
        body = src[start:end]
        # Skip past the signature line(s) to avoid counting the declaration itself
        body_after_sig = body[body.find("{") + 1 :] if "{" in body else body
        calls = set(call_pat.findall(body_after_sig)) - {name}
        for callee in sorted(calls):
            edges.append((callee, name))

    return nodes, edges


# ---------------------------------------------------------------------------
# Lean 4 parsing
# ---------------------------------------------------------------------------

LEAN_DECL = re.compile(
    r"^(?:theorem|lemma|noncomputable def|def|axiom)\s+(\w+)",
    re.MULTILINE,
)
LEAN_SORRY = re.compile(r"\bsorry\b")
LEAN_BARE_AXIOM = re.compile(
    r"^axiom\s+(\w+)",
    re.MULTILINE,
)


def parse_lean(path: Path) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Return (nodes, edges) extracted from a Lean 4 file.

    nodes: {name: {"axiom": bool, "file": str}}
    edges: [(callee, caller)]  — prerequisite-first, matching extra_edges convention
    """
    src = path.read_text(encoding="utf-8")
    decls = [(m.group(1), m.start()) for m in LEAN_DECL.finditer(src)]
    if not decls:
        return {}, []

    # Collect names of bare `axiom` declarations so we can mark them without
    # needing to inspect a body for `sorry`.
    bare_axiom_names = {m.group(1) for m in LEAN_BARE_AXIOM.finditer(src)}

    decls_with_end: list[tuple[str, int, int]] = []
    for i, (name, start) in enumerate(decls):
        end = decls[i + 1][1] if i + 1 < len(decls) else len(src)
        decls_with_end.append((name, start, end))

    nodes: dict[str, dict] = {}
    for name, start, end in decls_with_end:
        body = src[start:end]
        is_axiom = name in bare_axiom_names or bool(LEAN_SORRY.search(body))
        nodes[name] = {"axiom": is_axiom, "file": path.name, "prover": "lean"}

    lemma_names = set(nodes)
    ref_pat = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in lemma_names) + r")\b"
    )

    edges: list[tuple[str, str]] = []
    for name, start, end in decls_with_end:
        body = src[start:end]
        # Trim past the colon/`:=` to get to the proof body
        body_after_decl = body[body.find(":=") + 2 :] if ":=" in body else body
        calls = set(ref_pat.findall(body_after_decl)) - {name}
        for callee in sorted(calls):
            edges.append((callee, name))

    return nodes, edges


# ---------------------------------------------------------------------------
# Concept collapsing
# ---------------------------------------------------------------------------


def load_concepts(path: Path) -> dict[str, str]:
    """Load a YAML concepts file and return {lemma_name: concept_id}.

    YAML format (no external dependency — parsed manually):

        concepts:
          SYM:
            label: "d-Sep Symmetry"
            members: [DSep_Symmetry, dSep_symmetry]
          KAHN_PROOF:
            label: "Topological Sort Correctness"
            members: [KahnsAlgorithm_Correct, kahnSort_spec]
    """
    import json  # noqa: PLC0415

    text = path.read_text(encoding="utf-8")

    # Try JSON first (superset-compatible subset), then naive YAML
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Minimal YAML parser: only handles the structure above
        data = _parse_minimal_yaml(text)

    mapping: dict[str, str] = {}  # lemma -> concept_id
    for concept_id, info in data.get("concepts", {}).items():
        for member in info.get("members", []):
            mapping[member] = concept_id
    return mapping


def _parse_minimal_yaml(text: str) -> dict:
    """Parse the narrow YAML subset used by proof_dag_concepts.yaml."""
    result: dict = {"concepts": {}, "extra_edges": []}
    current_concept: str | None = None
    in_members = False
    in_extra_edges = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped.startswith("#") or not stripped:
            continue
        if stripped == "concepts:":
            in_extra_edges = False
            continue
        if stripped == "extra_edges:":
            in_extra_edges = True
            current_concept = None
            continue
        if in_extra_edges:
            if stripped.startswith("- "):
                # "- FROM --> TO"
                edge = stripped[2:].strip()
                parts = [p.strip() for p in edge.split("-->")]
                if len(parts) == 2:
                    result["extra_edges"].append({"from": parts[0], "to": parts[1]})
            continue
        if indent == 2 and stripped.endswith(":"):
            current_concept = stripped[:-1]
            result["concepts"][current_concept] = {"label": current_concept, "members": []}
            in_members = False
        elif indent == 4 and stripped.startswith("label:") and current_concept:
            result["concepts"][current_concept]["label"] = stripped[6:].strip().strip('"')
        elif indent == 4 and stripped.startswith("class:") and current_concept:
            result["concepts"][current_concept]["class"] = stripped[6:].strip().strip('"')
        elif indent == 4 and stripped == "members:":
            in_members = True
        elif indent == 6 and stripped.startswith("- ") and in_members and current_concept:
            result["concepts"][current_concept]["members"].append(stripped[2:].strip())
        else:
            in_members = False

    return result


def collapse_to_concepts(
    all_nodes: dict[str, dict],
    all_edges: list[tuple[str, str]],
    concepts_path: Path,
    concept_labels: dict[str, str],
    drop_unmapped: bool = True,
) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Collapse raw lemma nodes into concept groups.

    Returns new (nodes, edges) where:
    - Each concept becomes one node whose status is 'axiom' if ANY member is unproved.
    - Edges between concepts are aggregated; self-loops are dropped.
    - If drop_unmapped=True (default), lemmas not listed in the mapping are silently
      dropped — only the named concept nodes appear in the output.
    """
    mapping = load_concepts(concepts_path)
    known_concepts = set(concept_labels.keys())

    def concept_of(name: str) -> str | None:
        if name in mapping:
            return mapping[name]
        if drop_unmapped:
            return None  # discard
        return name  # keep as-is

    # Build concept node metadata: track proved/total counts per prover
    concept_nodes: dict[str, dict] = {}
    for name, meta in all_nodes.items():
        cid = concept_of(name)
        if cid is None:
            continue
        p = meta["prover"]  # "dafny" or "lean"
        proved = 0 if meta["axiom"] else 1
        if cid not in concept_nodes:
            concept_nodes[cid] = {
                "axiom": meta["axiom"],
                "label": concept_labels.get(cid, cid),
                "prover": p,
                "dafny_proved": proved if p == "dafny" else 0,
                "dafny_total": 1 if p == "dafny" else 0,
                "lean_proved": proved if p == "lean" else 0,
                "lean_total": 1 if p == "lean" else 0,
            }
        else:
            if meta["axiom"]:
                concept_nodes[cid]["axiom"] = True
            if p == "dafny":
                concept_nodes[cid]["dafny_proved"] += proved
                concept_nodes[cid]["dafny_total"] += 1
            else:
                concept_nodes[cid]["lean_proved"] += proved
                concept_nodes[cid]["lean_total"] += 1
            # Mark as "both" if we see both provers
            existing = concept_nodes[cid]["prover"]
            if existing != p and existing != "both":
                concept_nodes[cid]["prover"] = "both"

    # Ensure every concept defined in the file appears, even if no members were found
    # (allows manual-only concepts like K, REACH whose members may not exist in code)
    for cid, label in concept_labels.items():
        if cid not in concept_nodes:
            concept_nodes[cid] = {
                "axiom": False, "label": label, "prover": "dafny",
                "dafny_proved": 0, "dafny_total": 0,
                "lean_proved": 0, "lean_total": 0,
            }

    # Aggregate edges — only between known concepts
    concept_edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for caller, callee in all_edges:
        src = concept_of(caller)
        dst = concept_of(callee)
        if src is None or dst is None:
            continue
        if src != dst and (src, dst) not in seen:
            concept_edges.append((src, dst))
            seen.add((src, dst))

    return concept_nodes, concept_edges


# ---------------------------------------------------------------------------
# Mermaid generation
# ---------------------------------------------------------------------------

CLASSDEFS = """\
    classDef bothProved fill:#d6f5d6,stroke:#2d8a2d,color:#000,padding:12px,font-size:13px
    classDef leanOnly   fill:#d6eeff,stroke:#1a6fa8,color:#000,padding:12px,font-size:13px
    classDef dafnyOnly  fill:#fff8d6,stroke:#a08800,color:#000,padding:12px,font-size:13px
    classDef unproved   fill:#ffd6d6,stroke:#cc3333,color:#000,padding:12px,font-size:13px
    classDef partial    fill:#ffe8b2,stroke:#cc7700,color:#000,padding:12px,font-size:13px
    classDef algo       fill:#ead6ff,stroke:#6622cc,color:#000,padding:12px,font-size:13px
    classDef mathlib    fill:#ffe0b2,stroke:#cc6600,color:#000,padding:12px,font-size:13px"""


def cycles_in(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Return one representative cycle per reachable back-edge.
    Returns [] if the graph is a DAG.

    Each returned list is a closed walk [v0, v1, ..., v0].  Only the *first*
    cycle found in each DFS tree is returned — use for diagnostics, not
    exhaustive enumeration.
    """
    from collections import defaultdict

    adj: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for a, b in edges:
        adj[a].append(b)
        nodes.add(a)
        nodes.add(b)

    UNVISITED, DONE = 0, 2
    state: dict[str, int] = defaultdict(int)
    found: list[list[str]] = []

    def dfs(u: str, path: list[str], on_stack: set[str]) -> bool:
        state[u] = 1  # in-progress
        on_stack.add(u)
        for v in adj[u]:
            if v in on_stack:
                # Back-edge: v is an ancestor in the current path
                idx = path.index(v)
                found.append(path[idx:] + [v])
                return True
            if state[v] == UNVISITED:
                if dfs(v, path + [v], on_stack):
                    return True
        on_stack.discard(u)
        state[u] = DONE
        return False

    for n in sorted(nodes):
        if state[n] == UNVISITED:
            dfs(n, [n], set())

    return found


def sanitize(name: str) -> str:
    """Make a valid Mermaid node ID.

    Prefix with 'n_' so the ID never collides with a Mermaid reserved keyword
    (graph, end, style, interpolate, default, …).
    """
    return "n_" + re.sub(r"[^A-Za-z0-9_]", "_", name)


def build_mermaid(
    all_nodes: dict[str, dict],
    all_edges: list[tuple[str, str]],
    filter_pat: re.Pattern | None = None,
) -> str:
    if filter_pat:
        keep = {n for n in all_nodes if filter_pat.search(n)}
        neighbour_edges = [(a, b) for a, b in all_edges if a in keep or b in keep]
        keep |= {a for a, b in neighbour_edges} | {b for a, b in neighbour_edges}
        nodes = {n: v for n, v in all_nodes.items() if n in keep}
        edges = [(a, b) for a, b in all_edges if a in keep and b in keep]
    else:
        nodes = all_nodes
        edges = all_edges

    # De-duplicate edges
    edges = list(dict.fromkeys(edges))

    lines = ["```mermaid", "flowchart TD", CLASSDEFS]

    for name, meta in sorted(nodes.items()):
        nid = sanitize(name)
        label = meta.get("label", name.replace("_", " "))
        prover = meta.get("prover", "dafny")
        is_axiom = meta.get("axiom", False)
        dp = meta.get("dafny_proved", 0)
        dt = meta.get("dafny_total", 0)
        lp = meta.get("lean_proved", 0)
        lt = meta.get("lean_total", 0)
        cls_override = meta.get("class")  # explicit override from concepts file
        # Append per-prover fractions when member count data is available
        if not cls_override and (dt > 0 or lt > 0):
            parts = []
            if dt > 0:
                parts.append(f"D: {dp}/{dt}")
            if lt > 0:
                parts.append(f"L: {lp}/{lt}")
            label = f"{label}\n({' · '.join(parts)} proved)"
        total_proved = dp + lp
        total_all = dt + lt
        # A prover has "fully proved" a concept when all its members are proved.
        dafny_done = dt > 0 and dp == dt
        lean_done = lt > 0 and lp == lt
        if cls_override:
            cls = cls_override
        elif dafny_done and lean_done:
            cls = "bothProved"
        elif dafny_done and not lean_done:
            # Dafny fully proved; Lean either has sorry/axiom members or none
            cls = "dafnyOnly"
        elif lean_done and not dafny_done:
            # Lean fully proved; Dafny either has {:axiom} members or none
            cls = "leanOnly"
        elif total_all > 0 and 0 < total_proved < total_all:
            # Mixed within a single prover (some proved, some not)
            cls = "partial"
        elif is_axiom:
            cls = "unproved"
        elif prover == "both":
            # No members matched but concept spans both provers by heuristic
            cls = "bothProved"
        elif prover == "lean":
            cls = "leanOnly"
        else:
            cls = "dafnyOnly"
        lines.append(f'    {nid}["{label}"]:::{cls}')

    lines.append("")
    for caller, callee in edges:
        lines.append(f"    {sanitize(caller)} --> {sanitize(callee)}")

    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Mermaid DAG from Dafny/Lean sources")
    ap.add_argument("--dafny-only", action="store_true")
    ap.add_argument("--lean-only", action="store_true")
    ap.add_argument("--output", metavar="FILE", help="Write output to FILE instead of stdout")
    ap.add_argument(
        "--filter",
        metavar="REGEX",
        help="Only include nodes whose names match REGEX (plus one-hop neighbours)",
    )
    ap.add_argument(
        "--concepts",
        metavar="FILE",
        help="YAML file mapping lemma names to high-level concept groups (see proof_dag_concepts.yaml)",
    )
    ap.add_argument(
        "--src",
        metavar="DIR",
        default="src",
        help="Root source directory (default: src)",
    )
    args = ap.parse_args()

    src_root = Path(args.src)
    all_nodes: dict[str, dict] = {}
    all_edges: list[tuple[str, str]] = []

    if not args.lean_only:
        for dfy in sorted((src_root / "dafny").glob("*.dfy")):
            n, e = parse_dafny(dfy)
            all_nodes.update(n)
            all_edges.extend(e)

    if not args.dafny_only:
        lean_dir = src_root / "lean"
        for lean in sorted(lean_dir.rglob("*.lean")):
            n, e = parse_lean(lean)
            all_nodes.update(n)
            all_edges.extend(e)

    if args.concepts:
        concepts_path = Path(args.concepts)
        # Load labels from YAML for display
        import json  # noqa: PLC0415
        text = concepts_path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _parse_minimal_yaml(text)
        concept_labels = {cid: info.get("label", cid) for cid, info in data.get("concepts", {}).items()}
        concept_classes = {cid: info.get("class") for cid, info in data.get("concepts", {}).items()}
        all_nodes, all_edges = collapse_to_concepts(all_nodes, all_edges, concepts_path, concept_labels)
        # Apply explicit class overrides from concepts file
        for cid, cls in concept_classes.items():
            if cls and cid in all_nodes:
                all_nodes[cid]["class"] = cls
        # Append hand-specified semantic edges from extra_edges section
        seen_edges = set(all_edges)
        for entry in data.get("extra_edges", []):
            src, dst = entry["from"], entry["to"]
            if (src, dst) not in seen_edges:
                all_edges.append((src, dst))
                seen_edges.add((src, dst))

    filter_pat = re.compile(args.filter) if args.filter else None
    mermaid = build_mermaid(all_nodes, all_edges, filter_pat)

    if args.output:
        out = Path(args.output)
        out.write_text(mermaid + "\n", encoding="utf-8")
        print(f"Wrote {out}", file=sys.stderr)
        # Also splice the diagram into any Markdown files that contain the
        # auto-generated diagram markers adjacent to this output file.
        _splice_into_narratives(out, mermaid)
    else:
        print(mermaid)


_BEGIN_MARKER = "<!-- begin auto-generated diagram"
_END_MARKER = "<!-- end auto-generated diagram -->"


def _splice_into_narratives(dag_path: Path, mermaid: str) -> None:
    """Replace the content between the auto-generated diagram markers in any
    sibling *.md file that references *dag_path* by name."""
    docs_dir = dag_path.parent
    tag = dag_path.name
    for md in sorted(docs_dir.glob("*.md")):
        if md == dag_path:
            continue
        text = md.read_text(encoding="utf-8")
        begin_idx = text.find(_BEGIN_MARKER)
        if begin_idx == -1:
            continue
        # Check that this marker references our output file
        marker_line_end = text.index("\n", begin_idx)
        marker_line = text[begin_idx:marker_line_end]
        if tag not in marker_line:
            continue
        end_idx = text.find(_END_MARKER, begin_idx)
        if end_idx == -1:
            continue
        end_idx_after = end_idx + len(_END_MARKER)
        new_text = (
            text[:begin_idx]
            + f"{_BEGIN_MARKER} ({tag}) -->\n\n"
            + mermaid
            + f"\n\n{_END_MARKER}"
            + text[end_idx_after:]
        )
        md.write_text(new_text, encoding="utf-8")
        print(f"Spliced diagram into {md}", file=sys.stderr)


if __name__ == "__main__":
    main()
