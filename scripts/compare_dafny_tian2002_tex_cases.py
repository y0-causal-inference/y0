#!/usr/bin/env python3
"""Compare extracted Dafny Tian-2002 runtime against Python Tian on tex-defined cases.

This script uses the graph cases encoded in docs/shpitser-figures.tex:
- Figure 1(a), Figure 1(b)
- Figure 2(a) through Figure 2(h)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from y0.algorithm.tian_id import (
    compute_c_factor_conditioning_on_topological_predecessors,
    identify_district_variables,
)
from y0.dsl import Product, Sum, Variable
from y0.graph import NxMixedGraph


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extracted_dir() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_tian2002_extracted_py"


def _as_list(value: object) -> list[str]:
    if hasattr(value, "Elements"):
        return [str(item) for item in value.Elements]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return [str(value)]


def _as_iterable(value: object) -> list[object]:
    if hasattr(value, "Elements"):
        return list(value.Elements)
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _ir_node_to_jsonable(node: object) -> dict[str, Any]:
    cls_name = node.__class__.__name__

    def _get_field(obj: object, *names: str) -> object:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(f"none of fields {names!r} found on {type(obj)!r}")

    if cls_name.endswith("IRNode_IRSum"):
        return {
            "type": "sum",
            "over": _as_list(_get_field(node, "over", "_0")),
            "body": _ir_node_to_jsonable(_get_field(node, "body", "_1")),
        }
    if cls_name.endswith("IRNode_IRProduct"):
        factors = _get_field(node, "factors", "_0")
        return {
            "type": "product",
            "factors": [_ir_node_to_jsonable(f) for f in _as_iterable(factors)],
        }
    if cls_name.endswith("IRNode_IRProb"):
        return {
            "type": "prob",
            "vars": _as_list(_get_field(node, "vars_", "vars", "_0")),
            "given": _as_list(_get_field(node, "given", "_1")),
            "intervened": _as_list(_get_field(node, "intervened", "_2")),
        }
    if cls_name.endswith("IRNode_IRFrac"):
        return {
            "type": "frac",
            "numer": _ir_node_to_jsonable(_get_field(node, "numer", "_0")),
            "denom": _ir_node_to_jsonable(_get_field(node, "denom", "_1")),
        }
    if cls_name.endswith("IRNode_IRFailHedge"):
        return {
            "type": "fail_hedge",
            "f_nodes": _as_list(_get_field(node, "F__nodes", "F_nodes", "_0")),
            "fprime_nodes": _as_list(_get_field(node, "Fprime__nodes", "Fprime_nodes", "_1")),
        }
    return {"type": "unknown", "repr": repr(node)}


def _load_extracted() -> tuple[Any, Any]:
    extracted_dir = _extracted_dir()
    if not extracted_dir.exists():
        raise FileNotFoundError(f"missing extracted directory: {extracted_dir}")

    sys.path.insert(0, str(extracted_dir))
    import _dafny  # type: ignore
    import IDTian2002Extracted  # type: ignore

    return _dafny, IDTian2002Extracted


def _run_python_tian_identify(
    graph: NxMixedGraph,
    outcomes: set[Variable],
    treatments: set[Variable],
) -> dict[str, Any]:
    ordering = list(graph.topological_sort())
    d_set = graph.remove_in_edges(treatments).ancestors_inclusive(outcomes)
    d_sub = graph.subgraph(d_set)
    comps_d = [set(component) for component in d_sub.districts()]
    comps_g = [set(component) for component in graph.districts()]

    factors = []
    for d_i in comps_d:
        c_i = next((component for component in comps_g if d_i <= component), None)
        if c_i is None:
            return {"ok": False, "reason": "missing C_i superset"}

        q_c_i = compute_c_factor_conditioning_on_topological_predecessors(
            district=c_i,
            graph_probability=graph.joint_probability(),
            ordering=ordering,
        )
        value = identify_district_variables(
            input_variables=d_i,
            input_district=c_i,
            district_probability=q_c_i,
            graph=graph,
            ordering=ordering,
        )
        if value is None:
            return {
                "ok": False,
                "reason": "identify_district_variables FAIL",
                "d_i": sorted(v.name for v in d_i),
                "c_i": sorted(v.name for v in c_i),
            }
        factors.append(value)

    body = factors[0] if len(factors) == 1 else Product.safe(factors)
    over = [v for v in ordering if v in (d_set - outcomes)]
    expr = body if not over else Sum.safe(body, over)
    return {
        "ok": True,
        "expr": expr.to_text(),
        "sum_over": [v.name for v in over],
    }


def _run_extracted_tian(
    _dafny: Any,
    extracted: Any,
    graph: NxMixedGraph,
    outcomes: set[Variable],
    treatments: set[Variable],
    graph_id: str,
) -> dict[str, Any]:
    ordering = list(graph.topological_sort())
    edge_ctor = extracted.Edge_Edge
    directed = _dafny.SeqWithoutIsStrInference(
        [edge_ctor(u.name, v.name) for u, v in graph.directed.edges()]
    )
    undirected = _dafny.SeqWithoutIsStrInference(
        [edge_ctor(u.name, v.name) for u, v in graph.undirected.edges()]
    )
    all_nodes = _dafny.SeqWithoutIsStrInference([v.name for v in ordering])
    out = {v.name for v in outcomes}
    trt = {v.name for v in treatments}
    ords = _dafny.SeqWithoutIsStrInference([v.name for v in ordering])

    ok, doc = extracted.default__.IDTianToIR(
        graph_id, directed, undirected, all_nodes, out, trt, ords
    )
    ir = _ir_node_to_jsonable(doc.result)
    return {
        "ok": bool(ok),
        "ir_type": ir.get("type"),
        "ir": ir,
    }


def _build_tex_cases() -> list[dict[str, Any]]:
    w1 = Variable("W1")
    w2 = Variable("W2")
    x = Variable("X")
    y1 = Variable("Y1")
    y2 = Variable("Y2")
    y = Variable("Y")
    z = Variable("Z")
    z1 = Variable("Z1")
    z2 = Variable("Z2")
    w = Variable("W")

    return [
        {
            "name": "fig1a_tex",
            "expected_identifiable_tex": True,
            "graph": NxMixedGraph.from_edges(
                directed=[(w1, x), (x, y1), (w2, y2)],
                undirected=[(w1, y1), (w1, w2), (w2, x), (w1, y2)],
            ),
            "outcomes": {y1, y2},
            "treatments": {x},
        },
        {
            "name": "fig1b_tex",
            "expected_identifiable_tex": False,
            "graph": NxMixedGraph.from_edges(
                directed=[(w1, x), (x, y1), (w1, w2), (w2, y2)],
                undirected=[(w1, y1), (w1, w2), (w2, x), (x, y2)],
            ),
            "outcomes": {y1, y2},
            "treatments": {x},
        },
        {
            "name": "fig2a_tex",
            "expected_identifiable_tex": False,
            "graph": NxMixedGraph.from_edges(directed=[(x, y)], undirected=[(x, y)]),
            "outcomes": {y},
            "treatments": {x},
        },
        {
            "name": "fig2b_tex",
            "expected_identifiable_tex": False,
            "graph": NxMixedGraph.from_edges(directed=[(x, z), (z, y)], undirected=[(x, z)]),
            "outcomes": {y},
            "treatments": {x},
        },
        {
            "name": "fig2c_tex",
            "expected_identifiable_tex": False,
            "graph": NxMixedGraph.from_edges(
                directed=[(x, z), (z, y), (x, y)], undirected=[(x, z)]
            ),
            "outcomes": {y},
            "treatments": {x},
        },
        {
            "name": "fig2d_tex",
            "expected_identifiable_tex": False,
            "graph": NxMixedGraph.from_edges(
                directed=[(x, y), (z, y)], undirected=[(x, z), (z, y)]
            ),
            "outcomes": {y},
            "treatments": {x},
        },
        {
            "name": "fig2e_tex",
            "expected_identifiable_tex": False,
            "graph": NxMixedGraph.from_edges(
                directed=[(x, z), (z, y)], undirected=[(x, z), (x, y)]
            ),
            "outcomes": {y},
            "treatments": {x},
        },
        {
            "name": "fig2f_tex",
            "expected_identifiable_tex": False,
            "graph": NxMixedGraph.from_edges(
                directed=[(x, z), (z, y)], undirected=[(z, y), (x, y)]
            ),
            "outcomes": {y},
            "treatments": {x},
        },
        {
            "name": "fig2g_tex",
            "expected_identifiable_tex": False,
            "graph": NxMixedGraph.from_edges(
                directed=[(x, z1), (x, z2), (z1, y), (z2, y)],
                undirected=[(z1, z2), (z1, y)],
            ),
            "outcomes": {y},
            "treatments": {x},
        },
        {
            "name": "fig2h_tex",
            "expected_identifiable_tex": False,
            "graph": NxMixedGraph.from_edges(
                directed=[(z, x), (x, w), (w, y)],
                undirected=[(z, w), (x, y), (z, x)],
            ),
            "outcomes": {y},
            "treatments": {x},
        },
    ]


def _build_report() -> dict[str, Any]:
    _dafny, extracted = _load_extracted()

    rows = []
    for case in _build_tex_cases():
        py = _run_python_tian_identify(case["graph"], case["outcomes"], case["treatments"])
        ex = _run_extracted_tian(
            _dafny,
            extracted,
            case["graph"],
            case["outcomes"],
            case["treatments"],
            case["name"],
        )
        rows.append(
            {
                "case": case["name"],
                "expected_identifiable_tex": case["expected_identifiable_tex"],
                "python_identifiable": bool(py["ok"]),
                "extracted_identifiable": bool(ex["ok"]),
                "python_vs_extracted_agree": bool(py["ok"]) == bool(ex["ok"]),
                "python_vs_tex_agree": bool(py["ok"]) == bool(case["expected_identifiable_tex"]),
                "extracted_vs_tex_agree": bool(ex["ok"]) == bool(case["expected_identifiable_tex"]),
                "python_expr_or_reason": py.get("expr", py.get("reason")),
                "extracted_ir_type": ex.get("ir_type"),
                "python_details": py,
                "extracted_details": ex,
            }
        )

    summary = {
        "cases_run": len(rows),
        "python_extracted_disagreements": [
            r["case"] for r in rows if not r["python_vs_extracted_agree"]
        ],
        "tex_label_disagreements_python": [r["case"] for r in rows if not r["python_vs_tex_agree"]],
        "tex_label_disagreements_extracted": [
            r["case"] for r in rows if not r["extracted_vs_tex_agree"]
        ],
    }

    return {"summary": summary, "rows": rows}


def main() -> int:
    """Build and print the tex-case comparison report, optionally writing JSON output."""
    parser = argparse.ArgumentParser(
        description="Compare extracted Dafny Tian-2002 runtime against Python Tian on tex-defined cases."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON report.",
    )
    args = parser.parse_args()

    try:
        report = _build_report()
    except FileNotFoundError as error:
        sys.stderr.write(f"{error}\n")
        return 1

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
