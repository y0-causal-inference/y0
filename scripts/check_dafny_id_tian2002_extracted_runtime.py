#!/usr/bin/env python3
"""Smoke-check extracted Dafny Tian-2002 runtime vs existing Python Tian implementation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from y0.algorithm.tian_id import identify_district_variables
from y0.dsl import Expression, P, Sum, X, Y
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
            "factors": [_ir_node_to_jsonable(f) for f in factors.Elements],
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


def main() -> int:
    """Run smoke checks comparing extracted Tian CIdentify with Python implementation."""
    try:
        _dafny, extracted = _load_extracted()
    except FileNotFoundError as error:
        sys.stderr.write(f"{error}\n")
        return 1

    graph = NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[(X, Y)])
    ordering = [X, Y]
    q_t = P(X, Y)

    # Existing Tian Python implementation (c-identify equivalent entry)
    py_fail = identify_district_variables(
        input_variables={Y},
        input_district={X, Y},
        district_probability=q_t,
        graph=graph,
        ordering=ordering,
    )
    py_success = identify_district_variables(
        input_variables={X},
        input_district={X, Y},
        district_probability=q_t,
        graph=graph,
        ordering=ordering,
    )

    edge_ctor = extracted.Edge_Edge
    ir_prob_ctor = extracted.IRNode_IRProb

    directed_edges = _dafny.SeqWithoutIsStrInference([edge_ctor("X", "Y")])
    undirected_edges = _dafny.SeqWithoutIsStrInference([edge_ctor("X", "Y")])
    dafny_ordering = _dafny.SeqWithoutIsStrInference(["X", "Y"])
    q_t_ir = ir_prob_ctor(
        _dafny.SeqWithoutIsStrInference(["X", "Y"]),
        _dafny.SeqWithoutIsStrInference([]),
        _dafny.SeqWithoutIsStrInference([]),
    )

    ex_fail_ok, ex_fail_value, ex_fail_f, ex_fail_fp = extracted.default__.CIdentify(
        directed_edges,
        undirected_edges,
        dafny_ordering,
        {"Y"},
        {"X", "Y"},
        q_t_ir,
        8,
    )

    ex_success_ok, ex_success_value, ex_success_f, ex_success_fp = extracted.default__.CIdentify(
        directed_edges,
        undirected_edges,
        dafny_ordering,
        {"X"},
        {"X", "Y"},
        q_t_ir,
        8,
    )

    expected_success_expr = Sum.safe(P(X, Y), [Y])
    payload = {
        "python": {
            "fail_case": {
                "is_fail": py_fail is None,
                "value": None if py_fail is None else py_fail.to_text(),
            },
            "success_case": {
                "is_fail": py_success is None,
                "value": None if py_success is None else py_success.to_text(),
                "matches_expected": isinstance(py_success, Expression)
                and py_success == expected_success_expr,
            },
        },
        "extracted": {
            "fail_case": {
                "is_fail": not bool(ex_fail_ok),
                "value": _ir_node_to_jsonable(ex_fail_value),
                "fail_F": sorted(str(x) for x in ex_fail_f),
                "fail_Fprime": sorted(str(x) for x in ex_fail_fp),
            },
            "success_case": {
                "is_fail": not bool(ex_success_ok),
                "value": _ir_node_to_jsonable(ex_success_value),
                "fail_F": sorted(str(x) for x in ex_success_f),
                "fail_Fprime": sorted(str(x) for x in ex_success_fp),
            },
        },
    }

    checks = {
        "fail_case_agreement": (py_fail is None) and (not bool(ex_fail_ok)),
        "success_case_agreement": isinstance(py_success, Expression)
        and py_success == expected_success_expr
        and bool(ex_success_ok)
        and payload["extracted"]["success_case"]["value"]["type"] == "sum"
        and payload["extracted"]["success_case"]["value"]["over"] == ["Y"],
    }
    payload["checks"] = checks
    payload["all_checks_passed"] = all(checks.values())

    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if payload["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
