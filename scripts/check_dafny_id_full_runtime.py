#!/usr/bin/env python3
"""Smoke-check consolidated extracted Dafny ID runtime and print JSON payloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extracted_dir() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_full_extracted_py"


def _dafny_seq_to_list(value: object) -> list[str]:
    if hasattr(value, "Elements"):
        return [str(item) for item in value.Elements]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    raise TypeError(f"unsupported Dafny sequence type: {type(value)!r}")


def _to_jsonable(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "_asdict"):
        as_dict = value._asdict
        if callable(as_dict):
            raw = as_dict()
            if isinstance(raw, dict):
                out: dict[str, object] = {}
                for k, v in raw.items():
                    key = k[:-1] if k.endswith("_") else k
                    key = key.replace("__", "_")
                    out[str(key)] = _to_jsonable(v)
                cname = type(value).__name__
                if cname == "IRNode_IRSum":
                    out.setdefault("tag", "sum")
                elif cname == "IRNode_IRProduct":
                    out.setdefault("tag", "product")
                elif cname == "IRNode_IRProb":
                    out.setdefault("tag", "prob")
                elif cname == "IRNode_IRFrac":
                    out.setdefault("tag", "frac")
                elif cname == "IRNode_IRFailHedge":
                    out.setdefault("tag", "fail")
                    out.setdefault("kind", "hedge")
                return out
    if hasattr(value, "Elements") and hasattr(value, "isStr"):
        elems = list(value.Elements)
        if value.isStr is True:
            return "".join(str(x) for x in elems)
        return [_to_jsonable(x) for x in elems]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(x) for x in value]
    if hasattr(value, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _run_case(
    runtime: object,
    module: object,
    *,
    graph_id: str,
    directed_edges: list[tuple[str, str]],
    undirected_edges: list[tuple[str, str]],
    all_nodes: list[str],
    outcomes: set[str],
    treatments: set[str],
    ordering: list[str],
) -> dict[str, object]:
    edge_ctor = getattr(module, "Edge_Edge")
    run = getattr(module, "default__").IDToIR

    d_edges = runtime.SeqWithoutIsStrInference([edge_ctor(a, b) for a, b in directed_edges])
    u_edges = runtime.SeqWithoutIsStrInference([edge_ctor(a, b) for a, b in undirected_edges])
    ok, doc = run(
        graph_id,
        d_edges,
        u_edges,
        runtime.SeqWithoutIsStrInference(all_nodes),
        runtime.Set(outcomes),
        runtime.Set(treatments),
        runtime.SeqWithoutIsStrInference(ordering),
    )
    if not ok:
        raise RuntimeError(f"full runtime rejected smoke case: {graph_id}")
    payload = _to_jsonable(doc)
    if not isinstance(payload, dict):
        raise TypeError("unexpected non-dict payload")
    return payload


def main() -> int:
    extracted_dir = _extracted_dir()
    if not extracted_dir.exists():
        sys.stderr.write(f"missing extracted directory: {extracted_dir}\n")
        return 1

    sys.path.insert(0, str(extracted_dir))
    import _dafny
    import IDFullExtracted

    cases = {
        "identifiable": _run_case(
            _dafny,
            IDFullExtracted,
            graph_id="full.identifiable.line1",
            directed_edges=[("X", "Y")],
            undirected_edges=[],
            all_nodes=["X", "Y"],
            outcomes={"Y"},
            treatments=set(),
            ordering=["X", "Y"],
        ),
        "line3_recursive_like": _run_case(
            _dafny,
            IDFullExtracted,
            graph_id="full.line3.recursive_like",
            directed_edges=[("Z", "X"), ("X", "Y")],
            undirected_edges=[],
            all_nodes=["Z", "X", "Y"],
            outcomes={"Y"},
            treatments={"X"},
            ordering=["Z", "X", "Y"],
        ),
        "line7_recursive_like": _run_case(
            _dafny,
            IDFullExtracted,
            graph_id="full.line7.recursive_like",
            directed_edges=[("X", "W"), ("W", "Y")],
            undirected_edges=[("X", "W"), ("W", "Y")],
            all_nodes=["X", "W", "Y"],
            outcomes={"Y"},
            treatments={"X"},
            ordering=["X", "W", "Y"],
        ),
        "hedge_fail": _run_case(
            _dafny,
            IDFullExtracted,
            graph_id="full.hedge.fail",
            directed_edges=[("X", "Y")],
            undirected_edges=[("X", "Y")],
            all_nodes=["X", "Y"],
            outcomes={"Y"},
            treatments={"X"},
            ordering=["X", "Y"],
        ),
    }

    sys.stdout.write(f"{json.dumps(cases, indent=2, sort_keys=True)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
