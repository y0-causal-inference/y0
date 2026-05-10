#!/usr/bin/env python3
"""Smoke-check extracted Dafny Line-7 transform and print JSON payload."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extracted_dir() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_line7_extracted_py"


def _dafny_seq_to_list(value: object) -> list[str]:
    if hasattr(value, "Elements"):
        return [str(item) for item in value.Elements]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    raise TypeError(f"unsupported Dafny sequence type: {type(value)!r}")


def main() -> int:
    """Run one line-7 transform invocation and print transform JSON."""
    extracted_dir = _extracted_dir()
    if not extracted_dir.exists():
        sys.stderr.write(f"missing extracted directory: {extracted_dir}\n")
        return 1

    sys.path.insert(0, str(extracted_dir))
    import _dafny
    import IDLine7Extracted
    from IDLine7Extracted import Edge_Edge

    undirected_edges = _dafny.SeqWithoutIsStrInference([
        Edge_Edge("X", "W"),
        Edge_Edge("W", "Y"),
    ])
    all_nodes = _dafny.SeqWithoutIsStrInference(["X", "W", "Y"])
    treatments = _dafny.Set({"X"})
    ordering = _dafny.SeqWithoutIsStrInference(["X", "W", "Y"])

    ok, district_nodes, reduced_treatments = IDLine7Extracted.default__.IDLine7Transform(
        undirected_edges,
        all_nodes,
        treatments,
        ordering,
    )
    if not ok:
        sys.stderr.write("extracted runtime rejected line-7 smoke transform\n")
        return 1

    payload = {
        "ok": bool(ok),
        "district_nodes": _dafny_seq_to_list(district_nodes),
        "reduced_treatments": _dafny_seq_to_list(reduced_treatments),
    }
    sys.stdout.write(f"{json.dumps(payload, indent=2, sort_keys=True)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())