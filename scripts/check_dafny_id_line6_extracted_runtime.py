#!/usr/bin/env python3
"""Smoke-check extracted Dafny Line-6 runtime and print emitted IR JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from y0.algorithm.identify.id_ir_to_dsl import (
    canonicalize_and_validate_doc,
    dafny_ir_doc_to_jsonable,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extracted_dir() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_line6_extracted_py"


def main() -> int:
    """Run a one-shot extracted runtime invocation and print canonical IR JSON."""
    extracted_dir = _extracted_dir()
    if not extracted_dir.exists():
        sys.stderr.write(f"missing extracted directory: {extracted_dir}\n")
        return 1

    sys.path.insert(0, str(extracted_dir))
    import _dafny
    import IDLine6Extracted
    from IDLine6Extracted import Edge_Edge

    undirected_edges = _dafny.SeqWithoutIsStrInference(
        [
            Edge_Edge("Z", "Y"),
        ]
    )
    all_nodes = _dafny.SeqWithoutIsStrInference(["X", "Z", "Y"])
    outcomes = _dafny.Set({"Y"})
    treatments = _dafny.Set({"X"})
    ordering = _dafny.SeqWithoutIsStrInference(["X", "Z", "Y"])

    ok, doc = IDLine6Extracted.default__.IDLine6ToIR(
        "smoke_line6",
        undirected_edges,
        all_nodes,
        outcomes,
        treatments,
        ordering,
    )
    if not ok:
        sys.stderr.write("extracted runtime rejected line-6 smoke query\n")
        return 1

    payload = dafny_ir_doc_to_jsonable(doc)
    canonical = canonicalize_and_validate_doc(payload)
    sys.stdout.write(f"{json.dumps(canonical, indent=2, sort_keys=True)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
