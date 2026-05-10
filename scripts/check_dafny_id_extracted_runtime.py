#!/usr/bin/env python3
"""Smoke-check extracted Dafny Line-1 runtime and print emitted IR JSON."""

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
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_line1_extracted_py"


def main() -> int:
    """Run a one-shot extracted runtime invocation and print canonical IR JSON."""
    extracted_dir = _extracted_dir()
    if not extracted_dir.exists():
        sys.stderr.write(f"missing extracted directory: {extracted_dir}\n")
        return 1

    sys.path.insert(0, str(extracted_dir))
    import _dafny
    import IDLine1Extracted

    ordering = _dafny.SeqWithoutIsStrInference(["X", "Y"])
    ok, doc = IDLine1Extracted.default__.IDLine1ToIR(
        "smoke_line1",
        {"Y"},
        set(),
        ordering,
    )
    if not ok:
        sys.stderr.write("extracted runtime rejected line-1 smoke query\n")
        return 1

    payload = dafny_ir_doc_to_jsonable(doc)
    canonical = canonicalize_and_validate_doc(payload)
    sys.stdout.write(f"{json.dumps(canonical, indent=2, sort_keys=True)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
