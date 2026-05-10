"""Bridge for calling extracted Dafny ID Line-1 runtime."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from y0.algorithm.identify.id_ir_to_dsl import (
    canonicalize_and_validate_doc,
    dafny_ir_doc_to_jsonable,
    ir_doc_to_expression,
)
from y0.algorithm.identify.utils import Identification
from y0.dsl import Expression, Variable

__all__ = [
    "ExtractedLine1UnavailableError",
    "ExtractedLine2UnavailableError",
    "identify_line1_from_extracted",
    "identify_line2_from_extracted",
    "supports_query_line1",
    "supports_query_line2",
]

_EXTRACTED_MODULE_NAME = "IDLine1Extracted"
_EXTRACTED_METHOD_NAME = "IDLine1ToIR"
_ENV_EXTRACTED_DIR = "Y0_DAFNY_ID_LINE1_PY_DIR"

_EXTRACTED_MODULE_NAME_L2 = "IDLine2Extracted"
_EXTRACTED_METHOD_NAME_L2 = "IDLine2ToIR"
_ENV_EXTRACTED_DIR_L2 = "Y0_DAFNY_ID_LINE2_PY_DIR"


class ExtractedLine1UnavailableError(RuntimeError):
    """Raised when extracted Line-1 runtime is unavailable."""


class ExtractedLine2UnavailableError(RuntimeError):
    """Raised when extracted Line-2 runtime is unavailable."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_extracted_dir() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_line1_extracted_py"


def _resolve_extracted_dir() -> Path:
    value = os.environ.get(_ENV_EXTRACTED_DIR)
    return Path(value).expanduser().resolve() if value else _default_extracted_dir()


def _load_extracted_module() -> ModuleType:
    extracted_dir = _resolve_extracted_dir()
    if not extracted_dir.exists():
        raise ExtractedLine1UnavailableError(
            f"extracted line-1 directory not found at {extracted_dir}. "
            "Run scripts/build_dafny_id_extracted.sh first."
        )
    extracted_dir_str = str(extracted_dir)
    if extracted_dir_str not in sys.path:
        sys.path.insert(0, extracted_dir_str)
    try:
        return importlib.import_module(_EXTRACTED_MODULE_NAME)
    except Exception as error:  # pragma: no cover - import errors are environment-dependent
        raise ExtractedLine1UnavailableError("failed to import extracted Dafny module") from error


def supports_query_line1(identification: Identification) -> bool:
    """Return true when query can be handled by extracted Line-1 runtime."""
    return not identification.treatments and not identification.conditions


def identify_line1_from_extracted(
    identification: Identification,
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run extracted Dafny Line-1 runtime and translate its IR into a y0 expression."""
    if not supports_query_line1(identification):
        raise ExtractedLine1UnavailableError("query is not a supported Line-1 form")

    module = _load_extracted_module()
    runner = getattr(module, "default__", None)
    method = getattr(runner, _EXTRACTED_METHOD_NAME, None)
    if method is None:
        raise ExtractedLine1UnavailableError(
            f"extracted module does not provide {_EXTRACTED_METHOD_NAME}"
        )

    try:
        dafny_runtime = importlib.import_module("_dafny")
    except Exception as error:  # pragma: no cover - environment-dependent import
        raise ExtractedLine1UnavailableError(
            "missing extracted Dafny runtime package _dafny"
        ) from error

    order = (
        tuple(ordering) if ordering is not None else tuple(identification.graph.topological_sort())
    )
    order_names = [variable.name for variable in order]
    outcomes = {variable.name for variable in identification.outcomes}
    treatments = {variable.name for variable in identification.treatments}

    seq_ctor = getattr(dafny_runtime, "SeqWithoutIsStrInference", None)
    if seq_ctor is None:
        raise ExtractedLine1UnavailableError("_dafny.SeqWithoutIsStrInference is unavailable")

    ok, dafny_doc = method(
        "runtime_line1",
        outcomes,
        treatments,
        seq_ctor(order_names),
    )
    if not ok:
        raise ExtractedLine1UnavailableError("extracted runtime rejected query")

    doc_json = dafny_ir_doc_to_jsonable(dafny_doc)
    canonical = canonicalize_and_validate_doc(doc_json)
    return ir_doc_to_expression(canonical)


def _default_extracted_dir_l2() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_line2_extracted_py"


def _resolve_extracted_dir_l2() -> Path:
    value = os.environ.get(_ENV_EXTRACTED_DIR_L2)
    return Path(value).expanduser().resolve() if value else _default_extracted_dir_l2()


def _load_extracted_module_l2() -> ModuleType:
    extracted_dir = _resolve_extracted_dir_l2()
    if not extracted_dir.exists():
        raise ExtractedLine2UnavailableError(
            f"extracted line-2 directory not found at {extracted_dir}. "
            "Run scripts/build_dafny_id_line2_extracted.sh first."
        )
    extracted_dir_str = str(extracted_dir)
    if extracted_dir_str not in sys.path:
        sys.path.insert(0, extracted_dir_str)
    try:
        return importlib.import_module(_EXTRACTED_MODULE_NAME_L2)
    except Exception as error:  # pragma: no cover - import errors are environment-dependent
        raise ExtractedLine2UnavailableError("failed to import extracted Dafny module") from error


def supports_query_line2(identification: Identification) -> bool:
    """Return true when query can be handled by extracted Line-2 runtime.

    Line 2 applies when V ≠ An(Y)_G (there are variables not in the ancestor set of Y).
    This means the graph can be simplified to just ancestors of Y.
    """
    outcomes_set = set(identification.outcomes)
    ancestors = identification.graph.ancestors_inclusive(outcomes_set)
    all_nodes = set(identification.graph.nodes())
    return all_nodes != ancestors


def identify_line2_from_extracted(
    identification: Identification,
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run extracted Dafny Line-2 runtime and translate its IR into a y0 expression."""
    if not supports_query_line2(identification):
        raise ExtractedLine2UnavailableError("query is not a supported Line-2 form")

    module = _load_extracted_module_l2()
    runner = getattr(module, "default__", None)
    method = getattr(runner, _EXTRACTED_METHOD_NAME_L2, None)
    if method is None:
        raise ExtractedLine2UnavailableError(
            f"extracted module does not provide {_EXTRACTED_METHOD_NAME_L2}"
        )

    try:
        dafny_runtime = importlib.import_module("_dafny")
    except Exception as error:  # pragma: no cover - environment-dependent import
        raise ExtractedLine2UnavailableError(
            "missing extracted Dafny runtime package _dafny"
        ) from error

    order = (
        tuple(ordering) if ordering is not None else tuple(identification.graph.topological_sort())
    )
    order_names = [variable.name for variable in order]
    outcomes = {variable.name for variable in identification.outcomes}
    treatments = {variable.name for variable in identification.treatments}

    seq_ctor = getattr(dafny_runtime, "SeqWithoutIsStrInference", None)
    set_ctor = getattr(dafny_runtime, "Set", None)
    if seq_ctor is None or set_ctor is None:
        raise ExtractedLine2UnavailableError("_dafny sequence/set constructors unavailable")

    # Convert graph edges to Dafny Edge objects
    edges: list[Any] = []
    for src, tgt in identification.graph.directed.edges():
        edge_class = getattr(module, "Edge_Edge", None)
        if edge_class is None:
            raise ExtractedLine2UnavailableError("Edge class not found in extracted module")
        edges.append(edge_class(src, tgt))

    all_nodes = list(identification.graph.nodes())

    ok, dafny_doc = method(
        "runtime_line2",
        seq_ctor(edges),
        seq_ctor(all_nodes),
        set_ctor(outcomes),
        set_ctor(treatments),
        seq_ctor(order_names),
    )
    if not ok:
        raise ExtractedLine2UnavailableError(
            "extracted runtime rejected query (Line 2 does not apply)"
        )

    doc_json = dafny_ir_doc_to_jsonable(dafny_doc)
    canonical = canonicalize_and_validate_doc(doc_json)
    return ir_doc_to_expression(canonical)
