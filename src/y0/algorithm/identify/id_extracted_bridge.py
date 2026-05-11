"""Bridge for calling extracted Dafny ID Line-by-Line runtime."""

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
from y0.algorithm.identify.id_std import identify as identify_handwritten
from y0.algorithm.identify.utils import Identification
from y0.dsl import Expression, P, Product, Variable

__all__ = [
    "ExtractedFullUnavailableError",
    "ExtractedLine1UnavailableError",
    "ExtractedLine2UnavailableError",
    "ExtractedLine3UnavailableError",
    "ExtractedLine4UnavailableError",
    "ExtractedLine5UnavailableError",
    "ExtractedLine6UnavailableError",
    "ExtractedLine7UnavailableError",
    "identify_full_from_extracted",
    "identify_line1_from_extracted",
    "identify_line2_from_extracted",
    "identify_line3_from_extracted",
    "identify_line4_from_extracted",
    "identify_line5_from_extracted",
    "identify_line6_from_extracted",
    "identify_line7_from_extracted",
    "supports_query_line1",
    "supports_query_line2",
    "supports_query_line3",
    "supports_query_line4",
    "supports_query_line5",
    "supports_query_line6",
    "supports_query_line7",
]

_EXTRACTED_MODULE_NAME = "IDLine1Extracted"
_EXTRACTED_METHOD_NAME = "IDLine1ToIR"
_ENV_EXTRACTED_DIR = "Y0_DAFNY_ID_LINE1_PY_DIR"

_EXTRACTED_MODULE_NAME_FULL = "IDFullExtracted"
_EXTRACTED_METHOD_NAME_FULL = "IDToIR"
_ENV_EXTRACTED_DIR_FULL = "Y0_DAFNY_ID_FULL_PY_DIR"

_EXTRACTED_MODULE_NAME_L2 = "IDLine2Extracted"
_EXTRACTED_METHOD_NAME_L2 = "IDLine2ToIR"
_ENV_EXTRACTED_DIR_L2 = "Y0_DAFNY_ID_LINE2_PY_DIR"

_EXTRACTED_MODULE_NAME_L3 = "IDLine3Extracted"
_EXTRACTED_METHOD_NAME_L3 = "IDLine3Transform"
_ENV_EXTRACTED_DIR_L3 = "Y0_DAFNY_ID_LINE3_PY_DIR"

_EXTRACTED_MODULE_NAME_L4 = "IDLine4Extracted"
_EXTRACTED_METHOD_NAME_L4 = "IDLine4ToIR"
_ENV_EXTRACTED_DIR_L4 = "Y0_DAFNY_ID_LINE4_PY_DIR"

_EXTRACTED_MODULE_NAME_L5 = "IDLine5Extracted"
_EXTRACTED_METHOD_NAME_L5 = "IDLine5ToIR"
_ENV_EXTRACTED_DIR_L5 = "Y0_DAFNY_ID_LINE5_PY_DIR"

_EXTRACTED_MODULE_NAME_L6 = "IDLine6Extracted"
_EXTRACTED_METHOD_NAME_L6 = "IDLine6ToIR"
_ENV_EXTRACTED_DIR_L6 = "Y0_DAFNY_ID_LINE6_PY_DIR"

_EXTRACTED_MODULE_NAME_L7 = "IDLine7Extracted"
_EXTRACTED_METHOD_NAME_L7 = "IDLine7Transform"
_ENV_EXTRACTED_DIR_L7 = "Y0_DAFNY_ID_LINE7_PY_DIR"


class ExtractedLine1UnavailableError(RuntimeError):
    """Raised when extracted Line-1 runtime is unavailable."""


class ExtractedFullUnavailableError(RuntimeError):
    """Raised when consolidated extracted full runtime is unavailable."""


class ExtractedLine2UnavailableError(RuntimeError):
    """Raised when extracted Line-2 runtime is unavailable."""


class ExtractedLine3UnavailableError(RuntimeError):
    """Raised when extracted Line-3 runtime is unavailable."""


class ExtractedLine4UnavailableError(RuntimeError):
    """Raised when extracted Line-4 runtime is unavailable."""


class ExtractedLine5UnavailableError(RuntimeError):
    """Raised when extracted Line-5 runtime is unavailable."""


class ExtractedLine6UnavailableError(RuntimeError):
    """Raised when extracted Line-6 runtime is unavailable."""


class ExtractedLine7UnavailableError(RuntimeError):
    """Raised when extracted Line-7 runtime is unavailable."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_extracted_dir() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_line1_extracted_py"


def _default_extracted_dir_full() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_full_extracted_py"


def _resolve_extracted_dir() -> Path:
    value = os.environ.get(_ENV_EXTRACTED_DIR)
    return Path(value).expanduser().resolve() if value else _default_extracted_dir()


def _resolve_extracted_dir_full() -> Path:
    value = os.environ.get(_ENV_EXTRACTED_DIR_FULL)
    return Path(value).expanduser().resolve() if value else _default_extracted_dir_full()


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


def _load_extracted_module_full() -> ModuleType:
    extracted_dir = _resolve_extracted_dir_full()
    if not extracted_dir.exists():
        raise ExtractedFullUnavailableError(
            f"extracted full-runtime directory not found at {extracted_dir}. "
            "Run scripts/build_dafny_id_full_extracted.sh first."
        )
    extracted_dir_str = str(extracted_dir)
    if extracted_dir_str not in sys.path:
        sys.path.insert(0, extracted_dir_str)
    try:
        return importlib.import_module(_EXTRACTED_MODULE_NAME_FULL)
    except Exception as error:  # pragma: no cover - import errors are environment-dependent
        raise ExtractedFullUnavailableError("failed to import extracted full Dafny module") from error


def identify_full_from_extracted(
    identification: Identification,
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run consolidated extracted Dafny runtime and translate IR into a y0 expression."""
    if identification.conditions:
        raise ExtractedFullUnavailableError("query conditions are not supported by full extracted runtime")

    module = _load_extracted_module_full()
    runner = getattr(module, "default__", None)
    method = getattr(runner, _EXTRACTED_METHOD_NAME_FULL, None)
    if method is None:
        raise ExtractedFullUnavailableError(
            f"extracted module does not provide {_EXTRACTED_METHOD_NAME_FULL}"
        )

    try:
        dafny_runtime = importlib.import_module("_dafny")
    except Exception as error:  # pragma: no cover - environment-dependent import
        raise ExtractedFullUnavailableError("missing extracted Dafny runtime package _dafny") from error

    order = (
        tuple(ordering) if ordering is not None else tuple(identification.graph.topological_sort())
    )
    order_names = [variable.name for variable in order]
    outcomes = {variable.name for variable in identification.outcomes}
    treatments = {variable.name for variable in identification.treatments}

    seq_ctor = getattr(dafny_runtime, "SeqWithoutIsStrInference", None)
    set_ctor = getattr(dafny_runtime, "Set", None)
    edge_class = getattr(module, "Edge_Edge", None)
    if seq_ctor is None or set_ctor is None or edge_class is None:
        raise ExtractedFullUnavailableError(
            "_dafny sequence/set constructors or edge class are unavailable"
        )

    directed_edges: list[Any] = []
    for src, tgt in identification.graph.directed.edges():
        directed_edges.append(edge_class(src.name, tgt.name))

    undirected_edges: list[Any] = []
    for src, tgt in identification.graph.undirected.edges():
        undirected_edges.append(edge_class(src.name, tgt.name))

    ok, dafny_doc = method(
        "runtime_full",
        seq_ctor(directed_edges),
        seq_ctor(undirected_edges),
        seq_ctor(order_names),
        set_ctor(outcomes),
        set_ctor(treatments),
        seq_ctor(order_names),
    )
    if not ok:
        raise ExtractedFullUnavailableError("extracted full runtime rejected query")

    doc_json = dafny_ir_doc_to_jsonable(dafny_doc)
    canonical = canonicalize_and_validate_doc(doc_json)
    return ir_doc_to_expression(canonical)


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


def _default_extracted_dir_l3() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_line3_extracted_py"


def _resolve_extracted_dir_l3() -> Path:
    value = os.environ.get(_ENV_EXTRACTED_DIR_L3)
    return Path(value).expanduser().resolve() if value else _default_extracted_dir_l3()


def _load_extracted_module_l3() -> ModuleType:
    extracted_dir = _resolve_extracted_dir_l3()
    if not extracted_dir.exists():
        raise ExtractedLine3UnavailableError(
            f"extracted line-3 directory not found at {extracted_dir}. "
            "Run scripts/build_dafny_id_line3_extracted.sh first."
        )
    extracted_dir_str = str(extracted_dir)
    if extracted_dir_str not in sys.path:
        sys.path.insert(0, extracted_dir_str)
    try:
        return importlib.import_module(_EXTRACTED_MODULE_NAME_L3)
    except Exception as error:  # pragma: no cover - import errors are environment-dependent
        raise ExtractedLine3UnavailableError("failed to import extracted Dafny module") from error


def supports_query_line3(identification: Identification) -> bool:
    """Return true when query can be handled by extracted Line-3 transform."""
    if not identification.treatments or identification.conditions:
        return False

    graph = identification.graph
    nodes = set(graph.nodes())
    if not nodes:
        return False

    outcomes_and_ancestors = graph.ancestors_inclusive(set(identification.outcomes))
    if nodes != outcomes_and_ancestors:
        return False

    no_effect = graph.get_no_effect_on_outcomes(identification.treatments, identification.outcomes)
    return bool(no_effect)


def identify_line3_from_extracted(
    identification: Identification,
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run extracted Dafny Line-3 transform, then recurse with handwritten identify."""
    if not supports_query_line3(identification):
        raise ExtractedLine3UnavailableError("query is not a supported Line-3 form")

    module = _load_extracted_module_l3()
    runner = getattr(module, "default__", None)
    method = getattr(runner, _EXTRACTED_METHOD_NAME_L3, None)
    if method is None:
        raise ExtractedLine3UnavailableError(
            f"extracted module does not provide {_EXTRACTED_METHOD_NAME_L3}"
        )

    try:
        dafny_runtime = importlib.import_module("_dafny")
    except Exception as error:  # pragma: no cover - environment-dependent import
        raise ExtractedLine3UnavailableError(
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
        raise ExtractedLine3UnavailableError("_dafny sequence/set constructors unavailable")

    edge_class = getattr(module, "Edge_Edge", None)
    if edge_class is None:
        raise ExtractedLine3UnavailableError("Edge class not found in extracted module")

    directed_edges: list[Any] = []
    for src, tgt in identification.graph.directed.edges():
        directed_edges.append(edge_class(src.name, tgt.name))

    all_nodes = [variable.name for variable in identification.graph.nodes()]

    ok, no_effect_nodes_raw, expanded_treatments_raw = method(
        seq_ctor(directed_edges),
        seq_ctor(all_nodes),
        set_ctor(outcomes),
        set_ctor(treatments),
        seq_ctor(order_names),
    )
    if not ok:
        raise ExtractedLine3UnavailableError(
            "extracted runtime rejected query (Line 3 does not apply)"
        )

    no_effect_nodes = {Variable(name) for name in _dafny_seq_to_str_list(no_effect_nodes_raw)}
    if not no_effect_nodes:
        raise ExtractedLine3UnavailableError("line-3 transform returned empty no-effect set")

    expanded_treatments = {
        Variable(name) for name in _dafny_seq_to_str_list(expanded_treatments_raw)
    }
    if not expanded_treatments:
        raise ExtractedLine3UnavailableError("line-3 transform returned empty treatment set")

    transformed_identification = Identification.from_parts(
        outcomes=identification.outcomes,
        treatments=expanded_treatments,
        estimand=identification.estimand,
        graph=identification.graph,
    )
    return identify_handwritten(transformed_identification, ordering=order)


def _default_extracted_dir_l5() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_line5_extracted_py"


def _resolve_extracted_dir_l5() -> Path:
    value = os.environ.get(_ENV_EXTRACTED_DIR_L5)
    return Path(value).expanduser().resolve() if value else _default_extracted_dir_l5()


def _load_extracted_module_l5() -> ModuleType:
    extracted_dir = _resolve_extracted_dir_l5()
    if not extracted_dir.exists():
        raise ExtractedLine5UnavailableError(
            f"extracted line-5 directory not found at {extracted_dir}. "
            "Run scripts/build_dafny_id_line5_extracted.sh first."
        )
    extracted_dir_str = str(extracted_dir)
    if extracted_dir_str not in sys.path:
        sys.path.insert(0, extracted_dir_str)
    try:
        return importlib.import_module(_EXTRACTED_MODULE_NAME_L5)
    except Exception as error:  # pragma: no cover - import errors are environment-dependent
        raise ExtractedLine5UnavailableError("failed to import extracted Dafny module") from error


def supports_query_line5(identification: Identification) -> bool:
    """Return true when query can be handled by extracted Line-5 runtime.

    Line 5 applies only after lines 1-4 fail to apply. We enforce those preconditions
    conservatively to keep generated behavior aligned with handwritten ID.
    """
    if not identification.treatments or identification.conditions:
        return False

    graph = identification.graph
    nodes = set(graph.nodes())
    if not nodes:
        return False

    if graph.get_no_effect_on_outcomes(identification.treatments, identification.outcomes):
        return False

    graph_without_treatments = graph.remove_nodes_from(identification.treatments)
    districts_without_treatment = graph_without_treatments.districts()
    if len(districts_without_treatment) != 1:
        return False

    return graph.districts() == {frozenset(nodes)}


def identify_line5_from_extracted(
    identification: Identification,
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run extracted Dafny Line-5 runtime and translate its hedge IR."""
    if not supports_query_line5(identification):
        raise ExtractedLine5UnavailableError("query is not a supported Line-5 form")

    module = _load_extracted_module_l5()
    runner = getattr(module, "default__", None)
    method = getattr(runner, _EXTRACTED_METHOD_NAME_L5, None)
    if method is None:
        raise ExtractedLine5UnavailableError(
            f"extracted module does not provide {_EXTRACTED_METHOD_NAME_L5}"
        )

    try:
        dafny_runtime = importlib.import_module("_dafny")
    except Exception as error:  # pragma: no cover - environment-dependent import
        raise ExtractedLine5UnavailableError(
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
        raise ExtractedLine5UnavailableError("_dafny sequence/set constructors unavailable")

    edge_class = getattr(module, "Edge_Edge", None)
    if edge_class is None:
        raise ExtractedLine5UnavailableError("Edge class not found in extracted module")

    undirected_edges: list[Any] = []
    for src, tgt in identification.graph.undirected.edges():
        undirected_edges.append(edge_class(src.name, tgt.name))

    all_nodes = [variable.name for variable in identification.graph.nodes()]

    ok, dafny_doc = method(
        "runtime_line5",
        seq_ctor(undirected_edges),
        seq_ctor(all_nodes),
        set_ctor(outcomes),
        set_ctor(treatments),
        seq_ctor(order_names),
    )
    if not ok:
        raise ExtractedLine5UnavailableError(
            "extracted runtime rejected query (Line 5 does not apply)"
        )

    doc_json = dafny_ir_doc_to_jsonable(dafny_doc)
    canonical = canonicalize_and_validate_doc(doc_json)
    return ir_doc_to_expression(canonical)


def _default_extracted_dir_l4() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_line4_extracted_py"


def _resolve_extracted_dir_l4() -> Path:
    value = os.environ.get(_ENV_EXTRACTED_DIR_L4)
    return Path(value).expanduser().resolve() if value else _default_extracted_dir_l4()


def _load_extracted_module_l4() -> ModuleType:
    extracted_dir = _resolve_extracted_dir_l4()
    if not extracted_dir.exists():
        raise ExtractedLine4UnavailableError(
            f"extracted line-4 directory not found at {extracted_dir}. "
            "Run scripts/build_dafny_id_line4_extracted.sh first."
        )
    extracted_dir_str = str(extracted_dir)
    if extracted_dir_str not in sys.path:
        sys.path.insert(0, extracted_dir_str)
    try:
        return importlib.import_module(_EXTRACTED_MODULE_NAME_L4)
    except Exception as error:  # pragma: no cover - import errors are environment-dependent
        raise ExtractedLine4UnavailableError("failed to import extracted Dafny module") from error


def supports_query_line4(identification: Identification) -> bool:
    """Return true when query can be handled by extracted Line-4 runtime.

    This implementation intentionally targets a conservative frontdoor-small
    decomposition shape and otherwise falls back to handwritten logic.
    """
    if not identification.treatments or identification.conditions:
        return False

    graph = identification.graph
    nodes = set(graph.nodes())
    if len(nodes) != 3 or len(identification.outcomes) != 1 or len(identification.treatments) != 1:
        return False

    outcomes_and_ancestors = graph.ancestors_inclusive(set(identification.outcomes))
    if nodes != outcomes_and_ancestors:
        return False

    if graph.get_no_effect_on_outcomes(identification.treatments, identification.outcomes):
        return False

    graph_without_treatments = graph.remove_nodes_from(identification.treatments)
    if len(graph_without_treatments.districts()) <= 1:
        return False

    treatment = next(iter(identification.treatments))
    outcome = next(iter(identification.outcomes))
    others = sorted(nodes - {treatment, outcome}, key=lambda variable: variable.name)
    if len(others) != 1:
        return False
    mediator = others[0]

    directed_edges = set(graph.directed.edges())
    return (treatment, mediator) in directed_edges and (mediator, outcome) in directed_edges


def identify_line4_from_extracted(
    identification: Identification,
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run extracted Dafny Line-4 runtime and translate its IR into a y0 expression."""
    if not supports_query_line4(identification):
        raise ExtractedLine4UnavailableError("query is not a supported Line-4 form")

    module = _load_extracted_module_l4()
    runner = getattr(module, "default__", None)
    method = getattr(runner, _EXTRACTED_METHOD_NAME_L4, None)
    if method is None:
        raise ExtractedLine4UnavailableError(
            f"extracted module does not provide {_EXTRACTED_METHOD_NAME_L4}"
        )

    try:
        dafny_runtime = importlib.import_module("_dafny")
    except Exception as error:  # pragma: no cover - environment-dependent import
        raise ExtractedLine4UnavailableError(
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
        raise ExtractedLine4UnavailableError("_dafny sequence/set constructors unavailable")

    edge_class = getattr(module, "Edge_Edge", None)
    if edge_class is None:
        raise ExtractedLine4UnavailableError("Edge class not found in extracted module")

    directed_edges: list[Any] = []
    for src, tgt in identification.graph.directed.edges():
        directed_edges.append(edge_class(src.name, tgt.name))

    undirected_edges: list[Any] = []
    for src, tgt in identification.graph.undirected.edges():
        undirected_edges.append(edge_class(src.name, tgt.name))

    all_nodes = [variable.name for variable in identification.graph.nodes()]

    ok, dafny_doc = method(
        "runtime_line4",
        seq_ctor(directed_edges),
        seq_ctor(undirected_edges),
        seq_ctor(all_nodes),
        set_ctor(outcomes),
        set_ctor(treatments),
        seq_ctor(order_names),
    )
    if not ok:
        raise ExtractedLine4UnavailableError(
            "extracted runtime rejected query (Line 4 does not apply)"
        )

    doc_json = dafny_ir_doc_to_jsonable(dafny_doc)
    canonical = canonicalize_and_validate_doc(doc_json)
    return ir_doc_to_expression(canonical)


def _default_extracted_dir_l6() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_line6_extracted_py"


def _resolve_extracted_dir_l6() -> Path:
    value = os.environ.get(_ENV_EXTRACTED_DIR_L6)
    return Path(value).expanduser().resolve() if value else _default_extracted_dir_l6()


def _load_extracted_module_l6() -> ModuleType:
    extracted_dir = _resolve_extracted_dir_l6()
    if not extracted_dir.exists():
        raise ExtractedLine6UnavailableError(
            f"extracted line-6 directory not found at {extracted_dir}. "
            "Run scripts/build_dafny_id_line6_extracted.sh first."
        )
    extracted_dir_str = str(extracted_dir)
    if extracted_dir_str not in sys.path:
        sys.path.insert(0, extracted_dir_str)
    try:
        return importlib.import_module(_EXTRACTED_MODULE_NAME_L6)
    except Exception as error:  # pragma: no cover - import errors are environment-dependent
        raise ExtractedLine6UnavailableError("failed to import extracted Dafny module") from error


def supports_query_line6(identification: Identification) -> bool:
    """Return true when query can be handled by extracted Line-6 runtime."""
    if not identification.treatments or identification.conditions:
        return False

    graph = identification.graph
    nodes = set(graph.nodes())
    if not nodes:
        return False

    outcomes_and_ancestors = graph.ancestors_inclusive(set(identification.outcomes))
    if nodes != outcomes_and_ancestors:
        return False

    if graph.get_no_effect_on_outcomes(identification.treatments, identification.outcomes):
        return False

    graph_without_treatments = graph.remove_nodes_from(identification.treatments)
    districts_without_treatment = graph_without_treatments.districts()
    if len(districts_without_treatment) != 1:
        return False

    if graph.districts() == {frozenset(nodes)}:
        return False

    district_without_treatment = next(iter(districts_without_treatment))
    return district_without_treatment in graph.districts()


def identify_line6_from_extracted(
    identification: Identification,
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run extracted Dafny Line-6 runtime and translate its IR into a y0 expression."""
    if not supports_query_line6(identification):
        raise ExtractedLine6UnavailableError("query is not a supported Line-6 form")

    module = _load_extracted_module_l6()
    runner = getattr(module, "default__", None)
    method = getattr(runner, _EXTRACTED_METHOD_NAME_L6, None)
    if method is None:
        raise ExtractedLine6UnavailableError(
            f"extracted module does not provide {_EXTRACTED_METHOD_NAME_L6}"
        )

    try:
        dafny_runtime = importlib.import_module("_dafny")
    except Exception as error:  # pragma: no cover - environment-dependent import
        raise ExtractedLine6UnavailableError(
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
        raise ExtractedLine6UnavailableError("_dafny sequence/set constructors unavailable")

    edge_class = getattr(module, "Edge_Edge", None)
    if edge_class is None:
        raise ExtractedLine6UnavailableError("Edge class not found in extracted module")

    undirected_edges: list[Any] = []
    for src, tgt in identification.graph.undirected.edges():
        undirected_edges.append(edge_class(src.name, tgt.name))

    all_nodes = [variable.name for variable in identification.graph.nodes()]

    ok, dafny_doc = method(
        "runtime_line6",
        seq_ctor(undirected_edges),
        seq_ctor(all_nodes),
        set_ctor(outcomes),
        set_ctor(treatments),
        seq_ctor(order_names),
    )
    if not ok:
        raise ExtractedLine6UnavailableError(
            "extracted runtime rejected query (Line 6 does not apply)"
        )

    doc_json = dafny_ir_doc_to_jsonable(dafny_doc)
    canonical = canonicalize_and_validate_doc(doc_json)
    return ir_doc_to_expression(canonical)


def _default_extracted_dir_l7() -> Path:
    return _repo_root() / ".cache" / "y0" / "dafny" / "id_line7_extracted_py"


def _resolve_extracted_dir_l7() -> Path:
    value = os.environ.get(_ENV_EXTRACTED_DIR_L7)
    return Path(value).expanduser().resolve() if value else _default_extracted_dir_l7()


def _load_extracted_module_l7() -> ModuleType:
    extracted_dir = _resolve_extracted_dir_l7()
    if not extracted_dir.exists():
        raise ExtractedLine7UnavailableError(
            f"extracted line-7 directory not found at {extracted_dir}. "
            "Run scripts/build_dafny_id_line7_extracted.sh first."
        )
    extracted_dir_str = str(extracted_dir)
    if extracted_dir_str not in sys.path:
        sys.path.insert(0, extracted_dir_str)
    try:
        return importlib.import_module(_EXTRACTED_MODULE_NAME_L7)
    except Exception as error:  # pragma: no cover - import errors are environment-dependent
        raise ExtractedLine7UnavailableError("failed to import extracted Dafny module") from error


def _district_product(district: set[Variable], ordering: Sequence[Variable]) -> Expression:
    return Product.safe(P(variable | ordering[: ordering.index(variable)]) for variable in district)


def _dafny_seq_to_str_list(value: Any) -> list[str]:
    if hasattr(value, "Elements"):
        return [str(item) for item in value.Elements]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    raise TypeError(f"unsupported Dafny sequence type: {type(value)!r}")


def supports_query_line7(identification: Identification) -> bool:
    """Return true when query can be handled by extracted Line-7 transform."""
    if not identification.treatments or identification.conditions:
        return False

    graph = identification.graph
    nodes = set(graph.nodes())
    if not nodes:
        return False

    outcomes_and_ancestors = graph.ancestors_inclusive(set(identification.outcomes))
    if nodes != outcomes_and_ancestors:
        return False

    if graph.get_no_effect_on_outcomes(identification.treatments, identification.outcomes):
        return False

    graph_without_treatments = graph.remove_nodes_from(identification.treatments)
    districts_without_treatment = graph_without_treatments.districts()
    if len(districts_without_treatment) != 1:
        return False

    if graph.districts() == {frozenset(nodes)}:
        return False

    district_without_treatment = next(iter(districts_without_treatment))
    if district_without_treatment in graph.districts():
        return False

    return any(district_without_treatment < district for district in graph.districts())


def identify_line7_from_extracted(
    identification: Identification,
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run extracted Dafny Line-7 transform, then recurse with handwritten identify."""
    if not supports_query_line7(identification):
        raise ExtractedLine7UnavailableError("query is not a supported Line-7 form")

    module = _load_extracted_module_l7()
    runner = getattr(module, "default__", None)
    method = getattr(runner, _EXTRACTED_METHOD_NAME_L7, None)
    if method is None:
        raise ExtractedLine7UnavailableError(
            f"extracted module does not provide {_EXTRACTED_METHOD_NAME_L7}"
        )

    try:
        dafny_runtime = importlib.import_module("_dafny")
    except Exception as error:  # pragma: no cover - environment-dependent import
        raise ExtractedLine7UnavailableError(
            "missing extracted Dafny runtime package _dafny"
        ) from error

    order = (
        tuple(ordering) if ordering is not None else tuple(identification.graph.topological_sort())
    )
    order_names = [variable.name for variable in order]
    treatments = {variable.name for variable in identification.treatments}

    seq_ctor = getattr(dafny_runtime, "SeqWithoutIsStrInference", None)
    set_ctor = getattr(dafny_runtime, "Set", None)
    if seq_ctor is None or set_ctor is None:
        raise ExtractedLine7UnavailableError("_dafny sequence/set constructors unavailable")

    edge_class = getattr(module, "Edge_Edge", None)
    if edge_class is None:
        raise ExtractedLine7UnavailableError("Edge class not found in extracted module")

    undirected_edges: list[Any] = []
    for src, tgt in identification.graph.undirected.edges():
        undirected_edges.append(edge_class(src.name, tgt.name))

    all_nodes = [variable.name for variable in identification.graph.nodes()]

    ok, district_nodes_raw, reduced_treatments_raw = method(
        seq_ctor(undirected_edges),
        seq_ctor(all_nodes),
        set_ctor(treatments),
        seq_ctor(order_names),
    )
    if not ok:
        raise ExtractedLine7UnavailableError(
            "extracted runtime rejected query (Line 7 does not apply)"
        )

    district_nodes = {Variable(name) for name in _dafny_seq_to_str_list(district_nodes_raw)}
    reduced_treatments = {Variable(name) for name in _dafny_seq_to_str_list(reduced_treatments_raw)}
    filtered_order = tuple(variable for variable in order if variable in district_nodes)
    if not filtered_order:
        raise ExtractedLine7UnavailableError("line-7 transform produced empty district ordering")

    transformed_identification = Identification.from_parts(
        outcomes=identification.outcomes,
        treatments=reduced_treatments,
        estimand=_district_product(district_nodes, filtered_order),
        graph=identification.graph.subgraph(district_nodes),
    )
    return identify_handwritten(transformed_identification, ordering=filtered_order)
