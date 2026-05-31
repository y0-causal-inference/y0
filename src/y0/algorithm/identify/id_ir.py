"""Validation and canonicalization helpers for ID intermediate representation."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "IRValidationError",
    "canonicalize_ir_document",
    "validate_ir_document",
]


class IRValidationError(ValueError):
    """Raised when an ID IR payload violates schema constraints."""


_VALID_NODE_TAGS = {"sum", "product", "prob", "frac", "fail"}


def _fail(message: str) -> None:
    raise IRValidationError(message)
    return None


def _as_mapping(payload: Any, *, where: str) -> Any:
    """Ensure payload is a mapping; return as-is for further processing."""
    if not isinstance(payload, Mapping):
        _fail(f"{where} must be a mapping")
    return payload


def _as_string_list(
    values: Any, *, where: str, sort: bool = True, unique: bool = True
) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        _fail(f"{where} must be a list of strings")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            _fail(f"{where} entries must be strings")
        result.append(value)

    if unique and len(result) != len(set(result)):
        _fail(f"{where} must not contain duplicates")

    if sort:
        return sorted(set(result)) if unique else sorted(result)
    return result


def _canonicalize_query(query_payload: Any) -> dict[str, Any]:
    query = _as_mapping(query_payload, where="query")

    graph_id = query.get("graph_id")
    if not isinstance(graph_id, str) or not graph_id:
        _fail("query.graph_id must be a non-empty string")

    outcomes = _as_string_list(query.get("outcomes", []), where="query.outcomes")
    treatments = _as_string_list(query.get("treatments", []), where="query.treatments")
    ordering = _as_string_list(
        query.get("ordering", []), where="query.ordering", sort=False, unique=True
    )

    return {
        "graph_id": graph_id,
        "outcomes": outcomes,
        "treatments": treatments,
        "ordering": ordering,
    }


def _node_sort_key(node: Mapping[str, Any]) -> tuple[str, str]:
    tag = str(node["tag"])
    if tag == "prob":
        vars_key = ",".join(node.get("vars", []))
        return tag, f"{vars_key}|{json.dumps(node, sort_keys=True)}"
    return tag, json.dumps(node, sort_keys=True)


def _is_structural_zero(node: Mapping[str, Any]) -> bool:
    # The v1 ID IR does not model an explicit zero node.
    return False


def _canonicalize_fail_node(node: Mapping[str, Any], allow_fail: bool) -> dict[str, Any]:
    """Canonicalize a 'fail' node, validating its structure and fields."""
    if not allow_fail:
        _fail("fail nodes are only allowed as the root result")
    kind = node.get("kind")
    if kind != "hedge":
        _fail("fail.kind must be 'hedge'")
    f_nodes = _as_string_list(node.get("F_nodes", []), where="fail.F_nodes")
    fprime_nodes = _as_string_list(node.get("Fprime_nodes", []), where="fail.Fprime_nodes")
    return {
        "tag": "fail",
        "kind": "hedge",
        "F_nodes": f_nodes,
        "Fprime_nodes": fprime_nodes,
    }


def _canonicalize_prob_node(node: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a 'prob' node, validating variables, given, and intervened fields."""
    vars_list = _as_string_list(node.get("vars", []), where="prob.vars")
    if not vars_list:
        _fail("prob.vars must not be empty")
    given = _as_string_list(node.get("given", []), where="prob.given")
    intervened = _as_string_list(node.get("intervened", []), where="prob.intervened")
    if set(given) & set(intervened):
        _fail("prob.given and prob.intervened must be disjoint")
    return {
        "tag": "prob",
        "vars": vars_list,
        "given": given,
        "intervened": intervened,
    }


def _canonicalize_sum_node(node: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a 'sum' node, merging adjacent sums and validating fields."""
    over = _as_string_list(node.get("over", []), where="sum.over")
    if not over:
        _fail("sum.over must not be empty")
    body = _canonicalize_node(node.get("body"), allow_fail=False)
    # Merge adjacent sums to keep a canonical right-associated form.
    while body.get("tag") == "sum":
        nested_over = body["over"]
        over = sorted(set(over) | set(nested_over))
        body = body["body"]
    return {
        "tag": "sum",
        "over": over,
        "body": body,
    }


def _canonicalize_product_node(node: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a 'product' node, validating and sorting factors."""
    factors_raw: Sequence[Any] | None = node.get("factors")
    if factors_raw is None:
        _fail("product.factors must be a list")
        return {"tag": "product", "factors": []}  # for mypy
    if not isinstance(factors_raw, Sequence) or isinstance(factors_raw, (str, bytes)):
        _fail("product.factors must be a list")
        return {"tag": "product", "factors": []}  # for mypy
    factors = [_canonicalize_node(factor, allow_fail=False) for factor in factors_raw]
    if not factors:
        _fail("product.factors must contain at least one factor")
        return {"tag": "product", "factors": []}  # for mypy
    factors.sort(key=_node_sort_key)
    return {
        "tag": "product",
        "factors": factors,
    }


def _canonicalize_frac_node(node: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a 'frac' node, validating numerator and denominator."""
    numer = _canonicalize_node(node.get("numer"), allow_fail=False)
    denom = _canonicalize_node(node.get("denom"), allow_fail=False)
    if denom.get("tag") == "fail" or _is_structural_zero(denom):
        _fail("frac.denom cannot be structurally zero or fail")
    return {
        "tag": "frac",
        "numer": numer,
        "denom": denom,
    }


def _canonicalize_node(node_payload: Any, *, allow_fail: bool) -> dict[str, Any]:
    """Canonicalize a node in the IR, dispatching to the appropriate handler by tag."""
    node = _as_mapping(node_payload, where="node")
    tag = node.get("tag")
    if tag not in _VALID_NODE_TAGS:
        _fail(f"unknown node tag: {tag!r}")
        return {}  # for mypy
    if tag == "fail":
        return _canonicalize_fail_node(node, allow_fail)
    if tag == "prob":
        return _canonicalize_prob_node(node)
    if tag == "sum":
        return _canonicalize_sum_node(node)
    if tag == "product":
        return _canonicalize_product_node(node)
    if tag == "frac":
        return _canonicalize_frac_node(node)

    _fail(f"unsupported node tag: {tag!r}")
    return {}  # for mypy


def validate_ir_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize one ID IR document."""
    doc = _as_mapping(copy.deepcopy(payload), where="document")

    version = doc.get("version")
    if not isinstance(version, str) or not version:
        _fail("version must be a non-empty string")

    engine = doc.get("engine")
    if engine != "id":
        _fail("engine must be 'id'")

    query = _canonicalize_query(doc.get("query"))
    result = _canonicalize_node(doc.get("result"), allow_fail=True)

    return {
        "version": version,
        "engine": "id",
        "query": query,
        "result": result,
    }


def canonicalize_ir_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize one ID IR document after validating its schema."""
    return validate_ir_document(payload)
