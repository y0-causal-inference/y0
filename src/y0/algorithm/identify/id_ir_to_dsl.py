"""Translate validated ID IR documents into y0 DSL expressions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from y0.algorithm.identify.utils import Unidentifiable
from y0.dsl import Distribution, Expression, P, Product, Sum, Variable

from .id_ir import canonicalize_ir_document

__all__ = [
    "build_oracle_case_from_doc",
    "canonicalize_and_validate_doc",
    "dafny_ir_doc_to_jsonable",
    "ir_doc_to_expression",
]


def _to_jsonable(value: object) -> object:
    """Recursively convert a Python object into JSON-compatible primitives."""
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(mapped_value) for key, mapped_value in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "_asdict"):
        as_dict = value._asdict
        if callable(as_dict):
            return _to_jsonable(as_dict())

    if hasattr(value, "__dict__"):
        return {
            key: _to_jsonable(attr_value)
            for key, attr_value in vars(value).items()
            if not key.startswith("_")
        }

    raise TypeError(f"unsupported IR value type for JSON conversion: {type(value)!r}")


def dafny_ir_doc_to_jsonable(doc: object) -> dict[str, object]:
    """Convert an extracted Dafny IRDoc object into a v1 JSON dictionary."""
    jsonable = _to_jsonable(doc)
    if not isinstance(jsonable, dict):
        raise TypeError("converted IR doc must be a dictionary")
    return jsonable


def canonicalize_and_validate_doc(doc_json: dict[str, object]) -> dict[str, object]:
    """Validate and canonicalize an ID IR document payload."""
    return canonicalize_ir_document(doc_json)


def build_oracle_case_from_doc(
    *,
    case_id: str,
    module: str,
    anchor_symbol: str,
    anchor_line: int,
    doc_json: dict[str, object],
) -> dict[str, object]:
    """Wrap one canonicalized IR document into a stable oracle fixture case."""
    canonical = canonicalize_and_validate_doc(doc_json)
    return {
        "case_id": case_id,
        "module": module,
        "anchor": {"symbol": anchor_symbol, "line": anchor_line},
        "query": canonical["query"],
        "expectation": {
            "kind": "ir",
            "value": canonical,
        },
    }


def _variables(names: Sequence[str]) -> tuple[Variable, ...]:
    return tuple(Variable(name) for name in names)


def _node_to_expression(node: Mapping[str, Any]) -> Expression:
    tag = node["tag"]

    if tag == "prob":
        children = _variables(node.get("vars", []))
        given = _variables(node.get("given", []))
        intervened = _variables(node.get("intervened", []))
        distribution = Distribution(children=frozenset(children), parents=frozenset(given))
        if intervened:
            return P(distribution, interventions=intervened)
        return P(distribution)

    if tag == "sum":
        body = _node_to_expression(node["body"])
        return Sum.safe(expression=body, ranges=_variables(node.get("over", [])))

    if tag == "product":
        factors = tuple(_node_to_expression(factor) for factor in node.get("factors", []))
        return Product.safe(factors)

    if tag == "frac":
        numer = _node_to_expression(node["numer"])
        denom = _node_to_expression(node["denom"])
        return numer / denom

    if tag == "fail":
        f_nodes = set(_variables(node.get("F_nodes", [])))
        fprime_nodes = set(_variables(node.get("Fprime_nodes", [])))
        raise Unidentifiable(f_nodes, fprime_nodes)

    raise ValueError(f"unsupported IR node tag: {tag!r}")


def ir_doc_to_expression(doc_json: Mapping[str, object]) -> Expression:
    """Translate one ID IR document into a y0 DSL expression.

    Raises:
        Unidentifiable: if the IR root is a hedge failure node.
    """
    canonical = canonicalize_ir_document(doc_json)
    result = canonical["result"]
    return _node_to_expression(result)
