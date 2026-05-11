"""Parity checks between handwritten and generated ID engines."""

from __future__ import annotations

import pytest

from y0.algorithm.identify import Identification, Unidentifiable
from y0.algorithm.identify import id_generated as id_generated_module
from y0.algorithm.identify.id_dispatch import identify_with_engine
from y0.dsl import Expression, P, Probability, Variable
from y0.graph import NxMixedGraph
from y0.mutate.canonicalize_expr import canonical_expr_equal, canonicalize


def _identifiable_case() -> tuple[NxMixedGraph, Probability]:
    x = Variable("X")
    y = Variable("Y")
    graph = NxMixedGraph.from_edges(directed=[(x, y)])
    query = P(y @ ~x)
    return graph, query


def _unidentifiable_case() -> tuple[NxMixedGraph, Probability]:
    x = Variable("X")
    y = Variable("Y")
    graph = NxMixedGraph.from_edges(directed=[(x, y)], undirected=[(x, y)])
    query = P(y @ ~x)
    return graph, query


@pytest.fixture(autouse=True)
def _disable_full_runtime_for_legacy_tests(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Keep existing line-route tests deterministic under Phase 4 routing changes."""
    if "full" in request.node.name:
        return

    def _full_unavailable(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        raise id_generated_module.ExtractedFullUnavailableError("disabled for legacy test")

    monkeypatch.setattr(id_generated_module, "identify_full_from_extracted", _full_unavailable)


def test_generated_matches_handwritten_identifiable() -> None:
    """Generated and handwritten engines should agree on identifiable queries."""
    graph, query = _identifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)

    handwritten = identify_with_engine(identification, engine="handwritten")
    generated = identify_with_engine(identification, engine="generated")

    if not isinstance(generated, Expression):
        pytest.fail("generated engine did not return an Expression")

    if not canonical_expr_equal(handwritten, generated):
        pytest.fail(
            "generated and handwritten expressions differ after canonicalization: "
            f"handwritten={handwritten.to_text()} generated={generated.to_text()}"
        )


def test_generated_matches_handwritten_unidentifiable_class() -> None:
    """Generated and handwritten engines should both fail on unidentifiable queries."""
    graph, query = _unidentifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)

    with pytest.raises(Unidentifiable):
        identify_with_engine(identification, engine="handwritten")

    with pytest.raises(Unidentifiable):
        identify_with_engine(identification, engine="generated")


def test_generated_expression_canonicalization_stable() -> None:
    """Generated engine output should canonicalize stably."""
    graph, query = _identifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)
    generated = identify_with_engine(identification, engine="generated")
    ordering = sorted(generated.get_variables(), key=lambda variable: variable.name)
    once = canonicalize(generated, ordering=ordering)
    twice = canonicalize(once, ordering=ordering)
    if once != twice:
        pytest.fail("generated engine expression canonicalization is not stable")


def _flatten_variable_names(value: object) -> set[str]:
    """Extract variable names from exception payload sets with nested districts."""
    names: set[str] = set()
    if isinstance(value, set | frozenset):
        for item in value:
            names.update(_flatten_variable_names(item))
        return names
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return {name}
    return names


def test_generated_full_line3_recursive_end_to_end() -> None:
    """Full-runtime recursive line-3-like query should return a deterministic expression."""
    x = Variable("X")
    y = Variable("Y")
    z = Variable("Z")
    graph = NxMixedGraph.from_edges(
        directed=[(z, x), (x, y)],
        undirected=[(z, x)],
    )
    query = P(y @ ~x)
    identification = Identification.from_expression(graph=graph, query=query)

    try:
        once = id_generated_module.identify_full_from_extracted(identification)
        twice = id_generated_module.identify_full_from_extracted(identification)
    except id_generated_module.ExtractedFullUnavailableError:
        pytest.skip("full extracted runtime unavailable")

    if not isinstance(once, Expression):
        pytest.fail("generated full runtime line-3 case did not return an Expression")
    if not canonical_expr_equal(once, twice):
        pytest.fail("generated full runtime line-3 case is not deterministic between runs")


def test_generated_full_line7_recursive_end_to_end() -> None:
    """Full-runtime recursive line-7-like query should fail with consistent hedge witness."""
    x = Variable("X")
    w = Variable("W")
    y = Variable("Y")
    graph = NxMixedGraph.from_edges(
        directed=[(x, w), (w, y)],
        undirected=[(x, w), (w, y)],
    )
    query = P(y @ ~x)
    identification = Identification.from_expression(graph=graph, query=query)

    try:
        with pytest.raises(Unidentifiable) as generated_error:
            id_generated_module.identify_full_from_extracted(identification)
    except id_generated_module.ExtractedFullUnavailableError:
        pytest.skip("full extracted runtime unavailable")

    generated_nodes = _flatten_variable_names(generated_error.value.args[0])
    generated_witness = _flatten_variable_names(generated_error.value.args[1])
    if generated_nodes != {"X", "W", "Y"}:
        pytest.fail(f"unexpected generated line-7 node set: {generated_nodes!r}")
    if generated_witness != {"W", "Y"}:
        pytest.fail(f"unexpected generated line-7 witness set: {generated_witness!r}")


def test_generated_prefers_full_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should route through full extracted runtime before line routes."""
    graph, query = _identifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_full(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        calls.append("full")
        return P(Variable("Y") @ ~Variable("X"))

    def _unexpected_supports_line1(identification: Identification) -> bool:
        del identification
        pytest.fail("line support checks should not run when full runtime succeeds")

    monkeypatch.setattr(id_generated_module, "identify_full_from_extracted", _fake_full)
    monkeypatch.setattr(id_generated_module, "supports_query_line1", _unexpected_supports_line1)

    result = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(result, P(Variable("Y") @ ~Variable("X"))):
        pytest.fail("generated engine did not return full-runtime expression")
    if calls != ["full"]:
        pytest.fail(f"unexpected full-runtime routing calls: {calls!r}")


def test_generated_falls_back_to_line_compat_when_full_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated engine should use line compatibility routing when full runtime is unavailable."""
    x = Variable("X")
    y = Variable("Y")
    graph = NxMixedGraph.from_edges(directed=[(x, y)])
    query = P(y)
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _full_unavailable(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        calls.append("full")
        raise id_generated_module.ExtractedFullUnavailableError("missing full runtime")

    def _supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return True

    def _line1(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        calls.append("line1")
        return P(y)

    monkeypatch.setenv("Y0_DAFNY_ID_LINE_COMPAT", "1")
    monkeypatch.setattr(id_generated_module, "identify_full_from_extracted", _full_unavailable)
    monkeypatch.setattr(id_generated_module, "supports_query_line1", _supports_line1)
    monkeypatch.setattr(id_generated_module, "identify_line1_from_extracted", _line1)

    result = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(result, P(y)):
        pytest.fail("generated engine did not return line-compat expression")
    if calls != ["full", "supports_line1", "line1"]:
        pytest.fail(f"unexpected fallback routing calls: {calls!r}")


def test_generated_falls_back_to_handwritten_when_full_unavailable_and_compat_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated engine should fall back to handwritten path when compat routes are disabled."""
    graph, query = _identifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _full_unavailable(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        calls.append("full")
        raise id_generated_module.ExtractedFullUnavailableError("missing full runtime")

    def _unexpected_supports_line1(identification: Identification) -> bool:
        del identification
        pytest.fail("line compatibility should be skipped when disabled")

    monkeypatch.setenv("Y0_DAFNY_ID_LINE_COMPAT", "0")
    monkeypatch.setattr(id_generated_module, "identify_full_from_extracted", _full_unavailable)
    monkeypatch.setattr(id_generated_module, "supports_query_line1", _unexpected_supports_line1)

    handwritten = identify_with_engine(identification, engine="handwritten")
    generated = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(handwritten, generated):
        pytest.fail("generated handwritten fallback does not match handwritten expression")
    if calls != ["full"]:
        pytest.fail(f"unexpected no-compat routing calls: {calls!r}")


def test_generated_prefers_extracted_for_line1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should route line-1 queries through extracted runtime."""
    x = Variable("X")
    y = Variable("Y")
    graph = NxMixedGraph.from_edges(directed=[(x, y)])
    query = P(y)
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports")
        return True

    def _fake_extracted(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        calls.append("extracted")
        return P(y)

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "identify_line1_from_extracted", _fake_extracted)

    result = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(result, P(y)):
        pytest.fail("generated engine did not return extracted line-1 expression")
    if calls != ["supports", "extracted"]:
        pytest.fail(f"unexpected generated routing calls: {calls!r}")


def test_generated_falls_back_when_not_line1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should fallback to handwritten runtime for non-line1 queries."""
    graph, query = _identifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports")
        return False

    def _unexpected_extracted(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        pytest.fail("extracted path should not be called for non-line1 query")

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "identify_line1_from_extracted", _unexpected_extracted)

    handwritten = identify_with_engine(identification, engine="handwritten")
    generated = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(handwritten, generated):
        pytest.fail("generated fallback path does not match handwritten expression")
    if calls != ["supports"]:
        pytest.fail(f"unexpected generated fallback routing calls: {calls!r}")


def test_generated_prefers_extracted_for_line2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should route line-2 queries through extracted runtime when available."""
    # Line 2 applies when V ≠ An(Y): a graph with nodes not in the ancestral set
    x = Variable("X")
    y = Variable("Y")
    graph = NxMixedGraph.from_edges(directed=[(x, y)])
    query = P(y @ ~x)
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return True

    def _fake_extracted_line2(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        calls.append("extracted_line2")
        return P(y @ ~x)

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(id_generated_module, "identify_line2_from_extracted", _fake_extracted_line2)

    result = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(result, P(y @ ~x)):
        pytest.fail("generated engine did not return extracted line-2 expression")
    if calls != ["supports_line1", "supports_line2", "extracted_line2"]:
        pytest.fail(f"unexpected generated routing calls: {calls!r}")


def test_generated_falls_back_when_not_line2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should fallback when line 2 is not available."""
    graph, query = _identifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return False

    def _unexpected_extracted_line2(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        pytest.fail("extracted line-2 path should not be called")

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(
        id_generated_module, "identify_line2_from_extracted", _unexpected_extracted_line2
    )

    handwritten = identify_with_engine(identification, engine="handwritten")
    generated = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(handwritten, generated):
        pytest.fail("generated fallback path does not match handwritten expression")
    if calls != ["supports_line1", "supports_line2"]:
        pytest.fail(f"unexpected generated fallback routing calls: {calls!r}")


def test_generated_prefers_extracted_for_line3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should route line-3 queries through extracted transform."""
    x = Variable("X")
    y = Variable("Y")
    z = Variable("Z")
    graph = NxMixedGraph.from_edges(
        directed=[(z, x), (x, y)],
        undirected=[(z, x)],
    )
    query = P(y @ ~x)
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return False

    def _fake_supports_line3(identification: Identification) -> bool:
        del identification
        calls.append("supports_line3")
        return True

    def _fake_extracted_line3(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        calls.append("extracted_line3")
        return P(y @ ~x)

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(id_generated_module, "supports_query_line3", _fake_supports_line3)
    monkeypatch.setattr(id_generated_module, "identify_line3_from_extracted", _fake_extracted_line3)

    result = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(result, P(y @ ~x)):
        pytest.fail("generated engine did not return extracted line-3 expression")
    if calls != ["supports_line1", "supports_line2", "supports_line3", "extracted_line3"]:
        pytest.fail(f"unexpected generated routing calls: {calls!r}")


def test_generated_falls_back_when_not_line3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should fallback when line 3 is not available."""
    graph, query = _identifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return False

    def _fake_supports_line3(identification: Identification) -> bool:
        del identification
        calls.append("supports_line3")
        return False

    def _fake_supports_line4(identification: Identification) -> bool:
        del identification
        calls.append("supports_line4")
        return False

    def _fake_supports_line5(identification: Identification) -> bool:
        del identification
        calls.append("supports_line5")
        return False

    def _fake_supports_line6(identification: Identification) -> bool:
        del identification
        calls.append("supports_line6")
        return False

    def _fake_supports_line7(identification: Identification) -> bool:
        del identification
        calls.append("supports_line7")
        return False

    def _unexpected_extracted_line3(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        pytest.fail("extracted line-3 path should not be called")

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(id_generated_module, "supports_query_line3", _fake_supports_line3)
    monkeypatch.setattr(id_generated_module, "supports_query_line4", _fake_supports_line4)
    monkeypatch.setattr(id_generated_module, "supports_query_line5", _fake_supports_line5)
    monkeypatch.setattr(id_generated_module, "supports_query_line6", _fake_supports_line6)
    monkeypatch.setattr(id_generated_module, "supports_query_line7", _fake_supports_line7)
    monkeypatch.setattr(
        id_generated_module, "identify_line3_from_extracted", _unexpected_extracted_line3
    )

    handwritten = identify_with_engine(identification, engine="handwritten")
    generated = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(handwritten, generated):
        pytest.fail("generated fallback path does not match handwritten expression")
    if calls != [
        "supports_line1",
        "supports_line2",
        "supports_line3",
        "supports_line4",
        "supports_line5",
        "supports_line6",
        "supports_line7",
    ]:
        pytest.fail(f"unexpected generated fallback routing calls: {calls!r}")


def test_generated_prefers_extracted_for_line5(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should route line-5 queries through extracted runtime."""
    x = Variable("X")
    y = Variable("Y")
    graph = NxMixedGraph.from_edges(directed=[(x, y)], undirected=[(x, y)])
    query = P(y @ ~x)
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return False

    def _fake_supports_line3(identification: Identification) -> bool:
        del identification
        calls.append("supports_line3")
        return False

    def _fake_supports_line4(identification: Identification) -> bool:
        del identification
        calls.append("supports_line4")
        return False

    def _fake_supports_line5(identification: Identification) -> bool:
        del identification
        calls.append("supports_line5")
        return True

    def _fake_extracted_line5(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        calls.append("extracted_line5")
        raise Unidentifiable({x, y}, {y})

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(id_generated_module, "supports_query_line3", _fake_supports_line3)
    monkeypatch.setattr(id_generated_module, "supports_query_line4", _fake_supports_line4)
    monkeypatch.setattr(id_generated_module, "supports_query_line5", _fake_supports_line5)
    monkeypatch.setattr(id_generated_module, "identify_line5_from_extracted", _fake_extracted_line5)

    with pytest.raises(Unidentifiable):
        identify_with_engine(identification, engine="generated")

    if calls != [
        "supports_line1",
        "supports_line2",
        "supports_line3",
        "supports_line4",
        "supports_line5",
        "extracted_line5",
    ]:
        pytest.fail(f"unexpected generated routing calls: {calls!r}")


def test_generated_falls_back_when_not_line5(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should fallback when line 5 is not available."""
    graph, query = _identifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return False

    def _fake_supports_line3(identification: Identification) -> bool:
        del identification
        calls.append("supports_line3")
        return False

    def _fake_supports_line4(identification: Identification) -> bool:
        del identification
        calls.append("supports_line4")
        return False

    def _fake_supports_line5(identification: Identification) -> bool:
        del identification
        calls.append("supports_line5")
        return False

    def _unexpected_extracted_line5(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        pytest.fail("extracted line-5 path should not be called")

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(id_generated_module, "supports_query_line3", _fake_supports_line3)
    monkeypatch.setattr(id_generated_module, "supports_query_line4", _fake_supports_line4)
    monkeypatch.setattr(id_generated_module, "supports_query_line5", _fake_supports_line5)
    monkeypatch.setattr(
        id_generated_module, "identify_line5_from_extracted", _unexpected_extracted_line5
    )

    handwritten = identify_with_engine(identification, engine="handwritten")
    generated = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(handwritten, generated):
        pytest.fail("generated fallback path does not match handwritten expression")
    if calls != [
        "supports_line1",
        "supports_line2",
        "supports_line3",
        "supports_line4",
        "supports_line5",
    ]:
        pytest.fail(f"unexpected generated fallback routing calls: {calls!r}")


def test_generated_prefers_extracted_for_line4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should route line-4 queries through extracted runtime."""
    x = Variable("X")
    y = Variable("Y")
    z = Variable("Z")
    graph = NxMixedGraph.from_edges(
        directed=[(x, z), (z, y)],
        undirected=[(x, y)],
    )
    query = P(y @ ~x)
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return False

    def _fake_supports_line3(identification: Identification) -> bool:
        del identification
        calls.append("supports_line3")
        return False

    def _fake_supports_line4(identification: Identification) -> bool:
        del identification
        calls.append("supports_line4")
        return True

    def _fake_extracted_line4(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        calls.append("extracted_line4")
        return P(y @ ~x)

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(id_generated_module, "supports_query_line3", _fake_supports_line3)
    monkeypatch.setattr(id_generated_module, "supports_query_line4", _fake_supports_line4)
    monkeypatch.setattr(id_generated_module, "identify_line4_from_extracted", _fake_extracted_line4)

    result = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(result, P(y @ ~x)):
        pytest.fail("generated engine did not return extracted line-4 expression")
    if calls != [
        "supports_line1",
        "supports_line2",
        "supports_line3",
        "supports_line4",
        "extracted_line4",
    ]:
        pytest.fail(f"unexpected generated routing calls: {calls!r}")


def test_generated_falls_back_when_not_line4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should fallback when line 4 is not available."""
    graph, query = _identifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return False

    def _fake_supports_line3(identification: Identification) -> bool:
        del identification
        calls.append("supports_line3")
        return False

    def _fake_supports_line4(identification: Identification) -> bool:
        del identification
        calls.append("supports_line4")
        return False

    def _unexpected_extracted_line4(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        pytest.fail("extracted line-4 path should not be called")

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(id_generated_module, "supports_query_line3", _fake_supports_line3)
    monkeypatch.setattr(id_generated_module, "supports_query_line4", _fake_supports_line4)
    monkeypatch.setattr(
        id_generated_module, "identify_line4_from_extracted", _unexpected_extracted_line4
    )

    handwritten = identify_with_engine(identification, engine="handwritten")
    generated = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(handwritten, generated):
        pytest.fail("generated fallback path does not match handwritten expression")
    if calls != ["supports_line1", "supports_line2", "supports_line3", "supports_line4"]:
        pytest.fail(f"unexpected generated fallback routing calls: {calls!r}")


def test_generated_prefers_extracted_for_line6(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should route line-6 queries through extracted runtime."""
    x = Variable("X")
    y = Variable("Y")
    z = Variable("Z")
    graph = NxMixedGraph.from_edges(
        directed=[(x, z), (z, y)],
        undirected=[(z, y)],
    )
    query = P(y @ ~x)
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return False

    def _fake_supports_line3(identification: Identification) -> bool:
        del identification
        calls.append("supports_line3")
        return False

    def _fake_supports_line4(identification: Identification) -> bool:
        del identification
        calls.append("supports_line4")
        return False

    def _fake_supports_line5(identification: Identification) -> bool:
        del identification
        calls.append("supports_line5")
        return False

    def _fake_supports_line6(identification: Identification) -> bool:
        del identification
        calls.append("supports_line6")
        return True

    def _fake_extracted_line6(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        calls.append("extracted_line6")
        return P(y @ ~x)

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(id_generated_module, "supports_query_line3", _fake_supports_line3)
    monkeypatch.setattr(id_generated_module, "supports_query_line4", _fake_supports_line4)
    monkeypatch.setattr(id_generated_module, "supports_query_line5", _fake_supports_line5)
    monkeypatch.setattr(id_generated_module, "supports_query_line6", _fake_supports_line6)
    monkeypatch.setattr(id_generated_module, "identify_line6_from_extracted", _fake_extracted_line6)

    result = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(result, P(y @ ~x)):
        pytest.fail("generated engine did not return extracted line-6 expression")
    if calls != [
        "supports_line1",
        "supports_line2",
        "supports_line3",
        "supports_line4",
        "supports_line5",
        "supports_line6",
        "extracted_line6",
    ]:
        pytest.fail(f"unexpected generated routing calls: {calls!r}")


def test_generated_falls_back_when_not_line6(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should fallback when line 6 is not available."""
    graph, query = _identifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return False

    def _fake_supports_line3(identification: Identification) -> bool:
        del identification
        calls.append("supports_line3")
        return False

    def _fake_supports_line4(identification: Identification) -> bool:
        del identification
        calls.append("supports_line4")
        return False

    def _fake_supports_line5(identification: Identification) -> bool:
        del identification
        calls.append("supports_line5")
        return False

    def _fake_supports_line6(identification: Identification) -> bool:
        del identification
        calls.append("supports_line6")
        return False

    def _fake_supports_line7(identification: Identification) -> bool:
        del identification
        calls.append("supports_line7")
        return False

    def _unexpected_extracted_line6(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        pytest.fail("extracted line-6 path should not be called")

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(id_generated_module, "supports_query_line3", _fake_supports_line3)
    monkeypatch.setattr(id_generated_module, "supports_query_line4", _fake_supports_line4)
    monkeypatch.setattr(id_generated_module, "supports_query_line5", _fake_supports_line5)
    monkeypatch.setattr(id_generated_module, "supports_query_line6", _fake_supports_line6)
    monkeypatch.setattr(id_generated_module, "supports_query_line7", _fake_supports_line7)
    monkeypatch.setattr(
        id_generated_module, "identify_line6_from_extracted", _unexpected_extracted_line6
    )

    handwritten = identify_with_engine(identification, engine="handwritten")
    generated = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(handwritten, generated):
        pytest.fail("generated fallback path does not match handwritten expression")
    if calls != [
        "supports_line1",
        "supports_line2",
        "supports_line3",
        "supports_line4",
        "supports_line5",
        "supports_line6",
        "supports_line7",
    ]:
        pytest.fail(f"unexpected generated fallback routing calls: {calls!r}")


def test_generated_prefers_extracted_for_line7(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should route line-7 queries through extracted transform."""
    x = Variable("X")
    w = Variable("W")
    y = Variable("Y")
    graph = NxMixedGraph.from_edges(
        directed=[(x, w), (w, y)],
        undirected=[(x, w), (w, y)],
    )
    query = P(y @ ~x)
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return False

    def _fake_supports_line3(identification: Identification) -> bool:
        del identification
        calls.append("supports_line3")
        return False

    def _fake_supports_line4(identification: Identification) -> bool:
        del identification
        calls.append("supports_line4")
        return False

    def _fake_supports_line5(identification: Identification) -> bool:
        del identification
        calls.append("supports_line5")
        return False

    def _fake_supports_line6(identification: Identification) -> bool:
        del identification
        calls.append("supports_line6")
        return False

    def _fake_supports_line7(identification: Identification) -> bool:
        del identification
        calls.append("supports_line7")
        return True

    def _fake_extracted_line7(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        calls.append("extracted_line7")
        return P(y @ ~x)

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(id_generated_module, "supports_query_line3", _fake_supports_line3)
    monkeypatch.setattr(id_generated_module, "supports_query_line4", _fake_supports_line4)
    monkeypatch.setattr(id_generated_module, "supports_query_line5", _fake_supports_line5)
    monkeypatch.setattr(id_generated_module, "supports_query_line6", _fake_supports_line6)
    monkeypatch.setattr(id_generated_module, "supports_query_line7", _fake_supports_line7)
    monkeypatch.setattr(id_generated_module, "identify_line7_from_extracted", _fake_extracted_line7)

    result = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(result, P(y @ ~x)):
        pytest.fail("generated engine did not return extracted line-7 expression")
    if calls != [
        "supports_line1",
        "supports_line2",
        "supports_line3",
        "supports_line4",
        "supports_line5",
        "supports_line6",
        "supports_line7",
        "extracted_line7",
    ]:
        pytest.fail(f"unexpected generated routing calls: {calls!r}")


def test_generated_falls_back_when_not_line7(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated engine should fallback when line 7 is not available."""
    graph, query = _identifiable_case()
    identification = Identification.from_expression(graph=graph, query=query)

    calls: list[str] = []

    def _fake_supports_line1(identification: Identification) -> bool:
        del identification
        calls.append("supports_line1")
        return False

    def _fake_supports_line2(identification: Identification) -> bool:
        del identification
        calls.append("supports_line2")
        return False

    def _fake_supports_line3(identification: Identification) -> bool:
        del identification
        calls.append("supports_line3")
        return False

    def _fake_supports_line4(identification: Identification) -> bool:
        del identification
        calls.append("supports_line4")
        return False

    def _fake_supports_line5(identification: Identification) -> bool:
        del identification
        calls.append("supports_line5")
        return False

    def _fake_supports_line6(identification: Identification) -> bool:
        del identification
        calls.append("supports_line6")
        return False

    def _fake_supports_line7(identification: Identification) -> bool:
        del identification
        calls.append("supports_line7")
        return False

    def _unexpected_extracted_line7(
        identification: Identification,
        *,
        ordering: list[Variable] | None = None,
    ) -> Expression:
        del identification, ordering
        pytest.fail("extracted line-7 path should not be called")

    monkeypatch.setattr(id_generated_module, "supports_query_line1", _fake_supports_line1)
    monkeypatch.setattr(id_generated_module, "supports_query_line2", _fake_supports_line2)
    monkeypatch.setattr(id_generated_module, "supports_query_line3", _fake_supports_line3)
    monkeypatch.setattr(id_generated_module, "supports_query_line4", _fake_supports_line4)
    monkeypatch.setattr(id_generated_module, "supports_query_line5", _fake_supports_line5)
    monkeypatch.setattr(id_generated_module, "supports_query_line6", _fake_supports_line6)
    monkeypatch.setattr(id_generated_module, "supports_query_line7", _fake_supports_line7)
    monkeypatch.setattr(
        id_generated_module, "identify_line7_from_extracted", _unexpected_extracted_line7
    )

    handwritten = identify_with_engine(identification, engine="handwritten")
    generated = identify_with_engine(identification, engine="generated")
    if not canonical_expr_equal(handwritten, generated):
        pytest.fail("generated fallback path does not match handwritten expression")
    if calls != [
        "supports_line1",
        "supports_line2",
        "supports_line3",
        "supports_line4",
        "supports_line5",
        "supports_line6",
        "supports_line7",
    ]:
        pytest.fail(f"unexpected generated fallback routing calls: {calls!r}")
