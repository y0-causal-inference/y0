"""Conformance checks for ID oracle generator prerequisites."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol, cast

from y0.algorithm.identify.id_oracle_types import load_fixture, save_fixture


class _GeneratorModule(Protocol):
    """Typed subset of the generator module surface used by this test."""

    def parse_dafny_file(self, path: Path) -> _ParsedSpec:
        """Parse a Dafny file and return the parsed specification."""
        ...


class _ParsedLemma(Protocol):
    """Typed subset of parsed Dafny lemmas used by this test."""

    name: str


class _ParsedSpec(Protocol):
    """Typed subset of parsed Dafny spec used by this test."""

    module: str
    lemmas: list[_ParsedLemma]


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_generator_module() -> _GeneratorModule:
    script_path = _root() / "scripts" / "generate_dafny_conformance_tests.py"
    spec = importlib.util.spec_from_file_location("_dafny_generator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load generator module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(_GeneratorModule, module)


def test_identification_module_is_parsed() -> None:
    """The generator parser should extract declarations from identification.dfy."""
    module = _load_generator_module()
    parse_dafny_file = module.parse_dafny_file
    spec = parse_dafny_file(_root() / "src" / "dafny" / "identification.dfy")

    if spec.module != "Identification":
        raise AssertionError(f"unexpected module name: {spec.module!r}")
    if not spec.lemmas:
        raise AssertionError("expected parsed lemmas from identification.dfy")

    lemma_names = {lemma.name for lemma in spec.lemmas}
    if "ID_Line4" not in lemma_names or "ID_Line5" not in lemma_names:
        raise AssertionError("expected ID_Line4 and ID_Line5 in parsed lemmas")


def test_id_fixture_serialization_is_stable(tmp_path: Path) -> None:
    """Saving the ID fixture twice should produce byte-identical output."""
    fixture_path = _root() / "tests" / "data" / "generated" / "dafny_oracle" / "id_cases.v1.json"
    fixture = load_fixture(fixture_path)

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    save_fixture(first, fixture)
    save_fixture(second, fixture)

    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()
    if first_bytes != second_bytes:
        raise AssertionError("fixture serialization should be deterministic")
