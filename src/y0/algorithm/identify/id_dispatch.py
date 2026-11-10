"""Runtime dispatch between handwritten and generated ID engines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .id_generated import identify_generated
from .id_std import identify as identify_handwritten
from .utils import Identification
from ...dsl import Expression, Variable

__all__ = [
    "IDEngine",
    "identify_with_engine",
]

IDEngine = Literal["handwritten", "generated"]


def identify_with_engine(
    identification: Identification,
    *,
    engine: IDEngine = "handwritten",
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run the requested ID engine for one identification query."""
    if engine == "handwritten":
        return identify_handwritten(identification, ordering=ordering)
    if engine == "generated":
        return identify_generated(identification, ordering=ordering)
    raise ValueError(f"unknown ID engine: {engine}")
