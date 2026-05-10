"""Generated ID engine entrypoint.

This module exposes the Phase 4 generated-engine surface while the concrete
implementation is still sourced from the verified handwritten ID runtime.
"""

from __future__ import annotations

from collections.abc import Sequence

from .id_std import identify as identify_handwritten
from .utils import Identification
from ...dsl import Expression, Variable

__all__ = [
    "identify_generated",
]


def identify_generated(
    identification: Identification,
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run the generated ID engine.

    The current Phase 4 slice keeps behavior equivalent by delegating to the
    handwritten implementation through a thin adapter boundary.
    """
    return identify_handwritten(identification, ordering=ordering)
