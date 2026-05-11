"""Generated ID engine entrypoint.

This module exposes the Phase 4 generated-engine surface while the concrete
implementation is still sourced from the verified handwritten ID runtime.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from .id_extracted_bridge import (
    ExtractedFullUnavailableError,
    ExtractedLine1UnavailableError,
    ExtractedLine2UnavailableError,
    ExtractedLine3UnavailableError,
    ExtractedLine4UnavailableError,
    ExtractedLine5UnavailableError,
    ExtractedLine6UnavailableError,
    ExtractedLine7UnavailableError,
    identify_full_from_extracted,
    identify_line1_from_extracted,
    identify_line2_from_extracted,
    identify_line3_from_extracted,
    identify_line4_from_extracted,
    identify_line5_from_extracted,
    identify_line6_from_extracted,
    identify_line7_from_extracted,
    supports_query_line1,
    supports_query_line2,
    supports_query_line3,
    supports_query_line4,
    supports_query_line5,
    supports_query_line6,
    supports_query_line7,
)
from .id_std import identify as identify_handwritten
from .utils import Identification
from ...dsl import Expression, Variable

__all__ = [
    "identify_generated",
]


def _line_compat_enabled() -> bool:
    value = os.environ.get("Y0_DAFNY_ID_LINE_COMPAT", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _identify_via_line_compat(
    identification: Identification,
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression | None:
    if supports_query_line1(identification):
        try:
            return identify_line1_from_extracted(identification, ordering=ordering)
        except ExtractedLine1UnavailableError:
            pass
    if supports_query_line2(identification):
        try:
            return identify_line2_from_extracted(identification, ordering=ordering)
        except ExtractedLine2UnavailableError:
            pass
    if supports_query_line3(identification):
        try:
            return identify_line3_from_extracted(identification, ordering=ordering)
        except ExtractedLine3UnavailableError:
            pass
    if supports_query_line4(identification):
        try:
            return identify_line4_from_extracted(identification, ordering=ordering)
        except ExtractedLine4UnavailableError:
            pass
    if supports_query_line5(identification):
        try:
            return identify_line5_from_extracted(identification, ordering=ordering)
        except ExtractedLine5UnavailableError:
            pass
    if supports_query_line6(identification):
        try:
            return identify_line6_from_extracted(identification, ordering=ordering)
        except ExtractedLine6UnavailableError:
            pass
    if supports_query_line7(identification):
        try:
            return identify_line7_from_extracted(identification, ordering=ordering)
        except ExtractedLine7UnavailableError:
            pass
    return None


def identify_generated(
    identification: Identification,
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run the generated ID engine.

    Phase 4 prefers the consolidated extracted runtime and preserves compatibility
    by optionally routing through line-by-line extracted slices before handwritten
    fallback.
    """
    try:
        return identify_full_from_extracted(identification, ordering=ordering)
    except ExtractedFullUnavailableError:
        pass

    if _line_compat_enabled():
        compat_result = _identify_via_line_compat(identification, ordering=ordering)
        if compat_result is not None:
            return compat_result

    return identify_handwritten(identification, ordering=ordering)
