# -*- coding: utf-8 -*-

"""A parser for causaleffect probability expressions based on :mod:`pyparsing`."""

from ...dsl import Expression

__all__ = [
    'parse_causaleffect',
]


def parse_causaleffect(s: str) -> Expression:
    """Parse a causaleffect probability expression."""
    raise NotImplementedError
