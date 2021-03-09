# -*- coding: utf-8 -*-

"""Utilities."""

from itertools import chain, combinations
from typing import Collection, Iterable, Optional, TypeVar

__all__ = [
    'powerset',
]

X = TypeVar('X')


def powerset(
    iterable: Iterable[X],
    start: int = 0,
    stop: Optional[int] = None,
) -> Iterable[Collection[X]]:
    """Get successively longer combinations of the source.

    :param iterable: List to get combinations from
    :param start: smallest combination to get (default 0)
    :param stop: Largest combination to get (None means length of the list and is the default)
    :return: Iterator of powerset of values.

    .. seealso: :func:`more_iterools.powerset` for a non-constrainable implementation
    """
    s = list(iterable)
    if stop is None:
        stop = len(s) + 1
    return chain.from_iterable(combinations(s, r) for r in range(start, stop))
