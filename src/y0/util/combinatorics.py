# -*- coding: utf-8 -*-

"""Utilities."""

import math
from itertools import chain, combinations
from typing import Any, Collection, Iterable, Mapping, Optional, TypeVar

from tqdm import tqdm

__all__ = [
    "powerset",
]

X = TypeVar("X")


def powerset(
    iterable: Iterable[X],
    start: int = 0,
    stop: Optional[int] = None,
    *,
    reverse: bool = False,
    use_tqdm: Optional[bool] = False,
    tqdm_kwargs: Optional[Mapping[str, Any]] = None,
) -> Iterable[Collection[X]]:
    """Get successively longer combinations of the source.

    :param iterable: List to get combinations from
    :param start: smallest combination to get (default 0)
    :param stop: Largest combination to get (None means length of the list and is the default)
    :param reverse: Should the bigger powersets be returned first?
    :param use_tqdm: Should a progress bar be shown
    :param tqdm_kwargs: Options for tqdm
    :return: Iterator of powerset of values.

    .. seealso: :func:`more_iterools.powerset` for a non-constrainable implementation
    """
    s = list(iterable)
    n = len(s)
    if stop is None:
        stop = n + 1

    if reverse:
        rv = chain.from_iterable(combinations(s, n - r) for r in range(start, stop))
    else:
        rv = chain.from_iterable(combinations(s, r) for r in range(start, stop))

    if use_tqdm:
        total = sum(math.comb(n, k) for k in range(start, stop))
        rv = tqdm(rv, disable=use_tqdm, total=total, **(tqdm_kwargs or {}))

    return rv
