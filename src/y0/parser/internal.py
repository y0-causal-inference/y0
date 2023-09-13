# -*- coding: utf-8 -*-

"""A parser for y0 internal DSL probability expressions based on Python's :func:`eval` function."""

import itertools as itt
import string

from y0.dsl import PP, Expression, P, Q, Sum, Variable

__all__ = [
    "parse_y0",
]

LOCALS = {
    "P": P,
    "PROB": P,
    "Prob": P,
    "PROBABILITY": P,
    "Probability": P,
    "SUM": Sum,
    "Sum": Sum,
    "Q": Q,
    "QFactor": Q,
    "PP": PP,
}

for letter in itt.chain(string.ascii_uppercase, ["Pi", "π"]):
    if letter in {"P", "Q"}:
        continue
    LOCALS[letter] = Variable(letter)
    for index in range(10):
        name = f"{letter}{index}"
        LOCALS[name] = Variable(name)

        name_underscored = f"{letter}_{index}"
        LOCALS[name_underscored] = Variable(name_underscored)


def parse_y0(s) -> Expression:
    """Parse a valid Python expression using the :mod:`y0.dsl` objects, written in a string.

    :param s: The string to parse. Should be a valid Python expression given ``from y0.dsl import *``.
        Variables of the form A-Z, A0-Z0, A_0-Z_0, A1-Z1, A_1-Z_1, ..., A9-Z9, A_9-Z_9 are available.
    :return: An expression object.

    >>> from y0.parser import parse_y0
    >>> from y0.dsl import P, PP, A, B, Sum, Pi1
    >>> parse_y0('Sum[B](P(A|B) * P(B))') == Sum[B](P(A|B) * P(B))
    True
    >>> parse_y0('PP[π1](A)') == PP[Pi1](A)
    True
    """
    return eval(s, {}, LOCALS)  # noqa:S307
