# -*- coding: utf-8 -*-

"""A parser for y0 internal DSL probability expressions based on Python's :func:`eval` function."""

import string

from y0.dsl import Expression, P, Sum, Variable

__all__ = [
    'parse_y0',
]

LOCALS = {
    'P': P,
    'PROB': P,
    'PROBABILITY': P,
    'Probability': P,
    'SUM': Sum,
    'Sum': Sum,
}

for letter in string.ascii_uppercase:
    if letter == 'P':
        continue
    LOCALS[letter] = Variable(letter)
    for index in range(4):
        name = f'{letter}{index}'
        LOCALS[name] = Variable(name)

        name_underscored = f'{letter}_{index}'
        LOCALS[name_underscored] = Variable(name_underscored)


def parse_y0(s) -> Expression:
    """Parse a valid Python expression using the :mod:`y0.dsl` objects, written in a string.

    :param s: The string to parse. Should be a valid Python expression given ``from y0.dsl import *``.
        Variables of the form A-Z, A1-Z1, A_1-Z_1, ..., A4-Z4, A_4-Z_4 are available.
    :return: An expression object.

    >>> from y0.parser import parse_y0
    >>> from y0.dsl import P, A, B, Sum
    >>> parse_y0('Sum[B](P(A|B) * P(B))') == Sum[B](P(A|B) * P(B))
    True
    """
    return eval(s, {}, LOCALS)  # noqa:S307
