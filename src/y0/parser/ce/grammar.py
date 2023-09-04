# -*- coding: utf-8 -*-

"""A parser for Craig-like probability expressions based on :mod:`pyparsing`."""

import logging

from pyparsing import (
    Forward,
    Group,
    OneOrMore,
    Optional,
    ParseException,
    ParseResults,
    StringEnd,
    StringStart,
    Suppress,
)

from .utils import probability_pe, qfactor_pe, variables_pe
from ...dsl import Expression, Fraction, Product, Sum

__all__ = [
    "parse_causaleffect",
    "grammar",
]

logger = logging.getLogger(__name__)
expr = Forward()


def _make_sum(_s, _l, tokens: ParseResults) -> Expression:
    return Sum.safe(
        ranges=tokens["ranges"].asList() if "ranges" in tokens else [],
        expression=tokens["expression"],
    )


def _make_frac(_s, _l, tokens: ParseResults) -> Fraction:
    return Fraction(
        numerator=tokens["numerator"],
        denominator=tokens["denominator"],
    )


def _make_product(_s, _l, tokens: ParseResults) -> Expression:
    return Product.safe(tokens.asList())


# auto-product
rr = OneOrMore(probability_pe | qfactor_pe | expr).setParseAction(_make_product)

sum_pe = (
    Suppress("\\sum_{")
    + Optional(Group(variables_pe).setResultsName("ranges"))
    + Suppress("}")
    + rr.setResultsName("expression")
)
sum_pe.setName("sum")
sum_pe.setParseAction(_make_sum)

fraction_pe = (
    Suppress("\\frac_{")
    + rr.setResultsName("numerator")
    + Suppress("}{")
    + rr.setResultsName("denominator")
    + Suppress("}")
)
fraction_pe.setName("fraction")
fraction_pe.setParseAction(_make_frac)

expr << (probability_pe | qfactor_pe | sum_pe | fraction_pe)

# TODO enable products?

grammar = StringStart() + expr + StringEnd()
grammar.setName("probabilityGrammar")


def parse_causaleffect(s: str) -> Expression:
    """Parse a causaleffect probability expression."""
    try:
        x = grammar.parseString(s)
    except ParseException:
        logger.warning("could not parse %s", s)
        raise
    else:
        return x.asList()[0]
