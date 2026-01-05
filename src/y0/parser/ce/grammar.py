"""A parser for Craig-like probability expressions based on :mod:`pyparsing`."""

import logging
from typing import cast

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
    "grammar",
    "parse_causaleffect",
]

logger = logging.getLogger(__name__)
expr = Forward()


def _make_sum(_s, _l, tokens: ParseResults) -> Expression:  # type:ignore[no-untyped-def]
    return Sum.safe(
        ranges=tokens["ranges"].asList() if "ranges" in tokens else [],
        expression=tokens["expression"],
    )


def _make_frac(_s, _l, tokens: ParseResults) -> Fraction:  # type:ignore[no-untyped-def]
    return Fraction(
        numerator=tokens["numerator"],
        denominator=tokens["denominator"],
    )


def _make_product(_s, _l, tokens: ParseResults) -> Expression:  # type:ignore[no-untyped-def]
    return Product.safe(tokens.asList())


# auto-product
rr = OneOrMore(probability_pe | qfactor_pe | expr).set_parse_action(_make_product)

sum_pe = (
    Suppress("\\sum_{")
    + Optional(Group(variables_pe).set_results_name("ranges"))
    + Suppress("}")
    + rr.set_results_name("expression")
)
sum_pe.set_name("sum")
sum_pe.set_parse_action(_make_sum)

fraction_pe = (
    Suppress("\\frac_{")
    + rr.set_results_name("numerator")
    + Suppress("}{")
    + rr.set_results_name("denominator")
    + Suppress("}")
)
fraction_pe.set_name("fraction")
fraction_pe.set_parse_action(_make_frac)

expr << (probability_pe | qfactor_pe | sum_pe | fraction_pe)

# TODO enable products?

grammar = StringStart() + expr + StringEnd()
grammar.set_name("probabilityGrammar")


def parse_causaleffect(s: str) -> Expression:
    """Parse a causaleffect probability expression."""
    try:
        x = grammar.parse_string(s)
    except ParseException:
        logger.warning("could not parse %s", s)
        raise
    else:
        return cast(Expression, x.asList()[0])
