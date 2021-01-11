# -*- coding: utf-8 -*-

"""A parser for probability expressions."""

from pyparsing import Forward, Group, OneOrMore, Optional, ParseResults, StringEnd, StringStart, Suppress

from .dsl import Expression, Fraction, Product, Sum
from .parser_utils import probability_pe, variables_pe

__all__ = [
    'parse',
    'grammar',
]

expr = Forward()


def _make_sum(_s, _l, tokens: ParseResults) -> Sum:
    return Sum(
        ranges=tokens['ranges'].asList() if 'ranges' in tokens else [],
        expression=tokens['expression'],
    )


def _make_frac(_s, _l, tokens: ParseResults) -> Fraction:
    return Fraction(
        numerator=tokens['numerator'],
        denominator=tokens['denominator'],
    )


def _make_product(_s, _l, tokens: ParseResults) -> Expression:
    tokens = tokens.asList()
    if len(tokens) == 1:
        return tokens[0]
    else:
        return Product(tokens)


# auto-product
rr = OneOrMore(probability_pe | expr).setParseAction(_make_product)

sum_pe = (
    Suppress('[')
    + Suppress('sum_{')
    + Optional(Group(variables_pe).setResultsName('ranges'))
    + Suppress('}')
    + rr.setResultsName('expression')
    + Suppress(']')
)
sum_pe.setName('sum')
sum_pe.setParseAction(_make_sum)

fraction_pe = (
    Suppress('frac_{')
    + rr.setResultsName('numerator')
    + Suppress('}{')
    + rr.setResultsName('denominator')
    + Suppress('}')
)
fraction_pe.setName('fraction')
fraction_pe.setParseAction(_make_frac)

expr << (probability_pe | sum_pe | fraction_pe)

# TODO enable products?

grammar = StringStart() + expr + StringEnd()
grammar.setName('probabilityGrammar')


def parse(s: str) -> Expression:
    """Parse an expression."""
    x = grammar.parseString(s)
    return x.asList()[0]
