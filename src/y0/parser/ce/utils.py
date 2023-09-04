# -*- coding: utf-8 -*-

"""A parser for causaleffect probability expressions based on :mod:`pyparsing`."""

from pyparsing import (
    Group,
    Optional,
    ParseResults,
    Suppress,
    Word,
    alphanums,
    alphas,
    delimitedList,
    nums,
)

from ...dsl import Distribution, Probability, QFactor, Variable, _sorted_variables


def _make_variable(_s, _l, tokens: ParseResults) -> Variable:
    name = tokens["name"]
    if "subscript" in tokens:
        name += "_" + tokens["subscript"]
    return Variable(name=name)


subscript = Suppress("_{") + Word(nums)("subscript") + Suppress("}")
variable_pe = Word(alphas, alphanums)("name") + Optional(subscript)
variable_pe.setParseAction(_make_variable)
variable_pe.setName("variable")


def _unpack(_s, _l, tokens: ParseResults):
    return tokens[0]


variables_pe = delimitedList(Group(variable_pe).setParseAction(_unpack))
_children_pe = Group(variables_pe).setResultsName("children")
_parents_pe = Group(Optional(Suppress("|") + variables_pe)).setResultsName("parents")
probability_pe = Suppress("P(") + _children_pe + _parents_pe + Suppress(")")


def _make_probability(_s, _l, tokens: ParseResults) -> Probability:
    return Probability(
        Distribution(
            children=_sorted_variables(tokens["children"].asList()),
            parents=_sorted_variables(tokens["parents"].asList()),
        )
    )


probability_pe.setParseAction(_make_probability)
probability_pe.setName("probability")

qfactor_pe = (
    Suppress("Q[\\{")
    + Group(variables_pe).setResultsName("codomain")
    + Suppress("\\}](")
    + Group(variables_pe).setResultsName("domain")
    + Suppress(")")
)


def _make_q(_s, _l, tokens: ParseResults) -> QFactor:
    return QFactor(
        codomain=frozenset(tokens["codomain"].asList()),
        domain=frozenset(tokens["domain"].asList()),
    )


qfactor_pe.setParseAction(_make_q)
qfactor_pe.setName("qfactor")
