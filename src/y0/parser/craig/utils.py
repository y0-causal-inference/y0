# -*- coding: utf-8 -*-

"""Utilities for the parser for Craig-like probability expressions."""

from pyparsing import Group, Optional, ParseResults, Suppress, Word, alphas, delimitedList

from ...dsl import CounterfactualVariable, Distribution, Intervention, Probability, QFactor, Variable

__all__ = [
    'probability_pe',
    'variables_pe',
    'qfactor_pe',
]


def _parse_variable(_s, _l, tokens: ParseResults):
    return Variable(tokens[0])


letter = Word(alphas)


def _set_star(_s, _l, tokens: ParseResults):
    return 1 == len(tokens.asList())


def _make_intervention(_s, _l, tokens: ParseResults):
    return Intervention(name=tokens['name'], star=bool(tokens['star']))


def _unpack(_s, _l, tokens: ParseResults):
    return tokens[0]


def _make_variable(_s, _l, tokens: ParseResults) -> Variable:
    if 'interventions' not in tokens:
        return Variable(name=tokens['name'])
    return CounterfactualVariable(
        name=tokens['name'],
        interventions=tuple(tokens['interventions'].asList()),
    )


def _make_probability(_s, _l, tokens: ParseResults) -> Probability:
    children, parents = tokens['children'].asList(), tokens['parents'].asList()
    if not children:
        raise ValueError
    return Probability(Distribution(children=tuple(children), parents=tuple(parents)))


def _make_q(_s, _l, tokens: ParseResults) -> QFactor:
    codomain, domain = tokens['codomain'].asList(), tokens['domain'].asList()
    return QFactor(codomain=tuple(codomain), domain=tuple(domain))


# The suffix "pe" refers to :class:`pyparsing.ParserElement`, which is the
#  class in pyparsing that everything inherits from
star_pe = Optional('*')('star').setParseAction(_set_star)
intervention_pe = letter('name') + star_pe
intervention_pe.setParseAction(_make_intervention)
interventions_pe = Optional(
    Suppress('_{')
    + delimitedList(Group(intervention_pe).setParseAction(_unpack)).setResultsName('interventions')
    + Suppress('}'),
)
variable_pe = letter('name') + interventions_pe
variable_pe.setParseAction(_make_variable)
variable_pe.setName('variable')

variables_pe = delimitedList(Group(variable_pe).setParseAction(_unpack))
_children_pe = Group(variables_pe).setResultsName('children')
_parents_pe = Group(Optional(Suppress('|') + variables_pe)).setResultsName('parents')
probability_pe = Suppress('P(') + _children_pe + _parents_pe + Suppress(')')
probability_pe.setParseAction(_make_probability)
probability_pe.setName('probability')

qfactor_pe = (
    Suppress('Q[')
    + Group(variables_pe).setResultsName('codomain')
    + Suppress('](')
    + Group(variables_pe).setResultsName('domain')
    + Suppress(')')
)
qfactor_pe.setParseAction(_make_q)
qfactor_pe.setName('qfactor')
