# -*- coding: utf-8 -*-

"""A parser for causaleffect probability expressions based on :mod:`pyparsing`."""

from pyparsing import Group, Optional, ParseResults, Suppress, Word, alphanums, alphas, delimitedList, nums

from ..craig.utils import _make_probability, _make_q, _unpack
from ...dsl import Variable


def _make_variable(_s, _l, tokens: ParseResults) -> Variable:
    name = tokens['name']
    if 'subscript' in tokens:
        name += ('_' + tokens['subscript'])
    return Variable(name=name)


subscript = Suppress('_{') + Word(nums)('subscript') + Suppress('}')
variable_pe = Word(alphas, alphanums)('name') + Optional(subscript)
variable_pe.setParseAction(_make_variable)
variable_pe.setName('variable')

variables_pe = delimitedList(Group(variable_pe).setParseAction(_unpack))
_children_pe = Group(variables_pe).setResultsName('children')
_parents_pe = Group(Optional(Suppress('|') + variables_pe)).setResultsName('parents')
probability_pe = Suppress('P(') + _children_pe + _parents_pe + Suppress(')')
probability_pe.setParseAction(_make_probability)
probability_pe.setName('probability')

qfactor_pe = (
    Suppress('Q[\\{')
    + Group(variables_pe).setResultsName('codomain')
    + Suppress('\\}](')
    + Group(variables_pe).setResultsName('domain')
    + Suppress(')')
)
qfactor_pe.setParseAction(_make_q)
qfactor_pe.setName('qfactor')
