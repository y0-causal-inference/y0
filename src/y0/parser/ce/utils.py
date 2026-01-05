"""A parser for causaleffect probability expressions based on :mod:`pyparsing`."""

from pyparsing import (
    DelimitedList,
    Group,
    Optional,
    ParseResults,
    Suppress,
    Word,
    alphanums,
    alphas,
    nums,
)

from ...dsl import Distribution, Probability, QFactor, Variable, _sorted_variables


def _make_variable(_s, _l, tokens: ParseResults) -> Variable:  # type:ignore[no-untyped-def]
    name = tokens["name"]
    if "subscript" in tokens:
        name += "_" + tokens["subscript"]
    return Variable(name=name)


subscript = Suppress("_{") + Word(nums)("subscript") + Suppress("}")
variable_pe = Word(alphas, alphanums)("name") + Optional(subscript)
variable_pe.set_parse_action(_make_variable)
variable_pe.set_name("variable")


def _unpack(_s, _l, tokens: ParseResults):  # type:ignore[no-untyped-def]
    return tokens[0]


variables_pe = DelimitedList(Group(variable_pe).set_parse_action(_unpack))
_children_pe = Group(variables_pe).set_results_name("children")
_parents_pe = Group(Optional(Suppress("|") + variables_pe)).set_results_name("parents")
probability_pe = Suppress("P(") + _children_pe + _parents_pe + Suppress(")")


def _make_probability(_s, _l, tokens: ParseResults) -> Probability:  # type:ignore[no-untyped-def]
    return Probability(
        Distribution(
            children=_sorted_variables(tokens["children"].asList()),
            parents=_sorted_variables(tokens["parents"].asList()),
        )
    )


probability_pe.set_parse_action(_make_probability)
probability_pe.set_name("probability")

qfactor_pe = (
    Suppress("Q[\\{")
    + Group(variables_pe).set_results_name("codomain")
    + Suppress("\\}](")
    + Group(variables_pe).set_results_name("domain")
    + Suppress(")")
)


def _make_q(_s, _l, tokens: ParseResults) -> QFactor:  # type:ignore[no-untyped-def]
    return QFactor(
        codomain=frozenset(tokens["codomain"].asList()),
        domain=frozenset(tokens["domain"].asList()),
    )


qfactor_pe.set_parse_action(_make_q)
qfactor_pe.set_name("qfactor")
