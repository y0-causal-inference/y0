# -*- coding: utf-8 -*-

"""Predicates for expressions."""

from .dsl import Expression, Fraction, Probability, Product, Sum, Variable
from .graph import NxMixedGraph

__all__ = [
    "has_markov_postcondition",
    "is_good_control",
    "is_bad_control",
]


def has_markov_postcondition(expression: Expression) -> bool:
    """Check that the expression is a sum/product of markov kernels.

    :param expression: Any expression
    :return: if the expression satisfies the sum/product of markov kernels condition
    :raises TypeError: if an object with an invalid type is passed
    """
    if isinstance(expression, Probability):
        return expression.distribution.is_markov_kernel()
    elif isinstance(expression, Product):
        return all(has_markov_postcondition(subexpr) for subexpr in expression.expressions)
    elif isinstance(expression, Sum):
        return has_markov_postcondition(expression.expression)
    elif isinstance(expression, Fraction):
        return has_markov_postcondition(expression.numerator) and has_markov_postcondition(
            expression.denominator
        )
    else:
        raise TypeError


def _control_precondition(graph: NxMixedGraph, query: Probability, variable: Variable):
    if missing := query.get_variables().difference(graph.nodes()):
        raise ValueError(f"Query variables missing: {missing}")
    if variable not in graph.nodes():
        raise ValueError(f"Test variable missing: {variable}")
    # TODO does this need to be extended to check that the
    #  query and variable aren't counterfactual?


def is_good_control(graph: NxMixedGraph, query: Probability, variable: Variable) -> bool:
    """Return if the variable is a good control."""
    _control_precondition(graph, query, variable)

    raise NotImplementedError


def is_bad_control(graph: NxMixedGraph, query: Probability, variable: Variable) -> bool:
    """Return if the variable is a good control."""
    _control_precondition(graph, query, variable)

    raise NotImplementedError
