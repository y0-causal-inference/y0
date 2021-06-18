# -*- coding: utf-8 -*-

"""Predicates for expressions."""

from .dsl import Expression, Fraction, Probability, Product, Sum

__all__ = [
    "has_markov_postcondition",
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
