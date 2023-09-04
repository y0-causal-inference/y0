# -*- coding: utf-8 -*-

"""Calculate the complexity of an expression."""

from typing import Iterable

from .dsl import (
    CounterfactualVariable,
    Expression,
    Fraction,
    Intervention,
    One,
    Probability,
    Product,
    QFactor,
    Sum,
    Variable,
    Zero,
)

__all__ = [
    "complexity",
]

CONST_CONST = 1.0  # yo dawg
VAR_CONSTANT = 1.0  # yo cat
SUM_CONST = 0.0
FRAC_CONST = 0.0
PROD_CONST = 0.0
PROB_CONST = 0.0
Q_CONST = 0.0


def complexity(expr: Expression) -> float:
    """Calculate the "complexity" of the expression, where a bigger result means more complex.

    Currently, the complexity of the expression is defined recursively, where all of the
    parts of each expression are added up, counting the number of variables in each.

    :param expr: The input expression
    :return: The complexity of the expression
    :raises TypeError:
        Raised if an invalid expression type is used
    """
    if isinstance(expr, (One, Zero)):
        return CONST_CONST
    if isinstance(expr, Fraction):
        return FRAC_CONST + complexity(expr.numerator) + complexity(expr.denominator)
    if isinstance(expr, Product):
        return PROD_CONST + sum(complexity(subexpr) for subexpr in expr.expressions)
    if isinstance(expr, Sum):
        return SUM_CONST + range_complexity(expr.ranges) + complexity(expr.expression)
    if isinstance(expr, QFactor):
        return Q_CONST + range_complexity(expr.domain) + range_complexity(expr.codomain)
    if isinstance(expr, Probability):
        return PROB_CONST + probability_complexity(expr)
    raise TypeError(f"Unhandled expression type: {expr.__class__.__name__}")


def range_complexity(variables: Iterable[Variable]) -> float:
    """Return the complexity of a sequence of variables."""
    return sum(variable_complexity(v) for v in variables)


def variable_complexity(variable: Variable) -> float:
    """Return the complexity of a single variable."""
    if isinstance(variable, CounterfactualVariable):
        return VAR_CONSTANT + sum(variable_complexity(v) for v in variable.interventions)
    elif isinstance(variable, Intervention):
        return VAR_CONSTANT
    else:
        return VAR_CONSTANT


def probability_complexity(expr: Probability) -> float:
    """Return the complexity of a probability expression."""
    return range_complexity(expr.children) + range_complexity(expr.parents)
