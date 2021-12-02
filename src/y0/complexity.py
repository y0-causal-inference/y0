# -*- coding: utf-8 -*-

"""Calculate the complexity of an expression."""

from typing import Sequence

from .dsl import (
    Expression,
    Fraction,
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
SUM_CONST = 0.0
FRAC_CONST = 0.0
PROD_CONST = 0.0
PROB_CONST = 0.0
Q_CONST = 0.0


def complexity(expr: Expression) -> float:
    """Calculate the complexity of the given expression, recursively."""
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
        return PROB_CONST + prob_complexity(expr)
    raise TypeError(f"Unhandled expression type: {expr.__class__.__name__}")


def range_complexity(variables: Sequence[Variable]) -> float:
    """Return the complexity of a sequence of variables."""
    return sum(var_complexity(v) for v in variables)


def var_complexity(variable: Variable) -> float:
    """Return the complexity of a single variable."""
    return 1.0  # FIXME


def prob_complexity(expr: Probability) -> float:
    raise NotImplementedError
