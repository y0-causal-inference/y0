"""Contraction functions."""

from operator import attrgetter

from y0.dsl import Distribution, Expression, Fraction, Probability
from y0.mutate.utils import Applier

__all__ = ["contract", "recursive_contract"]


def contract(expression: Expression) -> Expression:
    """Contract a fraction.

    :param expression: A fraction
    :returns: A bayesian compressed probability, if possible

    >>> from y0.dsl import P, A, B, C, D
    >>> assert P(A | B) == contract(P(A, B) / P(B))
    >>> assert P(A | B, C) == contract(P(A, B, C) / P(B, C))
    >>> assert P(B, C | A) == contract(P(A, B, C) / P(A))
    >>> assert P(A, B, C) / P(D) == contract(P(A, B, C) / P(D))
    """
    if not (
        isinstance(expression, Fraction)
        and isinstance(expression.numerator, Probability)
        and isinstance(expression.denominator, Probability)
        and not expression.numerator.parents
        and not expression.denominator.parents
        and set(expression.denominator.children).issubset(expression.numerator.children)
    ):
        return expression
    children = set(expression.numerator.children).difference(expression.denominator.children)
    parents = set(expression.numerator.children).intersection(expression.denominator.children)
    return expression.numerator._new(
        Distribution(
            children=tuple(sorted(children, key=attrgetter("name"))),
            parents=tuple(sorted(parents, key=attrgetter("name"))),
        )
    )


class _Contracter(Applier):
    def apply_fraction(self, expression: Fraction) -> Expression:
        """Contract a fraction."""
        return contract(expression)


def recursive_contract(e: Expression) -> Expression:
    """Recursively contract fractions."""
    return _Contracter().apply_expression(e)
