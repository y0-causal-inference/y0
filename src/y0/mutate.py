# -*- coding: utf-8 -*-

"""Operations for mutating and simplifying expressions."""

from .dsl import Distribution, Expression, Fraction, P, Probability, Product


def expand_simple_bayes(p: Probability) -> Fraction:
    """Given P(Y|X,...), return P(X,...|Y)*P(Y)/P(X,...)."""
    if not p.distribution.is_markov_kernel():
        raise NotImplementedError

    y = p.distribution.children
    x = p.distribution.parents
    return P(Distribution(x, y)) * P(y) / P(x)


def _check_pair(e: Expression) -> bool:
    if not isinstance(e, Product):
        return False
    if 2 != len(e.expressions):
        return False

    a, b = e.expressions

    if not isinstance(a, Probability):
        return False
    if not isinstance(b, Probability):
        return False
    return check_partial_bayes(a, b) or check_partial_bayes(b, a)


def check_partial_bayes(a: Probability, b: Probability) -> bool:
    """Check if a and b take the form P(X,...|Y,...)P(Y,...)"""
    return (
        not b.distribution.is_conditioned()
        and a.distribution.parents == b.distribution.children
    )


def is_expanded_simple_bayes(frac: Fraction) -> bool:
    if not isinstance(frac, Fraction):
        return False


def reduce_simple_bayes(p: Product) -> Probability:
    """Given return P(X|Y)*P(Y)/P(X), return P(Y|X).

    .. note:: The multiplication can be in any order
    """
