# -*- coding: utf-8 -*-

"""Operations for mutating and simplifying expressions."""

from .dsl import Distribution, Fraction, P, Probability, Product


def expand_simple_bayes(p: Probability) -> Fraction:
    """Given P(Y|X,...), return P(X,...|Y)*P(Y)/P(X)."""
    if not p.distribution.is_markov_kernel():
        raise NotImplementedError

    """P(Y|X1,X2) = P(X1,X2|Y)P(Y)/P(X1,X2)"""

    y = p.distribution.children
    x = p.distribution.parents

    reverse_conditional = Distribution(
        children=p.distribution.parents,
        parents=p.distribution.children,
    )

    return P(reverse_conditional) * P(y) / P(x)


def is_expanded_simple_bayes(frac: Fraction) -> bool:
    if not isinstance(frac, Fraction):
        return False


def reduce_simple_bayes(p: Product) -> Probability:
    """Given return P(X|Y)*P(Y)/P(X), return P(Y|X).

    .. note:: The multiplication can be in any order
    """
