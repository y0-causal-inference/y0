# -*- coding: utf-8 -*-

"""Operations for mutating and simplifying expressions."""

from .dsl import ConditionalProbability, Expression, P, Probability, Product, Fraction


def is_simple_bayes(p: Probability) -> bool:
    """Return if the probability follows the form P(Y|X)."""
        isinstance(p.probability, ConditionalProbability)
        and 1 == len(p.probability.parents)
    )


def expand_simple_bayes(p: Probability) -> Fraction:
    """Given P(Y|X), return P(X|Y)*P(Y)/P(X)."""
    if not is_simple_bayes(p):
        raise NotImplementedError

    y = p.probability.child
    x = p.probability.parents[0]
    return P(x | y) * P(y) / P(x)


def is_expanded_simple_bayes(frac: Fraction) -> bool:
    if not isinstance(frac, Fraction):
        return False

def reduce_simple_bayes(p: Product) -> Probability:
    """Given return P(X|Y)*P(Y)/P(X), return P(Y|X).

    .. note:: The multiplication can be in any order
    """
