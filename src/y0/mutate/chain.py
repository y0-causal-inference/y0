# -*- coding: utf-8 -*-

"""Operations for mutating and simplifying expressions."""

import warnings

from ..dsl import (
    Distribution,
    Expression,
    Fraction,
    OrderingHint,
    Probability,
    Product,
    ensure_ordering,
)

__all__ = [
    "chain_expand",
    "fraction_expand",
    "bayes_expand",
]


def chain_expand(
    p: Probability, *, reorder: bool = True, ordering: OrderingHint = None
) -> Expression:
    r"""Expand a probability distribution to a product of conditional probabilities on single variables.

    :param p: The given probability expression
    :param reorder: Should the variables be reordered with respect to the ordering before expanding? This is important
        because there are a variety of equivalent expressions that can't be directly matched.
    :param ordering: An ordering to be used if ``reorder`` is true. If none, automatically generates a canonical
        ordering using :func:`y0.dsl.ensure_ordering`.
    :return: A product representing the expanded distribution, in which each probability term is a markov kernel

    :raises ValueError: if the ordering is passed explicitly and it does not cover all variables

    Two variables:

    .. math::
        P(X,Y)=P(X|Y)*P(Y)

    >>> from y0.dsl import P, X, Y
    >>> assert chain_expand(P(X, Y)) == P(X | Y) * P(Y)

    The recurrence relation for many variables is defined as:

    .. math::
        P(X_n,\dots,X_1) = P(X_n|X_{n-1},\dots,X_1) \times P(X_{n-1},\dots,X_1)

    >>> from y0.dsl import P, A, X, Y, Z
    >>> assert chain_expand(P(X, Y, Z)) == P(X | Y, Z) * P(Y | Z) * P(Z)

    Extra conditions come along for the ride.

    >>> assert chain_expand(P(X, Y, Z | A)) == P(X | Y, Z, A) * P(Y | Z, A) * P(Z | A)
    """
    if reorder:
        _ordering = ensure_ordering(p, ordering=ordering)
        if any(v not in _ordering for v in p.children):
            raise ValueError
        ordered_children = tuple(v for v in _ordering if v in p.children)
    else:
        ordered_children = p.children

    return Product.safe(
        p._new(
            Distribution(children=(ordered_children[i],)).given(
                ordered_children[i + 1 :] + p.parents
            )
        )
        for i in range(len(ordered_children))
    )


def fraction_expand(p: Probability) -> Expression:
    r"""Expand a probability distribution with fractions.

    :param p: The given probability expression
    :returns: A fraction representing the joint distribution

    The simple case has one child variable ($A$) and one parent variable ($B$):

    .. math::
        P(A | B) = \frac{P(A,B)}{P(B)}

    >>> from y0.dsl import P, A, B, Sum
    >>> from y0.mutate.chain import fraction_expand
    >>> assert fraction_expand(P(A | B)) == P(A, B) / P(B)

    If there are no conditions (i.e., parents), then the probability
    is returned without modification.

    >>> assert fraction_expand(P(A, B)) == P(A, B)

    In general, with many children $Y_i$ and many parents $X_i$:

    .. math::
        P(Y_1,\dots,Y_n | X_1, \dots, X_m) = \frac{P(Y_1,\dots,Y_n,X_1,\dots,X_m)}{P(X_1,\dots,X_m)}
    """
    if not p.parents:
        return p
    return Fraction(p.uncondition(), p._new(Distribution.safe(p.parents)))


def bayes_expand(p: Probability) -> Expression:
    r"""Expand a probability distribution using Bayes' theorem.

    :param p: The given probability expression, with arbitrary number of children and parents
    :returns: A fraction representing the joint distribution

    .. math::
        P(Y_1,\dots,Y_n|X_1,\dots,X_m)
        = \frac{P(Y_1,\dots,Y_n,X_1,\dots,X_m)}{\sum_{Y_1,\dots,Y_n} P(Y_1,\dots,Y_n,X_1,\dots,X_m)}

    >>> from y0.dsl import P, A, B, C, Sum
    >>> from y0.mutate.chain import bayes_expand
    >>> assert bayes_expand(P(A | B)) == P(A, B) / Sum[A](P(A, B)

    If there are no conditions (i.e., parents), then the probability
    is returned without modification.

    >>> assert bayes_expand(P(A, B)) == P(A, B)

    .. note:: This expansion will create a different but equal expression to :func:`fraction_expand`.
    """
    if not p.parents:
        return p
    warnings.warn(
        "Bayes expansion is now auto-normalized to fraction expansion "
        "since introducing new rules in Sum.safe in "
        "https://github.com/y0-causal-inference/y0/pull/159. Simply use fraction_expand() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return p.uncondition().normalize_marginalize(p.children)
