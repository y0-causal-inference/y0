# -*- coding: utf-8 -*-

"""Operations for mutating and simplifying expressions."""

from ..dsl import Distribution, Fraction, OrderingHint, P, Probability, Product, ensure_ordering

__all__ = [
    'chain_expand',
    'fraction_expand',
]


def chain_expand(p: Probability, *, reorder: bool = True, ordering: OrderingHint = None) -> Product:
    r"""Expand a probability distribution to a product of conditional probabilities on single variables.

    :param p: The given probability expression
    :param reorder: Should the variables be reordered with respect to the ordering before expanding? This is important
        because there are a variety of equivalent expressions that can't be directly matched.
    :param ordering: An ordering to be used if ``reorder`` is true. If none, automatically generates a canonical
        ordering using :func:`y0.dsl.get_canonical_ordering`.
    :return: A product representing the expanded distribution, in which each probability term is a markov kernel

    :raises NotImplementedError: if the distribution has conditions
    :raises ValueError: if the ordering is passed explicitly and it does not cover all variables

    Two variables:

    .. math::
        P(X,Y)=P(X|Y)*P(Y)

    >>> from y0.dsl import P, X, Y
    >>> assert chain_expand(P(X | Y)) == P(X | Y) * P(Y)

    The recurrence relation for many variables is defined as:

    .. math::
        P(X_n,\dots,X_1) = P(X_n|X_{n-1},\dots,X_1) \times P(X_{n-1},\dots,X_1)

    >>> from y0.dsl import P, X, Y, Z
    >>> chain_expand(P(X, Y, Z)) == P(X | (Y, Z)) * P(Y | Z) * P(Z)
    """
    # TODO how does this change for a conditioned distribution P(X, Y | Z)? Does Z just come along for the ride?
    if p.distribution.is_conditioned():
        raise NotImplementedError

    if reorder:
        _ordering = ensure_ordering(p, ordering=ordering)
        if any(v not in _ordering for v in p.distribution.children):
            raise ValueError
        ordered_children = tuple(
            v
            for v in _ordering
            if v in p.distribution.children
        )
    else:
        ordered_children = p.distribution.children

    return Product(tuple(
        P(Distribution(children=(ordered_children[i],)).given(ordered_children[i + 1:]))
        for i in range(len(ordered_children))
    ))


def fraction_expand(p: Probability) -> Fraction:
    r"""Expand a probability distribution with fractions.

    :param p: The given probability expression
    :returns: A fraction representing the joint distribution

    :raises NotImplementedError: if the distribution is not a markov kernel

    One condition:

    .. math::
        P(A | B) = \frac{P(A,B)}{P(B)}

    Many conditions:

    .. math::
        P(A | X_n, \dots, X_1) = \frac{P(A,X_n,\dots,X_1)}{P(X_n,\dots,X_1)}
    """
    if not p.distribution.is_markov_kernel():
        raise NotImplementedError
    return P(p.distribution.children + p.distribution.parents) / P(p.distribution.parents)
