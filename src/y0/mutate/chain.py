# -*- coding: utf-8 -*-

"""Operations for mutating and simplifying expressions."""

from ..dsl import Distribution, OrderingHint, P, Probability, Product, get_canonical_ordering

__all__ = [
    'chain_expand',
]


def chain_expand(p: Probability, *, reorder: bool = True, ordering: OrderingHint = None) -> Product:
    """Expand a joint probability to a product of conditional probabilities on single variables.

    :param p: The given probability expression
    :param reorder: Should the variables be reordered with respect to the ordering before expanding? This is important
        because there are a variety of equivalent expressions that can't be directly matched.
    :param ordering: An ordering to be used if ``reorder`` is true. If none, automatically generates a canonical
        ordering using :func:`y0.dsl.get_canonical_ordering`.
    :return: A product representing the expanded distribution, in which each probability term is a markov kernel
    """
    # TODO how does this change for a conditioned distribution P(X, Y | Z)? Does Z just come along for the ride?
    assert not p.distribution.is_conditioned()

    if reorder:
        _ordering = get_canonical_ordering(p, ordering=ordering)
        assert all(v in _ordering for v in p.distribution.children)
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
