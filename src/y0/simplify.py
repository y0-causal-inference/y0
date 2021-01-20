# -*- coding: utf-8 -*-

"""An implementation of the simplification algorithm from [tikka2018]_.

.. [tikka2018] Tikka, S., & Karvanen, J. (2018). `Simplifying probabilistic expressions in causal
   inference <https://arxiv.org/abs/1806.07082>`_. arXiv 1806.07082.
"""

from typing import Final, Sequence, Union

from ananke.graphs import ADMG

from .dsl import Expression, Fraction, Probability, Product, Sum
from .graph import NxMixedGraph

__all__ = [
    'simplify',
]


def has_markov_postcondition(expression: Expression) -> bool:
    """Check that the expression is a sum/product of markov kernels.

    :param expression: Any expression
    :return: if the expression satisfies the sum/product of markov kernels condition
    """
    if isinstance(expression, Probability):
        return expression.distribution.is_markov_kernel()
    elif isinstance(expression, Product):
        return all(
            has_markov_postcondition(subexpr)
            for subexpr in expression.expressions
        )
    elif isinstance(expression, Sum):
        return has_markov_postcondition(expression.expression)
    elif isinstance(expression, Fraction):
        return has_markov_postcondition(expression.numerator) and has_markov_postcondition(expression.denominator)
    else:
        raise TypeError


def simplify(expression: Expression, ordering: Union[NxMixedGraph, ADMG, Sequence[str]]) -> Expression:
    """Simplify an expression."""
    simplifier = Simplifier(ordering)
    rv = simplifier.simplify(expression)
    assert has_markov_postcondition(rv)
    return rv


class Simplifier:
    """A data structure to support application of the simplify algorithm."""

    #: The constant topological ordering
    ordering: Final[Sequence[str]]

    def __init__(self, ordering: Union[NxMixedGraph, ADMG, Sequence[str]]):
        if isinstance(ordering, NxMixedGraph):
            ordering = ordering.to_admg()
        if isinstance(ordering, ADMG):
            ordering = ordering.topological_sort()
        self.ordering = ordering

    # The simplify function is implemented inside a class since there is shared state between recursive calls
    #  and this makes it much easier to reference rather than making the simplify function itself have many
    #  arguments
    def simplify(self, expression: Expression) -> Expression:
        """Simplify the expression given the internal topological ordering."""
        raise NotImplementedError
