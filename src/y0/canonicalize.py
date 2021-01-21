# -*- coding: utf-8 -*-

"""Implementation of the canonicalization algorithm."""

from typing import Sequence, Union

from .dsl import Distribution, Expression, Fraction, Probability, Product, Sum, Variable
from .predicates import has_markov_postcondition

__all__ = [
    'canonicalize',
]


def canonicalize(expression: Expression, ordering: Sequence[Union[str, Variable]]) -> Expression:
    """Canonicalize an expression that meets the markov condition with respect to the given ordering.

    :param expression: An expression to canonicalize
    :param ordering: A toplogical ordering
    :return: A canonical expression
    :raises ValueError: if the expression does not pass the markov postcondition
    :raises ValueError: if the ordering has duplicates
    """
    if not has_markov_postcondition(expression):
        raise ValueError(f'can not canonicalize expression that does not have the markov postcondition: {expression}')

    ordering = _upgrade_ordering(ordering)
    if len(set(ordering)) != len(ordering):
        raise ValueError(f'ordering has duplicates: {ordering}')

    canonicalizer = Canonicalizer(ordering)
    return canonicalizer.canonicalize(expression)


def _upgrade_ordering(variables: Sequence[Union[str, Variable]]) -> Sequence[Variable]:
    return tuple(
        Variable(variable) if isinstance(variable, str) else variable
        for variable in variables
    )


class Canonicalizer:
    """A data structure to support application of the canonicalize algorithm."""

    def __init__(self, ordering: Sequence[Variable]) -> None:
        """Initialize the canonicalizer.

        :param ordering: A topological ordering over the variables appearing in the expression.
        """
        self.ordering = ordering
        self.ordering_level = {
            variable: level
            for level, variable in enumerate(self.ordering)
        }

    def canonicalize(self, expression: Expression) -> Expression:
        """Canonicalize an expression.

        :param expression: An uncanonicalized expression
        :return: A canonicalized expression
        :raises TypeError: if an object with an invalid type is passed
        """
        if isinstance(expression, Probability):
            return Probability(Distribution(
                children=expression.distribution.children,
                parents=tuple(sorted(expression.distribution.parents, key=self.ordering_level.__getitem__)),
            ))
        elif isinstance(expression, Sum):
            return Sum(
                expression=self.canonicalize(expression.expression),
                ranges=expression.ranges,
            )
        elif isinstance(expression, Product):
            probabilities = []
            other = []
            for subexpr in expression.expressions:
                subexpr = self.canonicalize(subexpr)
                if isinstance(subexpr, Probability):
                    probabilities.append(subexpr)
                else:
                    other.append(subexpr)
            probabilities = sorted(probabilities, key=lambda p: p.distribution.children[0].name)
            other = sorted(other, key=self._nonatomic_key)
            return Product((*probabilities, *other))
        elif isinstance(expression, Fraction):
            return Fraction(
                numerator=self.canonicalize(expression.numerator),
                denominator=self.canonicalize(expression.denominator),
            )
        else:
            raise TypeError

    def _nonatomic_key(self, expression: Expression):
        raise NotImplementedError('nonatomic sort not implemented')
