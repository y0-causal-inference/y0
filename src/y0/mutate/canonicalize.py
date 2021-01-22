# -*- coding: utf-8 -*-

"""Implementation of the canonicalization algorithm."""

from typing import Sequence, Union

from ..dsl import Distribution, Expression, Fraction, Probability, Product, Sum, Variable, _upgrade_ordering
from ..predicates import has_markov_postcondition

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


def _sort_probability_key(probability: Probability) -> str:
    return probability.distribution.children[0].name


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

    def _canonicalize_probability(self, expression: Probability) -> Probability:
        return Probability(Distribution(
            children=expression.distribution.children,
            parents=tuple(sorted(expression.distribution.parents, key=self.ordering_level.__getitem__)),
        ))

    def canonicalize(self, expression: Expression) -> Expression:
        """Canonicalize an expression.

        :param expression: An uncanonicalized expression
        :return: A canonicalized expression
        :raises TypeError: if an object with an invalid type is passed
        """
        if isinstance(expression, Probability):  # atomic
            return self._canonicalize_probability(expression)
        elif isinstance(expression, Sum):
            if isinstance(expression.expression, Probability):  # also atomic
                return expression

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
            probabilities = sorted(probabilities, key=_sort_probability_key)

            # If other is empty, this is also atomic
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
        """Generate a sort key for a *canonical* expression.

        :param expression: A canonical expression
        :returns: A tuple in which the first element is the integer priority for the expression
            and the rest depends on the expression type.
        :raises TypeError: if an invalid expression type is given
        """
        if isinstance(expression, Probability):
            return 0, expression.distribution.children[0].name
        elif isinstance(expression, Sum):
            return 1, *self._nonatomic_key(expression.expression)
        elif isinstance(expression, Product):
            inner_keys = (self._nonatomic_key(sexpr) for sexpr in expression.expressions)
            return 2, *inner_keys
        elif isinstance(expression, Fraction):
            return 3, self._nonatomic_key(expression.numerator), self._nonatomic_key(expression.denominator)
        else:
            raise TypeError
