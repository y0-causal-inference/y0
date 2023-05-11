# -*- coding: utf-8 -*-

"""Implementation of the canonicalization algorithm."""

from operator import attrgetter
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

from ..dsl import (
    CounterfactualVariable,
    Distribution,
    Expression,
    Fraction,
    One,
    Probability,
    Product,
    Sum,
    Variable,
    Zero,
    ensure_ordering,
)

__all__ = [
    "canonicalize",
    "canonical_expr_equal",
]


def canonicalize(
    expression: Expression, ordering: Optional[Sequence[Union[str, Variable]]] = None
) -> Expression:
    """Canonicalize an expression that meets the markov condition with respect to the given ordering.

    :param expression: An expression to canonicalize
    :param ordering: A toplogical ordering. If none is given, it is assigned by sort order of the variable names.
    :return: A canonical expression
    """
    canonicalizer = Canonicalizer(ensure_ordering(expression, ordering=ordering))
    return canonicalizer.canonicalize(expression)


def _sort_probability_key(probability: Probability) -> Tuple[str, ...]:
    return tuple(child.name for child in probability.children)


class Canonicalizer:
    """A data structure to support application of the canonicalize algorithm."""

    ordering: Sequence[Variable]
    ordering_level: Mapping[str, int]

    def __init__(self, ordering: Sequence[Variable]) -> None:
        """Initialize the canonicalizer.

        :param ordering: A topological ordering over the variables appearing in the expression.

        :raises ValueError: if the ordering has duplicates
        """
        if len(set(ordering)) != len(ordering):
            raise ValueError(f"ordering has duplicates: {ordering}")

        self.ordering = ordering
        self.ordering_level = {variable.name: level for level, variable in enumerate(self.ordering)}

    def _canonicalize_probability(self, expression: Probability) -> Probability:
        return Probability(
            Distribution(
                children=self._sorted(expression.children),
                parents=self._sorted(expression.parents),
            )
        )

    def _sorted(self, variables: Tuple[Variable, ...]) -> Tuple[Variable, ...]:
        return tuple(
            sorted(
                (self._canonicalize_variable(variable) for variable in variables),
                key=self._sorted_key,
            )
        )

    def _canonicalize_variable(self, variable: Variable) -> Variable:
        if isinstance(variable, CounterfactualVariable):
            return CounterfactualVariable(
                name=variable.name,
                star=variable.star,
                interventions=tuple(sorted(variable.interventions)),
            )
        else:
            return variable

    def _sorted_key(self, variable: Variable) -> int:
        return self.ordering_level[variable.name]

    def canonicalize(self, expression: Expression) -> Expression:
        """Canonicalize an expression.

        :param expression: An uncanonicalized expression
        :return: A canonicalized expression
        :raises TypeError: if an object with an invalid type is passed
        """
        if isinstance(expression, Probability):  # atomic
            return self._canonicalize_probability(expression)
        elif isinstance(expression, Sum):
            if not expression.ranges:  # flatten unnecessary sum
                return self.canonicalize(expression.expression)
            return Sum(
                expression=self.canonicalize(expression.expression),
                ranges=self._sorted(expression.ranges),
            )
        elif isinstance(expression, Product):
            if 1 == len(expression.expressions):  # flatten unnecessary product
                return self.canonicalize(expression.expressions[0])

            probabilities = []
            other = []
            for subexpr in _flatten_product(expression):
                subexpr = self.canonicalize(subexpr)
                if isinstance(subexpr, Probability):
                    probabilities.append(subexpr)
                elif isinstance(subexpr, One):
                    continue
                elif isinstance(subexpr, Zero):
                    return Zero()  # If there's an explicit zero, then return zero
                else:
                    other.append(subexpr)
            if not probabilities and not other:
                return One()

            probabilities = sorted(probabilities, key=_sort_probability_key)
            # If other is empty, this is also atomic
            other = sorted(other, key=self._nonatomic_key)
            expressions = (*probabilities, *other)
            if len(expressions) == 1:
                return expressions[0]
            return Product(expressions)
        elif isinstance(expression, Fraction):
            numerator = self.canonicalize(expression.numerator)
            # TODO check if there's a zero in numerator, then return zero if so
            denominator = self.canonicalize(expression.denominator)
            if isinstance(denominator, One):
                return numerator
            if numerator == denominator:
                return One()
            return numerator / denominator  # TODO
        elif isinstance(expression, (One, Zero)):
            return expression
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
            return 0, expression.children[0].name
        elif isinstance(expression, Sum):
            return 1, *self._nonatomic_key(expression.expression)
        elif isinstance(expression, Product):
            inner_keys = (self._nonatomic_key(sexpr) for sexpr in expression.expressions)
            return 2, *inner_keys
        elif isinstance(expression, Fraction):
            return (
                3,
                self._nonatomic_key(expression.numerator),
                self._nonatomic_key(expression.denominator),
            )
        elif isinstance(expression, (One, Zero)):
            return 4, expression.to_text()
            # TODO find an example where this happens
        else:
            raise TypeError


def _flatten_product(product: Product) -> Iterable[Expression]:
    for expression in product.expressions:
        if isinstance(expression, Product):
            yield from _flatten_product(expression)
        else:
            yield expression


def canonical_expr_equal(left: Expression, right: Expression) -> bool:
    """Return True if two expressions are equal after canonicalization."""
    ordering = sorted(left.get_variables() | right.get_variables(), key=attrgetter("name"))
    return canonicalize(left, ordering) == canonicalize(right, ordering)
