# -*- coding: utf-8 -*-

"""Implementation of the canonicalization algorithm."""

from typing import Collection, Iterable, Mapping, Optional, Sequence, Tuple, Union

from ..dsl import (
    Distribution,
    Expression,
    Fraction,
    One,
    Probability,
    Product,
    Sum,
    Variable,
    Zero,
    _variable_sort_key,
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
        return expression._new(
            Distribution(
                children=self._sorted(expression.children),
                parents=self._sorted(expression.parents),
            )
        )

    def _sorted(self, variables: Collection[Variable]) -> Tuple[Variable, ...]:
        return tuple(sorted(variables, key=self._sorted_key))

    def _canonicalize_variable(self, variable: Variable) -> Variable:
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
            return Sum.safe(
                expression=self.canonicalize(expression.expression),
                ranges=expression.ranges,
                simplify=True,
            )
        elif isinstance(expression, Product):
            # note: safe already sorts
            return Product.safe(
                self.canonicalize(subexpr) for subexpr in _flatten_product(expression)
            )
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


def _flatten_product(product: Product) -> Iterable[Expression]:
    for expression in product.expressions:
        if isinstance(expression, Product):
            yield from _flatten_product(expression)
        else:
            yield expression


def canonical_expr_equal(left: Expression, right: Expression) -> bool:
    """Return True if two expressions are equal after canonicalization."""
    ordering = sorted(left.get_variables() | right.get_variables(), key=_variable_sort_key)
    return canonicalize(left, ordering) == canonicalize(right, ordering)
