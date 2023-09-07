"""Utilities for mutation functionality."""

from y0.dsl import Expression, Sum, Product, Fraction, Probability, QFactor

__all__ = [
    "Applier",
]


class Applier:
    """A class for building recursive mutation strategies."""

    def apply(self, expression: Expression) -> Expression:
        """Mutate a generic expression."""
        if isinstance(expression, Sum):
            return self.apply_sum(expression)
        if isinstance(expression, Product):
            return self.apply_product(expression)
        if isinstance(expression, Fraction):
            return self.apply_fraction(expression)
        if isinstance(expression, Probability):
            return self.apply_probability(expression)
        if isinstance(expression, QFactor):
            return self.apply_q(expression)
        return expression

    def apply_sum(self, expression: Sum) -> Expression:
        """Mutate a sum."""
        return Sum(expression=self.apply(expression.expression), ranges=expression.ranges)

    def apply_product(self, expression: Product) -> Expression:
        """Mutate a product."""
        return Product.safe(self.apply(e) for e in expression.expressions)

    def apply_fraction(self, expression: Fraction) -> Expression:
        """Mutate a fraction."""
        return Fraction(
            numerator=self.apply(expression.numerator),
            denominator=self.apply(expression.denominator),
        )

    def apply_q(self, expression: QFactor) -> Expression:
        """Mutate a probability."""
        return expression

    def apply_probability(self, expression: Probability) -> Expression:
        """Mutate a probability."""
        return expression
