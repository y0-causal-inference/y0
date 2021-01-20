# -*- coding: utf-8 -*-

"""An implementation of the simplification algorithm from [tikka2018]_.

.. [tikka2018] Tikka, S., & Karvanen, J. (2018). `Simplifying probabilistic expressions in causal
   inference <https://arxiv.org/abs/1806.07082>`_. arXiv 1806.07082.
"""

from typing import Optional, Sequence, Union

from ananke.graphs import ADMG

from .dsl import Distribution, Expression, Fraction, Probability, Product, Sum, Variable
from .graph import NxMixedGraph

__all__ = [
    'simplify',
]


def has_markov_postcondition(expression: Expression) -> bool:
    """Check that the expression is a sum/product of markov kernels.

    :param expression: Any expression
    :return: if the expression satisfies the sum/product of markov kernels condition
    :raises TypeError: if an object with an invalid type is passed
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
        raise NotImplementedError


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


def simplify(
    expression: Expression,
    graph: Union[NxMixedGraph, ADMG],
    ordering: Optional[Sequence[Union[str, Variable]]],
) -> Expression:
    """Simplify an expression.

    :param expression: An unsimplified expression
    :param graph: A graph
    :param ordering: An optional topological ordering. If not provided, one will be generated from the graph.
    :returns: A simplified expression
    :raises ValueError: if the simplified expression does not satisfy the markov postcondition
    """
    simplifier = Simplifier(graph=graph, ordering=_upgrade_ordering(ordering) if ordering is not None else None)
    rv = simplifier.simplify(expression)
    if not has_markov_postcondition(rv):
        raise ValueError('raised if the simplified expression does not satisfy the markov postcondition')
    return rv


class Simplifier:
    """A data structure to support application of the simplify algorithm."""

    #: The constant topological ordering
    ordering: Sequence[Variable]

    def __init__(
        self,
        graph: Union[NxMixedGraph, ADMG],
        ordering: Optional[Sequence[Variable]] = None,
    ) -> None:
        if isinstance(graph, NxMixedGraph):
            self.graph = graph.to_admg()
        elif isinstance(graph, ADMG):
            self.graph = graph
        else:
            raise TypeError

        if ordering is None:
            self.ordering = [Variable(name) for name in self.graph.topological_sort()]
        else:
            self.ordering = ordering

    # The simplify function is implemented inside a class since there is shared state between recursive calls
    #  and this makes it much easier to reference rather than making the simplify function itself have many
    #  arguments
    def simplify(self, expression: Expression) -> Expression:
        """Simplify the expression given the internal topological ordering."""
        raise NotImplementedError
