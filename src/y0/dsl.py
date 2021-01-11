# -*- coding: utf-8 -*-

"""An internal domain-specific language for probability expressions."""

from __future__ import annotations

import functools
import itertools as itt
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, TypeVar, Union

__all__ = [
    'Variable',
    'Intervention',
    'CounterfactualVariable',
    'ConditionalProbability',
    'JointProbability',
    'P',
    'Probability',
    'Sum',
    'Product',
    'Fraction',
    'Expression',
    'One',
    'A', 'B', 'C', 'D', 'Q', 'S', 'T', 'W', 'X', 'Y', 'Z',
]

X = TypeVar('X')
XList = Union[X, List[X]]


def _upgrade_variables(variables: XList[Variable]) -> List[Variable]:
    return [variables] if isinstance(variables, Variable) else variables


class _Mathable(ABC):
    @abstractmethod
    def to_text(self) -> str:
        """Output this DSL object in internal string format."""

    @abstractmethod
    def to_latex(self) -> str:
        """Output this DSL object in LaTeX string format."""

    def __str__(self) -> str:
        return self.to_text()


@dataclass(frozen=True)
class Variable(_Mathable):
    """A variable, typically with a single letter."""

    #: The name of the variable
    name: str

    def __post_init__(self):
        if self.name == 'P':
            raise ValueError('trust me, P is a bad variable name.')

    def to_text(self) -> str:  # noqa:D102
        return self.name

    def to_latex(self) -> str:  # noqa:D102
        return self.to_text()

    def intervene(self, variables: XList[Variable]) -> CounterfactualVariable:
        """Intervene on this variable with the given variable(s).

        :param variables: The variable(s) used to extend this variable as it is changed to a
            counterfactual variable
        :returns: A new counterfactual variable over this variable with the given intervention(s).

        .. note:: This function can be accessed with the matmult @ operator.
        """
        return CounterfactualVariable(
            name=self.name,
            interventions=_upgrade_variables(variables),
        )

    def __matmul__(self, variables: XList[Variable]) -> CounterfactualVariable:
        return self.intervene(variables)

    def given(self, parents: XList[Variable]) -> ConditionalProbability:
        """Create a distribution in which this variable is conditioned on the given variable(s).

        :param parents: A variable or list of variables to include as conditions in the new conditional distribution
        :returns: A new conditional probability distribution

        .. note:: This function can be accessed with the or | operator.
        """
        return ConditionalProbability(
            child=self,
            parents=_upgrade_variables(parents),
        )

    def __or__(self, parents: XList[Variable]) -> ConditionalProbability:
        return self.given(parents)

    def joint(self, children: XList[Variable]) -> JointProbability:
        """Create a joint distribution between this variable and the given variable(s).

        :param children: The variable(s) for use with this variable in a joint distribution
        :returns: A new joint distribution over this variable and the given variables.

        .. note:: This function can be accessed with the and & operator.
        """
        return JointProbability([self, *_upgrade_variables(children)])

    def __and__(self, children: XList[Variable]) -> JointProbability:
        return self.joint(children)

    def invert(self) -> Intervention:
        """Create an :class:`Intervention` variable that is different from what was observed (with a star)."""
        return Intervention(name=self.name, star=True)

    def __invert__(self) -> Intervention:
        return self.invert()

    @classmethod
    def __class_getitem__(cls, item) -> Variable:
        return Variable(item)


@dataclass(frozen=True)
class Intervention(Variable):
    """An intervention variable.

    An intervention variable is usually used as a subscript in a :class:`CounterfactualVariable`.
    """

    #: The name of the intervention
    name: str
    #: If true, indicates this intervention represents a value different from what was observed
    star: bool = False

    def to_text(self) -> str:  # noqa:D102
        return f'{self.name}*' if self.star else self.name

    def to_latex(self) -> str:  # noqa:D102
        return f'{self.name}^*' if self.star else self.name

    def invert(self) -> Intervention:
        """Create an :class:`Intervention` variable that is different from what was observed (with a star)."""
        return Intervention(name=self.name, star=not self.star)


@dataclass(frozen=True)
class CounterfactualVariable(Variable):
    """A counterfactual variable.

    Counterfactual variables are like normal variables, but can have a list of interventions.
    Each intervention is either the same as what was observed (no star) or different from what
    was observed (star).
    """

    #: The name of the counterfactual variable
    name: str
    #: The interventions on the variable. Should be non-empty
    interventions: List[Variable]

    def to_text(self) -> str:  # noqa:D102
        intervention_latex = ','.join(intervention.to_text() for intervention in self.interventions)
        return f'{self.name}_{{{intervention_latex}}}'

    def to_latex(self) -> str:  # noqa:D102
        intervention_latex = ','.join(intervention.to_latex() for intervention in self.interventions)
        return f'{self.name}_{{{intervention_latex}}}'

    def intervene(self, variables: XList[Variable]) -> CounterfactualVariable:
        """Intervene on this counterfactual variable with the given variable(s).

        :param variables: The variable(s) used to extend this counterfactual variable's
            current interventions
        :returns: A new counterfactual variable with both this counterfactual variable's interventions
            and the given intervention(s)

        .. note:: This function can be accessed with the matmult @ operator.
        """
        variables = typing.cast(List[Variable], _upgrade_variables(variables))  # type: ignore
        self._raise_for_overlapping_interventions(variables)
        return CounterfactualVariable(
            name=self.name,
            interventions=[*self.interventions, *variables],
        )

    def _raise_for_overlapping_interventions(self, variables: List[Variable]) -> None:
        """Raise an error if any of the given variables are already listed in interventions in this counterfactual.

        :param variables: Variables to check for overlap
        :raises ValueError: If there are overlapping variables given.
        """
        overlaps = {
            new
            for old, new in itt.product(self.interventions, variables)
            if old.name == new.name
        }
        if overlaps:
            raise ValueError(f'Overlapping interventions in new interventions: {overlaps}')

    def invert(self) -> Intervention:
        """Raise an error, since counterfactuals can't be inverted the same as normal variables or interventions."""
        raise NotImplementedError


@dataclass
class JointProbability(_Mathable):
    """A joint probability distribution over several variables."""

    children: List[Variable]

    def to_text(self) -> str:  # noqa:D102
        return ','.join(child.to_text() for child in self.children)

    def to_latex(self) -> str:  # noqa:D102
        return ','.join(child.to_latex() for child in self.children)

    def joint(self, children: XList[Variable]) -> JointProbability:
        """Create a joint distribution between the variables in this distribution the given variable(s).

        :param children: The variable(s) with which this joint distribution is extended
        :returns: A new joint distribution over all previous and given variables.

        .. note:: This function can be accessed with the and & operator.
        """
        return JointProbability([
            *self.children,
            *_upgrade_variables(children),
        ])

    def __and__(self, children: XList[Variable]) -> JointProbability:
        return self.joint(children)


@dataclass
class ConditionalProbability(_Mathable):
    """A conditional distribution over a single child variable and one or more parent conditional variables."""

    child: Variable
    parents: List[Variable]

    def to_text(self) -> str:  # noqa:D102
        parents = ','.join(parent.to_text() for parent in self.parents)
        return f'{self.child.to_text()}|{parents}'

    def to_latex(self) -> str:  # noqa:D102
        parents = ','.join(parent.to_latex() for parent in self.parents)
        return f'{self.child.to_latex()}|{parents}'

    def given(self, parents: XList[Variable]) -> ConditionalProbability:
        """Create a new conditional distribution with this distribution's children, parents, and the given parent(s).

        :param parents: A variable or list of variables to include as conditions in the new conditional distribution,
            in addition to the variables already in this conditional distribution
        :returns: A new conditional probability distribution

        .. note:: This function can be accessed with the or | operator.
        """
        return ConditionalProbability(
            child=self.child,
            parents=[*self.parents, *_upgrade_variables(parents)],
        )

    def __or__(self, parents: XList[Variable]) -> ConditionalProbability:
        return self.given(parents)


class Expression(_Mathable, ABC):
    """The abstract class representing all expressions."""

    def _repr_latex_(self) -> str:  # hack for auto-display of latex in jupyter notebook
        return f'${self.to_latex()}$'

    def __mul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError


class Probability(Expression):
    """The probability over a distribution."""

    def __init__(
        self,
        probability: Union[Variable, List[Variable], ConditionalProbability, JointProbability],
        *args: Variable,
    ) -> None:
        """Create a probability expression over the given variable(s) or distribution.

        :param probability: If given a :class:`ConditionalProbability` or :class:`JointProbability`,
            creates a probability expression directly over the distribution. If given variable or
            list of variables, conveniently creates a :class:`JointProbability` over the variable(s)
            first.
        :param args: If the first argument (``probability``) was given as a single variable, the
            ``args`` variadic argument can be used to specify a list of additiona variables.
        :raises ValueError: If varidic args are used incorrectly (i.e., in combination with a
            list of variables, :class:`ConditionalProbability`, or :class:`JointProbability`.

        .. note:: This class is so commonly used, that it is aliased as :class:`P`.

        Creation with a :class:`ConditionalProbability`:

        >>> from y0.dsl import P, A, B
        >>> P(A | B)

        Creation with a :class:`JointProbability`:

        >>> from y0.dsl import P, A, B
        >>> P(A & B)

        Creation with a single :class:`Variable`:

        >>> from y0.dsl import P, A
        >>> P(A)

        Creation with a list of :class:`Variable`:

        >>> from y0.dsl import P, A, B
        >>> P([A, B])

        Creation with a list of :class:`Variable`: using variadic arguments:

        >>> from y0.dsl import P, A, B
        >>> P(A, B)
        """
        if isinstance(probability, Variable):
            if not args:
                probability = [probability]
            elif not all(isinstance(p, Variable) for p in args):
                raise ValueError
            else:
                probability = [probability, *args]
        if isinstance(probability, list):
            probability = JointProbability(probability)
        self.probability = probability

    def to_text(self) -> str:  # noqa:D102
        return f'P({self.probability.to_text()})'

    def to_latex(self) -> str:  # noqa:D102
        return f'P({self.probability.to_latex()})'

    def __repr__(self):
        return f'P({repr(self.probability)})'

    def __eq__(self, other):
        return isinstance(other, Probability) and self.probability == other.probability

    def __mul__(self, other: Expression) -> Expression:
        if isinstance(other, Product):
            return Product([self, *other.expressions])
        elif isinstance(other, Fraction):
            return Fraction(self * other.numerator, other.denominator)
        else:
            return Product([self, other])

    def __truediv__(self, expression: Expression) -> Fraction:
        return Fraction(self, expression)


P = Probability


@dataclass
class Product(Expression):
    """Represent the product of several probability expressions."""

    expressions: List[Expression]

    def to_text(self):  # noqa:D102
        return ' '.join(expression.to_text() for expression in self.expressions)

    def to_latex(self):  # noqa:D102
        return ' '.join(expression.to_latex() for expression in self.expressions)

    def __mul__(self, other: Expression):
        if isinstance(other, Product):
            return Product([*self.expressions, *other.expressions])
        elif isinstance(other, Fraction):
            return Fraction(self * other.numerator, other.denominator)
        else:
            return Product([*self.expressions, other])

    def __truediv__(self, expression: Expression) -> Fraction:
        return Fraction(self, expression)


@dataclass
class Sum(Expression):
    """Represent the sum over an expression over an optional set of variables."""

    #: The expression over which the sum is done
    expression: Expression
    #: The variables over which the sum is done. Defaults to an empty list, meaning no variables.
    ranges: List[Variable] = field(default_factory=list)

    def to_text(self) -> str:  # noqa:D102
        ranges = ','.join(r.to_text() for r in self.ranges)
        return f'[ sum_{{{ranges}}} {self.expression.to_text()} ]'

    def to_latex(self) -> str:  # noqa:D102
        ranges = ','.join(r.to_latex() for r in self.ranges)
        return rf'\sum_{{{ranges}}} {self.expression.to_latex()}'

    def __mul__(self, expression: Expression):
        if isinstance(expression, Product):
            return Product([self, *expression.expressions])
        else:
            return Product([self, expression])

    def __truediv__(self, expression: Expression) -> Fraction:
        return Fraction(self, expression)

    @classmethod
    def __class_getitem__(cls, ranges: Union[Variable, Tuple[Variable, ...]]) -> Callable[[Expression], Sum]:
        """Create a partial sum object over the given ranges.

        :param ranges: The variables over which the partial sum will be done
        :returns: A partial :class:`Sum` that can be called solely on an expression

        Example single variable sum:

        >>> from y0.dsl import Sum, P, A, B
        >>> Sum[B](P(A | B) * P(B))

        Example multiple variable sum:

        >>> from y0.dsl import Sum, P, A, B, C
        >>> Sum[B, C](P(A | B) * P(B))
        """
        return functools.partial(Sum, ranges=_prepare_ranges(ranges))


def _prepare_ranges(ranges: Union[Variable, Tuple[Variable, ...]]) -> List[Variable]:
    if isinstance(ranges, tuple):
        return list(ranges)
    elif isinstance(ranges, Variable):  # a single element is not given as a tuple, such as in Sum[T]
        return [ranges]
    else:
        raise TypeError


@dataclass
class Fraction(Expression):
    """Represents a fraction of two expressions."""

    #: The expression in the numerator of the fraction
    numerator: Expression
    #: The expression in the denominator of the fraction
    denominator: Expression

    def to_text(self) -> str:  # noqa:D102
        return f'frac_{{{self.numerator.to_text()}}}{{{self.denominator.to_text()}}}'

    def to_latex(self) -> str:  # noqa:D102
        return rf'\frac{{{self.numerator.to_latex()}}}{{{self.denominator.to_latex()}}}'

    def __mul__(self, expression: Expression) -> Fraction:
        if isinstance(expression, Fraction):
            return Fraction(self.numerator * expression.numerator, self.denominator * expression.denominator)
        else:
            return Fraction(self.numerator * expression, self.denominator)

    def __truediv__(self, expression: Expression) -> Fraction:
        if isinstance(expression, Fraction):
            return Fraction(self.numerator * expression.denominator, self.denominator * expression.numerator)
        else:
            return Fraction(self.numerator, self.denominator * expression)


class One(Expression):
    """The multiplicative identity (1)."""

    def to_text(self) -> str:  # noqa:D102
        return '1'

    def to_latex(self) -> str:  # noqa:D102
        return '1'

    def __rmul__(self, expression: Expression) -> Expression:
        return expression

    def __mul__(self, expression: Expression) -> Expression:
        return expression

    def __truediv__(self, other: Expression) -> Fraction:
        return Fraction(self, other)


A, B, C, D, Q, S, T, W, X, Y, Z = map(Variable, 'ABCDQSTWXYZ')  # type: ignore
