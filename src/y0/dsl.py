# -*- coding: utf-8 -*-

"""An internal domain-specific language for probability expressions."""

from __future__ import annotations

import functools
import itertools as itt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, TypeVar, Union

__all__ = [
    'Variable',
    'V',
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


def _upgrade_variables(variables):
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
    """Represents a variable, typically with a single letter."""

    #: The name of the variable
    name: str

    def __post_init__(self):
        if self.name == 'P':
            raise ValueError('trust me, P is a bad variable name.')

    def to_text(self) -> str:  # noqa:D102
        return self.name

    def to_latex(self) -> str:  # noqa:D102
        return self.to_text()

    def intervene(self, interventions: XList[Union[Variable, Intervention]]) -> CounterfactualVariable:
        """Intervene on the given variable or variables.

        :param interventions: A variable (or intervention) instance or list of variables (or interventions).

        .. note:: This function can be accessed with the matmult @ operator.
        """
        return CounterfactualVariable(
            name=self.name,
            interventions=_upgrade_variables(interventions),
        )

    def __matmul__(self, interventions: XList[Intervention]):
        return self.intervene(interventions)

    def given(self, parents: XList[Variable]) -> ConditionalProbability:
        return ConditionalProbability(
            child=self,
            parents=_upgrade_variables(parents),
        )

    def __or__(self, parents: XList[Variable]) -> ConditionalProbability:
        return self.given(parents)

    def joint(self, children: XList[Variable]) -> JointProbability:
        return JointProbability([self, *_upgrade_variables(children)])

    def __and__(self, children: XList[Variable]) -> JointProbability:
        return self.joint(children)

    def star(self) -> Intervention:
        return Intervention(name=self.name, star=True)

    def __invert__(self):
        return self.star()

    @classmethod
    def __class_getitem__(cls, item) -> Variable:
        return Variable(item)


V = Variable


@dataclass(frozen=True)
class Intervention(Variable):
    #: The name of the intervention
    name: str
    #: If true, indicates this intervention represents a value different from what was observed
    star: bool = False

    def to_text(self) -> str:  # noqa:D102
        return f'{self.name}*' if self.star else self.name

    def to_latex(self) -> str:  # noqa:D102
        return f'{self.name}^*' if self.star else self.name

    def __invert__(self):
        return Intervention(name=self.name, star=not self.star)


@dataclass(frozen=True)
class CounterfactualVariable(Variable):
    #: The name of the counterfactual variable
    name: str
    #: The interventions on the variable. Should be non-empty
    interventions: List[Intervention]

    def to_text(self) -> str:  # noqa:D102
        intervention_latex = ','.join(intervention.to_text() for intervention in self.interventions)
        return f'{self.name}_{{{intervention_latex}}}'

    def to_latex(self) -> str:  # noqa:D102
        intervention_latex = ','.join(intervention.to_latex() for intervention in self.interventions)
        return f'{self.name}_{{{intervention_latex}}}'

    def intervene(self, interventions: XList[Intervention]) -> CounterfactualVariable:
        interventions = _upgrade_variables(interventions)
        self._raise_for_overlapping_interventions(interventions)
        return CounterfactualVariable(
            name=self.name,
            interventions=[*self.interventions, *interventions],
        )

    def _raise_for_overlapping_interventions(self, new_interventions: List[Intervention]):
        overlaps = {
            new
            for old, new in itt.product(self.interventions, new_interventions)
            if old.name == new.name
        }
        if overlaps:
            raise ValueError(f'Overlapping interventions in new interventions: {overlaps}')


@dataclass
class JointProbability(_Mathable):
    children: List[Variable]

    def to_text(self) -> str:  # noqa:D102
        return ','.join(child.to_text() for child in self.children)

    def to_latex(self) -> str:  # noqa:D102
        return ','.join(child.to_latex() for child in self.children)

    def joint(self, children: XList[Variable]) -> JointProbability:
        return JointProbability([
            *self.children,
            *_upgrade_variables(children),
        ])

    def __and__(self, children: XList[Variable]) -> JointProbability:
        return self.joint(children)


@dataclass
class ConditionalProbability(_Mathable):
    child: Variable
    parents: List[Variable]

    def to_text(self) -> str:  # noqa:D102
        parents = ','.join(parent.to_text() for parent in self.parents)
        return f'{self.child.to_text()}|{parents}'

    def to_latex(self) -> str:  # noqa:D102
        parents = ','.join(parent.to_latex() for parent in self.parents)
        return f'{self.child.to_latex()}|{parents}'

    def given(self, parents: XList[Variable]) -> ConditionalProbability:
        return ConditionalProbability(
            child=self.child,
            parents=[*self.parents, *_upgrade_variables(parents)]
        )

    def __or__(self, parents: XList[Variable]) -> ConditionalProbability:
        return self.given(parents)


class Expression(_Mathable, ABC):
    def _repr_latex_(self) -> str:  # hack for auto-display of latex in jupyter notebook
        return f'${self.to_latex()}$'

    def __mul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError


class Probability(Expression):
    def __init__(
        self,
        probability: Union[Variable, List[Variable], ConditionalProbability, JointProbability],
        *args,
    ):
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
    expression: Expression
    # The variables over which the sum is done. Defaults to an empty list, meaning no variables.
    ranges: List[Variable] = field(default_factory=list)

    def to_text(self) -> str:  # noqa:D102
        ranges = ','.join(r.to_text() for r in self.ranges)
        return f'[ sum_{{{ranges}}} {self.expression.to_text()} ]'

    def to_latex(self) -> str:  # noqa:D102
        ranges = ','.join(r.to_latex() for r in self.ranges)
        return rf'\sum_{{{ranges}}} {self.expression.to_latex()}'

    def __mul__(self, expression: Expression):
        if isinstance(expression, Product):
            return Product([self, expression.expressions])
        else:
            return Product([self, expression])

    def __truediv__(self, expression: Expression) -> Fraction:
        return Fraction(self, expression)

    @classmethod
    def __class_getitem__(cls, ranges: Union[Variable, Tuple[Variable, ...]]) -> Callable[[Expression], Sum]:
        if isinstance(ranges, tuple):
            ranges = list(ranges)
        else:  # a single element is not given as a tuple, such as in Sum[T]
            ranges = [ranges]
        return functools.partial(Sum, ranges=ranges)


@dataclass
class Fraction(Expression):
    numerator: Expression
    denominator: Expression

    def to_text(self) -> str:  # noqa:D102
        return f'frac_{{{self.numerator.to_text()}}}{{{self.denominator.to_text()}}}'

    def to_latex(self) -> str:  # noqa:D102
        return rf'\frac{{{self.numerator.to_latex()}}}{{{self.denominator.to_latex()}}}'

    def __mul__(self, expression: Expression):
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


A, B, C, D, Q, S, T, W, X, Y, Z = map(Variable, 'ABCDQSTWXYZ')
