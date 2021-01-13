# -*- coding: utf-8 -*-

"""An internal domain-specific language for probability expressions."""

from __future__ import annotations

import functools
import itertools as itt
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Set, Tuple, TypeVar, Union

__all__ = [
    'Variable',
    'Intervention',
    'CounterfactualVariable',
    'Distribution',
    'P',
    'Probability',
    'Sum',
    'Product',
    'Fraction',
    'Expression',
    'One',
    'A', 'B', 'C', 'D', 'Q', 'S', 'T', 'W', 'X', 'Y', 'Z',
    'get_variable_names',
]

X = TypeVar('X')
XList = Union[X, List[X]]


def _upgrade_variables(variables: XList[Variable]) -> List[Variable]:
    return [variables] if isinstance(variables, Variable) else variables


class _Mathable(ABC):
    @abstractmethod
    def to_text(self) -> str:
        """Output this DSL object in the internal string format."""

    @abstractmethod
    def to_latex(self) -> str:
        """Output this DSL object in the LaTeX string format."""

    def _repr_latex_(self) -> str:  # hack for auto-display of latex in jupyter notebook
        return f'${self.to_latex()}$'

    def __str__(self) -> str:
        return self.to_text()

    def get_variables(self) -> Set[Variable]:
        """Get the set of variables used in this expression."""
        return set()


@dataclass(frozen=True)
class Variable(_Mathable):
    """A variable, typically with a single letter."""

    #: The name of the variable
    name: str

    def __post_init__(self):
        if self.name == 'P':
            raise ValueError('trust me, P is a bad variable name.')

    def to_text(self) -> str:
        """Output this variable in the internal string format."""
        return self.name

    def to_latex(self) -> str:
        """Output this variable in the LaTeX string format."""
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
            interventions=[var.as_intervention()
                           for var in _upgrade_variables(variables)],
        )

    def __matmul__(self, variables: XList[Variable]) -> CounterfactualVariable:
        return self.intervene(variables)

    def given(self, parents: Union[XList[Variable], Distribution]) -> Distribution:
        """Create a distribution in which this variable is conditioned on the given variable(s).

        The new distribution is a Markov Kernel.

        :param parents: A variable or list of variables to include as conditions in the new conditional distribution
        :returns: A new conditional probability distribution
        :raises TypeError: If a distribution is given as the parents that contains conditionals

        .. note:: This function can be accessed with the or | operator.
        """
        if not isinstance(parents, Distribution):
            return Distribution(
                children=[self],
                parents=_upgrade_variables(parents),
            )
        elif parents.is_conditioned():
            raise TypeError('can not be given a distribution that has conditionals')
        else:
            # The parents variable is actually a Distribution instance with no parents,
            #  so its children become the parents for the new Markov Kernel distribution
            return Distribution(
                children=[self],
                parents=parents.children,  # don't think about this too hard
            )

    def __or__(self, parents: XList[Variable]) -> Distribution:
        return self.given(parents)

    def joint(self, children: XList[Variable]) -> Distribution:
        """Create a joint distribution between this variable and the given variable(s).

        :param children: The variable(s) for use with this variable in a joint distribution
        :returns: A new joint distribution over this variable and the given variables.

        .. note:: This function can be accessed with the and & operator.
        """
        return Distribution(
            children=[self, *_upgrade_variables(children)],
        )

    def __and__(self, children: XList[Variable]) -> Distribution:
        return self.joint(children)

    def invert(self) -> Intervention:
        """Create an :class:`Intervention` variable that is different from what was observed (with a star)."""
        return Intervention(name=self.name, star=True)

    def __invert__(self) -> Intervention:
        return self.invert()

    @classmethod
    def __class_getitem__(cls, item) -> Variable:
        return Variable(item)

    def get_variables(self) -> Set[Variable]:
        """Get the set of variables used in this expression."""
        return {self}

    def as_intervention(self, star=False) -> Intervention:
        """Make a clone of this variable as an intervention."""
        return Intervention(self.name, star)


@dataclass(frozen=True)
class Intervention(Variable):
    """An intervention variable.

    An intervention variable is usually used as a subscript in a :class:`CounterfactualVariable`.
    """

    #: The name of the intervention
    name: str
    #: If true, indicates this intervention represents a value different from what was observed
    star: bool = False

    def to_text(self) -> str:
        """Output this intervention variable in the internal string format."""
        return f'{self.name}*' if self.star else self.name

    def to_latex(self) -> str:
        """Output this intervention variable in the LaTeX string format."""
        return f'{self.name}^*' if self.star else self.name

    def invert(self) -> Intervention:
        """Create an :class:`Intervention` variable that is different from what was observed (with a star)."""
        return Intervention(name=self.name, star=not self.star)

    def as_intervention(self, star=None) -> Intervention:
        """Return this variable as an intervention.

        :param star: Specify a new star-value to obtain a clone.
        :returns: the intervention itself or a clone.
        """
        if star is None or star == self.star:
            return self
        return Intervention(self.name, star)


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

    def __hash__(self):
        return hash(self.to_text())

    def to_text(self) -> str:
        """Output this counterfactual variable in the internal string format."""
        intervention_latex = ','.join(intervention.to_text()
                                      for intervention in self.interventions)
        return f'{self.name}_{{{intervention_latex}}}'

    def to_latex(self) -> str:
        """Output this counterfactual variable in the LaTeX string format."""
        intervention_latex = ','.join(intervention.to_latex()
                                      for intervention in self.interventions)
        return f'{self.name}_{{{intervention_latex}}}'

    def intervene(self, variables: XList[Variable]) -> CounterfactualVariable:
        """Intervene on this counterfactual variable with the given variable(s).

        :param variables: The variable(s) used to extend this counterfactual variable's
            current interventions
        :returns: A new counterfactual variable with both this counterfactual variable's interventions
            and the given intervention(s)

        .. note:: This function can be accessed with the matmult @ operator.
        """
        variables = typing.cast(
            List[Variable], _upgrade_variables(variables))  # type: ignore
        self._raise_for_overlapping_interventions(variables)
        return CounterfactualVariable(
            name=self.name,
            interventions=[var.as_intervention()
                           for var in [*self.interventions, *variables]],
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
            raise ValueError(
                f'Overlapping interventions in new interventions: {overlaps}')

    def invert(self) -> Intervention:
        """Raise an error, since counterfactuals can't be inverted the same as normal variables or interventions."""
        raise NotImplementedError

    def as_intervention(self, star=False) -> Intervention:
        """Return this variable as an intervention. Illegal in this case."""
        raise RuntimeError("No nested interventions")

    def get_variables(self) -> Set[Variable]:
        """Get the set of variables used in this expression."""
        base = super(CounterfactualVariable, self).get_variables()
        for var in self.interventions:
            base.update(var.get_variables())
        return base


@dataclass
class Distribution(_Mathable):
    """A general distribution over several child variables, conditioned by several parents."""

    children: List[Variable]
    parents: List[Variable] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.children, Variable):
            self.children = [self.children]
        if not self.children:
            raise ValueError('distribution must have at least one child')
        if isinstance(self.parents, Variable):
            self.parents = [self.parents]

    def to_text(self) -> str:
        """Output this distribution in the internal string format."""
        children = ','.join(child.to_text() for child in self.children)
        if self.parents:
            parents = ','.join(parent.to_text() for parent in self.parents)
            return f'{children}|{parents}'
        else:
            return children

    def to_latex(self) -> str:
        """Output this distribution in the LaTeX string format."""
        children = ','.join(child.to_latex() for child in self.children)
        parents = ','.join(parent.to_latex() for parent in self.parents)
        return f'{children}|{parents}'

    def is_conditioned(self) -> bool:
        """Return if this distribution is conditioned."""
        return 0 < len(self.parents)

    def is_markov_kernel(self) -> bool:
        """Return if this distribution a markov kernel -> one child variable and one or more conditionals."""
        return len(self.children) == 1

    def joint(self, children: XList[Variable]) -> Distribution:
        """Create a new distribution including the given child variables.

        :param children: The variable(s) with which this distribution's children are extended
        :returns: A new distribution.

        .. note:: This function can be accessed with the and & operator.
        """
        return Distribution(
            children=[*self.children, *_upgrade_variables(children)],
            parents=self.parents,
        )

    def __and__(self, children: XList[Variable]) -> Distribution:
        return self.joint(children)

    def given(self, parents: Union[XList[Variable], Distribution]) -> Distribution:
        """Create a new mixed distribution additionally conditioned on the given parent variables.

        :param parents: The variable(s) with which this distribution's parents are extended
        :returns: A new distribution
        :raises TypeError: If a distribution is given as the parents that contains conditionals

        .. note:: This function can be accessed with the or | operator.
        """
        if not isinstance(parents, Distribution):
            return Distribution(
                children=self.children,
                parents=[*self.parents, *_upgrade_variables(parents)],
            )
        elif parents.is_conditioned():
            raise TypeError('can not be given a distribution that has conditionals')
        else:
            # The parents variable is actually a Distribution instance with no parents,
            #  so its children get appended as parents for the new mixed distribution
            return Distribution(
                children=self.children,
                parents=[*self.parents, *parents.children],  # don't think about this too hard
            )

    def __or__(self, parents: XList[Variable]) -> Distribution:
        return self.given(parents)

    def get_variables(self) -> Set[Variable]:
        """Get the set of variables used in this expression."""
        base = self.children.get_variables()
        for var in self.parents:
            base.update(var.get_variables())
        return base


class Expression(_Mathable, ABC):
    """The abstract class representing all expressions."""

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __truediv__(self, other):
        pass


class Probability(Expression):
    """The probability over a distribution."""

    def __init__(
        self,
        distribution: Union[Variable, List[Variable], Distribution],
        *args: Variable,
    ) -> None:
        """Create a probability expression over the given variable(s) or distribution.

        :param distribution: If given a :class:`Distribution`, creates a probability expression
            directly over the distribution. If given variable or list of variables, conveniently
            creates a :class:`Distribtion` with the variable(s) as children.
        :param args: If the first argument (``distribution``) was given as a single variable, the
            ``args`` variadic argument can be used to specify a list of additional variables.
        :raises ValueError: If varidic args are used incorrectly (i.e., in combination with a
            list of variables or :class:`Distribution`.

        .. note:: This class is so commonly used, that it is aliased as :class:`P`.

        Creation with a conditional distribution:

        >>> from y0.dsl import P, A, B
        >>> P(A | B)

        Creation with a joint distribution:

        >>> from y0.dsl import P, A, B
        >>> P(A & B)

        Creation with a mixed joint/conditional distribution:

        >>> from y0.dsl import P, A, B, C
        >>> P(A & B | C)

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
        if isinstance(distribution, Variable):
            if not args:
                distribution = [distribution]
            elif not all(isinstance(p, Variable) for p in args):
                raise ValueError
            else:
                distribution = [distribution, *args]
        if isinstance(distribution, list):
            distribution = Distribution(children=distribution)
        self.distribution = distribution

    def to_text(self) -> str:
        """Output this probability in the internal string format."""
        return f'P({self.distribution.to_text()})'

    def to_latex(self) -> str:
        """Output this probability in the LaTeX string format."""
        return f'P({self.distribution.to_latex()})'

    def __repr__(self):
        return f'P({repr(self.distribution)})'

    def __eq__(self, other):
        return isinstance(other, Probability) and self.distribution == other.distribution

    def __mul__(self, other: Expression) -> Expression:
        if isinstance(other, Product):
            return Product([self, *other.expressions])
        elif isinstance(other, Fraction):
            return Fraction(self * other.numerator, other.denominator)
        else:
            return Product([self, other])

    def __truediv__(self, expression: Expression) -> Fraction:
        return Fraction(self, expression)

    def get_variables(self) -> Set[Variable]:
        """Get the set of variables used in this expression."""
        return self.probability.get_variables()


P = Probability


@dataclass
class Product(Expression):
    """Represent the product of several probability expressions."""

    expressions: List[Expression]

    def to_text(self):
        """Output this product in the internal string format."""
        return ' '.join(expression.to_text() for expression in self.expressions)

    def to_latex(self):
        """Output this product in the LaTeX string format."""
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

    def get_variables(self) -> Set[Variable]:
        """Get the set of variables used in this expression."""
        base = set()
        for var in self.expressions:
            base.update(var.get_variables())
        return base


@dataclass
class Sum(Expression):
    """Represent the sum over an expression over an optional set of variables."""

    #: The expression over which the sum is done
    expression: Expression
    #: The variables over which the sum is done. Defaults to an empty list, meaning no variables.
    ranges: List[Variable] = field(default_factory=list)

    def to_text(self) -> str:
        """Output this sum in the internal string format."""
        ranges = ','.join(r.to_text() for r in self.ranges)
        return f'[ sum_{{{ranges}}} {self.expression.to_text()} ]'

    def to_latex(self) -> str:
        """Output this sum in the LaTeX string format."""
        ranges = ','.join(r.to_latex() for r in self.ranges)
        return rf'\sum_{{{ranges}}} {self.expression.to_latex()}'

    def __mul__(self, expression: Expression):
        if isinstance(expression, Product):
            return Product([self, *expression.expressions])
        else:
            return Product([self, expression])

    def __truediv__(self, expression: Expression) -> Fraction:
        return Fraction(self, expression)

    def get_variables(self) -> Set[Variable]:
        """Get the set of variables used in this expression."""
        base = self.expression.get_variables()
        for var in self.ranges:
            base.update(var.get_variables())
        return base

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
    # a single element is not given as a tuple, such as in Sum[T]
    elif isinstance(ranges, Variable):
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

    def to_text(self) -> str:
        """Output this fraction in the internal string format."""
        return f'frac_{{{self.numerator.to_text()}}}{{{self.denominator.to_text()}}}'

    def to_latex(self) -> str:
        """Output this fraction in the LaTeX string format."""
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

    def get_variables(self) -> Set[Variable]:
        """Get the set of variables used in this expression."""
        base = self.numerator.get_variables()
        base.update(self.denominator.get_variables())
        return base


class One(Expression):
    """The multiplicative identity (1)."""

    def to_text(self) -> str:
        """Output this identity variable in the internal string format."""
        return '1'

    def to_latex(self) -> str:
        """Output this identity instance in the LaTeX string format."""
        return '1'

    def __rmul__(self, expression: Expression) -> Expression:
        return expression

    def __mul__(self, expression: Expression) -> Expression:
        return expression

    def __truediv__(self, other: Expression) -> Fraction:
        return Fraction(self, other)


A, B, C, D, Q, S, T, W, X, Y, Z = map(Variable, 'ABCDQSTWXYZ')  # type: ignore


# TODO get_variable_names is not an informative name that gives insight into what
#  this function does. Needs to be renamed.
def get_variable_names(expression: _Mathable) -> Set[str]:
    """Get the variable names."""
    return {
        variable.name
        for variable in expression.get_variables()
    }
