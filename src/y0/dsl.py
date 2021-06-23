# -*- coding: utf-8 -*-

"""An internal domain-specific language for probability expressions."""

from __future__ import annotations

import functools
import itertools as itt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from inspect import isgenerator
from operator import attrgetter
from typing import (
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

__all__ = [
    "Variable",
    "Intervention",
    "CounterfactualVariable",
    "Distribution",
    "P",
    "Probability",
    "Sum",
    "Product",
    "Fraction",
    "Expression",
    "One",
    "Q",
    "QFactor",
    "A",
    "B",
    "C",
    "D",
    "R",
    "S",
    "T",
    "W",
    "X",
    "Y",
    "Z",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
    "Y1",
    "Y2",
    "Y3",
    "Y4",
    "Y5",
    "Y6",
    "Z1",
    "Z2",
    "Z3",
    "Z4",
    "Z5",
    "Z6",
    # Helpers
    "ensure_ordering",
]

X = TypeVar("X")
XSeq = Union[X, Sequence[X]]


def _upgrade_variables(variables: XSeq[Variable]) -> Tuple[Variable, ...]:
    return (variables,) if isinstance(variables, Variable) else tuple(variables)


def _to_interventions(variables: Sequence[Variable]) -> Tuple[Intervention, ...]:
    return tuple(
        variable
        if isinstance(variable, Intervention)
        else Intervention(name=variable.name, star=False)
        for variable in variables
    )


class _Mathable(ABC):
    @abstractmethod
    def to_text(self) -> str:
        """Output this DSL object in the internal string format."""

    @abstractmethod
    def to_latex(self) -> str:
        """Output this DSL object in the LaTeX string format."""

    def _repr_latex_(self) -> str:  # hack for auto-display of latex in jupyter notebook
        return f"${self.to_latex()}$"

    def __str__(self) -> str:
        return self.to_text()

    @abstractmethod
    def _iter_variables(self) -> Iterable[Variable]:
        """Iterate over variables."""

    def get_variables(self) -> Set[Variable]:
        """Get the set of variables used in this expression."""
        return set(self._iter_variables())


@dataclass(frozen=True)
class Variable(_Mathable):
    """A variable, typically with a single letter."""

    #: The name of the variable
    name: str

    def __post_init__(self):
        if self.name in {"P", "Q"}:
            raise ValueError(f"trust me, {self.name} is a bad variable name.")

    @classmethod
    def norm(cls, name: Union[str, Variable]) -> Variable:
        """Automatically upgrade a string to a variable."""
        if isinstance(name, str):
            return Variable(name)
        elif isinstance(name, Variable):
            return name
        else:
            raise TypeError

    def to_text(self) -> str:
        """Output this variable in the internal string format."""
        return self.name

    def to_latex(self) -> str:
        """Output this variable in the LaTeX string format."""
        return self.to_text()

    def intervene(self, variables: XSeq[Variable]) -> CounterfactualVariable:
        """Intervene on this variable with the given variable(s).

        :param variables: The variable(s) used to extend this variable as it is changed to a
            counterfactual variable
        :returns: A new counterfactual variable over this variable with the given intervention(s).

        .. note:: This function can be accessed with the matmult @ operator.
        """
        return CounterfactualVariable(
            name=self.name,
            interventions=_to_interventions(_upgrade_variables(variables)),
        )

    def __matmul__(self, variables: XSeq[Variable]) -> CounterfactualVariable:
        return self.intervene(variables)

    def given(self, parents: Union[XSeq[Variable], Distribution]) -> Distribution:
        """Create a distribution in which this variable is conditioned on the given variable(s).

        The new distribution is a Markov Kernel.

        :param parents: A variable or list of variables to include as conditions in the new conditional distribution
        :returns: A new conditional probability distribution
        :raises TypeError: If a distribution is given as the parents that contains conditionals

        .. note:: This function can be accessed with the or | operator.
        """
        if not isinstance(parents, Distribution):
            return Distribution(
                children=(self,),
                parents=_upgrade_variables(parents),
            )
        elif parents.is_conditioned():
            raise TypeError("can not be given a distribution that has conditionals")
        else:
            # The parents variable is actually a Distribution instance with no parents,
            #  so its children become the parents for the new Markov Kernel distribution
            return Distribution(
                children=(self,),
                parents=parents.children,  # don't think about this too hard
            )

    def __or__(self, parents: XSeq[Variable]) -> Distribution:
        return self.given(parents)

    def joint(self, children: XSeq[Variable]) -> Distribution:
        """Create a joint distribution between this variable and the given variable(s).

        :param children: The variable(s) for use with this variable in a joint distribution
        :returns: A new joint distribution over this variable and the given variables.

        .. note:: This function can be accessed with the and & operator.
        """
        return Distribution(
            children=(self, *_upgrade_variables(children)),
        )

    def __and__(self, children: XSeq[Variable]) -> Distribution:
        return self.joint(children)

    def invert(self) -> Intervention:
        """Create an :class:`Intervention` variable that is different from what was observed (with a star)."""
        return Intervention(name=self.name, star=True)

    def __invert__(self) -> Intervention:
        return self.invert()

    def __neg__(self) -> Intervention:
        return Intervention(name=self.name, star=False)

    @classmethod
    def __class_getitem__(cls, item) -> Variable:
        return Variable(item)

    def _iter_variables(self) -> Iterable[Variable]:
        """Get a set containing this variable."""
        yield self

    def __lt__(self, other):
        return str(self) < str(other)

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
        return f"{self.name}*" if self.star else self.name

    def to_latex(self) -> str:
        """Output this intervention variable in the LaTeX string format."""
        return f"{self.name}^*" if self.star else self.name

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
    interventions: Tuple[Intervention, ...]

    def __post_init__(self):
        if not self.interventions:
            raise ValueError("should give at least one intervention")
        for intervention in self.interventions:
            if not isinstance(intervention, Intervention):
                raise TypeError(
                    f"only Intervention instances are allowed."
                    f" Got: ({intervention.__class__.__name__}) {intervention}",
                )

    def get_parent(self) -> Variable:
        """Get the parent variable."""
        return Variable(name=self.name)

    def to_text(self) -> str:
        """Output this counterfactual variable in the internal string format."""
        intervention_latex = _list_to_text(self.interventions)
        return f"{self.name}_{{{intervention_latex}}}"

    def to_latex(self) -> str:
        """Output this counterfactual variable in the LaTeX string format."""
        intervention_latex = _list_to_latex(self.interventions)
        return f"{self.name}_{{{intervention_latex}}}"

    def intervene(self, variables: XSeq[Variable]) -> CounterfactualVariable:
        """Intervene on this counterfactual variable with the given variable(s).

        :param variables: The variable(s) used to extend this counterfactual variable's
            current interventions. Automatically converts variables to interventions.
        :returns: A new counterfactual variable with both this counterfactual variable's interventions
            and the given intervention(s)

        .. note:: This function can be accessed with the matmult @ operator.
        """
        _variables = _upgrade_variables(variables)
        _interventions = _to_interventions(_variables)
        self._raise_for_overlapping_interventions(_interventions)
        return CounterfactualVariable(
            name=self.name,
            interventions=(*self.interventions, *_interventions),
        )

    def _raise_for_overlapping_interventions(self, interventions: Iterable[Intervention]) -> None:
        """Raise an error if any of the given variables are already listed in interventions in this counterfactual.

        :param interventions: Interventions to check for overlap
        :raises ValueError: If there are overlapping variables given.
        """
        overlaps = {
            new
            for old, new in itt.product(self.interventions, interventions)
            if old.name == new.name
        }
        if overlaps:
            raise ValueError(f"Overlapping interventions in new interventions: {overlaps}")

    def invert(self) -> Intervention:
        """Raise an error, since counterfactuals can't be inverted the same as normal variables or interventions."""
        raise NotImplementedError

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the union of this variable and its interventions."""
        yield from super()._iter_variables()
        for intervention in self.interventions:
            yield from intervention._iter_variables()


@dataclass(frozen=True)
class Distribution(_Mathable):
    """A general distribution over several child variables, conditioned by several parents."""

    children: Tuple[Variable, ...]
    parents: Tuple[Variable, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if isinstance(self.children, (list, Variable)):
            raise TypeError(f"children of wrong type: {type(self.children)}")
        if isinstance(self.parents, (list, Variable)):
            raise TypeError
        if not self.children:
            raise ValueError("distribution must have at least one child")

    def to_text(self) -> str:
        """Output this distribution in the internal string format."""
        children = _list_to_text(self.children)
        if self.parents:
            parents = _list_to_text(self.parents)
            return f"{children}|{parents}"
        else:
            return children

    def to_latex(self) -> str:
        """Output this distribution in the LaTeX string format."""
        children = _list_to_latex(self.children)
        if self.parents:
            parents = _list_to_latex(self.parents)
            return f"{children}|{parents}"
        else:
            return children

    def is_conditioned(self) -> bool:
        """Return if this distribution is conditioned."""
        return 0 < len(self.parents)

    def is_markov_kernel(self) -> bool:
        """Return if this distribution a markov kernel -> one child variable and one or more conditionals."""
        return len(self.children) == 1

    def uncondition(self) -> Distribution:
        """Return a new distribution that is not conditioned on the parents."""
        return Distribution(
            children=(*self.children, *self.parents),
        )

    def joint(self, children: XSeq[Variable]) -> Distribution:
        """Create a new distribution including the given child variables.

        :param children: The variable(s) with which this distribution's children are extended
        :returns: A new distribution.

        .. note:: This function can be accessed with the and & operator.
        """
        return Distribution(
            children=(*self.children, *_upgrade_variables(children)),
            parents=self.parents,
        )

    def __and__(self, children: XSeq[Variable]) -> Distribution:
        return self.joint(children)

    def given(self, parents: Union[XSeq[Variable], Distribution]) -> Distribution:
        """Create a new mixed distribution additionally conditioned on the given parent variables.

        :param parents: The variable(s) with which this distribution's parents are extended
        :returns: A new distribution
        :raises TypeError: If a distribution is given as the parents that contains conditionals

        .. note:: This function can be accessed with the or | operator.
        """
        if not isinstance(parents, Distribution):
            return Distribution(
                children=self.children,
                parents=(*self.parents, *_upgrade_variables(parents)),
            )
        elif parents.is_conditioned():
            raise TypeError("can not be given a distribution that has conditionals")
        else:
            # The parents variable is actually a Distribution instance with no parents,
            #  so its children get appended as parents for the new mixed distribution
            return Distribution(
                children=self.children,
                parents=(
                    *self.parents,
                    *parents.children,
                ),  # don't think about this too hard
            )

    def __or__(self, parents: XSeq[Variable]) -> Distribution:
        return self.given(parents)

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the set of variables used in this distribution."""
        for variable in itt.chain(self.children, self.parents):
            yield from variable._iter_variables()


class Expression(_Mathable, ABC):
    """The abstract class representing all expressions."""

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __truediv__(self, other):
        pass


@dataclass(frozen=True)
class Probability(Expression):
    """The probability over a distribution."""

    #: The distribution over which the probability is expressed
    distribution: Distribution

    def to_text(self) -> str:
        """Output this probability in the internal string format."""
        return f"P({self.distribution.to_text()})"

    def to_latex(self) -> str:
        """Output this probability in the LaTeX string format."""
        return f"P({self.distribution.to_latex()})"

    def __repr__(self):
        return f"P({repr(self.distribution)})"

    def __mul__(self, other: Expression) -> Expression:
        if isinstance(other, Product):
            return Product((self, *other.expressions))
        elif isinstance(other, Fraction):
            return Fraction(self * other.numerator, other.denominator)
        else:
            return Product((self, other))

    def __truediv__(self, expression: Expression) -> Fraction:
        return Fraction(self, expression)

    def uncondition(self) -> Probability:
        """Return a new probability where the underlying distribution is no longer conditioned by the parents.

        :returns: A new probability over a distribution over the children and parents of the previous distribution

        >>> from y0.dsl import P, A, B
        >>> P(A | B).uncondition() == P(A, B)
        """
        return Probability(self.distribution.uncondition())

    def marginalize(self, ranges: XSeq[Variable]) -> Fraction:
        """Return this probability distribution, marginalized by the given variables.

        :param ranges: A variable or list of variables over which to marganilize this probability's distribution
        :returns: A fraction in which the denominator is represents the sum over the given ranges

        >>> from y0.dsl import P, A, B
        >>> P(A, B).marginalize(A) == P(A, B) / Sum[A](P(A, B))
        """
        return self / Sum(expression=self, ranges=_prepare_ranges(ranges))

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the set of variables used in the distribution in this probability."""
        yield from self.distribution._iter_variables()


def P(  # noqa:N802
    distribution: Union[
        str,
        Sequence[str],
        Set[str],
        Iterator[str],
        Variable,
        Sequence[Variable],
        Set[Variable],
        Iterator[Variable],
        Distribution,
    ],
    *args: Union[str, Variable],
) -> Probability:
    """Create a probability expression over the given variable(s) or distribution.

    :param distribution: If given a :class:`Distribution`, creates a probability expression
        directly over the distribution. If given variable or list of variables, conveniently
        creates a :class:`Distribtion` with the variable(s) as children.
    :param args: If the first argument (``distribution``) was given as a single variable, the
        ``args`` variadic argument can be used to specify a list of additional variables.
    :returns: A probability object
    :raises TypeError: If the distribution is the wrong type
    :raises ValueError: If varidic args are used incorrectly (i.e., in combination with a
        list of variables or :class:`Distribution`.

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

    Creation with a single string:

    >>> from y0.dsl import P, A
    >>> P('A')

    Creation with a list of :class:`Variable`:

    >>> from y0.dsl import P, A, B
    >>> P((A, B))

    Creation with a list of strings:

    >>> from y0.dsl import P, A, B
    >>> P(('A', 'B'))

    Creation with a list of :class:`Variable`: using variadic arguments:

    >>> from y0.dsl import P, A, B
    >>> P(A, B)

    Creation with a list of strings using variadic arguments:

    >>> from y0.dsl import P, A, B
    >>> P('A', 'B')

    Creation with a fancy generator of variables:

    >>> from y0.dsl import P, A, B
    >>> P(Variable(name) for name in 'AB')

    Creation with a fancy generator of strings:

    >>> from y0.dsl import P, A, B
    >>> P(name for name in 'AB')
    """
    if isinstance(distribution, Distribution):
        return Probability(distribution=distribution)

    children: Tuple[Variable, ...]
    if isinstance(distribution, (str, Variable)):
        children = (Variable.norm(distribution), *_upgrade_ordering(args))
    elif isinstance(distribution, (list, tuple, set)) or isgenerator(distribution):
        if args:
            raise ValueError("can not use variadic arguments with combination of first arg")
        children = _sorted_variables(_upgrade_ordering(distribution))
    else:
        raise TypeError(f"invalid distribution type: {type(distribution)} {distribution}")
    return Probability(distribution=Distribution(children=children))


@dataclass(frozen=True)
class Product(Expression):
    """Represent the product of several probability expressions."""

    expressions: Tuple[Expression, ...]

    @classmethod
    def safe(cls, it: Iterable[Expression]) -> Product:
        """Construct a product from any iterable of expressions."""
        return cls(expressions=tuple(it))

    def to_text(self):
        """Output this product in the internal string format."""
        return " ".join(expression.to_text() for expression in self.expressions)

    def to_latex(self):
        """Output this product in the LaTeX string format."""
        return " ".join(expression.to_latex() for expression in self.expressions)

    def __mul__(self, other: Expression):
        if isinstance(other, Product):
            return Product((*self.expressions, *other.expressions))
        elif isinstance(other, Fraction):
            return Fraction(self * other.numerator, other.denominator)
        else:
            return Product((*self.expressions, other))

    def __truediv__(self, expression: Expression) -> Fraction:
        return Fraction(self, expression)

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the union of the variables used in each expresison in this product."""
        for expression in self.expressions:
            yield from expression._iter_variables()


def _list_to_text(elements: Iterable[_Mathable]) -> str:
    return ",".join(element.to_text() for element in elements)


def _list_to_latex(elements: Iterable[_Mathable]) -> str:
    return ",".join(element.to_latex() for element in elements)


@dataclass(frozen=True)
class Sum(Expression):
    """Represent the sum over an expression over an optional set of variables."""

    #: The expression over which the sum is done
    expression: Expression
    #: The variables over which the sum is done. Defaults to an empty list, meaning no variables.
    ranges: Tuple[Variable, ...] = field(default_factory=tuple)

    @classmethod
    def safe(cls, *, expression: Expression, ranges: Iterable[Union[str, Variable]]) -> Sum:
        """Construct a sum from an expresion and a permissive set of things in the ranges."""
        return cls(expression=expression, ranges=_upgrade_ordering(ranges))

    def to_text(self) -> str:
        """Output this sum in the internal string format."""
        ranges = _list_to_text(self.ranges)
        return f"[ sum_{{{ranges}}} {self.expression.to_text()} ]"

    def to_latex(self) -> str:
        """Output this sum in the LaTeX string format."""
        ranges = _list_to_latex(self.ranges)
        return rf"\sum_{{{ranges}}} {self.expression.to_latex()}"

    def __mul__(self, expression: Expression):
        if isinstance(expression, Product):
            return Product((self, *expression.expressions))
        else:
            return Product((self, expression))

    def __truediv__(self, expression: Expression) -> Fraction:
        return Fraction(self, expression)

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the union of the variables used in the range of this sum and variables in its summand."""
        yield from self.expression._iter_variables()
        for variable in self.ranges:
            yield from variable._iter_variables()

    @classmethod
    def __class_getitem__(
        cls, ranges: Union[Variable, Tuple[Variable, ...]]
    ) -> Callable[[Expression], Sum]:
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


def _prepare_ranges(ranges: XSeq[Variable]) -> Tuple[Variable, ...]:
    if isinstance(ranges, Variable):  # a single element is not given as a tuple, such as in Sum[T]
        return (ranges,)
    return tuple(ranges)


@dataclass(frozen=True)
class Fraction(Expression):
    """Represents a fraction of two expressions."""

    #: The expression in the numerator of the fraction
    numerator: Expression
    #: The expression in the denominator of the fraction
    denominator: Expression

    def to_text(self) -> str:
        """Output this fraction in the internal string format."""
        return f"frac_{{{self.numerator.to_text()}}}{{{self.denominator.to_text()}}}"

    def to_latex(self) -> str:
        """Output this fraction in the LaTeX string format."""
        return rf"\frac{{{self.numerator.to_latex()}}}{{{self.denominator.to_latex()}}}"

    def __mul__(self, expression: Expression) -> Fraction:
        if isinstance(expression, Fraction):
            return Fraction(
                self.numerator * expression.numerator,
                self.denominator * expression.denominator,
            )
        else:
            return Fraction(self.numerator * expression, self.denominator)

    def __truediv__(self, expression: Expression) -> Fraction:
        if isinstance(expression, Fraction):
            return Fraction(
                self.numerator * expression.denominator,
                self.denominator * expression.numerator,
            )
        elif isinstance(expression, One):
            return self
        else:
            return Fraction(self.numerator, self.denominator * expression)

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the set of variables used in the numerator and denominator of this fraction."""
        yield from self.numerator._iter_variables()
        yield from self.denominator._iter_variables()

    def simplify(self) -> Expression:
        """Simplify this fraction."""
        if isinstance(self.denominator, One):
            return self.numerator
        if isinstance(self.numerator, One):
            return self
        if self.numerator == self.denominator:
            return One()
        if isinstance(self.numerator, Product) and isinstance(self.denominator, Product):
            return self._simplify_parts(self.numerator.expressions, self.denominator.expressions)
        elif isinstance(self.numerator, Product):
            return self._simplify_parts(self.numerator.expressions, [self.denominator])
        elif isinstance(self.denominator, Product):
            return self._simplify_parts([self.numerator], self.denominator.expressions)
        return self

    @classmethod
    def _simplify_parts(
        cls, numerator: Sequence[Expression], denominator: Sequence[Expression]
    ) -> Expression:
        """Calculate the minimum fraction.

        :param numerator: A sequence of expressions that are multiplied in the product in the numerator
        :param denominator: A sequence of expressions that are multiplied in the product in the denominator
        :returns: A simplified fraction.
        """
        new_numerator, new_denominator = cls._simplify_parts_helper(numerator, denominator)
        if new_numerator and new_denominator:
            return Fraction(
                _expression_or_product(new_numerator),
                _expression_or_product(new_denominator),
            )
        elif new_numerator:
            return _expression_or_product(new_numerator)
        elif new_denominator:
            return One() / _expression_or_product(new_denominator)
        else:
            return One()

    @staticmethod
    def _simplify_parts_helper(
        numerator: Sequence[Expression],
        denominator: Sequence[Expression],
    ) -> Tuple[Tuple[Expression, ...], Tuple[Expression, ...]]:
        numerator_cancelled = set()
        denominator_cancelled = set()
        for i, n_expr in enumerate(numerator):
            for j, d_expr in enumerate(denominator):
                if j in denominator_cancelled:
                    continue
                if n_expr == d_expr:
                    numerator_cancelled.add(i)
                    denominator_cancelled.add(j)
                    break
        return (
            tuple(expr for i, expr in enumerate(numerator) if i not in numerator_cancelled),
            tuple(expr for i, expr in enumerate(denominator) if i not in denominator_cancelled),
        )


def _expression_or_product(e: Sequence[Expression]) -> Expression:
    if not e:
        raise ValueError
    if 1 == len(e):
        return e[0]
    return Product(tuple(e))


class One(Expression):
    """The multiplicative identity (1)."""

    def to_text(self) -> str:
        """Output this identity variable in the internal string format."""
        return "1"

    def to_latex(self) -> str:
        """Output this identity instance in the LaTeX string format."""
        return "1"

    def __rmul__(self, expression: Expression) -> Expression:
        return expression

    def __mul__(self, expression: Expression) -> Expression:
        return expression

    def __truediv__(self, other: Expression) -> Fraction:
        return Fraction(self, other)

    def __eq__(self, other):
        return isinstance(other, One)  # all ones are equal

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the set of variables used in this expression."""
        return iter([])


@dataclass(frozen=True)
class QFactor(Expression):
    """A function from the variables in the domain to a probability function over variables in the codomain."""

    domain: Tuple[Variable, ...]
    codomain: Tuple[Variable, ...]

    @classmethod
    def __class_getitem__(cls, codomain: Union[Variable, Tuple[Variable, ...]]):
        """Create a partial Q Factor object over the given codomain.

        :param codomain: The variables over which the partial Q Factor will be done
        :returns: A partial :class:`QFactor` that can be called solely on an expression

        Example single variable codomain Q expression:

        >>> from y0.dsl import Sum, Q, A, B, C
        >>> Q[C](A, B)

        Example multiple variable codomain Q expression:

        >>> from y0.dsl import Sum, Q, A, B, C, D
        >>> Q[C, D](A, B)
        """

        def _helper(*domain: Variable):
            return QFactor(domain=domain, codomain=_prepare_ranges(codomain))

        return _helper

    def to_text(self) -> str:
        """Output this fraction in the internal string format."""
        codomain = _list_to_latex(self.codomain)
        domain = _list_to_text(self.domain)
        return f"Q[{codomain}]({domain})"

    def to_latex(self) -> str:
        """Output this fraction in the LaTeX string format."""
        codomain = _list_to_latex(self.codomain)
        domain = _list_to_text(self.domain)
        return rf"Q_{{{codomain}}}({{{domain}}})"

    def __mul__(self, other: Expression):
        if isinstance(other, Product):
            return Product((self, *other.expressions))
        elif isinstance(other, Fraction):
            return Fraction(self * other.numerator, other.denominator)
        else:
            return Product((self, other))

    def __truediv__(self, expression: Expression) -> Fraction:
        if isinstance(expression, Fraction):
            return Fraction(self * expression.denominator, expression.numerator)
        else:
            return Fraction(self, expression)

    def _iter_variables(self) -> Iterable[Variable]:
        yield from self.codomain
        yield from self.domain


Q = QFactor

A, B, C, D, R, S, T, W, X, Y, Z = map(Variable, "ABCDRSTWXYZ")  # type: ignore
V1, V2, V3, V4, V5, V6 = [Variable(f"V{i}") for i in range(1, 7)]
W1, W2, W3, W4, W5, W6 = [Variable(f"W{i}") for i in range(1, 7)]
Y1, Y2, Y3, Y4, Y5, Y6 = [Variable(f"Y{i}") for i in range(1, 7)]
Z1, Z2, Z3, Z4, Z5, Z6 = [Variable(f"Z{i}") for i in range(1, 7)]


def _upgrade_ordering(variables: Iterable[Union[str, Variable]]) -> Tuple[Variable, ...]:
    return tuple(Variable.norm(variable) for variable in variables)


OrderingHint = Optional[Sequence[Union[str, Variable]]]


def ensure_ordering(
    expression: Expression,
    *,
    ordering: OrderingHint = None,
) -> Sequence[Variable]:
    """Get a canonical ordering of the variables in the expression, or pass one through.

    The canonical ordering of the variables in a given expression is based on the alphabetical
    sort order of the variables based on their names.

    :param expression: The expression to get a canonical ordering from.
    :param ordering: A given ordering to pass through if not none, otherwise calculate it.
    :returns: The ordering
    """
    if ordering is not None:
        return _upgrade_ordering(ordering)
    # use alphabetical ordering
    return _sorted_variables(expression.get_variables())


def _sorted_variables(variables: Iterable[Variable]) -> Tuple[Variable, ...]:
    return tuple(sorted(variables, key=attrgetter("name")))


def _get_treatment_variables(variables: set[Variable]) -> set[Variable]:
    return {variable for variable in variables if isinstance(variable, Intervention)}


def _get_outcome_variables(variables: set[Variable]) -> set[Variable]:
    return {variable for variable in variables if not isinstance(variable, Intervention)}


def get_outcomes_and_treatments(*, query: Expression) -> tuple[set[Variable], set[Variable]]:
    """Get outcomes and treatments sets from the query expression."""
    variables = query.get_variables()
    return (
        _get_outcome_variables(variables),
        _get_treatment_variables(variables),
    )


def outcomes_and_treatments_to_query(
    *, outcomes: set[Variable], treatments: Optional[set[Variable]] = None
) -> Expression:
    """Create a query expression from a set of outcome and treatment variables."""
    if not treatments:
        return P(outcomes)
    return P(Variable.norm(y) @ _upgrade_ordering(treatments) for y in outcomes)
