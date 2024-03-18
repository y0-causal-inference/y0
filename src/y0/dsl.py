# -*- coding: utf-8 -*-

r"""An internal domain-specific language for probability expressions.

=======================  ====================================================================
Expression               Description
=======================  ====================================================================
:math:`P(A)`             The probability of A occurring
:math:`P(A^*)`           The probability of A not occurring
:math:`P(A, B)`          The joint probability of A and B occurring
:math:`P(A \mid B)`      The conditional probability of A given B occurring
:math:`P(A \mid B^*)`    The conditional probability of A occurring given B not occurring
:math:`P(A^* \mid B)`    The conditional probability of A not occurring given B occurring
:math:`P(A^* \mid B^*)`  The conditional probability of A not occurring given B not occurring
:math:`\sum_A P(A, B)`   The marginal probability of B
=======================  ====================================================================

Level 3 of Pearl's Causal Hierarchy.

==============================  =================================================
Expression                      Description
==============================  =================================================
:math:`P(Y_X \mid X^*, Y^*)`    Probability of sufficient causation
:math:`P(Y^*_{X^*} \mid X, Y)`  Probability of necessary causation
:math:`P(Y_X, Y^*_{X^*})`       Probability of necessary and sufficient causation
==============================  =================================================
"""

from __future__ import annotations

import functools
import itertools as itt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from operator import attrgetter
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

if TYPE_CHECKING:
    import sympy

__all__ = [
    "Element",
    "Variable",
    "Intervention",
    "CounterfactualVariable",
    "Distribution",
    "Event",
    "P",
    "Probability",
    "Sum",
    "Product",
    "Fraction",
    "Expression",
    "One",
    "Zero",
    "Q",
    "QFactor",
    "A",
    "AA",
    "B",
    "C",
    "D",
    "M",
    "R",
    "S",
    "T",
    "U",
    "W",
    "X",
    "Y",
    "Z",
    "U1",
    "U2",
    "U3",
    "U4",
    "U5",
    "U6",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "W0",
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
    "vmap_adj",
    "vmap_pairs",
    # Transport
    "PopulationProbability",
    "PP",
    "Pi1",
    "Pi2",
    "Pi3",
    "Pi4",
    "Pi5",
    "Pi6",
    "π1",
    "π2",
    "π3",
    "π4",
    "π5",
    "π6",
    "Population",
]

T_co = TypeVar("T_co", covariant=True)


def _to_interventions(variables: Sequence[Variable]) -> Tuple[Intervention, ...]:
    return tuple(
        (
            variable
            if isinstance(variable, Intervention)
            else Intervention(name=variable.name, star=False)
        )
        for variable in variables
    )


class Element(ABC):
    """An element in the y0 internal domain-speific language that can be converted to text, LaTeX, and code."""

    @abstractmethod
    def to_text(self) -> str:
        """Output this DSL object in the internal string format."""

    @abstractmethod
    def to_latex(self) -> str:
        """Output this DSL object in the LaTeX string format."""

    @abstractmethod
    def to_y0(self) -> str:
        """Output this DSL object as y0 python code."""

    def _repr_latex_(self) -> str:  # hack for auto-display of latex in jupyter notebook
        return f"${self.to_latex()}$"

    def __str__(self) -> str:
        return self.to_y0()

    def __repr__(self) -> str:
        return self.to_y0()

    @abstractmethod
    def _iter_variables(self) -> Iterable[Variable]:
        """Iterate over variables."""

    def get_variables(self) -> Set[Variable]:
        """Get the set of variables used in this expression."""
        return set(self._iter_variables())


@dataclass(frozen=True, order=True, repr=False)
class Variable(Element):
    """A variable, typically with a single letter."""

    #: The name of the variable
    name: str
    #: The star status of the variable. None means it's a variable,
    #: False means it's the same as the value for the variable,
    #: and True means it's a different value from the variable.
    star: Optional[bool] = None

    def __post_init__(self):
        if self.name in {"P", "Q", "PP"}:
            raise ValueError(f"trust me, {self.name} is a bad variable name.")

    @classmethod
    def norm(cls, name: Union[str, Variable]) -> Variable:
        """Automatically upgrade a string to a variable."""
        if isinstance(name, str):
            return Variable(name)
        elif isinstance(name, Variable):
            return name
        else:
            raise TypeError(f"({type(name)}) {name} is not valid")

    def get_base(self) -> Variable:
        """Return the base variable, with no other nonsense."""
        return Variable(self.name)

    def to_text(self) -> str:
        """Output this variable in the internal string format."""
        return self.name

    def to_sympy(self) -> "sympy.Symbol":
        """Get the object for sympy."""
        import sympy

        return sympy.Symbol(self.to_latex())

    def to_latex(self) -> str:
        """Output this variable in the LaTeX string format.

        :returns: The LaTeX representaton of this variable.

        >>> Variable('X').to_latex()
        'X'
        >>> Variable('X1').to_latex()
        'X_1'
        >>> Variable('X12').to_latex()
        'X_{12}'
        """
        # if it ends with a number, use that as a subscript
        ending_numeric = 0
        for c in reversed(self.name):
            if c.isnumeric():
                ending_numeric += 1
        if ending_numeric == 0:
            return self.name
        elif ending_numeric == 1:
            return f"{self.name[:-1]}_{self.name[-1]}"
        else:
            return f"{self.name[:-ending_numeric]}_{{{self.name[-ending_numeric:]}}}"

    def to_y0(self) -> str:
        """Output this variable instance as y0 internal DSL code."""
        if self.star is None:
            return self.name
        elif self.star:
            return f"+{self.name}"
        else:
            return f"-{self.name}"

    def intervene(self, variables: VariableHint) -> CounterfactualVariable:
        """Intervene on this variable with the given variable(s).

        :param variables: The variable(s) used to extend this variable as it is changed to a
            counterfactual variable
        :returns: A new counterfactual variable over this variable with the given intervention(s).

        .. note:: This function can be accessed with the matmult @ operator.
        """
        interventions = _to_interventions(_upgrade_variables(variables))
        return CounterfactualVariable(
            name=self.name,
            star=self.star,
            interventions=frozenset(interventions),
        )

    def __matmul__(self, variables: VariableHint) -> CounterfactualVariable:
        return self.intervene(variables)

    def given(self, parents: Union[VariableHint, Distribution]) -> Distribution:
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
                parents=_upgrade_ordering(parents),
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

    def __or__(self, parents: Union[VariableHint, Distribution]) -> Distribution:
        return self.given(parents)

    def joint(self, children: VariableHint) -> Distribution:
        """Create a joint distribution between this variable and the given variable(s).

        :param children: The variable(s) for use with this variable in a joint distribution
        :returns: A new joint distribution over this variable and the given variables.

        .. note:: This function can be accessed with the and & operator.
        """
        return Distribution(
            children=_upgrade_ordering((self, *_upgrade_variables(children))),
        )

    def __and__(self, children: VariableHint) -> Distribution:
        return self.joint(children)

    def _intervention(self, star: bool) -> Variable:
        return Intervention(name=self.name, star=star)

    def invert(self) -> Variable:
        """Create an :class:`Intervention` variable that is different from what was observed (with a star)."""
        return self._intervention(not self.star)

    def __invert__(self) -> Variable:
        return self.invert()

    def __pos__(self) -> Variable:
        return self._intervention(True)

    def __neg__(self) -> Variable:
        return self._intervention(False)

    @classmethod
    def __class_getitem__(cls, item) -> Variable:
        return Variable(item)

    def _iter_variables(self) -> Iterable[Variable]:
        """Get a set containing this variable."""
        yield self


VariableHint = Union[str, Variable, Iterable[Union[str, Variable]]]


@dataclass(frozen=True, order=True, repr=False)
class Intervention(Variable):
    """An intervention variable.

    An intervention variable is usually used as a subscript in a :class:`CounterfactualVariable`.
    """

    def __post_init__(self):
        if self.star is None:
            raise ValueError("Intervention must have a non-None star")

    def to_text(self) -> str:
        """Output this intervention variable in the internal string format."""
        return f"{self.name}*" if self.star else self.name

    def to_latex(self) -> str:
        """Output this intervention variable in the LaTeX string format."""
        latex = super().to_latex()
        return f"{latex}^*" if self.star else latex

    def to_y0(self) -> str:
        """Output this intervention instance as y0 internal DSL code."""
        mark = "+" if self.star else "-"
        return f"{mark}{self.name}"


@dataclass(frozen=True, order=True, repr=False)
class CounterfactualVariable(Variable):
    """A counterfactual variable.

    Counterfactual variables are like normal variables, but can have a list of interventions.
    Each intervention is either the same as what was observed (no star) or different from what
    was observed (star).
    """

    #: The interventions on the variable. Should be non-empty
    interventions: frozenset[Intervention] = field(default_factory=frozenset)

    def __post_init__(self):
        if not self.interventions:
            raise ValueError("should give at least one intervention")
        for intervention in self.interventions:
            if not isinstance(intervention, Intervention):
                raise TypeError(
                    f"only Intervention instances are allowed."
                    f" Got: ({intervention.__class__.__name__}) {intervention}",
                )

    def to_text(self) -> str:
        """Output this counterfactual variable in the internal string format."""
        intervention_latex = _list_to_text(_sort_interventions(self.interventions))
        return f"{self.name}_{{{intervention_latex}}}"

    def to_latex(self) -> str:
        """Output this counterfactual variable in the LaTeX string format.

        :returns: A latex representation of this counterfactual variable

        >>> (Variable('X') @ Variable('Y')).to_latex()
        '{X}_{Y}'
        >>> (Variable('X1') @ Variable('Y')).to_latex()
        '{X_1}_{Y}'
        >>> (Variable('X12') @ Variable('Y')).to_latex()
        '{X_{12}}_{Y}'
        """
        intervention_latex = _list_to_latex(_sort_interventions(self.interventions))
        prefix = "^*" if self.star else ""
        return f"{{{super().to_latex()}}}{prefix}_{{{intervention_latex}}}"

    def to_y0(self) -> str:
        """Output this counterfactual variable instance as y0 internal DSL code."""
        if self.star is None:
            prefix = ""
        elif self.star:
            prefix = "+"
        else:
            prefix = "-"
        if len(self.interventions) == 1:
            return f"{prefix}{self.name} @ {list(self.interventions)[0].to_y0()}"
        else:
            ins = ", ".join(i.to_y0() for i in _sort_interventions(self.interventions))
            return f"{prefix}{self.name} @ ({ins})"

    def is_event(self) -> bool:
        """Return if the counterfactual variable has a value."""
        return self.star is not None

    def has_tautology(self) -> bool:
        """Return if the counterfactual variable contain its own value in the subscript.

        :returns: True if we force a variable X to have a value x and the resulting value of X is x.
        :raises ValueError: if the counterfactual value doesn't have a value assigned
        """
        if not self.is_event():
            raise ValueError(
                "Can not determine the consistency of a counterfactual variable with no value assigned."
            )
        return any(self.name == i.name and self.star == i.star for i in self.interventions)

    def is_inconsistent(self) -> bool:
        """Return if the counterfactual variable violates the Axiom of Effectiveness.

        :returns: True if we force a variable X to have a value x and the resulting value of X is not x
        :raises ValueError: if the counterfactual value doesn't have a value assigned
        """
        if not self.is_event():
            raise ValueError(
                "Can not determine the consistency of a counterfactual variable with no value assigned."
            )
        return any(self.name == i.name and self.star != i.star for i in self.interventions)

    def intervene(self, variables: VariableHint) -> CounterfactualVariable:
        """Intervene on this counterfactual variable with the given variable(s).

        :param variables: The variable(s) used to extend this counterfactual variable's
            current interventions. Automatically converts variables to interventions.
        :returns: A new counterfactual variable with both this counterfactual variable's interventions
            and the given intervention(s).

        .. warning::

            Will raise a value error ff the value of a new intervention conflicts
            with the value of intervention already listed in this counterfactual.

        .. note:: This function can be accessed with the matmult @ operator.
        """
        _interventions = _to_interventions(_upgrade_ordering(variables))
        interventions = {*self.interventions, *_interventions}
        self._raise_for_overlapping_interventions(interventions)
        return CounterfactualVariable(
            name=self.name, star=self.star, interventions=frozenset(interventions)
        )

    @staticmethod
    def _raise_for_overlapping_interventions(interventions: Iterable[Intervention]) -> None:
        """Raise an error if there are two values of the same variable in the list of interventions.

        :param interventions: Interventions to check for overlap
        :raises ValueError: If there are overlapping variables given
        """
        overlaps = {
            (old, new)
            for old, new in itt.product(interventions, repeat=2)
            if old.name == new.name and old.star != new.star
        }
        if overlaps:
            raise ValueError(f"Overlapping interventions in new interventions: {overlaps}")

    def _with_star(self, star: bool) -> CounterfactualVariable:
        return CounterfactualVariable(
            name=self.name,
            star=star,
            interventions=self.interventions,
        )

    def invert(self) -> CounterfactualVariable:
        """Invert the value of the counterfactual variable."""
        return self._with_star(not self.star)

    def __pos__(self) -> CounterfactualVariable:
        return self._with_star(True)

    def __neg__(self) -> CounterfactualVariable:
        return self._with_star(False)

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the union of this variable and its interventions."""
        yield from super()._iter_variables()
        for intervention in self.interventions:
            yield from intervention._iter_variables()


@dataclass(frozen=True)
class Distribution(Element):
    """A general distribution over several child variables, conditioned by several parents.

    P(X | Y) means that X is a child and Y is a parent.
    """

    children: Tuple[Variable, ...]
    parents: Tuple[Variable, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if isinstance(self.children, (list, Variable)):
            raise TypeError(f"children of wrong type: {type(self.children)}")
        if isinstance(self.parents, (list, Variable)):
            raise TypeError
        if not self.children:
            raise ValueError("distribution must have at least one child")

    @classmethod
    def safe(
        cls,
        distribution: Union[VariableHint, Distribution],
        *args: Union[str, Variable, Distribution],
    ) -> Distribution:
        """Create a distribution the given variable(s) or distribution.

        :param distribution: If given a :class:`Distribution`, creates a probability expression
            directly over the distribution. If given variable or list of variables, conveniently
            creates a :class:`Distribution` with the variable(s) as children.
        :param args: If the first argument (``distribution``) was given as a single variable, the
            ``args`` variadic argument can be used to specify a list of additional variables.
        :returns: A Distribution object
        :raises ValueError: If invalid combination of arguments are given.
        """
        if isinstance(distribution, (str, Variable, Distribution)):
            extended_args = [distribution, *args]
            dist_pos = [i for i, e in enumerate(extended_args) if isinstance(e, Distribution)]

            # There are no distributions (e.g., no conditionals were given with the | already)
            if 0 == len(dist_pos):
                return Distribution(
                    children=_upgrade_ordering(cast(VariableHint, extended_args)),
                )

            # A single conditional was given. Everything before it should be considered
            # as child variables, and everything after as parent variables.
            elif 1 == len(dist_pos):
                i = dist_pos[0]
                pre = cast(Iterable[Union[str, Variable]], extended_args[:i])
                dist = cast(Distribution, extended_args[i])
                post = cast(Iterable[Union[str, Variable]], extended_args[i + 1 :])
                return Distribution(
                    children=_sorted_variables((*_upgrade_ordering(pre), *dist.children)),
                    parents=_sorted_variables((*dist.parents, *_upgrade_ordering(post))),
                )

            # Multiple conditionals were detected. This isn't allowed.
            else:
                raise ValueError("can not give multiple distribution objects")
        elif args:
            raise ValueError("can not use args/parents when giving an iterable as first argument")
        else:
            return Distribution(
                children=_upgrade_ordering(distribution),
            )

    def _to_x(self, func: Callable[[Iterable[Variable]], str]) -> str:
        children = func(self.children)
        if not self.parents:
            return children
        return f"{children} | {func(self.parents)}"

    def to_text(self) -> str:
        """Output this distribution in the internal string format."""
        return self._to_x(_list_to_text)

    def to_y0(self) -> str:
        """Output this distribution instance as y0 internal DSL code."""
        return self._to_x(_list_to_y0)

    def to_latex(self) -> str:
        """Output this distribution in the LaTeX string format."""
        return self._to_x(_list_to_latex)

    def is_conditioned(self) -> bool:
        """Return if this distribution is conditioned."""
        return 0 < len(self.parents)

    def is_markov_kernel(self) -> bool:
        """Return if this distribution a markov kernel -> one child variable and one or more conditionals."""
        return len(self.children) == 1

    def intervene(self, variables: VariableHint) -> Distribution:
        """Return a new distribution that has the given intervention(s) on all variables."""
        # check that the variables aren't in any of them yet
        variables = _upgrade_ordering(variables)
        return Distribution(
            children=tuple(child.intervene(variables) for child in self.children),
            parents=tuple(parent.intervene(variables) for parent in self.parents),
        )

    def __matmul__(self, variables: VariableHint) -> Distribution:
        return self.intervene(variables)

    def uncondition(self) -> Distribution:
        """Return a new distribution that is not conditioned on the parents."""
        return Distribution(
            children=(*self.children, *self.parents),
        )

    def joint(self, children: VariableHint) -> Distribution:
        """Create a new distribution including the given child variables.

        :param children: The variable(s) with which this distribution's children are extended
        :returns: A new distribution.

        .. note:: This function can be accessed with the and & operator.
        """
        return Distribution(
            children=_upgrade_ordering((*self.children, *_upgrade_variables(children))),
            parents=self.parents,
        )

    def __and__(self, children: VariableHint) -> Distribution:
        return self.joint(children)

    def given(self, parents: Union[VariableHint, Distribution]) -> Distribution:
        """Create a new mixed distribution additionally conditioned on the given parent variables.

        :param parents: The variable(s) with which this distribution's parents are extended
        :returns: A new distribution
        :raises TypeError: If a distribution is given as the parents that contains conditionals

        .. note:: This function can be accessed with the or | operator.
        """
        # TODO handle duplicate variables in the parents.
        if not isinstance(parents, Distribution):
            return Distribution(
                children=self.children,
                parents=_upgrade_ordering((*self.parents, *_upgrade_variables(parents))),
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

    def __or__(self, parents: Union[VariableHint, Distribution]) -> Distribution:
        return self.given(parents)

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the set of variables used in this distribution."""
        for variable in itt.chain(self.children, self.parents):
            yield from variable._iter_variables()


class Expression(Element, ABC):
    """The abstract class representing all expressions."""

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def _get_key(self) -> tuple:
        """Generate a sort key for a *canonical* expression.

        :returns: A tuple in which the first element is the integer priority for the expression
            and the rest depends on the expression type.
        """
        raise NotImplementedError

    def __lt__(self, other: Expression):
        return self._get_key() < other._get_key()

    def __truediv__(self, expression: Expression) -> Expression:
        """Divide this expression by another and create a fraction."""
        if isinstance(expression, One):
            return self
        elif isinstance(expression, Fraction):
            return Fraction(self * expression.denominator, expression.numerator)
        else:
            return Fraction(self, expression)

    def conditional(self, ranges: VariableHint) -> Expression:
        """Return this expression, conditioned by the given variables.

        :param ranges: A variable or list of variables over which to marginalize this expression
        :returns: A fraction in which the denominator is represents the sum over the given ranges

        >>> from y0.dsl import P, A, B
        >>> assert P(A, B).conditional(A) == P(A, B) / Sum[B](P(A, B))
        >>> assert P(A, B, C).conditional([A, B]) == P(A, B, C) / Sum[C](P(A, B, C))
        """
        ranges = _upgrade_ordering([r.get_base() for r in _upgrade_variables(ranges)])
        ranges_complement = set([c.get_base() for c in self._iter_variables()]) - set(ranges)
        return self.normalize_marginalize(ranges_complement)

    def normalize_marginalize(self, ranges: VariableHint) -> Expression:
        """Return this expression, normalized by this expression marginalized by the given variables."""
        return self / self.marginalize(ranges)

    def marginalize(self, ranges: VariableHint) -> Expression:
        """Return this expression, marginalizing out the given variables.

        :param ranges: A variable or list of variables over which to marginalize this expression
        :returns: The expression but summed over the given variables

        >>> from y0.dsl import P, A, B, C
        >>> assert P(A, B).marginalize(A) == Sum[A](P(A, B))
        >>> assert P(A, B, C).marginalize([A, B]) == Sum[A, B](P(A, B, C))
        """
        return Sum.safe(
            expression=self,
            ranges=_upgrade_ordering([r.get_base() for r in _upgrade_variables(ranges)]),
        )


@dataclass(frozen=True, repr=False)
class Probability(Expression):
    """The probability over a distribution."""

    #: The distribution over which the probability is expressed
    distribution: Distribution

    @classmethod
    def safe(
        cls,
        distribution: DistributionHint,
        *args: Union[str, Variable],
        interventions: Optional[VariableHint] = None,
    ) -> Probability:
        """Create a distribution the given variable(s) or distribution.

        :param distribution: If given a :class:`Distribution`, creates a probability expression
            directly over the distribution. If given variable or list of variables, conveniently
            creates a :class:`Distribution` with the variable(s) as children.
        :param args: If the first argument (``distribution``) was given as a single variable, the
            ``args`` variadic argument can be used to specify a list of additional variables.
        :param interventions: An optional variable or variables to use as interventions.
        :returns: A probability object
        """
        distribution = Distribution.safe(distribution, *args)
        if interventions is not None:
            distribution = distribution.intervene(interventions)
        return Probability(distribution)

    def _get_key(self):
        # TODO incorporate more information from children and parents
        return 0, self.children[0].name

    def to_text(self) -> str:
        """Output this probability in the internal string format."""
        return f"P({self.distribution.to_text()})"

    def _help_level_2_distribution(self):
        # if all parts of distribution have same intervention set, then put it out front
        intervention_sets = {
            x.interventions if isinstance(x, CounterfactualVariable) else tuple()
            for x in itt.chain(self.children, self.parents)
        }
        # check that there's only one intervention set and that it's not an empty one
        if len(intervention_sets) == 1 and (interventions := intervention_sets.pop()):
            unintervened_distribution = Distribution(
                parents=tuple(Variable(name=v.name, star=v.star) for v in self.parents),
                children=tuple(Variable(name=v.name, star=v.star) for v in self.children),
            )
            return interventions, unintervened_distribution
        else:
            return None, None

    def to_y0(self) -> str:
        """Output this probability instance as y0 internal DSL code."""
        interventions, unintervened_distribution = self._help_level_2_distribution()
        if not interventions:
            return f"P({self.distribution.to_y0()})"

        # only keep the + if necessary, otherwise show regular
        intervention_str = ",".join(
            f"+{intervention.name}" if intervention.star else intervention.name
            for intervention in interventions
        )
        return f"P[{intervention_str}]({unintervened_distribution.to_y0()})"

    def to_latex(self) -> str:
        """Output this probability in the LaTeX string format."""
        interventions, unintervened_distribution = self._help_level_2_distribution()
        if not interventions:
            return f"P({self.distribution.to_latex()})"

        intervention_str = ",".join(intervention.to_latex() for intervention in interventions)
        return f"P_{{{intervention_str}}}({unintervened_distribution.to_latex()})"

    @property
    def parents(self) -> Tuple[Variable, ...]:
        """Get the distribution's parents."""
        return self.distribution.parents

    @property
    def children(self) -> Tuple[Variable, ...]:
        """Get the distribution's children."""
        return self.distribution.children

    def is_conditioned(self) -> bool:
        """Return if this distribution is conditioned."""
        return self.distribution.is_conditioned()

    def is_markov_kernel(self) -> bool:
        """Return if this distribution a markov kernel -> one child variable and one or more conditionals."""
        return self.distribution.is_markov_kernel()

    def __mul__(self, other: Expression) -> Expression:
        if isinstance(other, Zero):
            return other
        elif isinstance(other, One):
            return self
        elif isinstance(other, Product):
            return Product.safe((self, *other.expressions))
        elif isinstance(other, Fraction):
            return Fraction(self * other.numerator, other.denominator)
        else:
            return Product.safe((self, other))

    def _new(self, distribution: Distribution):
        # This is implemented this way to make overriding easier
        return Probability(distribution)

    def intervene(self, variables: VariableHint) -> Probability:
        """Return a new probability where the underlying distribution has been intervened by the given variables."""
        return self._new(self.distribution.intervene(variables))

    def __matmul__(self, variables: VariableHint) -> Probability:
        return self.intervene(variables)

    def uncondition(self) -> Probability:
        """Return a new probability where the underlying distribution is no longer conditioned by the parents.

        :returns: A new probability over a distribution over the children and parents of the previous distribution

        >>> from y0.dsl import P, A, B
        >>> P(A | B).uncondition() == P(A, B)
        """
        return self._new(self.distribution.uncondition())

    def conditional(self, ranges: VariableHint) -> Expression:
        """Return this expression, conditioned by the given variables.

        :param ranges: A variable or list of variables over which to marginalize this expression
        :returns: A fraction in which the denominator is represents the sum over the given ranges

        >>> from y0.dsl import P, A, B
        >>> assert P(A, B).conditional(A) == P(A, B) / Sum[B](P(A, B))
        >>> assert P(A, B, C).conditional([A, B]) == P(A, B, C) / Sum[C](P(A, B, C))
        """
        ranges = _upgrade_ordering([r.get_base() for r in _upgrade_variables(ranges)])
        ranges_complement = set(
            [c.get_base() for c in self._iter_variables() if not isinstance(c, Intervention)]
        ) - set(ranges)
        return self.normalize_marginalize(ranges_complement)

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the set of variables used in the distribution in this probability."""
        yield from self.distribution._iter_variables()


DistributionHint = Union[VariableHint, Distribution]


class ProbabilityBuilderType:
    """A base class for building probability distributions."""

    def __call__(
        self,
        distribution: DistributionHint,
        *args: Union[str, Variable],
        interventions: Optional[VariableHint] = None,
    ) -> Probability:
        return Probability.safe(distribution, *args, interventions=interventions)

    def __getitem__(self, interventions: VariableHint):
        """Generate a probability builder closure.

        :param interventions: A variable or variables to intervene on using the do-calculus level 2
            rules, meaning they are all applied to all parent and children variables in the resulting
            expression
        :returns: A function with the same semantics as :meth:`__call__` such that you can build
            a probability expression.

        >>> from y0.dsl import P, W, X, Y, Z
        >>> assert P[X](Y) == P(Y @ X)
        >>> assert P[X](Y, Z) == P(Y @ X & Z @ X)
        >>> assert P[X](Y | Z) == P(Y @ X | Z @ X)
        >>> assert P[X](Y @ Z) == P(Y @ Z @ X)
        >>> assert P[X](Y @ Z | W) == P(Y @ Z @ X | W @ X)
        """
        return functools.partial(self, interventions=interventions)


P = ProbabilityBuilderType()
"""``P`` is a magical object of mystery and wonder that can be used to create :class:`Probability` instances.

It itself is a singleton instance of :class:`ProbabilityBuilderType` and can be used wither via the
:meth:`ProbabilityBuilderType.__call__`, as if it were a function like ``P(Y)`` or it can be used as
a combination with the :meth:`ProbabilityBuilderType.__getitem__` and a call, like ``P[X](Y)`` to
denote interventions using the do-Calculus $L_2$ notation. Here are some examples:

A univariate distribution can be created either with a string or a :class:`Variable`:

>>> from y0.dsl import P, A
>>> P('A') == P(A)

**Multivariate Distributions**

A joint distribution can be created with several strings or :class:`Variable` instances
with variadic arguments:

>>> from y0.dsl import P, A, B
>>> P(A, B) == P('A', 'B')

A joint distribution can also be created with a single argument that is either an iterable
of either strings or :class:`Variable` instances

>>> from y0.dsl import P, A, B
>>> P((A, B)) == P([A, B]) == P(('A', 'B')) == P(['A', 'B'])

This even extends to fancy generators, for which you can omit the parentheses:

Creation with a fancy generator of variables:

>>> from y0.dsl import P, A, B
>>> P(Variable(name) for name in 'AB') == P(name for name in 'AB') == P(A, B)

**Conditional Distributions**

Creation with a conditional distribution:

>>> from y0.dsl import P, A, B
>>> P(A | B)

Creation with a mixed joint/conditional distribution:

>>> from y0.dsl import P, A, B, C
>>> P(A & B | C)

**Specifying an Intervention with L2 do-Calculus Notation**

Intervene on a single variable:

>>> from y0.dsl import P, X, Y
>>> P[X](Y) == P(Y @ X)

Intervene on multiple children:

>>> from y0.dsl import P, X, Y, Z
>>> P[X](Y, Z) == P(Y @ X & Z @ X)

Intervene on multiple parents:

>>> from y0.dsl import P, W, X, Y, Z
>>> P[X](Y | (W, Z)) == P(Y @ X | (W @ X, Z @ X)):

Intervene on both children and parents:

>>> from y0.dsl import P, X, Y, Z
>>> P[X](Y | Z) == P(Y @ X | Z @ X)

Intervene on X on top of previous interventions:

>>> from y0.dsl import P, X, Y, Z
>>> P[X](Y @ Z) == P(Y @ X @ Z)

Allow mixing with L3, where each variable can have different interventions:

>>> from y0.dsl import P, W, X, Y, Z
>>> P[X](Y @ Z | W) == P(Y @ X @ Z | W @ X)

**Specifying Multiple Interventions with L2 do-Calculus Notation**

Multiple interventions on a single variable:

>>> from y0.dsl import P, X1, X2, Y
>>> P[X1, X2](Y) == P(Y @ X)

Multiple interventions  on multiple children:

>>> from y0.dsl import P, X1, X2, Y, Z
>>> P[X1, X2](Y, Z) == P(Y @ X1 @ X2 & Z @ X1 @ X2)

... and so on
"""


@dataclass(frozen=True, repr=False)
class Product(Expression):
    """Represent the product of several probability expressions."""

    expressions: Tuple[Expression, ...]

    def __post_init__(self):
        if len(self.expressions) < 2:
            raise ValueError("Product() must two or more expressions")

    @classmethod
    def safe(cls, expressions: Union[Expression, Iterable[Expression]]) -> Expression:
        """Construct a product from any iterable of expressions.

        :param expressions: An expression or iterable of expressions which should be multiplied
        :returns: A :class:`Product` object

        Standard usage, same as the normal ``__init__``:

        >>> from y0.dsl import Product, X, Y, A, P
        >>> Product.safe((P(X, Y), ))

        Use a list or other iterable:

        >>> Product.safe([P(X), P(Y | X)])

        Use an inline generator:

        >>> Product.safe(P(v) for v in [X, Y])

        Use a single expression:

        >>> Product.safe(P(X, Y))
        """
        if isinstance(expressions, Expression):
            return expressions
        # Remove multiplications of one
        expressions = tuple(expression for expression in expressions if expression != One())
        # If any multiplications are by zero, then return zero
        if any(expression == Zero() for expression in expressions):
            return Zero()
        if not expressions:
            return One()
        if len(expressions) == 1:
            return expressions[0]
        return cls(expressions=tuple(sorted(expressions)))

    def _get_key(self):
        inner_keys = (sexpr._get_key() for sexpr in self.expressions)
        return 2, *inner_keys

    def to_text(self):
        """Output this product in the internal string format."""
        return " ".join(expression.to_text() for expression in self.expressions)

    def to_y0(self) -> str:
        """Output this product instance as y0 internal DSL code."""
        return " * ".join(expr.to_y0() for expr in self.expressions)

    def to_latex(self):
        """Output this product in the LaTeX string format."""
        return " ".join(expression.to_latex() for expression in self.expressions)

    def __mul__(self, other: Expression):
        if isinstance(other, Zero):
            return other
        if isinstance(other, Product):
            return Product.safe((*self.expressions, *other.expressions))
        elif isinstance(other, Fraction):
            return Fraction(self * other.numerator, other.denominator)
        else:
            return Product.safe((*self.expressions, other))

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the union of the variables used in each expresison in this product."""
        for expression in self.expressions:
            yield from expression._iter_variables()


def _list_to_text(elements: Iterable[Element]) -> str:
    return ", ".join(element.to_text() for element in elements)


def _list_to_latex(elements: Iterable[Element]) -> str:
    return ", ".join(element.to_latex() for element in elements)


def _list_to_y0(elements: Iterable[Element]) -> str:
    return ", ".join(element.to_y0() for element in elements)


@dataclass(frozen=True, repr=False)
class Sum(Expression):
    """Represent the sum over an expression over an optional set of variables."""

    #: The expression over which the sum is done
    expression: Expression
    #: The variables over which the sum is done. Defaults to an empty list, meaning no variables.
    ranges: frozenset[Variable]

    def __post_init__(self):
        if not isinstance(self.ranges, frozenset):
            raise TypeError
        if not self.ranges:
            raise ValueError("Sum must have ranges")
        for r in self.ranges:
            if isinstance(r, (CounterfactualVariable, Intervention)):
                raise TypeError("Ranges must not be counterfactuals nor interventions")

    @classmethod
    def safe(
        cls,
        expression: Expression,
        ranges: Union[str, Variable, Iterable[Union[str, Variable]]],
        *,
        simplify: bool = False,
    ) -> Expression:
        """Construct a sum from an expression and a permissive set of things in the ranges.

        :param expression: The expression over which the sum is done
        :param ranges: The variable or list of variables over which the sum is done
        :param simplify: Should the sum be simplified using :func:`Sum.simplify`?
        :returns: A :class:`Sum` object

        Standard usage, same as the normal ``__init__``:

        >>> from y0.dsl import Sum, X, Y, A, P
        >>> Sum.safe(P(X, Y), (X,))

        Use a list or other iterable:

        >>> Sum.safe(P(X, Y), [X])

        Use a single variable:

        >>> Sum.safe(P(X, Y), X)
        """
        if isinstance(ranges, str):
            ranges = (Variable(ranges),)
        elif isinstance(ranges, Variable):
            ranges = (ranges,)
        else:
            ranges = _upgrade_ordering(ranges)
        if not ranges:
            return expression
        if isinstance(expression, Zero):
            return expression
        rv = cls(
            expression=expression,
            ranges=frozenset(ranges),
        )
        if simplify:
            return rv.simplify()
        return rv

    def simplify(self) -> Expression:
        """Simplify this sum."""
        expression = self.expression
        ranges = set(self.ranges)

        # Special case when ranges cover
        if isinstance(expression, Probability) and not expression.parents:  # i.e., no conditions
            children = {
                child.get_base(): child
                for child in expression.children
                # FIXME what happens if same name appears with multiple different counterfactual variables?
                #  this should actually evaluate to zero since that's impossible
            }
            if ranges == set(children):
                return One()
            elif ranges > set(children):
                keep = ranges - set(children)
                return Sum.safe(
                    expression=One(),
                    ranges=frozenset(v for k, v in children.items() if k in keep),
                )
            elif ranges < set(children):
                keep = set(children) - ranges
                return expression._new(
                    Distribution.safe(v for k, v in children.items() if k in keep)
                )
            else:  # partial or no overlap
                intersection = ranges.intersection(children)
                keep = set(children) - intersection
                prob = expression._new(
                    Distribution.safe(v for k, v in children.items() if k in keep)
                )
                return Sum.safe(
                    expression=prob,
                    ranges=ranges - intersection,
                )
        return self

    def _get_key(self):
        return 1, *self.expression._get_key()

    def _get_sorted_ranges(self) -> Sequence[Variable]:
        return sorted(self.ranges, key=attrgetter("name"))

    def to_text(self) -> str:
        """Output this sum in the internal string format."""
        ranges = _list_to_text(self._get_sorted_ranges())
        return f"[ sum_{{{ranges}}} {self.expression.to_text()} ]"

    def to_latex(self) -> str:
        """Output this sum in the LaTeX string format."""
        ranges = _list_to_latex(self._get_sorted_ranges())
        return rf"\sum\limits_{{{ranges}}} {self.expression.to_latex()}"

    def to_y0(self):
        """Output this sum instance as y0 internal DSL code."""
        if isinstance(self.expression, Fraction):
            s = self.expression.to_y0(parens=False)
        else:
            s = self.expression.to_y0()
        if not self.ranges:
            return f"Sum({s})"
        ranges = _list_to_y0(self._get_sorted_ranges())
        return f"Sum[{ranges}]({s})"

    def __mul__(self, expression: Expression):
        if isinstance(expression, Zero):
            return expression
        elif isinstance(expression, Product):
            return Product.safe((self, *expression.expressions))
        else:
            return Product.safe((self, expression))

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the union of the variables used in the range of this sum and variables in its summand."""
        yield from self.expression._iter_variables()
        for variable in self.ranges:
            yield from variable._iter_variables()

    @classmethod
    def __class_getitem__(cls, ranges: VariableHint) -> Callable[[Expression], Expression]:
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
        return functools.partial(Sum.safe, ranges=_upgrade_ordering(ranges))


@dataclass(frozen=True, repr=False)
class Fraction(Expression):
    """Represents a fraction of two expressions."""

    #: The expression in the numerator of the fraction
    numerator: Expression
    #: The expression in the denominator of the fraction
    denominator: Expression

    def __post_init__(self):
        if isinstance(self.denominator, Zero):
            raise ZeroDivisionError

    def _get_key(self):
        return (
            3,
            self.numerator._get_key(),
            self.denominator._get_key(),
        )

    def to_text(self) -> str:
        """Output this fraction in the internal string format."""
        return f"frac_{{{self.numerator.to_text()}}}{{{self.denominator.to_text()}}}"

    def to_latex(self) -> str:
        """Output this fraction in the LaTeX string format."""
        return rf"\frac{{{self.numerator.to_latex()}}}{{{self.denominator.to_latex()}}}"

    def to_y0(self, parens: bool = True) -> str:
        """Output this fraction as y0 internal DSL code."""
        s = f"({self.numerator.to_y0()} / {self.denominator.to_y0()})"
        return f"({s})" if parens else s

    def __mul__(self, expression: Expression) -> Expression:
        if isinstance(expression, Zero):
            return expression
        elif isinstance(expression, Fraction):
            return Fraction(
                self.numerator * expression.numerator,
                self.denominator * expression.denominator,
            )
        else:
            return Fraction(self.numerator * expression, self.denominator)

    def __truediv__(self, expression: Expression) -> Fraction:
        if isinstance(expression, One):
            return self
        elif isinstance(expression, Fraction):
            return Fraction(
                self.numerator * expression.denominator,
                self.denominator * expression.numerator,
            )
        else:
            return Fraction(self.numerator, self.denominator * expression)

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the set of variables used in the numerator and denominator of this fraction."""
        yield from self.numerator._iter_variables()
        yield from self.denominator._iter_variables()

    def flip(self) -> Fraction:
        """Exchange the numerator and denominator."""
        return Fraction(self.denominator, self.numerator)

    def simplify(self) -> Expression:
        """Simplify this fraction."""
        if isinstance(self.denominator, One):
            return self.numerator
        if isinstance(self.numerator, Zero):
            return self.numerator
        if isinstance(self.numerator, One):
            if isinstance(self.denominator, Fraction):
                return self.denominator.flip().simplify()
            else:
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
                Product.safe(new_numerator),
                Product.safe(new_denominator),
            )
        elif new_numerator:
            return Product.safe(new_numerator)
        elif new_denominator:
            return One() / Product.safe(new_denominator)
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


class One(Expression):
    """The multiplicative identity (1)."""

    def to_text(self) -> str:
        """Output this identity variable in the internal string format."""
        return "1"

    def to_latex(self) -> str:
        """Output this identity instance in the LaTeX string format."""
        return "1"

    def to_y0(self) -> str:
        """Output this identity instance as y0 internal DSL code."""
        return "One()"

    def _get_key(self):
        return 4, self.to_text()

    def __rmul__(self, expression: Expression) -> Expression:
        return expression

    def __mul__(self, expression: Expression) -> Expression:
        return expression

    def __eq__(self, other):
        return isinstance(other, One)  # all ones are equal

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the set of variables used in this expression."""
        return iter([])


class Zero(Expression):
    """The additive identity (0)."""

    def to_text(self) -> str:
        """Output this identity variable in the internal string format."""
        return "0"

    def to_latex(self) -> str:
        """Output this identity instance in the LaTeX string format."""
        return "0"

    def to_y0(self) -> str:
        """Output this identity instance as y0 internal DSL code."""
        return "Zero()"

    def _get_key(self):
        return 4, self.to_text()

    def __rmul__(self, expression: Expression) -> Expression:
        return self

    def __mul__(self, expression: Expression) -> Expression:
        return self

    def __truediv__(self, other: Expression) -> Expression:
        if isinstance(other, Zero):
            raise ZeroDivisionError
        return self

    def __eq__(self, other):
        return isinstance(other, Zero)  # all zeros are equal

    def _iter_variables(self) -> Iterable[Variable]:
        """Get the set of variables used in this expression."""
        return iter([])


class QBuilder(Protocol[T_co]):
    """A protocol for annotating the special class getitem functionality of the :class:`QFactor` class."""

    def __call__(self, arg: VariableHint, *args: Union[str, Variable]) -> T_co: ...


@dataclass(frozen=True, repr=False)
class QFactor(Expression):
    """A function from the variables in the domain to a probability function over variables in the codomain."""

    domain: frozenset[Variable]
    codomain: frozenset[Variable]

    @classmethod
    def safe(
        cls,
        domain: VariableHint,
        *args: Union[str, Variable],
        codomain: VariableHint,
    ) -> QFactor:
        """Create a Q factor with various input types."""
        return cls(
            domain=cls._prepare_domain(domain, *args),
            codomain=frozenset(_upgrade_variables(codomain)),
        )

    @staticmethod
    def _prepare_domain(
        arg: VariableHint,
        *args: Union[str, Variable],
    ) -> frozenset[Variable]:
        """Prepare a list of variables from a potentially unruly set of args and variadic args."""
        if isinstance(arg, (str, Variable)):
            return frozenset((Variable.norm(arg), *_upgrade_ordering(args)))
        if args:
            raise ValueError("can not use variadic arguments with combination of first arg")
        return frozenset(_sorted_variables(_upgrade_ordering(arg)))

    @classmethod
    def __class_getitem__(cls, codomain: Union[Variable, Iterable[Variable]]) -> QBuilder[QFactor]:
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
        return functools.partial(cls.safe, codomain=codomain)

    def _get_key(self) -> tuple:
        return -5, min(v.name for v in self.domain), min(v.name for v in self.codomain)

    def _sorted_codomain(self):
        return sorted(self.codomain, key=attrgetter("name"))

    def _sorted_domain(self):
        return sorted(self.domain, key=attrgetter("name"))

    def to_text(self) -> str:
        """Output this Q factor in the internal string format."""
        codomain = _list_to_text(self._sorted_codomain())
        domain = _list_to_text(self._sorted_domain())
        return f"Q[{codomain}]({domain})"

    def to_latex(self) -> str:
        """Output this Q factor in the LaTeX string format."""
        codomain = _list_to_latex(self._sorted_codomain())
        domain = _list_to_latex(self._sorted_domain())
        return rf"Q_{{{codomain}}}({{{domain}}})"

    def to_y0(self) -> str:
        """Output this Q factor instance as y0 internal DSL code."""
        codomain = _list_to_y0(self._sorted_codomain())
        domain = _list_to_y0(self._sorted_domain())
        return f"Q[{codomain}]({domain})"

    def __mul__(self, other: Expression):
        if isinstance(other, Product):
            return Product.safe((self, *other.expressions))
        elif isinstance(other, Fraction):
            return Fraction(self * other.numerator, other.denominator)
        else:
            return Product.safe((self, other))

    def _iter_variables(self) -> Iterable[Variable]:
        yield from self.codomain
        yield from self.domain


Q = QFactor

AA = Variable("AA")
A, B, C, D, E, F, G, M, R, S, T, U, W, X, Y, Z = map(Variable, "ABCDEFGMRSTUWXYZ")  # type: ignore
U1, U2, U3, U4, U5, U6 = [Variable(f"U{i}") for i in range(1, 7)]
V1, V2, V3, V4, V5, V6 = [Variable(f"V{i}") for i in range(1, 7)]
W0, W1, W2, W3, W4, W5, W6 = [Variable(f"W{i}") for i in range(7)]
M0, M1, M2, M3, M4, M5, M6 = [Variable(f"M{i}") for i in range(7)]
X1, X2, X3, X4, X5, X6 = [Variable(f"X{i}") for i in range(1, 7)]
Y1, Y2, Y3, Y4, Y5, Y6 = [Variable(f"Y{i}") for i in range(1, 7)]
Z1, Z2, Z3, Z4, Z5, Z6 = [Variable(f"Z{i}") for i in range(1, 7)]
π1, π2, π3, π4, π5, π6 = Pi1, Pi2, Pi3, Pi4, Pi5, Pi6 = [Variable(f"π{i}") for i in range(1, 7)]


def _sort_interventions(interventions: Iterable[Intervention]) -> Tuple[Intervention, ...]:
    return tuple(sorted(interventions, key=lambda i: (i.name, i.star)))


def _variable_sort_key(variable: Variable) -> tuple[str, str]:
    if isinstance(variable, CounterfactualVariable):
        return variable.name, ",".join(
            i.to_y0() for i in _sort_interventions(variable.interventions)
        )
    else:
        return variable.name, ""


def _sorted_variables(variables: Iterable[Variable]) -> Tuple[Variable, ...]:
    return tuple(sorted(variables, key=_variable_sort_key))


def _upgrade_variables(variables: VariableHint) -> Tuple[Variable, ...]:
    if isinstance(variables, str):
        return (Variable(variables),)
    elif isinstance(variables, Variable):
        return (variables,)
    else:
        return tuple(Variable.norm(variable) for variable in variables)


def _upgrade_ordering(variables: VariableHint) -> Tuple[Variable, ...]:
    return _sorted_variables(set(_upgrade_variables(variables)))


OrderingHint = Optional[Iterable[Union[str, Variable]]]


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


def vmap_pairs(edges: Iterable[Tuple[str, str]]) -> List[Tuple[Variable, Variable]]:
    """Map pair of strings to pairs of variables."""
    return [(Variable(source), Variable(target)) for source, target in edges]


def vmap_adj(adjacency_dict):
    """Map an adjacency dictionary of strings to variables."""
    return {
        Variable(source): [Variable(target) for target in targets]
        for source, targets in adjacency_dict.items()
    }


#: A conjunction of factual and counterfactual events
Event = Dict[Variable, Intervention]

Population = Variable


@dataclass(frozen=True, repr=False)
class PopulationProbability(Probability):
    """A probability that is annotated with a population.

    >>> from y0.dsl import PP, Pi1, Y, X
    >>> # Make a population-annotated probability of Y
    >>> PP[Pi1](Y)
    >>> # Make a conditioned population of Y @ X
    >>> PP[Pi1][X](Y)

    Related publications:
    - `Surrogate Outcomes and Transportability <https://arxiv.org/abs/1806.07172>`_ (Tikka and Karvanen, 2018)
    """

    population: Population

    def _new(self, distribution) -> PopulationProbability:
        return PopulationProbability(population=self.population, distribution=distribution)

    def _get_key(self):
        return -1, self.population, self.children[0].name

    def to_y0(self) -> str:
        """Output this probability instance as y0 internal DSL code."""
        interventions, unintervened_distribution = self._help_level_2_distribution()
        if not interventions:
            return f"P({self.distribution.to_y0()})"

        # only keep the + if necessary, otherwise show regular
        intervention_str = ",".join(
            f"+{intervention.name}" if intervention.star else intervention.name
            for intervention in interventions
        )
        return f"PP[{self.population.to_y0()}][{intervention_str}]({unintervened_distribution.to_y0()})"

    def to_latex(self) -> str:
        """Output this probability in the LaTeX string format."""
        interventions, unintervened_distribution = self._help_level_2_distribution()
        if self.population == TARGET_DOMAIN:
            pop_latex = r"\pi^\ast"
        else:
            pop_latex = self.population.to_latex()

        if not interventions:
            return f"P^{{{pop_latex}}}({self.distribution.to_latex()})"

        intervention_str = ",".join(intervention.to_latex() for intervention in interventions)
        return f"P_{{{intervention_str}}}^{{{pop_latex}}}({unintervened_distribution.to_latex()})"


class PopulationProbabilityBuilderType(ProbabilityBuilderType):
    """A magical type for building population probabilities."""

    def __init__(self, population: Population):
        """Initialize the builder with a given population."""
        self.population = population

    @classmethod
    def __class_getitem__(cls, population: Population) -> "PopulationProbabilityBuilderType":
        """Get a population probability builder class initialized with the given population."""
        return cls(population)

    def __call__(self, *args, **kwargs) -> PopulationProbability:  # noqa:D102
        probability = super().__call__(*args, **kwargs)
        return PopulationProbability(
            population=self.population, distribution=probability.distribution
        )


PP = PopulationProbabilityBuilderType


TARGET_DOMAIN = Population("pi*")
