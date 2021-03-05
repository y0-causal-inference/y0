# -*- coding: utf-8 -*-

"""Data structures."""

from __future__ import annotations

from typing import Iterable, NamedTuple, Optional, Tuple

from .dsl import Expression, Variable

__all__ = [
    'VermaConstraint',
    'ConditionalIndependency',
]


class VermaConstraint(NamedTuple):
    """Represent a Verma constraint."""

    lhs_cfactor: Expression
    lhs_expr: Expression
    rhs_cfactor: Expression
    rhs_expr: Expression
    variables: Tuple[Variable, ...]

    @classmethod
    def from_element(cls, element) -> VermaConstraint:
        """Extract content from each element in the vector returned by `verma.constraint`.

        :param element: An element in the in the vector returned by `verma.constraint`
        :returns: A Verma constraint tuple for the given element

        .. seealso:: Extracting from R objects https://rpy2.github.io/doc/v3.4.x/html/vector.html#extracting-items
        """
        from .parser import parse_causaleffect
        from .r_utils import _extract, _parse_vars
        print(element)
        return cls(
            rhs_cfactor=parse_causaleffect(_extract(element, 'rhs.cfactor')),
            rhs_expr=parse_causaleffect(_extract(element, 'rhs.expr')),
            lhs_cfactor=parse_causaleffect(_extract(element, 'lhs.cfactor')),
            lhs_expr=parse_causaleffect(_extract(element, 'lhs.expr')),
            variables=_parse_vars(element),
        )


class ConditionalIndependency(NamedTuple):
    """A conditional independency."""

    left: str
    right: str
    observations: Tuple[str, ...]

    def __repr__(self):
        left = self.left.name
        right = self.right.name
        observations = tuple(obs.name for obs in self.observations)
        return f"ConditionalIndependency('{left}', '{right}', {observations})"

    @property
    def is_canonical(self) -> bool:
        """Return if the conditional independency is canonical."""
        return self.left < self.right\
                and isinstance(self.observations, tuple)\
                and tuple(sorted(self.observations)) == (self.observations)

    @classmethod
    def create(
        cls,
        left: str,
        right: str,
        observations: Optional[Iterable[str]] = tuple(),
    ) -> ConditionalIndependency:
        """Create a canonical conditional independency."""

        left, right = sorted([left, right])

        observations = tuple(sorted(set(observations)))

        return cls(left, right, observations)
