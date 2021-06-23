# -*- coding: utf-8 -*-

"""Data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Iterable, NamedTuple, Optional, Tuple

from .constants import NodeType
from .dsl import Expression, Variable

__all__ = [
    "VermaConstraint",
    "DSeparationJudgement",
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
            rhs_cfactor=parse_causaleffect(_extract(element, "rhs.cfactor")),
            rhs_expr=parse_causaleffect(_extract(element, "rhs.expr")),
            lhs_cfactor=parse_causaleffect(_extract(element, "lhs.cfactor")),
            lhs_expr=parse_causaleffect(_extract(element, "lhs.expr")),
            variables=_parse_vars(element),
        )


@dataclass(frozen=True)
class DSeparationJudgement(Generic[NodeType]):
    """
    Record if a left/right pair are d-separated given the conditions.

    By default, acts like a boolean, but also caries evidence graph.
    """

    separated: bool
    left: NodeType
    right: NodeType
    conditions: Tuple[NodeType, ...]

    @classmethod
    def create(
        cls,
        left: NodeType,
        right: NodeType,
        conditions: Optional[Iterable[NodeType]] = None,
        *,
        separated: bool = True,
    ) -> DSeparationJudgement[NodeType]:
        """Create a d-separation judgement in canonical form."""
        left, right = sorted([left, right])
        if conditions is None:
            conditions = tuple()
        conditions = tuple(sorted(set(conditions)))
        return cls(separated, left, right, conditions)

    def __bool__(self) -> bool:
        return self.separated

    def __repr__(self) -> str:
        return f"{repr(self.separated)} -- '{self.left}' d-sep '{self.right}' conditioned on {self.conditions}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, DSeparationJudgement) and self.separated == other

    @property
    def is_canonical(self) -> bool:
        """Return if the conditional independency is in canonical form."""
        return (
            self.left < self.right
            and isinstance(self.conditions, tuple)
            and tuple(sorted(self.conditions)) == (self.conditions)
        )
