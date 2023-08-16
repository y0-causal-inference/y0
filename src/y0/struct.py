# -*- coding: utf-8 -*-

"""Data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, NamedTuple, Optional, Tuple

import pandas as pd

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

        return cls(
            rhs_cfactor=parse_causaleffect(_extract(element, "rhs.cfactor")),
            rhs_expr=parse_causaleffect(_extract(element, "rhs.expr")),
            lhs_cfactor=parse_causaleffect(_extract(element, "lhs.cfactor")),
            lhs_expr=parse_causaleffect(_extract(element, "lhs.expr")),
            variables=_parse_vars(element),
        )


@dataclass(frozen=True)
class DSeparationJudgement:
    """
    Record if a left/right pair are d-separated given the conditions.

    By default, acts like a boolean, but also caries evidence graph.
    """

    separated: bool
    left: Variable
    right: Variable
    conditions: Tuple[Variable, ...]

    @classmethod
    def create(
        cls,
        left: Variable,
        right: Variable,
        conditions: Optional[Iterable[Variable]] = None,
        *,
        separated: bool = True,
    ) -> DSeparationJudgement:
        """Create a d-separation judgement in canonical form."""
        left, right = sorted([left, right], key=str)
        if conditions is None:
            conditions = tuple()
        conditions = tuple(sorted(set(conditions), key=str))
        return cls(separated, left, right, conditions)

    def __bool__(self) -> bool:
        return self.separated

    @property
    def is_canonical(self) -> bool:
        """Return if the conditional independency is in canonical form."""
        return (
            self.left < self.right
            and isinstance(self.conditions, tuple)
            and tuple(sorted(self.conditions, key=str)) == (self.conditions)
        )

    def cressie_read(
        self, df: pd.DataFrame, boolean: bool = False, **kwargs
    ) -> Tuple[float, int, float]:
        """Calculate the Cressie Read statistic for conditional independence.

        :param df: A dataframe.
        :param boolean: Should results be returned as a pre-cutoff boolean?
        :param kwargs: Additional kwargs to pass to :func:`cressie_read`
        :returns:
            Tests the null hypothesis that X is independent of Y given Zs
            and returns a three-tuple of chi, dof, p_value
        :raises ValueError: if any parts of the judgement aren't in the dataframe's
            columns
        """
        from .util.stat_utils import cressie_read

        if self.left.name not in df.columns:
            raise ValueError(
                f"left variable {self.left.name} ({type(self.left.name)}) not in columns {list(df.columns)}"
            )
        if self.right.name not in df.columns:
            raise ValueError(
                f"right variable {self.right.name} ({type(self.right.name)}) not in columns {df.columns}"
            )
        for c in self.conditions:
            if c.name not in df.columns:
                raise ValueError(
                    f"conditional {c.name} ({type(c.name)}) not in columns {df.columns}"
                )

        return cressie_read(
            X=self.left.name,
            Y=self.right.name,
            Z={condition.name for condition in self.conditions},
            data=df,
            boolean=boolean,
            **kwargs,
        )
