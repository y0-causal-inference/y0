# -*- coding: utf-8 -*-

"""Data structures."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, NamedTuple, Optional, Tuple, Union

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

        :param element: An element in the vector returned by `verma.constraint`
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


@lru_cache
def get_conditional_independence_tests():
    """Get the conditional independence tests from :mod:`pgmpy.estimators.CITests`."""
    try:
        from pgmpy.estimators import CITests
    except ImportError as e:
        raise ImportError("Calculating falsifications requires `pip install pgmpy`.") from e

    return {
        "pearson": CITests.pearsonr,
        "chi-square": CITests.chi_square,
        "cressie_read": CITests.cressie_read,
        "freeman_tuckey": CITests.freeman_tuckey,
        "g_sq": CITests.g_sq,
        "log_likelihood": CITests.log_likelihood,
        "modified_log_likelihood": CITests.modified_log_likelihood,
        "power_divergence": CITests.power_divergence,
        "neyman": CITests.neyman,
    }


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

    def test(
        self, df: pd.DataFrame, boolean: bool = False, method: Optional[str] = None, **kwargs
    ) -> Union[Tuple[float, int, float], bool]:
        """Test for conditional independence, given some data.

        :param df: A dataframe.
        :param boolean: Should results be returned as a pre-cutoff boolean?
        :param method: Conditional independence from :mod:`pgmpy` to use. If none,
            defaults to :func:`pgmpy.estimators.CITests.cressie_read`.
        :param kwargs: Additional kwargs to pass to the estimator function
        :returns:
            Tests the null hypothesis that X is independent of Y given Zs.
            If ``boolean=False``, returns a three-tuple of chi, dof, p_value.
            If ``boolean=True``, make sure you also set ``significance_level=0.05`` or your preferred
            value, then returns simply a boolean if the test fails.
        :raises ValueError: if any parts of the judgement aren't in the dataframe's
            columns
        """
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

        tests = get_conditional_independence_tests()
        func = tests[method or "cressie_read"]
        return func(
            X=self.left.name,
            Y=self.right.name,
            Z={condition.name for condition in self.conditions},
            data=df,
            boolean=boolean,
            **kwargs,
        )
