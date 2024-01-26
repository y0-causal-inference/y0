# -*- coding: utf-8 -*-

"""Data structures."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Literal, NamedTuple, Optional, Tuple, Union, cast

import pandas as pd

from .dsl import Expression, Variable

__all__ = [
    "VermaConstraint",
    "DSeparationJudgement",
]

DEFAULT_SIGNIFICANCE = 0.01


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


CITest = Literal[
    "pearson",
    "chi-square",
    "cressie_read",
    "freeman_tuckey",
    "g_sq",
    "log_likelihood",
    "modified_log_likelihood",
    "power_divergence",
    "neyman",
]
DEFAULT_CONTINUOUS_CI_TEST: CITest = "pearson"
DEFAULT_DISCRETE_CI_TEST: CITest = "cressie_read"

CITestFunc = Callable


@lru_cache
def get_conditional_independence_tests() -> dict[CITest, CITestFunc]:
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


class CITestTuple(NamedTuple):
    """A tuple containing the results from a PGMPy conditional independency test.

    Note that continuous tests such as :func:`pgmpy.estimators.CITests.pearsonr`
    do not have an associated _degrees of freedom_ (dof), so this field is set
    to none in those cases.
    """

    statistic: float
    p_value: float
    dof: Optional[float] = None


CITestResult = Union[CITestTuple, bool]


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
            and tuple(sorted(self.conditions, key=str)) == self.conditions
        )

    def test(
        self,
        df: pd.DataFrame,
        *,
        boolean: bool = False,
        method: Optional[CITest] = None,
        significance_level: Optional[float] = None,
        _method_checked: bool = False,
    ) -> Union[bool, CITestTuple]:
        """Test for conditional independence, given some data.

        :param df: A dataframe.
        :param boolean: Should results be returned as a pre-cutoff boolean?
        :param method: Conditional independence from :mod:`pgmpy` to use. If none,
            defaults to :func:`pgmpy.estimators.CITests.cressie_read`.
        :param significance_level:
            The statistical tests employ this value for
            comparison with the p-value of the test to determine the independence of
            the tested variables. If none, defaults to 0.01. Only applied if ``boolean=True``.
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
            if c.name in {self.left.name, self.right.name}:
                raise ValueError(f"conditional {c.name} repeats one of the primary arguments")
            if c.name not in df.columns:
                raise ValueError(
                    f"conditional {c.name} ({type(c.name)}) not in columns {df.columns}"
                )
        if significance_level is None:
            significance_level = DEFAULT_SIGNIFICANCE

        method = _ensure_method(
            method,
            df[[self.left.name, self.right.name, *(c.name for c in self.conditions)]],
            skip=_method_checked,
        )
        tests: dict[CITest, CITestFunc] = get_conditional_independence_tests()
        func: CITestFunc = tests[method]
        result = func(
            X=self.left.name,
            Y=self.right.name,
            Z={condition.name for condition in self.conditions},
            data=df,
            boolean=boolean,
            significance_level=significance_level,
        )
        if boolean:
            return cast(bool, result)
        # Person's correlation returns a pair with the first element being the Person's correlation
        # and the second being the p-value. The other methods return a triple with the first element
        # being the Chi^2 statistic, the second being the p-value, and the third being the degrees of
        # freedom.
        if method == "pearson":
            statistic, p_value = result
            dof = None
        else:
            statistic, p_value, dof = result
        return CITestTuple(statistic=statistic, p_value=p_value, dof=dof)


def _ensure_method(method: Optional[CITest], df: pd.DataFrame, skip: bool = False) -> CITest:
    if skip:
        if method is None:
            raise RuntimeError
        return method
    # TODO extend to discrete but more than 2.
    #  see https://stats.stackexchange.com/questions/12273/how-to-test-if-my-data-is-discrete-or-continuous
    # TODO what happens when some variables are binary but others are continous?
    binary = _is_binary(df)
    if method is None:
        if binary:
            return DEFAULT_DISCRETE_CI_TEST
        else:
            return DEFAULT_CONTINUOUS_CI_TEST
    elif binary and method == "pearson":
        raise ValueError(
            f"using continuous data test ({method}) on binary data: {_summarize_df(df)}"
        )
    elif not binary and method != "pearson":
        raise ValueError(f"using binary data test ({method}) on continuous data")
    return method


def _summarize_df(df: pd.DataFrame):
    return {column: set(df[column].unique()) for column in df.columns}


def _is_binary(df: pd.DataFrame) -> bool:
    column_to_type = {column: _is_two_values(df[column]) for column in df.columns}
    return all(column_to_type.values())


def _is_two_values(series):
    values = set(series.unique())
    return values == {True, False} or values == {1, 0} or values == {1, -1}
