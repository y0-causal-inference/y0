# -*- coding: utf-8 -*-

"""Causal graphs have implications that can be tested in the context of a specific dataset.

This module includes algorithms to perform those tests.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Union

import pandas as pd
import statsmodels.stats.multitest
from tqdm.auto import tqdm

from .conditional_independencies import get_conditional_independencies
from ..graph import NxMixedGraph
from ..struct import CITest, DSeparationJudgement, _ensure_method

__all__ = [
    "get_graph_falsifications",
    "get_falsifications",
    "Falsifications",
]


@dataclass
class Falsifications:
    """A list of variables pairs that failed the D-separation and covariance test.

    Has an extra 'evidence' property that is a dictionary.

    - Keys are the d-separated variable pairs
    - Values are the covariances measured between them.
    """

    #: Sequence of implications that did not pass
    failures: pd.Series
    #: Collection of all implications tested
    evidence: pd.DataFrame


def get_graph_falsifications(
    graph: NxMixedGraph,
    df: pd.DataFrame,
    *,
    significance_level: Optional[float] = None,
    max_given: Optional[int] = None,
    verbose: bool = False,
    method: Optional[CITest] = None,
    sep: Optional[str] = None,
) -> Falsifications:
    """Test conditional independencies implied by a graph.

    :param graph: An ADMG
    :param df: Data to check for consistency with a causal implications
    :param significance_level: Significance for p-value test
    :param max_given: The maximum set size in the power set of the vertices minus the d-separable pairs
    :param verbose: If true, use tqdm for status updates.
    :param method: Conditional independence from :mod:`pgmpy` to use. If none,
        defaults to :func:`pgmpy.estimators.CITests.cressie_read`.
    :param sep: The separator between givens when outputting the dataframe
    :return: Falsifications report
    """
    judgements = get_conditional_independencies(graph, max_conditions=max_given, verbose=verbose)
    return get_falsifications(
        judgements=judgements,
        df=df,
        significance_level=significance_level,
        verbose=verbose,
        method=method,
        sep=sep,
    )


def get_falsifications(
    judgements: Union[NxMixedGraph, Iterable[DSeparationJudgement]],
    df: pd.DataFrame,
    *,
    significance_level: Optional[float] = None,
    verbose: bool = False,
    method: Optional[CITest] = None,
    correction: Optional[str] = None,
    sep: Optional[str] = None,
) -> Falsifications:
    """Test conditional independencies implied by a list of D-separation judgements.

    :param judgements: A list of D-separation judgements to check.
    :param df: Data to check for consistency with a causal implications
    :param verbose: If true, use tqdm for status updates.
    :param method: Conditional independence from :mod:`pgmpy` to use. If none,
        defaults to :func:`pgmpy.estimators.CITests.cressie_read`.
    :param correction: Method used for multiple hypothesis test correction. Defaults to ``holm``.
        See :func:`statsmodels.stats.multitest.multipletests` for possible methods.
    :param significance_level: Significance for p-value test, applied after multiple hypothesis testing correction
    :param sep: The separator between givens when outputting the dataframe
    :return: Falsifications report
    """
    if significance_level is None:
        significance_level = 0.05
    if correction is None:
        correction = "holm"
    if sep is None:
        sep = "|"
    # Make this loop explicit for clarity
    results = []
    method = _ensure_method(method, df)
    for judgement in tqdm(judgements, disable=not verbose, desc="Checking conditionals"):
        result = judgement.test(df, method=method, boolean=False)
        results.append(
            (
                judgement.left.name,
                judgement.right.name,
                sep.join(c.name for c in judgement.conditions),
                result.statistic,
                result.p_value,
                result.dof,
            )
        )
    evidence_df = pd.DataFrame(
        results,
        columns=["left", "right", "given", "stats", "p", "dof"],
    )
    if not results:
        reject = []
        p_adj = []
    else:
        reject, p_adj, _, _ = statsmodels.stats.multitest.multipletests(
            evidence_df["p"],
            alpha=significance_level,
            method=correction,
        )
    evidence_df["p_adj"] = p_adj
    evidence_df["p_adj_significant"] = reject
    evidence_df.sort_values("p_adj", ascending=True, inplace=True)
    evidence_df = evidence_df.sort_values("p_adj")
    failures_df = evidence_df.loc[evidence_df["p_adj_significant"], ["left", "right", "given"]]
    failures = failures_df.apply(tuple, axis="columns")
    return Falsifications(failures, evidence_df)
