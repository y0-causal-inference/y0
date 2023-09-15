# -*- coding: utf-8 -*-

"""Causal graphs have implications that can be tested in the context of a specific dataset.

This module includes algorithms to perform those tests.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Union

import pandas as pd
from tqdm.auto import tqdm

from .conditional_independencies import get_conditional_independencies
from ..graph import NxMixedGraph
from ..struct import DSeparationJudgement

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
    method: Optional[str] = None,
) -> Falsifications:
    """Test conditional independencies implied by a graph.

    :param graph: An ADMG
    :param df: Data to check for consistency with a causal implications
    :param significance_level: Significance for p-value test
    :param max_given: The maximum set size in the power set of the vertices minus the d-separable pairs
    :param verbose: If true, use tqdm for status updates.
    :param method: Conditional independence from :mod:`pgmpy` to use. If none,
        defaults to :func:`pgmpy.estimators.CITests.cressie_read`.
    :return: Falsifications report
    """
    judgements = get_conditional_independencies(graph, max_conditions=max_given, verbose=verbose)
    return get_falsifications(
        judgements=judgements,
        df=df,
        significance_level=significance_level,
        verbose=verbose,
        method=method,
    )


HB_LEVEL_NAME = "Holm–Bonferroni level"


def get_falsifications(
    judgements: Union[NxMixedGraph, Iterable[DSeparationJudgement]],
    df: pd.DataFrame,
    *,
    significance_level: Optional[float] = None,
    verbose: bool = False,
    method: Optional[str] = None,
) -> Falsifications:
    """Test conditional independencies implied by a list of D-separation judgements.

    :param judgements: A list of D-separation judgements to check.
    :param df: Data to check for consistency with a causal implications
    :param significance_level: Significance for p-value test
    :param verbose: If true, use tqdm for status updates.
    :param method: Conditional independence from :mod:`pgmpy` to use. If none,
        defaults to :func:`pgmpy.estimators.CITests.cressie_read`.
    :return: Falsifications report
    """
    if significance_level is None:
        significance_level = 0.05
    variances = {
        judgement: judgement.test(df, test=method)
        for judgement in tqdm(judgements, disable=not verbose, desc="Checking conditionals")
    }
    evidence = pd.DataFrame(
        [
            (
                judgement.left.name,
                judgement.right.name,
                "|".join(c.name for c in judgement.conditions),
                chi,
                p,
                dof,
            )
            for judgement, (chi, p, dof) in variances.items()
        ],
        columns=["left", "right", "given", "chi^2", "p", "dof"],
    )
    evidence.sort_values("p", ascending=True, inplace=True)
    evidence = (
        evidence.assign(
            **{HB_LEVEL_NAME: significance_level / pd.Series(range(len(evidence.index) + 1, 0, -1))}
        )
        .pipe(_assign_flags)
        .sort_values(["flagged", "dof"], ascending=False)
    )

    failures_df = evidence.loc[evidence["flagged"], ["left", "right", "given"]]
    failures = failures_df.apply(tuple, axis="columns")
    return Falsifications(failures, evidence)


def _assign_flags(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(flagged=(df["p"] < df[HB_LEVEL_NAME]))
