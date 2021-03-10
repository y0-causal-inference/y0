# -*- coding: utf-8 -*-

"""Causal graphs have implications that can be tested in the context of a specific dataset.

This module includes algorithms to perform those tests.
"""

from collections import abc
from typing import Iterable, Optional, Union

import pandas as pd
from ananke.graphs import SG
from tqdm import tqdm

from .conditional_independencies import get_conditional_independencies
from ..struct import DSeparationJudgement
from ..util.stat_utils import cressie_read


class Falsifications(abc.Sequence):
    """A list of variables pairs that failed the D-separation and covariance test.

    Has an extra 'evidence' property that is a dictionary.

    - Keys are the d-separated variable pairs
    - Values are the covariances measured between them.
    """

    def __init__(self, failures, evidence: pd.DataFrame):
        """Create Falsifications result.

        :param failures: Sequence of implications that did not pass
        :param evidence: Collection of all implications tested
        """
        self._failures = failures
        self.evidence = evidence

    def __getitem__(self, i):
        return self._failures[i]

    def __len__(self):
        return len(self._failures)

    def __repr__(self):
        return repr(self._failures) + "+evidence"


def falsifications(
    to_test: Union[SG, Iterable[DSeparationJudgement]],
    df: pd.DataFrame,
    significance_level: float = .05,
    max_given: Optional[int] = None,
    verbose: bool = False,
) -> Falsifications:
    """Test conditional independencies implied by a graph.

    :param to_test: Either a graph to generate d-separation from or a list of D-separations to check.
    :param df: Data to check for consistency with a causal implications
    :param significance_level: Significance for p-value test
    :param max_given: The maximum set size in the power set of the vertices minus the d-separable pairs
    :param verbose: If true, use tqdm for status updates.
    :return: Falsifications report
    """
    if isinstance(to_test, SG):
        to_test = get_conditional_independencies(to_test, max_conditions=max_given, verbose=verbose)

    variances = {
        (left, right, given): cressie_read(left, right, given, df, boolean=False)
        for _, left, right, given in tqdm(to_test, disable=not verbose, desc="Checking conditionals")
    }

    rows = [
        (left, right, given, chi, p, dof)
        for (left, right, given), (chi, dof, p) in variances.items()
    ]

    evidence = (
        pd.DataFrame(rows, columns=["left", "right", "given", "chi^2", "p", "dof"])
        .sort_values("p")
        .assign(**{"Holm–Bonferroni level": significance_level / pd.Series(range(len(rows) + 1, 0, -1))})
        .pipe(_assign_flags)
        .sort_values(["flagged", "dof"], ascending=False)
    )

    failures = evidence[evidence["flagged"]][["left", "right", "given"]].apply(tuple, axis="columns")
    return Falsifications(failures, evidence)


def _assign_flags(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(flagged=(df["p"] < df["Holm–Bonferroni level"]))
