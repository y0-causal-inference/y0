from collections import abc
from typing import Optional, TypeVar

import pandas as pd
from ananke.graphs import SG
from tqdm import tqdm

from .conditional_independencies import get_conditional_independencies
from ..util.stat_utils import cressie_read

X = TypeVar('X')


class Falsifications(abc.Sequence):
    """A list of variables pairs that failed the d_sepration+covariance test.

    Has an extra 'evidence' property that is a dictionary.
      * Keys are the d-separated variable pairs
      * Values are the covariances measured between them.

    verbose -- print each combination as its checked
    max_given -- Longest list of 'given' variables to check
    """

    def __init__(self, failures, evidence: pd.DataFrame):
        self._failures = failures
        self.evidence = evidence

    def __getitem__(self, i):
        return self._failures[i]

    def __len__(self):
        return len(self._failures)

    def __repr__(self):
        return repr(self._failures) + "+evidence"


def falsifications(
    graph: SG,
    df: pd.DataFrame,
    to_test: Optional = None,
    significance_level: float = .05,
    max_given: Optional[int] = None,
    verbose: bool = False,
) -> Falsifications:
    """

    :param graph: Graph to check for consistency with the data
    :param df: Data to check for consistency with the graph
    :param significance_level: Significance for p-value test
    :param stop: The maximum set size in the powerset of the verticies minus the d-seperable pairs
    :param verbose:
    :return:
    """
    def _unwrap(vs): return [v.name for v in vs]

    if to_test is None:
        to_test = list(get_conditional_independencies(graph, max_given=max_given, verbose=verbose))

    variances = {
        (left, right, given): cressie_read(left.name, right.name, _unwrap(given), df, boolean=False)
        for left, right, given in tqdm(to_test, disable=not verbose, desc="Checking conditionals")
    }

    rows = [
        (left, right, given, chi, p, dof)
        for (left, right, given), (chi, dof, p) in variances.items()
    ]

    evidence = pd.DataFrame(rows, columns=["left", "right", "given", "chi^2", "p", "dof"])\
        .sort_values("p")\
        .assign(**{"Holm–Bonferroni level": significance_level / pd.Series(range(len(rows) + 1, 0, -1))})\
        .pipe(lambda df: df.assign(flagged=(df["p"] < df["Holm–Bonferroni level"])))\
        .sort_values(["flagged", "dof"], ascending=False)

    failures = evidence[evidence["flagged"]][["left", "right", "given"]].apply(tuple, axis="columns")
    return Falsifications(failures, evidence)
