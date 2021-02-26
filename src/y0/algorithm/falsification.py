from collections import abc
from typing import Optional, TypeVar

import pandas as pd
from ananke.graphs import SG
from tqdm import tqdm

from .conditional_independencies import iter_d_separated
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
    significance_level: float = .05,
    max_given: Optional[int] = None,
    verbose: bool = False,
) -> Falsifications:
    """

    :param graph:
    :param df:
    :param significance_level:
    :param stop: The maximum set size in the powerset of the verticies minus the d-seperable pairs
    :param verbose:
    :return:
    """
    # TODO: Take G, [ConditionalIndependency...], df, etc. as params
    #       Test independencies passed
    # TODO: Make function G -> [ConditionalIndpeendency...]
    to_test = list(iter_d_separated(graph, max_given=max_given, verbose=verbose))

    variances = {
        (a, b, given): cressie_read(a, b, given, df, boolean=False)
        for a, b, given in tqdm(to_test, disable=not verbose, desc="Checking conditionals")
    }

    # TODO: Multiple-comparisons correction
    rows = [
        (a, b, given, chi, p, dof)
        for (a, b, given), (chi, dof, p) in variances.items()
    ]
    evidence = pd.DataFrame(rows, columns=["a", "b", "given", "chi^2", "p", "dof"]) \
        .pipe(lambda df: df.assign(flagged=(df["p-value"] < significance_level))) \
        .sort_values(["flagged", "dof"], ascending=False)

    failures = evidence[evidence["flagged"]][["A", "B", "Given"]].apply(tuple, axis="columns")
    return Falsifications(failures, evidence)
