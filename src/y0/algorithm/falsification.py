import copy
from collections import abc
from itertools import chain, combinations
from typing import Collection, Iterable, Optional, Sequence, Tuple, TypeVar

import networkx as nx
import pandas as pd
from ananke.graphs import SG
from tqdm import tqdm

from ..util.stat_utils import cressie_read

X = TypeVar('X')


class Result:
    """By default, acts like a boolean, but also caries evidence graph."""

    def __init__(self, separated: bool, a, b, given, evidence: nx.Graph):
        """separated -- T/F judgement
           a/b/given -- The question asked
           evidence -- The end graph
        """
        self.separated = separated
        self.a = a
        self.b = b
        self.given = given
        self.evidence = evidence

    def __bool__(self):
        return self.separated

    def __repr__(self):
        return f"{repr(self.separated)} ('{self.a}' d-sep '{self.b}' given {self.given})"

    def __eq__(self, other):
        return self.separated == other


def are_d_separated(graph: SG, a, b, *, given=frozenset()) -> Result:
    """Tests if nodes named by a & b are d-separated in G.

    Given conditions can be provided with the optional 'given' parameter.
    returns T/F and the final graph (as evidence)
    """
    named = {a, b}.union(given)

    # Filter to ancestors
    keep = graph.ancestors(named)
    graph = copy.deepcopy(graph.subgraph(keep))

    # Moralize (link parents of mentioned nodes)
    for u, v in get_augments(graph):
        graph.add_udedge(u, v)

    # disorient & remove givens
    evidence_graph = disorient(graph)

    keep = set(evidence_graph.nodes) - set(given)
    evidence_graph = evidence_graph.subgraph(keep)

    # check for path....
    separated = not nx.has_path(evidence_graph, a, b)  # If no path, then d-separated!

    return Result(separated, a, b, given=given, evidence=evidence_graph)


def disorient(graph: SG) -> nx.Graph:
    """Disorient the ananke segregated graph to a simple networkx graph."""
    rv = nx.Graph()
    rv.add_nodes_from(graph.vertices)
    rv.add_edges_from(chain(graph.di_edges, graph.ud_edges, graph.bi_edges))
    return rv


def get_augments(graph: SG):
    parents = [graph.parents([v]) for v in graph.vertices]
    augments = [*chain(*[combinations(nodes, 2) for nodes in parents if len(parents) > 1])]
    return augments


def powerset(iterable: Iterable[X], start: int = 0, stop: Optional[int] = None) -> Iterable[Collection[X]]:
    """Get successively longer combinations of the source.

    :param iterable: List to get combinations from
    :param start: smallest combination to get (default 0)
    :param stop: Largest combination to get (None means length of the list and is the default)

    .. seealso: :func:`more_iterools.powerset` for a non-constrainable implementation
    """
    s = list(iterable)
    if stop is None:
        stop = len(s) + 1
    return chain.from_iterable(combinations(s, r) for r in range(start, stop))


class Evidence(abc.Sequence):
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


EVIDENCE_COLUMNS = ["A", "B", "Given", "chi^2", "p-value", "dof"]


def iter_d_separated(
    graph: SG,
    *,
    stop: Optional[int] = None,
    verbose: bool = False,
) -> Iterable[Tuple[str, str, Sequence[str]]]:
    verticies = set(graph.vertices)
    for a, b in tqdm(combinations(verticies, 2), disable=not verbose, desc="Checking d-separation"):
        for given in powerset(verticies - {a, b}, stop=stop):
            if are_d_separated(graph, a, b, given=given):
                yield a, b, given


def falsifications(
    graph: SG,
    df: pd.DataFrame,
    significance_level: float = .05,
    stop: Optional[int] = None,
    verbose: bool = False,
) -> Evidence:
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
    to_test = list(iter_d_separated(graph, stop=stop, verbose=verbose))

    variances = {
        (a, b, given): cressie_read(a, b, given, df, boolean=False)
        for a, b, given in tqdm(to_test, disable=not verbose, desc="Checking conditionals")
    }

    # TODO: Multiple-comparisons correction
    rows = [
        (a, b, given, chi, p, dof)
        for (a, b, given), (chi, dof, p) in variances.items()
    ]
    evidence = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS) \
        .pipe(lambda df: df.assign(flagged=(df["p-value"] < significance_level))) \
        .sort_values(["flagged", "dof"], ascending=False)

    failures = evidence[evidence["flagged"]][["A", "B", "Given"]].apply(tuple, axis="columns")
    return Evidence(failures, evidence)
