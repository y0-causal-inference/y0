import copy
from collections import abc
from itertools import chain, combinations

import networkx as nx
import pandas as pd
from tqdm import tqdm

from ..util import CITests


class Result:
    """By default, acts like a boolean, but also caries evidence graph."""

    def __init__(self, separated, a, b, given, evidence):
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


def are_d_separated(G, a, b, *, given=frozenset()) -> Result:
    """Tests if nodes named by a & b are d-separated in G.
    Given conditions can be provided with the optional 'given' parameter.
    returns T/F and the final graph (as evidence)
    """
    named = set([a, b]).union(given)

    # Filter to ancestors
    keep = G.ancestors(named)
    G = copy.deepcopy(G.subgraph(keep))

    # Moralize (link parents of mentioned nodes)
    parents = [G.parents([v]) for v in G.vertices]
    augments = [*chain(*[combinations(nodes, 2) for nodes in parents if len(parents) > 1])]

    for edge in augments:
        G.add_udedge(*edge)

    # disorient & remove givens
    G2 = nx.Graph()
    G2.add_nodes_from(G.vertices)
    G2.add_edges_from(chain(G.di_edges, G.ud_edges, G.bi_edges))

    keep = set(G2.nodes) - set(given)
    G2 = G2.subgraph(keep)

    # check for path....
    separated = not nx.has_path(G2, a, b)   # If no path, then d-separated!

    return Result(separated, a, b, given, G2)


def all_combinations(source, min=0, max=None):
    """Get successively longer combinations of the source.
    source -- List to get combinations from
    min -- smallest combination to get (default 0)
    max -- Largest combination to get (None means length of the list and is the default)
    """
    max = len(source) if max is None else max
    return [*chain(*(combinations(source, size) for size in range(min, max + 1)))]


class Evidence(abc.Sequence):
    """A list of variables pairs that failed the d_sepration+covariance test.

    Has an extra 'evidence' property that is a dictionary.
      * Keys are the d-separated variable pairs
      * Values are the covariances measured between them.

    verbose -- print each combination as its checked
    max_given -- Longest list of 'given' variables to check
    """

    def __init__(self, failures, evidence):
        self._failures = failures
        self.evidence = evidence

    def __getitem__(self, i):
        return self._failures[i]

    def __len__(self):
        return len(self._failures)

    def __repr__(self):
        return repr(self._failures) + "+evidence"


def falsifications(G, df, significance_level=.05, max_given=None, verbose=False) -> Evidence:
    # TODO: Take G, [ConditionalIndependency...], df, etc. as params
    #       Test independencies passed
    # TODO: Make function G -> [ConditionalIndpeendency...]
    all_nodes = set(G.vertices)
    all_pairs = combinations(all_nodes, 2)

    to_test = [(a, b, given) for a, b in tqdm(all_pairs, desc="Checking d-separation")
               for given in all_combinations(all_nodes - {a, b}, max=max_given)
               if are_d_separated(G, a, b, given=given)]

    variances = {(a, b, given): CITests.cressie_read(a, b, given, df, boolean=False)
                 for a, b, given in tqdm(to_test, desc="Checking conditionals")}

    # TODO: Multiple-comparisons correction
    evidence = pd.DataFrame([(a, b, given, chi, p, dof)
                             for (a, b, given), (chi, dof, p)
                             in variances.items()],
                            columns=["A", "B", "Given", "chi^2", "p-value", "dof"])\
                 .pipe(lambda df: df.assign(flagged=(df["p-value"] < significance_level)))\
                 .sort_values(["flagged", "dof"], ascending=False)

    failures = evidence[evidence["flagged"]][["A", "B", "Given"]].apply(tuple, axis="columns")
    return Evidence(failures, evidence)
