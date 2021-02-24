# -*- coding: utf-8 -*-

"""An implementation to get conditional independencies of an ADMG."""

import copy
from itertools import combinations, chain
from typing import Set, Optional, Iterable, TypeVar, Collection

import networkx as nx
from tqdm import tqdm
from ananke.graphs import ADMG, SG

from ..struct import ConditionalIndependency

X = TypeVar('X')

__all__ = [
    'get_conditional_independencies',
]


def get_conditional_independencies(graph: ADMG,
                                   *,
                                   max_given: Optional[int] = None,
                                   verbose: bool = False) -> Set[ConditionalIndependency]:
    """Get the conditional independencies from the given ADMG.

    :param graph: An acyclic directed mixed graph
    :return: A set of conditional dependencies

    .. seealso:: Original issue https://github.com/y0-causal-inference/y0/issues/24
    """

    # TODO: This will list more constraints than correct.
    #   According to "On the Testable Implications of Causal Models with Hidden Variables"
    #   Jin Tian, Judea Pearl (2012), should only consider variables topolgoically 'before'
    #   in the constraint set.  This procedure looks at all variables.
    return {ConditionalIndependency.create(judgement.a, judgement.b, judgement.given)
            for judgement in iter_d_separated(graph, max_given=max_given, verbose=verbose)}


class DSeparationJudgement:
    """By default, acts like a boolean, but also caries evidence graph."""

    def __init__(self, separated: bool, a, b, given, evidence: Optional[nx.Graph] = None):
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


def powerset(iterable: Iterable[X],
             start: int = 0,
             stop: Optional[int] = None) -> Iterable[Collection[X]]:
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


def are_d_separated(graph: SG, a, b, *, given=frozenset()) -> DSeparationJudgement:
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

    return DSeparationJudgement(separated, a, b, given=given, evidence=evidence_graph)


def iter_d_separated(graph: SG,
                     *,
                     max_given: Optional[int] = None,
                     verbose: bool = False) -> Iterable[DSeparationJudgement]:

    verticies = set(graph.vertices)
    for a, b in tqdm(combinations(verticies, 2), disable=not verbose, desc="Checking d-separation"):
        for given in powerset(verticies - {a, b}, stop=max_given):
            if are_d_separated(graph, a, b, given=given):
                yield DSeparationJudgement(True, a, b, given)
