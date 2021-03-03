# -*- coding: utf-8 -*-

"""An implementation to get conditional independencies of an ADMG."""

import copy
from itertools import combinations, chain, groupby
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

    separations = DSeparationJudgement.minimal(iter_d_separated(graph, max_given=max_given, verbose=verbose))
    independencies = {ConditionalIndependency.create(judgement.a, judgement.b, judgement.given)
                      for judgement in separations}
    return independencies


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

    @classmethod
    def minimal(cls, dseps, policy=None):
        """Given some d-separations, reduces to a 'minimal' collection.

        For indepdencies of the form A _||_ B | {C1, C2, ...} the minmal collection will:
             * Have only one indepdency with the same A/B nodes.
             * Have the smallest set of C nodes
             * For sets of C nodes of the same size, replacement is made according to the 'policy' argument

        The default replacement policy is lexicographic.
        """
        def grouper(dsep): return (dsep.a, dsep.b)
        def size_order(dsep): return len(dsep.given)
        def lex_order(dsep): return ",".join(dsep.given)

        def same_length_as(ref):
            return lambda other: len(ref.given) == len(other.given)

        dseps = sorted(dseps, key=grouper)
        groups = {k: sorted(vs, key=size_order)
                  for k, vs in groupby(dseps, grouper)}
        short_groups = {k: filter(same_length_as(vs[0]), vs)
                        for k, vs in groups.items()}
        instances = [sorted(grp, key=lex_order)[0] for grp in short_groups.values()]
        return instances


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
